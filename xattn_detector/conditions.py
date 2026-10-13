"""Per-environment-step state logging, and the condition-violation onsets derived from it.

Two things are deliberately separated.

`StateLogger` records raw quantities every environment step -- object poses, contact flags, gripper opening,
joint margins, goal predicates. It decides nothing.

`evaluate` turns a recorded episode into per-condition boolean series and onsets, with thresholds passed in.
Keeping the thresholds out of collection means they can be revised from the measured distributions without
re-running a single rollout. That matters here: an earlier stage of this project set a gripper threshold from
the commanded action rather than the actual finger gap and mislabelled 9 of 80 frames, and the only reason it
was caught was that the raw gap had been recorded.

A condition onset is NOT the moment the episode became unrecoverable. It is the first step at which a named,
automatically checkable predicate held for `debounce` consecutive steps. An episode can register several, in
any order, and can register one and still succeed. They are stored per condition and never merged; `earliest`
is computed on demand, not stored as if it were "the" failure time.

Conditions (all from LiberoTask's own readers, no external judge):
  C1 grasp_empty_close   fingers closed with nothing between them
  C2 drop                an object that was held is resting on something again, no longer touched
  C3 wrong_object_grasp  the gripper (or hand) contacts an object no goal predicate mentions -- contact, not necessarily a grasp
  C4 object_tipped       an object's up-axis has rotated past a threshold from where it started
  C6 joint_limit         a joint is near its range end, or the wrist has wound up
"""
from __future__ import annotations

import numpy as np

CONDITIONS = ("C1_grasp_empty_close", "C2_drop", "C3_nontarget_contact", "C4_object_tipped", "C6_joint_limit")

DEFAULTS = dict(
    close_gap=0.04,      # opening() below this counts as "closed" -- MEASURE before trusting
    tilt_deg=45.0,       # object up-axis rotation from its initial orientation
    joint_margin_deg=5.0,
    wrist_deg=150.0,
    held_steps=3,        # how long an object must have been held before letting go counts as a drop
    debounce=3,          # consecutive steps a condition must hold before its onset is recorded
)


def quat2up(q):
    """Body z-axis in world coordinates, from a (w, x, y, z) quaternion."""
    w, x, y, z = q
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


class StateLogger:
    """Collects one episode's raw per-step state. Records; does not judge."""

    def __init__(self, task, max_objects: int = 8):
        self.task = task
        self.names = list(task.e.objects_dict.keys())[:max_objects]
        self.n = len(self.names)
        self.goal_names = [" ".join(p) for p in task.goal()]
        # objects any goal predicate mentions -- everything else is a bystander
        mentioned = {tok for p in task.goal() for tok in p}
        self.goal_objects = [o for o in self.names if o in mentioned]
        self.rows = []
        self.q0 = None

    def _object_flags(self):
        """gripper-contact and support flags for every object, from one pass over the contact list."""
        e = self.task.e
        m, d = e.sim.model, e.sim.data
        geom_of = {}
        for i, name in enumerate(self.names):
            for g in self.task.object_geoms(name):
                geom_of[g] = i
        touch = np.zeros(self.n, bool)
        support = np.zeros(self.n, bool)
        robot = np.zeros(self.n, bool)
        for k in range(d.ncon):
            c = d.contact[k]
            for ga, gb in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                i = geom_of.get(ga)
                if i is None:
                    continue
                other = m.body_id2name(m.geom_bodyid[gb]) or ""
                if "gripper" in other or "finger" in other:
                    touch[i] = True
                    robot[i] = True
                elif "robot" in other:
                    robot[i] = True
                elif gb not in geom_of:
                    support[i] = True  # resting against something that is not another tracked object
                else:
                    support[i] = True
        return touch, support, robot

    def record(self, step: int):
        t = self.task
        pos = np.zeros((self.n, 3), np.float32)
        quat = np.zeros((self.n, 4), np.float32)
        for i, name in enumerate(self.names):
            pos[i] = t.pos(name)
            quat[i] = t.quat(name)
        if self.q0 is None:
            self.q0 = quat.copy()
        touch, support, robot = self._object_flags()
        self.rows.append(dict(
            step=step, eef=t.eef().astype(np.float32),
            opening=np.float32(t.opening()),
            joint_margin=t.joint_margin_deg().astype(np.float32),
            wrist=np.float32(t.wrist_deg()),
            pos=pos, quat=quat, touch=touch, support=support, robot=robot,
            goal=np.array([t.predicate(p) for p in t.goal()], bool),
        ))

    def pack(self) -> dict:
        r = self.rows
        stack = lambda k: np.stack([x[k] for x in r])
        return dict(step=np.array([x["step"] for x in r], np.int32),
                    eef=stack("eef"), opening=stack("opening"), joint_margin=stack("joint_margin"),
                    wrist=stack("wrist"), obj_pos=stack("pos"), obj_quat=stack("quat"),
                    obj_touch=stack("touch"), obj_support=stack("support"), obj_robot=stack("robot"),
                    goal=stack("goal"), obj_quat0=self.q0,
                    object_names=np.array(self.names), goal_objects=np.array(self.goal_objects),
                    goal_names=np.array(self.goal_names))


# ---------------------------------------------------------------------------- derivation
def _onset(series: np.ndarray, debounce: int):
    """First index at which `series` has been true for `debounce` consecutive steps, else None."""
    run = 0
    for i, v in enumerate(series):
        run = run + 1 if v else 0
        if run >= debounce:
            return int(i - debounce + 1)
    return None


def evaluate(ep: dict, cfg: dict | None = None):
    """-> (per-condition boolean series, per-condition onset index or None).

    Indices are into the episode's own `step` array, so an onset converts to an environment step with
    `ep['step'][onset]`.
    """
    c = {**DEFAULTS, **(cfg or {})}
    names = [str(x) for x in ep["object_names"]]
    goal_objs = {str(x) for x in ep["goal_objects"]}
    touch, support = ep["obj_touch"], ep["obj_support"]
    T = len(ep["step"])
    series = {}

    closed = ep["opening"] < c["close_gap"]
    series["C1_grasp_empty_close"] = closed & ~touch.any(1)

    held = touch & ~support
    held_run = np.zeros_like(held, bool)
    for i in range(held.shape[1]):
        run = 0
        for t in range(T):
            run = run + 1 if held[t, i] else 0
            held_run[t, i] = run >= c["held_steps"]
    drop = np.zeros(T, bool)
    for t in range(1, T):
        for i in range(held.shape[1]):
            if held_run[t - 1, i] and not held[t, i] and support[t, i]:
                drop[t] = True
    series["C2_drop"] = drop

    bystander = np.array([n not in goal_objs for n in names], bool)
    series["C3_nontarget_contact"] = (touch & bystander[None, :]).any(1)

    up0 = np.stack([quat2up(q) for q in ep["obj_quat0"]])
    tilt = np.zeros((T, len(names)), np.float32)
    for t in range(T):
        for i in range(len(names)):
            u = quat2up(ep["obj_quat"][t, i])
            tilt[t, i] = np.degrees(np.arccos(np.clip(np.dot(u, up0[i]), -1, 1)))
    series["C4_object_tipped"] = (tilt > c["tilt_deg"]).any(1)

    series["C6_joint_limit"] = ((ep["joint_margin"].min(1) < c["joint_margin_deg"]) |
                                (np.abs(ep["wrist"]) > c["wrist_deg"]))

    onsets = {k: _onset(v, c["debounce"]) for k, v in series.items()}
    return series, onsets


def earliest_onset(onsets: dict):
    """The first condition to fire, as (name, index) -- computed on request, never stored as 'the' failure."""
    live = {k: v for k, v in onsets.items() if v is not None}
    if not live:
        return None, None
    k = min(live, key=live.get)
    return k, live[k]
