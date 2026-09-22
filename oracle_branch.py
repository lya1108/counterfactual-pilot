#!/usr/bin/env python3
"""Stage-2: does any single discrete action, inserted at the branch point, actually rescue the grasp?

Pilot 1 compared Gemma's proposal against `reference_correction`, which is a human guess, not ground
truth. There are only 8 actions and the simulator is deterministic, so instead of inventing a better
label we enumerate: run all 8 (times several durations) from the recorded branch point and let
robosuite say which ones recover. That produces the real answer set per frame, and it also answers
the prior question — whether the frame is recoverable by one action *at all*.

Branch protocol, per frame:

    restore state at t_commit-1   (the exact state behind the image Gemma was shown)
    → apply one discrete action for k steps
    → close the gripper (CLOSE_HOLD steps)
    → lift (LIFT_STEPS steps)
    → robosuite's own Lift._check_success()

Handing control to a *fresh* nominal controller instead was rejected: it re-approaches from scratch,
so it recovers from almost any state and all 8 actions would score alike. Closing and lifting
immediately is what makes the actions separable, and it matches the question actually being asked —
"instead of what it did next, what should it have done at this moment".

`NO_CHANGE` (straight to close+lift, no correction) is the control. It must fail on every frame: if
it does not, the branch is not reproducing the recorded failure and nothing else here is meaningful.

Nothing from ~/바탕화면/jungbeom is imported. The snapshot/restore recipe is the 4-part one verified
there (`fasi/envs/robosuite_env.py:163-185`, `docs/environment.md:41-43`), restated below.

    conda activate fasi && export MUJOCO_GL=egl
    python oracle_branch.py --dry-run
    python oracle_branch.py
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

import collect_failures as cf  # our own stage-1 file; constants and the fault specs live there  # noqa: E402
from prompts import ACTIONS  # noqa: E402

NO_CHANGE = "NO_CHANGE"

# label -> (axis index, sign) in world coordinates. World +y is image-right and +x points away from
# the robot base toward the camera; see prompts.py for how that maps onto the agentview image.
AXIS: dict[str, tuple[int, int]] = {
    "MOVE_FORWARD": (0, +1), "MOVE_BACKWARD": (0, -1),
    "MOVE_RIGHT": (1, +1), "MOVE_LEFT": (1, -1),
    "MOVE_UP": (2, +1), "MOVE_DOWN": (2, -1),
}
GRIP: dict[str, float] = {"OPEN_GRIPPER": -1.0, "CLOSE_GRIPPER": +1.0}


def make_action(label: str, hold_grip: float) -> np.ndarray:
    """A full-scale delta on one axis (±1 -> ±0.05 m per step under OSC_POSE control_delta).

    Translation labels keep whatever the faulty controller was commanding the gripper to do;
    gripper labels change the gripper and command no translation.
    """
    a = np.zeros(7, dtype=float)
    if label in AXIS:
        axis, sign = AXIS[label]
        a[axis] = float(sign)
        a[6] = hold_grip
    else:
        a[6] = GRIP[label]
    return a


# ------------------------------------------------------------------ snapshot / restore
def snapshot(env) -> dict:
    return {
        "sim_state": np.array(env.sim.get_state().flatten(), copy=True),
        "timestep": int(env.timestep),
        "cur_time": float(env.cur_time),
        "done": bool(env.done),
        "gripper_action": np.array(env.robots[0].gripper.current_action, copy=True),
        "qacc_warmstart": np.array(env.sim.data.qacc_warmstart, copy=True),
    }


def restore(env, s: dict) -> dict:
    env.sim.set_state_from_flattened(s["sim_state"].copy())
    env.sim.forward()
    env.timestep, env.cur_time, env.done = s["timestep"], s["cur_time"], s["done"]
    env.robots[0].gripper.current_action = s["gripper_action"].copy()
    env.sim.data.qacc_warmstart[:] = s["qacc_warmstart"]
    env._update_observables(force=True)
    return env._get_observations(force_update=True)


# ------------------------------------------------------------------ replay to the branch point
def replay_to_branch(env, spec: cf.FailureSpec, seed: int) -> tuple[dict, dict]:
    """Re-run the faulty controller and stop at the state the captured image was rendered from.

    Mirrors `collect_failures.run_episode`'s controller exactly; the caller verifies the result
    against the recorded metadata before anything is branched off it.
    """
    np.random.seed(int(seed))
    obs = env.reset()
    hold = np.zeros(7, dtype=float)
    hold[6] = -1.0
    for _ in range(cf.SETTLE_STEPS):
        obs, _r, _d, _i = env.step(hold)
    cube0 = np.array(obs["cube_pos"], dtype=float)

    phase, phase_t, close_t = "hover", 0, 0
    grip = spec.grip_on_approach
    t_commit: int | None = None
    prev: dict | None = None  # state/action at step t-1

    for t in range(cf.MAX_STEPS):
        eef = np.array(obs["robot0_eef_pos"], dtype=float)
        snap = snapshot(env)  # state at the start of step t

        if phase == "hover":
            target = cube0 + np.array([0.0, spec.y_offset, cf.HOVER_DZ])
            if np.linalg.norm(target - eef) < cf.POS_TOL or phase_t > cf.PHASE_BUDGET:
                phase, phase_t = "descend", 0
        elif phase == "descend":
            target = cube0 + np.array([0.0, spec.y_offset, spec.descend_dz])
            if t_commit is None and spec.grip_on_approach > 0 and (eef[2] - cube0[2]) <= 0.05:
                t_commit = t
            if np.linalg.norm(target - eef) < cf.POS_TOL or phase_t > cf.PHASE_BUDGET:
                phase, phase_t = "close", 0
        elif phase == "close":
            target = eef
            grip = 1.0
            if t_commit is None:
                t_commit = t
            close_t += 1
            if close_t >= cf.CLOSE_HOLD:
                phase, phase_t = "lift", 0
        else:
            target = cube0 + np.array([0.0, 0.0, cf.LIFT_DZ])
            grip = 1.0
            if phase_t > cf.LIFT_STEPS:
                break

        # the captured frame is the render of the state at t_commit-1, so that state is the branch
        if t_commit == t and prev is not None:
            return prev, {"cube0": cube0, "t_commit": t_commit}

        action = np.zeros(7, dtype=float)
        action[:3] = np.clip(cf.KP * (target - eef), -1.0, 1.0)
        action[6] = grip
        prev = {
            "snap": snap, "timestep": t, "eef_pos": [round(float(v), 5) for v in eef],
            "cube_pos": [round(float(v), 5) for v in cube0],
            "original_action": [round(float(v), 5) for v in action],
            "grip": float(grip),
        }
        obs, _r, _d, _i = env.step(action)
        phase_t += 1

    raise RuntimeError(f"{spec.name} seed={seed}: never reached a commit step")


# ------------------------------------------------------------------ one branched rollout
def run_branch(env, snap: dict, cube0: np.ndarray, hold_grip: float, label: str, k: int,
               second: tuple[str, int] | None = None) -> dict:
    obs = restore(env, snap)

    steps: list[tuple[str, int]] = [] if label == NO_CHANGE else [(label, k)]
    if second is not None:
        steps.append(second)
    for lab, n in steps:
        a = make_action(lab, hold_grip)
        for _ in range(n):
            obs, _r, _d, _i = env.step(a)

    eef = np.array(obs["robot0_eef_pos"], dtype=float)  # hold position while the fingers close
    close = np.zeros(7, dtype=float)
    close[6] = 1.0
    for _ in range(cf.CLOSE_HOLD):
        cur = np.array(obs["robot0_eef_pos"], dtype=float)
        close[:3] = np.clip(cf.KP * (eef - cur), -1.0, 1.0)
        obs, _r, _d, _i = env.step(close)

    grasped_after_close = bool(env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.cube))

    lift_target = cube0 + np.array([0.0, 0.0, cf.LIFT_DZ])
    lift = np.zeros(7, dtype=float)
    lift[6] = 1.0
    for _ in range(cf.LIFT_STEPS):
        cur = np.array(obs["robot0_eef_pos"], dtype=float)
        lift[:3] = np.clip(cf.KP * (lift_target - cur), -1.0, 1.0)
        obs, _r, _d, _i = env.step(lift)

    return {
        "action": label, "k": k,
        "action2": None if second is None else second[0], "k2": None if second is None else second[1],
        "success": bool(env._check_success()),
        "grasped": grasped_after_close,
        "final_cube_z": round(float(env.sim.data.body_xpos[env.cube_body_id][2]), 5),
    }


# ------------------------------------------------------------------ main
def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", default=str(here / "images_r2"),
                    help="frames to branch from; must have been collected with hard_reset on")
    ap.add_argument("--results-dir", default=str(here / "results"))
    ap.add_argument("--k", default="1,2,3,5,10", help="how many steps the corrective action is held")
    ap.add_argument("--pairs", action="store_true",
                    help="for frames no single action rescues, also sweep all 64 ordered action "
                         "pairs — separates 'unrecoverable frame' from 'one action cannot say it'")
    ap.add_argument("--k1", default="5,10,14", help="--pairs: steps for the first action")
    ap.add_argument("--k2", default="3", help="--pairs: steps for the second action")
    ap.add_argument("--image-size", type=int, default=256,
                    help="must match collection: render resolution perturbs the trajectory")
    ap.add_argument("--camera", default="agentview")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ks = [int(x) for x in args.k.split(",") if x.strip()]
    k1s = [int(x) for x in args.k1.split(",") if x.strip()]
    k2s = [int(x) for x in args.k2.split(",") if x.strip()]
    images_dir, results_dir = Path(args.images_dir), Path(args.results_dir)
    meta = {json.loads(l)["episode_id"]: json.loads(l)
            for l in (images_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    plan = cf.failure_plan()
    labels = [NO_CHANGE] + list(ACTIONS)
    n_roll = len(meta) * (1 + len(ACTIONS) * len(ks))

    if args.dry_run:
        print(f"[dry-run] frames      : {len(meta)} from {images_dir}")
        print(f"[dry-run] labels      : {len(labels)} ({NO_CHANGE} control + {len(ACTIONS)} actions)")
        print(f"[dry-run] k values    : {ks} steps held "
              f"(±1 is the OSC command cap, not displacement: ramp_ratio 0.2 means k=1 moves "
              f"~0.004 m and k=5 ~0.03-0.05 m)")
        if args.pairs:
            print(f"[dry-run] pair sweep  : k1={k1s} k2={k2s} over {len(ACTIONS)**2} ordered pairs, "
                  f"only for frames no single action rescues")
        print(f"[dry-run] rollouts    : {n_roll}")
        print(f"[dry-run] protocol    : restore(t_commit-1) -> action x k -> close x {cf.CLOSE_HOLD} "
              f"-> lift x {cf.LIFT_STEPS} -> _check_success()")
        print("[dry-run] OK — no environment was created.")
        return

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"oracle_{run_id}.jsonl"

    env = cf.make_env(args.image_size, args.camera, hard_reset=True)
    rows: list[dict] = []
    try:
        for ep, (spec, seed) in enumerate(plan):
            if ep not in meta:
                continue
            m = meta[ep]
            branch, info = replay_to_branch(env, spec, seed)

            # Refuse to branch off a state that is not the one behind the recorded image.
            mismatch = [key for key in ("timestep", "eef_pos", "cube_pos", "original_action")
                        if branch[key] != m[key]]
            if mismatch:
                raise SystemExit(
                    f"ep{ep:02d} {spec.name}: replayed branch state does not match metadata "
                    f"({', '.join(mismatch)}). Collect with hard_reset on and the same --image-size."
                )

            per_ep = []
            for label in labels:
                for k in (ks if label != NO_CHANGE else [0]):
                    r = run_branch(env, branch["snap"], info["cube0"], branch["grip"], label, k)
                    r.update({"episode_id": ep, "failure_type": spec.name, "reset_seed": seed,
                              "timestep": branch["timestep"],
                              "original_action_discrete": m["original_action_discrete"],
                              "reference_correction": m["reference_correction"], "run_id": run_id})
                    rows.append(r)
                    per_ep.append(r)

            ctrl = next(r for r in per_ep if r["action"] == NO_CHANGE)
            wins = sorted({r["action"] for r in per_ep if r["success"] and r["action"] != NO_CHANGE})
            flag = "" if not ctrl["success"] else "   !! control succeeded — branch is not the failure"
            print(f"  ep{ep:02d} {spec.name:22s} t={branch['timestep']:3d}  "
                  f"control={'SUCCESS' if ctrl['success'] else 'fail':7s}  "
                  f"recovering actions: {', '.join(wins) if wins else 'none'}{flag}")

            if args.pairs and not wins:
                pair_wins = []
                for l1 in ACTIONS:
                    for l2 in ACTIONS:
                        for k1 in k1s:
                            for k2 in k2s:
                                r = run_branch(env, branch["snap"], info["cube0"], branch["grip"],
                                               l1, k1, (l2, k2))
                                r.update({"episode_id": ep, "failure_type": spec.name,
                                          "reset_seed": seed, "timestep": branch["timestep"],
                                          "original_action_discrete": m["original_action_discrete"],
                                          "reference_correction": m["reference_correction"],
                                          "run_id": run_id})
                                rows.append(r)
                                if r["success"]:
                                    pair_wins.append((l1, k1, l2, k2))
                firsts = sorted({p[0] for p in pair_wins})
                print(f"       two-action sweep: {len(pair_wins)} rescuing sequences; "
                      f"viable first actions: {', '.join(firsts) if firsts else 'none'}")
    finally:
        env.close()

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summarize(rows, ks)
    print(f"\nresults -> {out_path}")


def summarize(rows: list[dict], ks: list[int]) -> None:
    eps = sorted({r["episode_id"] for r in rows})
    print(f"\n--- recoverability ({len(rows)} rollouts) ---")
    n_rec = 0
    for ep in eps:
        rs = [r for r in rows if r["episode_id"] == ep]
        ft = rs[0]["failure_type"]
        wins = sorted({r["action"] for r in rs
                       if r["success"] and r["action"] != NO_CHANGE and r["action2"] is None})
        grasp_only = sorted({r["action"] for r in rs if r["grasped"] and not r["success"]})
        n_rec += bool(wins)
        print(f"  ep{ep:02d} {ft:22s} recovers: {', '.join(wins) if wins else '(none)':40s} "
              f"ref={rs[0]['reference_correction']}")
        if not wins and grasp_only:
            print(f"       grasped but did not lift: {', '.join(grasp_only)}")
    print(f"\nframes recoverable by one action : {n_rec}/{len(eps)}")

    print("\n--- per action (how many frames it rescues, at any k) ---")
    for a in [NO_CHANGE] + list(ACTIONS):
        hits = sorted({r["episode_id"] for r in rows
                       if r["action"] == a and r["success"] and r["action2"] is None})
        print(f"  {a:14s} {len(hits)}/{len(eps)}  {hits if hits else ''}")

    print("\n--- per k (does the discrete vocabulary need the right magnitude?) ---")
    for k in ks:
        wins = {(r["episode_id"], r["action"]) for r in rows
                if r["k"] == k and r["success"] and r["action2"] is None}
        print(f"  k={k:<3d} steps   rescuing (frame, action) pairs: {len(wins)}")

    ref_ok = sorted({r["episode_id"] for r in rows
                     if r["success"] and r["action"] == r["reference_correction"]})
    print(f"\nreference_correction actually recovers on {len(ref_ok)}/{len(eps)} frames: {ref_ok}")


if __name__ == "__main__":
    main()
