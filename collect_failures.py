#!/usr/bin/env python3
"""Stage-1 pilot: collect pre-failure frames from a scripted Lift controller with injected faults.

This file is self-contained on purpose. Nothing from ~/바탕화면/jungbeom is imported or modified;
the settings below were *read* from that workspace and from the installed robosuite and are
restated here explicitly:

* env kwargs mirror ``fasi/envs/robosuite_env.py:92-107`` (RobosuiteSnapshotEnv.__init__):
  OSC_POSE controller, has_renderer=False, offscreen renderer on, control_freq=20,
  horizon=10**9 + ignore_done=True (episodes are budgeted here, not by robosuite),
  hard_reset=False, reward_shaping=False.
* the ``[::-1]`` vertical flip is robosuite's render convention, per ``robosuite_env.py:132-139``.
* seeding a reset through the global ``np.random`` state is per ``robosuite_env.py:142-147``.
* success is robosuite's own criterion, not ours: ``Lift._check_success()`` in
  robosuite/environments/manipulation/lift.py:417-428 is ``cube_z > table_offset_z + 0.04``
  with ``table_offset = (0, 0, 0.8)``, i.e. cube centre above 0.84 m.
* the optional ``--save-sim-state`` recipe (4 parts) is from ``robosuite_env.py:163-173`` and
  ``docs/environment.md:41-43``, where restoring all four made re-execution bit-identical.

Failure is never assumed: every episode runs to its budget and the frame is kept only if
``_check_success()`` is False at the end.

Run (conda fasi, MUJOCO_GL=egl):
    python collect_failures.py --dry-run
    python collect_failures.py
"""

from __future__ import annotations

import argparse
import json
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from prompts import ACTIONS  # noqa: E402

# --------------------------------------------------------------------------- scripted controller
KP = 10.0  # action = clip(KP * position_error, -1, 1); OSC_POSE control_delta maps ±1 -> ±0.05 m/step
HOVER_DZ = 0.10  # hover this far above the cube before descending
GRASP_DZ = 0.01  # nominal descend target above the cube centre
LIFT_DZ = 0.25
POS_TOL = 0.005
PHASE_BUDGET = 120  # per-phase step cap so an episode always terminates
# 30, not 15: measured on Lift seed 0 — after 15 close steps grip_qpos is 0.0252 and
# _check_grasp() is still False; by 30 it is 0.0224 and True.
CLOSE_HOLD = 30
LIFT_STEPS = 60
MAX_STEPS = 400
# The cube is spawned slightly above the table and settles during the first steps
# (cube_pos.z 0.8306 at reset -> 0.8205 once resting). Latching the target before it settles
# puts every descend ~1 cm too high, which alone is enough to lose the grasp.
SETTLE_STEPS = 10

# robosuite Lift facts (read from source, restated so this file stands alone)
TABLE_Z = 0.8
SUCCESS_MARGIN = 0.04


@dataclass
class FailureSpec:
    """One injected fault. `y_offset` is along world +y, which is exactly image-right in the
    fixed agentview camera (computed from table_arena.xml; see prompts.py). World x is the depth
    axis in this view and is visually entangled with up/down, so pilot 1 does not offset along x.
    """

    name: str
    y_offset: float  # metres along world y applied to the approach target
    descend_dz: float  # height above cube centre where the descent stops
    grip_on_approach: float  # -1 open, +1 already closed
    reference_correction: str
    reference_note: str


def failure_plan(types: list[str] | None = None,
                 seeds: list[int] | None = None) -> list[tuple[FailureSpec, int]]:
    """(spec, reset_seed) pairs -> 10 images by default. Each failure type demands a correction on a
    different axis of the 8-action vocabulary (horizontal / vertical / gripper), so that
    "does the choice change with the image" is actually observable.

    `types` / `seeds` override the default set, for follow-ups that need more frames of one type
    (e.g. separating "is the gripper on the left harder" from "is this frame just noisy"). The
    default call is unchanged, so recorded runs stay reproducible.
    """
    # ±0.08 m, not ±0.05: the open gripper's half-gap is 0.04 and the cube half-width 0.0204, so
    # anything under ~0.061 lands a finger on top of the cube and blocks the descent instead of
    # missing it cleanly.
    lateral_pos = FailureSpec(
        "lateral_offset_y_pos", +0.08, GRASP_DZ, -1.0,
        "MOVE_LEFT",
        "gripper is offset toward image-right of the cube (world +y), so the human-plausible "
        "correction is to move back left before closing",
    )
    lateral_neg = FailureSpec(
        "lateral_offset_y_neg", -0.08, GRASP_DZ, -1.0,
        "MOVE_RIGHT",
        "gripper is offset toward image-left of the cube (world -y), so the human-plausible "
        "correction is to move back right before closing",
    )
    too_high = FailureSpec(
        "close_too_high", 0.0, 0.08, -1.0,
        "MOVE_DOWN",
        "gripper is laterally aligned but stops ~8 cm above the cube and closes on air",
    )
    grip_early = FailureSpec(
        "gripper_closed_early", 0.0, GRASP_DZ, +1.0,
        "OPEN_GRIPPER",
        "gripper is already closed while descending onto the cube, so it knocks the cube "
        "instead of enclosing it",
    )
    specs = {s.name: s for s in (lateral_pos, lateral_neg, too_high, grip_early)}
    default_seeds = {
        "lateral_offset_y_pos": [0, 1], "lateral_offset_y_neg": [0, 1],
        "close_too_high": [0, 1, 2], "gripper_closed_early": [0, 1, 2],
    }
    names = types or list(default_seeds)
    unknown = [n for n in names if n not in specs]
    if unknown:
        raise ValueError(f"unknown failure type(s) {unknown}; available: {sorted(specs)}")
    return [(specs[n], s) for n in names for s in (seeds if seeds is not None else default_seeds[n])]


AXIS_LABELS = {
    0: ("MOVE_FORWARD", "MOVE_BACKWARD"),  # world +x = away from robot base, toward the camera
    1: ("MOVE_RIGHT", "MOVE_LEFT"),        # world +y = image right
    2: ("MOVE_UP", "MOVE_DOWN"),           # world +z
}


def discretize_action(action: np.ndarray) -> str:
    """Project the 7-D OSC action onto the same 8-label vocabulary Gemma must choose from.

    Used only to record what the robot *actually* did, so a Gemma reply that merely echoes the
    original action can be told apart from a genuine counterfactual proposal.
    """
    delta = np.asarray(action[:3], dtype=float)
    if np.linalg.norm(delta) < 0.1:  # essentially holding position -> the gripper command is the action
        return "CLOSE_GRIPPER" if float(action[6]) > 0 else "OPEN_GRIPPER"
    axis = int(np.argmax(np.abs(delta)))
    pos, neg = AXIS_LABELS[axis]
    return pos if delta[axis] > 0 else neg


# --------------------------------------------------------------------------- env
def make_env(image_size: int, camera: str, hard_reset: bool = True):
    """robosuite Lift with the kwargs restated from fasi/envs/robosuite_env.py:92-107.

    ``hard_reset`` differs from the referenced workspace, which uses False. Measured 2026-09-11 on
    Lift seed 0: with ``hard_reset=False`` a soft reset carries solver state across episodes, so the
    same (spec, seed) lands on a different trajectory depending on what ran before it in the same env
    — t_commit 45 with no prior episodes vs 46 after three, and the resting cube_z differs in the 4th
    decimal. With ``hard_reset=True`` all three histories give exactly t_commit 48, cube_z 0.82109777.
    Branching a counterfactual off a recorded frame needs that episode to be reproducible on its own,
    so this defaults to True; the cost on Lift is negligible.
    """
    import robosuite as suite
    from robosuite.controllers import load_controller_config

    return suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=[camera],
        camera_heights=image_size,
        camera_widths=image_size,
        control_freq=20,
        horizon=10**9,  # episodes are budgeted below, never by robosuite
        ignore_done=True,
        hard_reset=hard_reset,
        reward_shaping=False,
    )


def sim_state_snapshot(env) -> dict:
    """The 4-part recipe that made restore bit-exact in the referenced workspace
    (robosuite_env.py:163-173, docs/environment.md:41-43). Not restored in stage 1 — saved only
    so a later counterfactual-verification stage can branch from this exact frame."""
    return {
        "sim_state": np.array(env.sim.get_state().flatten(), copy=True),
        "timestep": int(env.timestep),
        "cur_time": float(env.cur_time),
        "done": bool(env.done),
        "gripper_action": np.array(env.robots[0].gripper.current_action, copy=True),
        "qacc_warmstart": np.array(env.sim.data.qacc_warmstart, copy=True),
    }


# --------------------------------------------------------------------------- rollout
@dataclass
class EpisodeResult:
    success: bool
    final_cube_z: float
    t_commit: int | None
    capture: dict | None = None
    trace: list = field(default_factory=list)


def run_episode(env, spec: FailureSpec, seed: int, camera: str, want_state: bool) -> EpisodeResult:
    """Scripted hover -> descend -> close -> lift, with `spec`'s fault injected.

    `t_commit` is the step at which the failure becomes irreversible; the frame kept for Gemma is
    the one observed at `t_commit - 1`, the last moment a different action could still help.
    """
    np.random.seed(int(seed))  # robosuite samples object placement from the global np.random state
    obs = env.reset()
    hold = np.zeros(7, dtype=float)
    hold[6] = -1.0
    for _ in range(SETTLE_STEPS):  # let the cube drop and come to rest before latching its pose
        obs, _r, _d, _i = env.step(hold)
    cube0 = np.array(obs["cube_pos"], dtype=float)  # latched: the cube is static until touched
    img_key = f"{camera}_image"

    phase, phase_t, close_t = "hover", 0, 0
    grip = spec.grip_on_approach
    t_commit: int | None = None
    capture: dict | None = None
    recent: deque = deque(maxlen=2)  # (t, frame, action, eef, phase); only step t-1 is ever needed

    for t in range(MAX_STEPS):
        eef = np.array(obs["robot0_eef_pos"], dtype=float)

        if phase == "hover":
            target = cube0 + np.array([0.0, spec.y_offset, HOVER_DZ])
            if np.linalg.norm(target - eef) < POS_TOL or phase_t > PHASE_BUDGET:
                phase, phase_t = "descend", 0
        elif phase == "descend":
            target = cube0 + np.array([0.0, spec.y_offset, spec.descend_dz])
            # a gripper that is already closed commits as soon as it is about to reach the cube
            if t_commit is None and spec.grip_on_approach > 0 and (eef[2] - cube0[2]) <= 0.05:
                t_commit = t
            if np.linalg.norm(target - eef) < POS_TOL or phase_t > PHASE_BUDGET:
                phase, phase_t = "close", 0
        elif phase == "close":
            target = eef  # hold position while the fingers close
            grip = 1.0
            if t_commit is None:
                t_commit = t
            close_t += 1
            if close_t >= CLOSE_HOLD:
                phase, phase_t = "lift", 0
        else:  # lift
            target = cube0 + np.array([0.0, 0.0, LIFT_DZ])
            grip = 1.0
            if phase_t > LIFT_STEPS:
                break

        # Grab the pre-commit frame the moment the failure turns irreversible. `recent` still ends
        # at step t-1 here (this step's frame is appended below), and t-1 is the last recoverable
        # moment. Doing this after the loop would fail: the episode runs on for ~70 more steps and
        # the rolling window no longer holds t_commit-1.
        if t_commit == t and capture is None and recent:
            c_t, c_frame, c_action, c_eef, c_phase = recent[-1]
            capture = {
                "timestep": c_t,
                "frame": c_frame,
                "original_action": [round(float(v), 5) for v in c_action],
                "original_action_discrete": discretize_action(c_action),
                "eef_pos": [round(float(v), 5) for v in c_eef],
                "cube_pos": [round(float(v), 5) for v in cube0],
                "phase": c_phase,
            }

        action = np.zeros(7, dtype=float)
        action[:3] = np.clip(KP * (target - eef), -1.0, 1.0)
        action[6] = grip

        frame = np.ascontiguousarray(obs[img_key][::-1])  # robosuite renders upside down
        recent.append((t, frame, action.copy(), eef.copy(), phase))

        obs, _reward, _done, _info = env.step(action)
        phase_t += 1

    final_cube_z = float(env.sim.data.body_xpos[env.cube_body_id][2])
    success = bool(env._check_success())

    state = sim_state_snapshot(env) if (want_state and capture is not None) else None
    return EpisodeResult(success, final_cube_z, t_commit, capture, trace=[state])


# --------------------------------------------------------------------------- main
def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", default=str(here / "images"))
    ap.add_argument("--camera", default="agentview")
    ap.add_argument("--image-size", type=int, default=256, help="≈258 prompt tokens on gemma4 at 256")
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--types", default=None,
                    help="comma-separated failure types to collect (default: all four)")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated reset seeds, applied to every selected type")
    ap.add_argument("--no-hard-reset", action="store_true",
                    help="use robosuite's soft reset (what the referenced workspace does). Faster, "
                         "but episodes stop being reproducible on their own — see make_env()")
    ap.add_argument("--save-sim-state", action="store_true",
                    help="also write the 4-part MuJoCo snapshot per frame (for a later restore stage)")
    ap.add_argument("--dry-run", action="store_true", help="validate plan and paths; never touch MuJoCo")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    plan = failure_plan(
        types=[t.strip() for t in args.types.split(",")] if args.types else None,
        seeds=[int(x) for x in args.seeds.split(",")] if args.seeds else None,
    )[: args.max_images]

    if args.dry_run:
        print(f"[dry-run] images_dir     : {images_dir}  (exists={images_dir.is_dir()})")
        print(f"[dry-run] camera         : {args.camera} @ {args.image_size}px")
        print(f"[dry-run] action labels  : {len(ACTIONS)} -> {', '.join(ACTIONS)}")
        print(f"[dry-run] planned frames : {len(plan)}")
        for i, (spec, seed) in enumerate(plan):
            print(f"  ep{i:02d} seed={seed} {spec.name:24s} y_off={spec.y_offset:+.3f} "
                  f"descend_dz={spec.descend_dz:.3f} grip={spec.grip_on_approach:+.0f} "
                  f"ref={spec.reference_correction}")
        types = sorted({s.name for s, _ in plan})
        print(f"[dry-run] failure types  : {len(types)} -> {types}")
        print("[dry-run] OK — no environment was created.")
        return

    images_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    env = make_env(args.image_size, args.camera, hard_reset=not args.no_hard_reset)
    meta_path = images_dir / "metadata.jsonl"
    run_started = datetime.now().isoformat(timespec="seconds")
    records, skipped = [], []

    try:
        # Control condition. The same controller with no fault injected must actually succeed,
        # otherwise the collected frames are just a broken controller and every `failure_reason`
        # is a false attribution. Checked on every run, never assumed.
        nominal = FailureSpec("nominal", 0.0, GRASP_DZ, -1.0, "-", "no fault injected")
        nom = [run_episode(env, nominal, s, args.camera, False) for s in (0, 1, 2)]
        n_ok = sum(r.success for r in nom)
        nominal_rate = n_ok / len(nom)
        print(f"nominal control check: {n_ok}/{len(nom)} lifted the cube "
              f"(final_cube_z={[round(r.final_cube_z, 4) for r in nom]}, threshold={TABLE_Z + SUCCESS_MARGIN})")
        if n_ok == 0:
            print("  !! the unperturbed controller never succeeds, so the injected faults are NOT "
                  "what causes these failures — fix the controller before trusting failure_reason")

        for ep, (spec, seed) in enumerate(plan):
            res = run_episode(env, spec, seed, args.camera, args.save_sim_state)
            tag = f"ep{ep:02d} seed={seed} {spec.name}"
            if res.capture is None:
                why = ("never reached a commit step" if res.t_commit is None
                       else f"commit at t={res.t_commit} but no preceding frame was retained")
                skipped.append({"episode_id": ep, "failure_type": spec.name, "seed": seed, "why": why})
                print(f"  SKIP {tag}: {why}")
                continue
            if res.success:  # never assume failure — robosuite says whether it failed
                skipped.append({"episode_id": ep, "failure_type": spec.name, "seed": seed,
                                "why": f"episode succeeded (cube_z={res.final_cube_z:.4f})"})
                print(f"  SKIP {tag}: episode SUCCEEDED (cube_z={res.final_cube_z:.4f}) — not a failure")
                continue

            t = res.capture["timestep"]
            fname = f"ep{ep:02d}_{spec.name}_t{t:04d}.png"
            Image.fromarray(res.capture["frame"]).save(images_dir / fname)

            rec = {
                "image_path": str(images_dir / fname),
                "image_file": fname,
                "episode_id": ep,
                "reset_seed": seed,
                "timestep": t,
                "t_commit": res.t_commit,
                "failure_type": spec.name,
                "failure_reason": spec.reference_note,
                "failure_verified_by": "robosuite Lift._check_success() == False at episode end",
                "nominal_success_rate": nominal_rate,  # same controller, fault removed
                "final_cube_z": round(res.final_cube_z, 5),
                "success_threshold_z": TABLE_Z + SUCCESS_MARGIN,
                "original_action": res.capture["original_action"],
                "original_action_discrete": res.capture["original_action_discrete"],
                "eef_pos": res.capture["eef_pos"],
                "cube_pos": res.capture["cube_pos"],
                "phase_at_capture": res.capture["phase"],
                # reference only — a human-plausible correction, NOT ground truth and never
                # shown to the model. Present so the pilot's outputs can be eyeballed quickly.
                "reference_correction": spec.reference_correction,
                "reference_correction_note": "human-plausible correction, not ground truth",
                "camera": args.camera,
                "image_size": args.image_size,
                "env": "Lift", "robots": "Panda", "controller": "OSC_POSE",
                "collected_at": run_started,
            }
            records.append(rec)
            if args.save_sim_state and res.trace[0] is not None:
                npz = images_dir / fname.replace(".png", "_simstate.npz")
                np.savez_compressed(npz, **res.trace[0])
                rec["sim_state_path"] = str(npz)
            print(f"  OK   {tag}: t_commit={res.t_commit} capture_t={t} "
                  f"orig={rec['original_action_discrete']:14s} cube_z={res.final_cube_z:.4f} -> {fname}")
    finally:
        env.close()

    with meta_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dist: dict[str, int] = {}
    for r in records:
        dist[r["failure_type"]] = dist.get(r["failure_type"], 0) + 1
    print(f"\nsaved {len(records)} images -> {images_dir}")
    print(f"metadata -> {meta_path}")
    print(f"per failure_type: {dist}")
    if skipped:
        print(f"skipped {len(skipped)}: {skipped}")


if __name__ == "__main__":
    main()
