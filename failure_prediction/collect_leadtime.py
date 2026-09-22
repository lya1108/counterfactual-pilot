#!/usr/bin/env python3
"""Failure-prediction lead-time set: paired success/failure frames at fixed distances from failure.

The question this set is built for: how early before a failure becomes irreversible can a VLM tell
that the attempt is going to fail? That needs frames at known distances from the failure, and a
matched success set — otherwise "predicts FAILURE" cannot be told apart from "always says FAILURE".

Definition of the failure timestep. ``t_fail = t_commit``, the step at which the fault becomes
irreversible, which is where ``collect_failures.run_episode`` already captures its single frame
(close phase begins, or -- for ``gripper_closed_early`` -- the closed gripper is about to reach the
cube). It is NOT the step where the lift visibly ends empty: that point is ~90 steps later and the
image is then trivially readable (gripper aloft, cube on the table). Using the irreversibility point
keeps the question the interesting one, and it matches the frame stage 2 branched from.

Frames kept per failure episode: ``t_fail - 20, -10, -5, -1``. An episode whose ``t_commit`` is
smaller than a lead time contributes nothing at that lead time and is recorded in ``skipped``.

Matching the success episodes. Progress is ``t / episode_len``, so failure episode *i* hands its
four progress fractions to success episode *i* (paired by index), and each success frame is taken at
``round(p * episode_len_success)``. The pair therefore shares progress-through-the-episode but NOT
the cube placement, since the 10 success episodes use 10 distinct reset seeds to be 10 independent
rollouts. Every success frame records its own phase and ``t_commit`` so the alternative reading --
"the classes differ by which control phase they are in, not by any failure evidence" -- can be
checked rather than assumed.

The controller, the env kwargs and the constants all come from ``collect_failures`` by import, so
there is one source of truth for the physics; only the recording differs (every frame is kept, not a
2-frame window). ``--verify`` re-runs ``collect_failures.run_episode`` on the same (spec, seed) and
asserts the same ``t_commit``, which is what makes that reuse safe to trust.

Chunks (``--chunk K --stride S``). A single frame cannot say whether the arm is still descending
or has come to rest, and the 2026-09-21 fact-injection run showed that is exactly what
``close_too_high`` turns on: its failing frame sits 8.5 cm above the cube and its matched success
frame sits *higher*, at 10.5 cm, so no function of one frame's state separates them. With ``--chunk``
each saved point also keeps the K-1 frames before it, spaced ``stride`` steps apart, in time order.
The window was chosen by measuring, not guessed: over the 20 steps before t_commit-1 the
``close_too_high`` arm moves 2.9 cm while its matched nominal moves 8.3-8.7 cm, and the same server
reads a 9 cm difference correctly in 4/4 order-flipped probes while a 2.3 cm one is at chance.
Shorter windows (stride 5, 10) leave the two classes 1.7 vs 4-6 cm apart; stride 30 reaches back into
the descent and they converge again (10.2 vs 11.6-13.0 cm).

Near the start of an episode the window is clipped at step 0, so the realised span can be shorter
than ``(K-1)*stride``; every record carries ``chunk_timesteps`` and ``chunk_span`` so an analysis can
condition on what was actually shown instead of on what was asked for.

Nothing from ~/바탕화면/jungbeom is imported or modified.

Run (conda fasi, MUJOCO_GL=egl):
    python collect_leadtime.py --dry-run
    python collect_leadtime.py --verify
    python collect_leadtime.py
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

# The stage 1-2 pilot lives next door and owns the scripted controller, the env kwargs and the
# prompt vocabulary. Importing rather than restating keeps one source of truth for the physics
# constants -- the same reason collect_leadtime imports collect_failures instead of copying its
# control loop. Only this path line is needed; nothing in ../counterfactual_pilot is modified.
import sys as _sys
_PILOT = str(Path(__file__).resolve().parent.parent / "counterfactual_pilot")
if _PILOT not in _sys.path:
    _sys.path.insert(0, _PILOT)

import collect_failures as cf  # noqa: E402

LEAD_TIMES = (20, 10, 5, 1)
N_SUCCESS = 10


@dataclass
class Rollout:
    """One full episode with every frame retained."""

    success: bool
    final_cube_z: float
    t_commit: int | None
    frames: list = field(default_factory=list)      # index == timestep
    phases: list = field(default_factory=list)
    eefs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    grip_qpos: list = field(default_factory=list)  # (2,) finger joint positions, per step

    @property
    def length(self) -> int:
        return len(self.frames)


def rollout(env, spec: cf.FailureSpec, seed: int, camera: str) -> Rollout:
    """Same control law as ``collect_failures.run_episode``; keeps every frame instead of two.

    Kept deliberately line-for-line comparable to that function. ``--verify`` asserts both agree on
    ``t_commit``, so a future edit to one that is not mirrored here shows up as a hard failure rather
    than as frames quietly taken at the wrong distance from the failure.
    """
    np.random.seed(int(seed))  # robosuite samples object placement from the global np.random state
    obs = env.reset()
    hold = np.zeros(7, dtype=float)
    hold[6] = -1.0
    for _ in range(cf.SETTLE_STEPS):  # let the cube settle before latching its pose
        obs, _r, _d, _i = env.step(hold)
    cube0 = np.array(obs["cube_pos"], dtype=float)
    img_key = f"{camera}_image"

    phase, phase_t, close_t = "hover", 0, 0
    grip = spec.grip_on_approach
    t_commit: int | None = None
    out = Rollout(False, 0.0, None)

    for t in range(cf.MAX_STEPS):
        eef = np.array(obs["robot0_eef_pos"], dtype=float)

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
        else:  # lift
            target = cube0 + np.array([0.0, 0.0, cf.LIFT_DZ])
            grip = 1.0
            if phase_t > cf.LIFT_STEPS:
                break

        action = np.zeros(7, dtype=float)
        action[:3] = np.clip(cf.KP * (target - eef), -1.0, 1.0)
        action[6] = grip

        out.frames.append(np.ascontiguousarray(obs[img_key][::-1]))  # robosuite renders upside down
        out.phases.append(phase)
        out.eefs.append(eef.copy())
        out.actions.append(action.copy())
        out.grip_qpos.append(np.asarray(obs["robot0_gripper_qpos"], dtype=float).copy())

        obs, _reward, _done, _info = env.step(action)
        phase_t += 1

    out.t_commit = t_commit
    out.final_cube_z = float(env.sim.data.body_xpos[env.cube_body_id][2])
    out.success = bool(env._check_success())
    out.cube0 = cube0
    return out


def chunk_timesteps(t: int, k: int, stride: int) -> list[int]:
    """The k timesteps ending at ``t``, spaced ``stride`` apart, clipped at the episode start.

    Clipping can collapse two requested steps onto the same frame near t=0; duplicates are dropped
    rather than shown twice, so a record's chunk can be shorter than k. The realised steps are
    always written to the metadata.
    """
    ts = sorted({max(t - i * stride, 0) for i in range(k)})
    return ts


def save_chunk(images_dir, Image, ro: Rollout, ts: list[int], stem: str) -> list[str]:
    """Write one file per chunk frame; returns the file names in time order."""
    names = []
    for n, tt in enumerate(ts):
        fn = f"{stem}_c{n}_t{tt:04d}.png"
        Image.fromarray(ro.frames[tt]).save(images_dir / fn)
        names.append(fn)
    return names


def frame_record(ro: Rollout, t: int, **extra) -> dict:
    """The per-frame fields shared by both classes."""
    return {
        "timestep": t,
        "progress": round(t / ro.length, 4),
        "phase": ro.phases[t],
        "episode_len": ro.length,
        "t_commit": ro.t_commit,
        "eef_pos": [round(float(v), 5) for v in ro.eefs[t]],
        "cube_pos": [round(float(v), 5) for v in ro.cube0],
        "gripper_cmd": round(float(ro.actions[t][6]), 3),
        "gripper_closed_cmd": bool(ro.actions[t][6] > 0),
        # The commanded state and the visible one are NOT the same thing, and conflating them put a
        # wrong label on 9 of the first 80 frames. A successful grasp is commanded closed while the
        # cube holds the fingers apart: measured on Lift, nominal close-phase frames sit at a gap of
        # 0.059-0.074 m, fully open is 0.079 m, and fingers genuinely pressed together are 0.001 m.
        # So a probe asking what the fingers *look* like has to be scored on the gap, not the command.
        "finger_gap_m": round(float(abs(ro.grip_qpos[t][0] - ro.grip_qpos[t][1])), 5),
        "fingers_pressed": bool(abs(ro.grip_qpos[t][0] - ro.grip_qpos[t][1]) < 0.01),
        "gripper_qpos": [round(float(v), 5) for v in ro.grip_qpos[t]],
        "action_discrete": cf.discretize_action(ro.actions[t]),
        "final_cube_z": round(ro.final_cube_z, 5),
        **extra,
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", default=str(here / "images_leadtime"))
    ap.add_argument("--camera", default="agentview")
    ap.add_argument("--image-size", type=int, default=256, help="≈258 prompt tokens on gemma4 at 256")
    ap.add_argument("--lead-times", default=",".join(str(x) for x in LEAD_TIMES))
    ap.add_argument("--n-success", type=int, default=N_SUCCESS)
    ap.add_argument("--chunk", type=int, default=1,
                    help="frames per saved point, in time order and ending at the lead point "
                         "(1 = the single-frame set)")
    ap.add_argument("--stride", type=int, default=10, help="steps between the frames of a chunk")
    ap.add_argument("--types", default=None, help="comma-separated failure types (default: all four)")
    ap.add_argument("--seeds", default=None, help="comma-separated reset seeds for the failure plan")
    ap.add_argument("--max-failures", type=int, default=10)
    ap.add_argument("--no-hard-reset", action="store_true", help="see collect_failures.make_env()")
    ap.add_argument("--verify", action="store_true",
                    help="cross-check t_commit against collect_failures.run_episode on every "
                         "failure episode (one extra rollout each) before collecting")
    ap.add_argument("--dry-run", action="store_true", help="validate plan and paths; never touch MuJoCo")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    leads = sorted({int(x) for x in args.lead_times.split(",")}, reverse=True)
    plan = cf.failure_plan(
        types=[t.strip() for t in args.types.split(",")] if args.types else None,
        seeds=[int(x) for x in args.seeds.split(",")] if args.seeds else None,
    )[: args.max_failures]
    succ_seeds = list(range(args.n_success))

    if args.dry_run:
        print(f"[dry-run] images_dir      : {images_dir}  (exists={images_dir.is_dir()})")
        print(f"[dry-run] camera          : {args.camera} @ {args.image_size}px")
        print(f"[dry-run] t_fail          : t_commit (irreversibility point, not the empty lift)")
        print(f"[dry-run] lead times      : {leads}")
        print(f"[dry-run] failure episodes: {len(plan)}")
        for i, (spec, seed) in enumerate(plan):
            print(f"    f{i:02d} seed={seed} {spec.name:24s} y_off={spec.y_offset:+.3f} "
                  f"descend_dz={spec.descend_dz:.3f} grip={spec.grip_on_approach:+.0f}")
        print(f"[dry-run] success episodes: {len(succ_seeds)} (nominal controller, seeds {succ_seeds})")
        print(f"[dry-run] pairing         : success i takes the four progress fractions of failure i")
        print(f"[dry-run] chunk           : {args.chunk} frame(s), stride {args.stride} "
              f"(span {(args.chunk - 1) * args.stride} steps, clipped at the episode start)")
        print(f"[dry-run] images           : {len(plan) * len(leads)} failure + "
              f"{len(succ_seeds) * len(leads)} success = "
              f"{(len(plan) + len(succ_seeds)) * len(leads)} (before any skip)")
        print("[dry-run] OK — no environment was created.")
        return

    images_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    env = cf.make_env(args.image_size, args.camera, hard_reset=not args.no_hard_reset)
    run_started = datetime.now().isoformat(timespec="seconds")
    records: list[dict] = []
    skipped: list[dict] = []
    nominal = cf.FailureSpec("nominal", 0.0, cf.GRASP_DZ, -1.0, "-", "no fault injected")

    try:
        # ---------------------------------------------------------------- failure episodes
        fail_ros: list[tuple[int, cf.FailureSpec, int, Rollout]] = []
        print(f"--- failure episodes ({len(plan)}) ---")
        for i, (spec, seed) in enumerate(plan):
            ro = rollout(env, spec, seed, args.camera)
            tag = f"f{i:02d} seed={seed} {spec.name}"
            if args.verify:
                ref = cf.run_episode(env, spec, seed, args.camera, False)
                if ref.t_commit != ro.t_commit:
                    raise SystemExit(
                        f"{tag}: t_commit disagrees with collect_failures.run_episode "
                        f"({ro.t_commit} vs {ref.t_commit}) — the two control loops have diverged"
                    )
            if ro.t_commit is None:
                skipped.append({"episode": tag, "why": "never reached a commit step"})
                print(f"  SKIP {tag}: never committed")
                continue
            if ro.success:  # never assume failure — robosuite decides
                skipped.append({"episode": tag, "why": f"episode SUCCEEDED (cube_z={ro.final_cube_z:.4f})"})
                print(f"  SKIP {tag}: SUCCEEDED (cube_z={ro.final_cube_z:.4f}) — not a failure")
                continue
            fail_ros.append((i, spec, seed, ro))
            print(f"  OK   {tag}: t_commit={ro.t_commit} len={ro.length} "
                  f"cube_z={ro.final_cube_z:.4f}{'  [verified]' if args.verify else ''}")

        # ---------------------------------------------------------------- success episodes
        print(f"\n--- success episodes ({len(succ_seeds)}, nominal controller) ---")
        succ_ros: list[tuple[int, int, Rollout]] = []
        for j, seed in enumerate(succ_seeds):
            ro = rollout(env, nominal, seed, args.camera)
            tag = f"s{j:02d} seed={seed} nominal"
            if not ro.success:
                skipped.append({"episode": tag, "why": f"nominal FAILED (cube_z={ro.final_cube_z:.4f})"})
                print(f"  SKIP {tag}: nominal controller FAILED (cube_z={ro.final_cube_z:.4f})")
                continue
            succ_ros.append((j, seed, ro))
            print(f"  OK   {tag}: t_commit={ro.t_commit} len={ro.length} cube_z={ro.final_cube_z:.4f}")

        if not succ_ros:
            raise SystemExit("the nominal controller never succeeded — there is no success class to "
                             "match against, and the failure frames cannot be attributed to the fault")

        # ---------------------------------------------------------------- save paired frames
        print(f"\n--- saving frames ---")
        for pair_id, (i, spec, seed, ro) in enumerate(fail_ros):
            s_entry = succ_ros[pair_id % len(succ_ros)]
            for lead in leads:
                t = ro.t_commit - lead
                if t < 0:
                    skipped.append({"episode": f"f{i:02d} {spec.name}", "lead_time": lead,
                                    "why": f"t_commit={ro.t_commit} is earlier than the lead time"})
                    print(f"  SKIP f{i:02d} lead-{lead}: t_commit={ro.t_commit} < {lead}")
                    continue
                stem = f"fail_p{pair_id:02d}_{spec.name}_lead{lead:02d}_t{t:04d}"
                fname = f"{stem}.png"
                Image.fromarray(ro.frames[t]).save(images_dir / fname)
                cts = chunk_timesteps(t, args.chunk, args.stride)
                cfiles = save_chunk(images_dir, Image, ro, cts, stem) if args.chunk > 1 else [fname]
                records.append(frame_record(
                    ro, t,
                    image_file=fname, image_path=str(images_dir / fname),
                    chunk_files=cfiles, chunk_timesteps=cts,
                    chunk_span=cts[-1] - cts[0], chunk_len=len(cts),
                    label="FAILURE", kind="failure", pair_id=pair_id, episode_id=f"f{i:02d}",
                    reset_seed=seed, failure_type=spec.name, failure_reason=spec.reference_note,
                    lead_time=lead, t_fail=ro.t_commit,
                    t_fail_definition="t_commit: the step at which the injected fault becomes irreversible",
                    verified_by="robosuite Lift._check_success() == False at episode end",
                ))

                # the matched success frame, at the same progress fraction
                j, s_seed, s_ro = s_entry
                p = t / ro.length
                st = min(int(round(p * s_ro.length)), s_ro.length - 1)
                sstem = f"succ_p{pair_id:02d}_s{j:02d}_lead{lead:02d}_t{st:04d}"
                sname = f"{sstem}.png"
                Image.fromarray(s_ro.frames[st]).save(images_dir / sname)
                scts = chunk_timesteps(st, args.chunk, args.stride)
                scfiles = save_chunk(images_dir, Image, s_ro, scts, sstem) if args.chunk > 1 else [sname]
                records.append(frame_record(
                    s_ro, st,
                    image_file=sname, image_path=str(images_dir / sname),
                    chunk_files=scfiles, chunk_timesteps=scts,
                    chunk_span=scts[-1] - scts[0], chunk_len=len(scts),
                    label="SUCCESS", kind="success", pair_id=pair_id, episode_id=f"s{j:02d}",
                    reset_seed=s_seed, failure_type="nominal", failure_reason="no fault injected",
                    lead_time=lead, t_fail=None,
                    matched_progress=round(p, 4), matched_to=f"f{i:02d}",
                    t_fail_definition="n/a (this episode succeeds)",
                    verified_by="robosuite Lift._check_success() == True at episode end",
                ))
                print(f"  f{i:02d} lead-{lead:<2d} t={t:3d} p={p:.3f} [{ro.phases[t]:7s}]  <->  "
                      f"s{j:02d} t={st:3d} [{s_ro.phases[st]:7s}]")
    finally:
        env.close()

    meta_path = images_dir / "metadata.jsonl"
    for r in records:
        r["camera"] = args.camera
        r["image_size"] = args.image_size
        r["env"] = "Lift"
        r["robots"] = "Panda"
        r["controller"] = "OSC_POSE"
        r["collected_at"] = run_started
    with meta_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_f = sum(1 for r in records if r["kind"] == "failure")
    n_s = len(records) - n_f
    print(f"\nsaved {len(records)} images ({n_f} failure / {n_s} success) -> {images_dir}")
    print(f"metadata -> {meta_path}")
    print(f"{'lead':>6s} {'failure':>9s} {'success':>9s}   failure phases / success phases")
    for lead in leads:
        fs = [r for r in records if r["lead_time"] == lead and r["kind"] == "failure"]
        ss = [r for r in records if r["lead_time"] == lead and r["kind"] == "success"]
        fp = {}
        sp = {}
        for r in fs:
            fp[r["phase"]] = fp.get(r["phase"], 0) + 1
        for r in ss:
            sp[r["phase"]] = sp.get(r["phase"], 0) + 1
        print(f"{lead:>6d} {len(fs):>9d} {len(ss):>9d}   {fp} / {sp}")
    if args.chunk > 1:
        short = [r for r in records if r["chunk_len"] < args.chunk]
        spans = sorted({r["chunk_span"] for r in records})
        print(f"chunk: {args.chunk} frames x stride {args.stride}; realised spans {spans}; "
              f"{len(short)} record(s) clipped short by the episode start")
    if skipped:
        print(f"\nskipped {len(skipped)}:")
        for s in skipped:
            print(f"  {s}")


if __name__ == "__main__":
    main()
