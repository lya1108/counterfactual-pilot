"""Pilot: how expensive is per-step state logging, and what do the raw quantities actually look like?

Thresholds are not guessed here. The point of this run is to produce the distributions that the thresholds in
conditions.DEFAULTS should be read off -- gripper opening when something is held versus when the fingers are
empty, object tilt, joint margin -- and to measure what the logging adds to rollout time before 800 episodes
are committed to it.

    conda activate fasi
    python xattn_detector/pilot_conditions.py --episodes 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "safe_detector"))

import jb  # noqa: E402

jb.enter()

from fasi.envs.libero_env import LiberoTask  # noqa: E402
from fasi.vla.pi05_steer import Pi05Steer  # noqa: E402

from conditions import CONDITIONS, StateLogger, earliest_onset, evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/vla/pi05_libero_finetuned")
    ap.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_goal", "libero_10"])
    ap.add_argument("--tasks", default="0,2,9")
    ap.add_argument("--inits", default="30,31")
    ap.add_argument("--exec", type=int, default=10)
    ap.add_argument("--out", default="data/pilot_cond")
    args = ap.parse_args()

    out = HERE / args.out
    out.mkdir(parents=True, exist_ok=True)
    tasks = [int(x) for x in args.tasks.split(",")]
    inits = [int(x) for x in args.inits.split(",")]

    vla = Pi05Steer(args.ckpt)
    eps, t_log, t_sim = [], 0.0, 0.0
    for suite in args.suites:
        for tid in tasks:
            task = LiberoTask(suite, tid)
            for init in inits:
                task.reset(init)
                log = StateLogger(task)
                t, ok = 0, False
                t_ep = time.time()
                while t < task.max_steps:
                    ag = task.obs["agentview_image"][::-1, ::-1].copy()
                    wr = task.obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
                    b = vla.batch_for(ag, wr, task.state8(), task.language)
                    g = torch.Generator().manual_seed(1000 * tid + 17 * init + t)
                    a = vla.denoise(b, torch.randn((1, vla.C, vla.A), generator=g), [], None)[0]
                    for j in range(args.exec):
                        t0 = time.time()
                        task.step(a[j])
                        t_sim += time.time() - t0
                        t += 1
                        t0 = time.time()
                        log.record(t)
                        t_log += time.time() - t0
                        if task.success():
                            ok = True
                            break
                    if ok:
                        break
                packed = log.pack()
                name = f"{suite}_t{tid}_i{init}"
                np.savez_compressed(out / f"{name}.npz", **packed)
                eps.append(dict(name=name, suite=suite, task=tid, init=init, success=ok, n_steps=t,
                                sec=round(time.time() - t_ep, 1),
                                bytes=(out / f"{name}.npz").stat().st_size, packed=packed))
                print(f"  {name}  {'S' if ok else 'F'}  {t} steps  {eps[-1]['sec']}s  "
                      f"{eps[-1]['bytes'] / 1024:.0f} KB", flush=True)
            task.close()

    n_steps = sum(e["n_steps"] for e in eps)
    print(f"\n=== 비용 ===")
    print(f"  환경 스텝 {n_steps}   시뮬레이션 {t_sim:.1f}s   상태 기록 {t_log:.1f}s "
          f"({t_log / max(t_sim, 1e-9):.0%} 추가)")
    print(f"  스텝당 기록 {t_log / n_steps * 1000:.2f} ms   에피소드당 저장 "
          f"{np.mean([e['bytes'] for e in eps]) / 1024:.0f} KB")
    print(f"  800 에피소드 환산: 저장 {np.mean([e['bytes'] for e in eps]) * 800 / 1024 ** 2:.0f} MB, "
          f"기록 시간 +{t_log / len(eps) * 800 / 60:.0f}분")

    print(f"\n=== 임계값을 정하기 위한 원시 분포 ===")
    op_held, op_free = [], []
    tilts, margins, wrists = [], [], []
    for e in eps:
        p = e["packed"]
        held = p["obj_touch"] & ~p["obj_support"]
        anyheld = held.any(1)
        op_held += list(p["opening"][anyheld])
        op_free += list(p["opening"][~anyheld])
        margins += list(p["joint_margin"].min(1))
        wrists += list(np.abs(p["wrist"]))
        from conditions import quat2up
        up0 = np.stack([quat2up(q) for q in p["obj_quat0"]])
        for t in range(len(p["step"])):
            for i in range(up0.shape[0]):
                u = quat2up(p["obj_quat"][t, i])
                tilts.append(np.degrees(np.arccos(np.clip(np.dot(u, up0[i]), -1, 1))))
    q = lambda v: np.percentile(v, [1, 25, 50, 75, 99]).round(4) if len(v) else "없음"
    print(f"  opening, 무언가 쥐고 있을 때 (n={len(op_held)}): {q(op_held)}")
    print(f"  opening, 쥐지 않았을 때   (n={len(op_free)}): {q(op_free)}")
    print(f"  joint margin 최소값               : {q(margins)}")
    print(f"  |wrist| deg                       : {q(wrists)}")
    print(f"  객체 tilt deg                     : {q(tilts)}")
    print("  (백분위 1 / 25 / 50 / 75 / 99)")

    print(f"\n=== 기본 임계값으로 본 조건 발생 ===")
    print(f"  {'episode':28s} {'결과':>4s} " + " ".join(f"{c.split('_')[0]:>4s}" for c in CONDITIONS) + "   최초")
    counts = {c: 0 for c in CONDITIONS}
    for e in eps:
        _, ons = evaluate(e["packed"])
        k, i = earliest_onset(ons)
        cells = []
        for c in CONDITIONS:
            v = ons[c]
            cells.append(f"{e['packed']['step'][v]:>4d}" if v is not None else "   .")
            counts[c] += v is not None
        first = f"{k.split('_')[0]}@{e['packed']['step'][i]}" if k else "없음"
        print(f"  {e['name']:28s} {'S' if e['success'] else 'F':>4s} " + " ".join(cells) + f"   {first}")
    print(f"\n  조건별 발생 에피소드 수 (총 {len(eps)}): " +
          ", ".join(f"{c.split('_')[0]}={counts[c]}" for c in CONDITIONS))

    (HERE / "results" / "pilot_conditions.json").write_text(json.dumps(
        dict(n_episodes=len(eps), n_steps=n_steps, log_ms_per_step=t_log / n_steps * 1000,
             kb_per_episode=float(np.mean([e["bytes"] for e in eps]) / 1024),
             counts=counts,
             episodes=[{k: v for k, v in e.items() if k != "packed"} for e in eps]),
        indent=2, ensure_ascii=False, default=float))
    print(f"\n-> {HERE / 'results' / 'pilot_conditions.json'}   상태 기록 -> {out}")


if __name__ == "__main__":
    main()
