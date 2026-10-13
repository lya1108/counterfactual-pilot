"""Is the mismatch found during collection run-to-run nondeterminism, or a difference in the replay loop?

collect_tokens.py found one episode in 69 whose end-effector path drifted from the recorded one, growing from
0 at call 1 to 1.85 cm by call 8 -- the signature of a tiny perturbation amplified by contact physics rather
than of a different starting state. This decides between the two explanations by replaying the SAME episode
several times in one process: if the replays differ from each other, nothing about the loop is at fault.

    conda activate fasi
    python xattn_detector/repro_check.py --suite libero_spatial --task 2 --init 17 --runs 3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "safe_detector"))
import jb  # noqa: E402

jb.enter()

from fasi.envs.libero_env import LiberoTask  # noqa: E402
from fasi.vla.pi05_steer import Pi05Steer  # noqa: E402


def rollout(vla, task, tid, init, calls, n_exec):
    """`calls` caps the number of VLA calls; a huge value means "to the end", as the original collection ran."""
    task.reset(init)
    t, eefs, acts, ok = 0, [], [], False
    for _ in range(calls):
        if t >= task.max_steps or ok:
            break
        ag = task.obs["agentview_image"][::-1, ::-1].copy()
        wr = task.obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
        b = vla.batch_for(ag, wr, task.state8(), task.language)
        g = torch.Generator().manual_seed(1000 * tid + 17 * init + t)
        a = vla.denoise(b, torch.randn((1, vla.C, vla.A), generator=g), [], None)[0]
        eefs.append(task.eef())
        acts.append(a[:n_exec].copy())
        for j in range(n_exec):
            task.step(a[j])
            t += 1
            if task.success():
                ok = True
                break
    return np.asarray(eefs), np.asarray(acts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", type=int, default=2)
    ap.add_argument("--init", type=int, default=17)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--calls", type=int, default=8)
    ap.add_argument("--exec", type=int, default=10)
    ap.add_argument("--warmup", default="", help="inits to run to completion in the same env before the target, "
                                                 "e.g. 0-16 -- tests whether reset() leaves residual state")
    ap.add_argument("--tap", default="none", choices=["none", "gpu", "cpu"],
                    help="install a token tap as collect_tokens.py does. `gpu` keeps the captured tensors on the "
                         "device (what TokenTap did); `cpu` moves them off immediately (what safe_detector's "
                         "LatentTap did). Holding device memory changes the caching allocator's state, which can "
                         "change kernel selection and therefore the last bits of the result.")
    args = ap.parse_args()

    warm = []
    for part in args.warmup.split(","):
        if "-" in part:
            a, b = part.split("-")
            warm += list(range(int(a), int(b) + 1))
        elif part.strip():
            warm.append(int(part))

    vla = Pi05Steer("outputs/vla/pi05_libero_finetuned")
    tap = None
    if args.tap != "none":
        sys.path.insert(0, str(HERE))
        from token_tap import TokenTap
        tap = TokenTap(vla.policy, pool=2)
        if args.tap == "cpu":  # mimic LatentTap: never hold device memory
            orig = tap._wrapped

            def cpu_wrapped(*a, **kw):
                out = tap._orig(*a, **kw)
                pre, suf = out[0]
                if suf is None and pre is not None:
                    tap.prefix, tap.prefix_pos = pre[0].detach().cpu(), kw.get("position_ids")
                elif pre is None and suf is not None:
                    tap.suffix.append(suf[0, -tap.chunk:].detach().cpu())
                return out

            tap.mod.forward = cpu_wrapped
    task = LiberoTask(args.suite, args.task)
    runs = []
    for _ in range(args.runs):
        for w in warm:  # replay the preceding episodes at FULL length, as the original collection did
            rollout(vla, task, args.task, w, 10 ** 6, args.exec)
        runs.append(rollout(vla, task, args.task, args.init, args.calls, args.exec))
    task.close()
    if warm:
        print(f"\n  (warmup: init {warm[0]}..{warm[-1]}를 매 반복 전에 끝까지 실행)")

    ref_path = (HERE.parent / "safe_detector" / "data" / args.suite /
                f"ep_{args.suite}_t{args.task}_i{args.init}_rot180.npz")
    print(f"\n{args.suite} t{args.task} i{args.init} — 같은 프로세스에서 {args.runs}회 반복\n")
    print("  비교                        최대 |Δeef| (m)      최대 |Δaction|")
    for i in range(1, len(runs)):
        d_e = np.abs(runs[i][0] - runs[0][0]).max()
        d_a = np.abs(runs[i][1] - runs[0][1]).max()
        print(f"  run{i} vs run0              {d_e:16.3e} {d_a:17.3e}")
    if ref_path.exists():
        ref = np.load(ref_path)["eef"][: args.calls]
        for i in range(len(runs)):
            print(f"  run{i} vs 원본 수집          {np.abs(runs[i][0] - ref).max():16.3e}")

    if len(runs) > 1:
        print("\n  호출별 |Δeef| (run1 vs run0):")
        print("   ", np.array2string(np.abs(runs[1][0] - runs[0][0]).max(1), precision=6, suppress_small=False))
        same = np.abs(runs[1][1] - runs[0][1]).max() == 0
        print(f"\n  정책 출력(액션)이 반복 간 완전 동일한가: {'예' if same else '아니오'}")
        print("  → 액션이 동일한데 eef가 갈리면 시뮬레이터 쪽, 액션부터 갈리면 GPU 커널 쪽 비결정성")
    elif ref_path.exists():
        print("\n  호출별 |Δeef| (run0 vs 원본):")
        print("   ", np.array2string(np.abs(runs[0][0] - ref).max(1), precision=9, suppress_small=False))


if __name__ == "__main__":
    main()
