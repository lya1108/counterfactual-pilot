"""Does a drifted replay still end in the same outcome? If not, its stored label cannot be reused.

collect_tokens.py flags an episode when the first 8 calls of the replay leave the recorded end-effector path.
A drifted path is not by itself a reason to distrust the label -- what matters is whether the episode still
ends the same way. That is only answerable by replaying the WHOLE episode, so this script does exactly that,
`--runs` times, and compares the outcome with the one safe_detector recorded.

Flagged episodes alone would not settle it: if outcomes also flip on episodes that replayed cleanly, the
problem is the benchmark's sensitivity rather than anything about the flagged set. So an equal-sized random
control group is drawn from the clean episodes and replayed the same way.

    conda activate fasi
    python xattn_detector/label_check.py --tokens data/tokens --runs 2 --control 20
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
sys.path.insert(0, str(HERE.parent / "safe_detector"))
import jb  # noqa: E402

jb.enter()

from fasi.envs.libero_env import LiberoTask  # noqa: E402
from fasi.vla.pi05_steer import Pi05Steer  # noqa: E402


def full_rollout(vla, task, tid, init, n_exec=10):
    """The loop safe_detector/collect_rollouts.py ran, to the end, returning the outcome it reaches."""
    task.reset(init)
    t, ok = 0, False
    while t < task.max_steps:
        ag = task.obs["agentview_image"][::-1, ::-1].copy()
        wr = task.obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
        b = vla.batch_for(ag, wr, task.state8(), task.language)
        g = torch.Generator().manual_seed(1000 * tid + 17 * init + t)
        a = vla.denoise(b, torch.randn((1, vla.C, vla.A), generator=g), [], None)[0]
        for j in range(n_exec):
            task.step(a[j])
            t += 1
            if task.success():
                ok = True
                break
        if ok:
            break
    return ok, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="data/tokens")
    ap.add_argument("--runs", type=int, default=2, help="replays per episode")
    ap.add_argument("--control", type=int, default=20, help="clean episodes to replay as a control group")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/label_check.json")
    args = ap.parse_args()

    tok = HERE / args.tokens if not Path(args.tokens).is_absolute() else Path(args.tokens)
    rows = [json.loads(l) for l in (tok / "episodes.jsonl").read_text().splitlines() if l.strip()]
    flagged = [r for r in rows if not r["repro_ok"]]
    clean = [r for r in rows if r["repro_ok"]]
    rng = np.random.default_rng(args.seed)
    control = [clean[i] for i in rng.choice(len(clean), min(args.control, len(clean)), replace=False)]
    print(f"총 {len(rows)} · 불일치 {len(flagged)} · 대조군 {len(control)} (일치 에피소드에서 무작위)")
    print(f"각 {args.runs}회 전체 재생 — 예상 {(len(flagged) + len(control)) * args.runs * 15 / 60:.0f}분\n", flush=True)

    vla = Pi05Steer("outputs/vla/pi05_libero_finetuned")
    out, task, cur = [], None, None
    t0 = time.time()
    for group, items in (("flagged", flagged), ("control", control)):
        for r in items:
            key = (r["suite"], r["task"])
            if key != cur:
                if task is not None:
                    task.close()
                task = LiberoTask(r["suite"], r["task"])
                cur = key
            res = [full_rollout(vla, task, r["task"], r["init"]) for _ in range(args.runs)]
            succ = [s for s, _ in res]
            agree = all(s == r["success"] for s in succ)
            stable = len(set(succ)) == 1
            out.append(dict(group=group, suite=r["suite"], task=r["task"], init=r["init"],
                            stored_success=r["success"], replays=succ, replay_t=[t for _, t in res],
                            stored_t=r["t_full"], agrees_with_stored=agree, self_consistent=stable,
                            d_eef=r["d_eef"]))
            mark = "OK " if agree else "불일치"
            print(f"  [{group:7s}] {r['suite']} t{r['task']} i{r['init']}  "
                  f"저장={r['success']}  재생={succ}  {mark}  (d_eef={r['d_eef']:.1e})", flush=True)
    if task is not None:
        task.close()

    print(f"\n{'=' * 72}\n{time.time() - t0:.0f}초")
    summary = {}
    for group in ("flagged", "control"):
        g = [o for o in out if o["group"] == group]
        if not g:
            continue
        agree = sum(o["agrees_with_stored"] for o in g)
        stable = sum(o["self_consistent"] for o in g)
        summary[group] = dict(n=len(g), agrees=agree, self_consistent=stable)
        print(f"  {group:8s} n={len(g):3d}   저장 라벨과 일치 {agree}/{len(g)}   "
              f"재생끼리 일관 {stable}/{len(g)}")
    print("\n  해석: 대조군도 비슷하게 흔들리면 라벨 불안정은 플래그와 무관한 벤치마크 특성이고,")
    print("        플래그 그룹만 흔들리면 플래그가 실제로 신뢰할 수 없는 라벨을 골라낸 것이다.")

    path = HERE / args.out if not Path(args.out).is_absolute() else Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(args=vars(args), summary=summary, episodes=out), indent=2, ensure_ascii=False))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
