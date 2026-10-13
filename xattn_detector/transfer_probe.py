"""Why does the within-task failure signal not reach a new task?

The timeline analysis left a clear split: on tasks the detector trained on, task-macro AUROC reaches 0.620 at
call 3; on tasks it has never seen, the same statistic sits at 0.46-0.52. The suspicion is that most of what
suf_mean encodes is WHICH TASK this is, and that the detector reads failure as a small deviation measured
against a task-specific baseline it only has for familiar tasks.

That is testable. First a variance decomposition: how much of suf_mean's variance lies between tasks versus
within a task, and how much of the within-task part lies between outcomes. Then four feature transforms, all
trained and scored identically:

  raw             what every result so far used
  task_center     subtract that task's own mean vector. On an unseen task the mean is computed from that task's
                  episodes WITHOUT their labels, so this is unsupervised adaptation, not leakage -- but it does
                  assume several rollouts of the new task are available before the detector is useful, which is
                  a real deployment cost and is reported as such.
  episode_center  subtract the episode's own first call, removing task and episode offsets together, using only
                  information available at the start of that rollout
  delta           consecutive-call differences: how the latent is MOVING rather than where it is

If task_center or episode_center transfers while raw does not, the signal was there all along and the detector
was spending its capacity on task identity. If none of them transfers, the within-task signal is task-specific
in substance, not merely offset.

    conda activate fasi
    python xattn_detector/transfer_probe.py --seeds 8
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "safe_detector"))

from buffer import Buffer  # noqa: E402
from online_runner import collect_split, episode_probs, fit_logreg, macro_auroc  # noqa: E402
from train_detector import auroc  # noqa: E402

TRANSFORMS = ("raw", "task_center", "episode_center", "delta")


def task_key(r):
    return f"{r['suite']}/t{r['task']}"


def variance_decomposition(X, M, recs):
    """Total / between-task / within-task variance of the per-call vectors, and the outcome share within tasks."""
    rows, keys, ys = [], [], []
    for i, r in enumerate(recs):
        for t in range(int(M[i].sum())):
            rows.append(X[i, t])
            keys.append(task_key(r))
            ys.append(r["y"])
    V = np.stack(rows)
    keys = np.array(keys)
    ys = np.array(ys)
    grand = V.mean(0)
    total = float(((V - grand) ** 2).sum(1).mean())

    between = 0.0
    within_parts, outcome_parts = [], []
    for k in np.unique(keys):
        m = keys == k
        mk = V[m].mean(0)
        between += m.sum() / len(V) * float(((mk - grand) ** 2).sum())
        within_parts.append((m.sum(), float(((V[m] - mk) ** 2).sum(1).mean())))
        yk = ys[m]
        if 0 < yk.sum() < len(yk):  # split the within-task part by outcome
            m1, m0 = V[m][yk == 1].mean(0), V[m][yk == 0].mean(0)
            p = yk.mean()
            outcome_parts.append((m.sum(), float(p * ((m1 - mk) ** 2).sum() + (1 - p) * ((m0 - mk) ** 2).sum())))
    w = sum(n for n, _ in within_parts)
    within = sum(n * v for n, v in within_parts) / w
    outcome = (sum(n * v for n, v in outcome_parts) / sum(n for n, _ in outcome_parts)) if outcome_parts else 0.0
    return dict(total=total, between_task=between, within_task=within, outcome_within_task=outcome)


def apply_transform(X, M, recs, name, task_means=None):
    """Return a transformed copy. `task_means` maps a task key to its mean vector (labels never consulted)."""
    Y = X.copy()
    if name == "raw":
        return Y, M
    if name == "task_center":
        for i, r in enumerate(recs):
            Y[i] = X[i] - task_means[task_key(r)]
        return Y, M
    if name == "episode_center":
        for i in range(len(recs)):
            Y[i] = X[i] - X[i, 0]
        return Y, M
    if name == "delta":
        Y = np.zeros_like(X)
        Y[:, 1:] = X[:, 1:] - X[:, :-1]
        M2 = M.copy()
        M2[:, 0] = 0.0  # the first call has no predecessor
        return Y, M2
    raise ValueError(name)


def task_means_of(X, M, recs):
    acc = collections.defaultdict(lambda: [np.zeros(X.shape[2]), 0])
    for i, r in enumerate(recs):
        n = int(M[i].sum())
        acc[task_key(r)][0] += X[i, :n].sum(0)
        acc[task_key(r)][1] += n
    return {k: v[0] / max(v[1], 1) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", default="data/online")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--calls", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=100.0)
    ap.add_argument("--select", default="macro")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    buf = Buffer.load(HERE / args.buffer)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr, Mtr, ytr, rtr = collect_split(buf, "train", calls=args.calls)
    Xva, Mva, yva, rva = collect_split(buf, "val", calls=args.calls)
    Xte, Mte, yte, rte = collect_split(buf, "test", calls=args.calls)
    off = [i for i, r in enumerate(rtr) if r.get("source") != "online"]
    onl = [i for i, r in enumerate(rtr) if r.get("source") == "online"]
    Xtr_o, Mtr_o, ytr_o, rtr_o = Xtr[off], Mtr[off], ytr[off], [rtr[i] for i in off]
    Xsn, Msn, ysn, rsn = Xtr[onl], Mtr[onl], ytr[onl], [rtr[i] for i in onl]

    print(f"학습 offline {len(rtr_o)}  |  평가: 학습한 task의 새 롤아웃 {len(rsn)}, "
          f"미학습 task {len(rte)}\n")

    print("=== suf_mean 분산 분해 (호출 단위 벡터) ===")
    for name, (X, M, R) in (("학습 offline", (Xtr_o, Mtr_o, rtr_o)),
                            ("미학습 task", (Xte, Mte, rte))):
        d = variance_decomposition(X, M, R)
        print(f"  {name:12s} 전체 {d['total']:9.1f}   task 간 {d['between_task']:9.1f} "
              f"({d['between_task'] / d['total']:5.1%})   task 내 {d['within_task']:9.1f} "
              f"({d['within_task'] / d['total']:5.1%})")
        print(f"  {'':12s} 그중 성공/실패가 설명하는 몫 {d['outcome_within_task']:8.2f} "
              f"(전체의 {d['outcome_within_task'] / d['total']:.2%}, task 내의 "
              f"{d['outcome_within_task'] / d['within_task']:.2%})")
    print()

    print("=== feature 변환별 task-macro AUROC (8시드 평균 ± sd) ===")
    print(f"  {'변환':14s} {'학습한 task (새 롤아웃)':>24s} {'미학습 task':>20s} {'전이 손실':>10s}")
    out = {}
    for name in TRANSFORMS:
        tm_tr = task_means_of(Xtr_o, Mtr_o, rtr_o)
        tm_va = task_means_of(Xva, Mva, rva)
        tm_sn = task_means_of(Xsn, Msn, rsn)
        tm_te = task_means_of(Xte, Mte, rte)  # unseen task: its own episodes, labels never used
        A, Ma = apply_transform(Xtr_o, Mtr_o, rtr_o, name, tm_tr)
        B, Mb = apply_transform(Xva, Mva, rva, name, tm_va)
        C, Mc = apply_transform(Xsn, Msn, rsn, name, tm_sn)
        D, Md = apply_transform(Xte, Mte, rte, name, tm_te)
        seen, unseen = [], []
        for seed in range(args.seeds):
            net, mu, sd, _ = fit_logreg(A, Ma, ytr_o, B, Mb, yva, rva, args, seed, device)
            seen.append(macro_auroc(episode_probs(net, mu, sd, C, Mc, device), ysn, rsn)[0])
            unseen.append(macro_auroc(episode_probs(net, mu, sd, D, Md, device), yte, rte)[0])
        f = lambda v: (float(np.nanmean(v)), float(np.nanstd(v, ddof=1)))
        out[name] = dict(seen=f(seen), unseen=f(unseen))
        s, u = out[name]["seen"], out[name]["unseen"]
        print(f"  {name:14s} {s[0]:16.3f}±{s[1]:.3f} {u[0]:14.3f}±{u[1]:.3f} {u[0] - s[0]:+10.3f}")

    path = HERE / "results" / f"transfer_probe_{args.tag}.json"
    path.write_text(json.dumps(dict(seeds=args.seeds, calls=args.calls, cells=out), indent=2,
                               ensure_ascii=False, default=float))
    print(f"\n-> {path}")
    print("\n  task_center 의 미학습 task 열은 그 task의 롤아웃 여러 개를 라벨 없이 먼저 모아야 계산된다 —")
    print("  새 task에 즉시 적용할 수 있는 수치가 아니라, '적응 기간을 주면 되는가'에 대한 답이다.")


if __name__ == "__main__":
    main()
