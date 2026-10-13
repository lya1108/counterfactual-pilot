"""Can a linear model read the task identity out of suf_mean?

The variance decomposition said 29-39% of suf_mean's variance sits between tasks. That is not the same question
as whether a classifier can USE it: a direction can hold little variance and still be perfectly separable, and
a large share can be spread in ways no linear map exploits. So this measures the thing directly -- multinomial
logistic regression from one call's suf_mean to which of the 24 tasks produced it.

Episodes are split by init, not by task, because predicting the identity of a task the model has never seen is
not a defined problem. Chance is 1/n_tasks, and a majority-class baseline is printed too since the tasks are
not equally represented.

    conda activate fasi
    python xattn_detector/task_id_probe.py
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "safe_detector"))

from buffer import Buffer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", default="data/online")
    ap.add_argument("--calls", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--holdout-inits", default="16,17,18,19", help="평가로 뺄 init (task가 아니라 init로 분할)")
    args = ap.parse_args()

    buf = Buffer.load(HERE / args.buffer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hold = {int(x) for x in args.holdout_inits.split(",")}

    recs = [r for r in buf.records if r["n_calls"] >= args.calls]
    keys = sorted({f"{r['suite']}/t{r['task']}" for r in recs})
    kidx = {k: i for i, k in enumerate(keys)}
    Xtr, ytr, Xte, yte = [], [], [], []
    for r in recs:
        f = buf.fetch(r, ("suf_mean",))["suf_mean"][: args.calls].astype(np.float32)
        lab = kidx[f"{r['suite']}/t{r['task']}"]
        (Xte if r["init"] in hold else Xtr).append(f)
        (yte if r["init"] in hold else ytr).append(np.full(len(f), lab))
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    Xte = np.concatenate(Xte); yte = np.concatenate(yte)
    n_cls = len(keys)
    counts = collections.Counter(yte.tolist())
    major = max(counts.values()) / len(yte)
    print(f"task {n_cls}개  학습 호출 {len(Xtr)}  평가 호출 {len(Xte)} (init {sorted(hold)} 보류)")
    print(f"우연 {1 / n_cls:.3%}   최빈 클래스 {major:.3%}\n")

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    xt = torch.from_numpy((Xtr - mu) / sd).to(device)
    yt = torch.from_numpy(ytr).long().to(device)
    xe = torch.from_numpy((Xte - mu) / sd).to(device)

    accs, top3s = [], []
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        net = nn.Linear(Xtr.shape[1], n_cls).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for _ in range(args.epochs):
            opt.zero_grad()
            nn.functional.cross_entropy(net(xt), yt).backward()
            opt.step()
        with torch.no_grad():
            logit = net(xe).cpu().numpy()
        pred = logit.argmax(1)
        accs.append(float((pred == yte).mean()))
        top3 = np.argsort(-logit, 1)[:, :3]
        top3s.append(float(np.mean([yte[i] in top3[i] for i in range(len(yte))])))
    print(f"task-ID 정확도   {np.mean(accs):.3%} ± {np.std(accs, ddof=1):.3%}   "
          f"(top-3 {np.mean(top3s):.3%})")
    print(f"우연 대비        {np.mean(accs) / (1 / n_cls):.0f}배")

    out = HERE / "results" / "task_id_probe.json"
    out.write_text(json.dumps(dict(n_tasks=n_cls, chance=1 / n_cls, majority=major,
                                   acc=[float(np.mean(accs)), float(np.std(accs, ddof=1))],
                                   top3=float(np.mean(top3s)), holdout_inits=sorted(hold)),
                              indent=2, ensure_ascii=False))
    print(f"\n-> {out}")
    print("  이 수치가 높다면 suf_mean 안에 task 정체성이 선형으로 꺼낼 수 있는 형태로 들어 있다는 뜻이고,")
    print("  failure detector 가 그 방향으로 용량을 쓸 여지가 실제로 존재한다는 의미다.")


if __name__ == "__main__":
    main()
