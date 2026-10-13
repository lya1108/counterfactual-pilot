"""Prospective C1 across prediction horizons. Guard interval fixed at 0; nothing else is varied.

Samples are the VLA calls where C1 is NOT already active, and the target is whether C1's onset falls in
(t, t+H]. Widening H changes two things at once -- it asks about a longer future AND it makes positives more
common -- so the positive rate is printed beside every row; a rise in AUPRC that merely tracks the positive
rate is not a gain.

AUPRC is reported because the positive class is rare: its floor is the positive rate itself, not 0.5, so the
two numbers have to be read together. Both are macro over held-out tasks, averaged the same way as AUROC.

    conda activate fasi
    python xattn_detector/c1_horizon.py --data data/cond200 --repeats 20
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "safe_detector"))

from c1_depth import build, target_horizon  # noqa: E402
from prospective_c1 import fit_eval  # noqa: E402
from train_detector import auroc  # noqa: E402

FEATURES = ("suf_mean", "state", "opening", "call_index", "random")


def average_precision(scores, y):
    """Area under precision-recall, computed as the mean precision at each positive (sklearn is unavailable)."""
    y = np.asarray(y, float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(-np.asarray(scores, float), kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / ys.sum())


def macro_both(scores, y, groups):
    by = collections.defaultdict(lambda: ([], []))
    for s, yy, g in zip(scores, y, groups):
        by[g][0].append(s)
        by[g][1].append(yy)
    aur, aup, rates = [], [], []
    for g, (a, b) in by.items():
        b = np.asarray(b, float)
        if 0 < b.sum() < len(b):
            aur.append(auroc(np.asarray(a), b))
            aup.append(average_precision(a, b))
            rates.append(b.mean())
    if not aur:
        return float("nan"), float("nan"), float("nan"), 0
    return float(np.mean(aur)), float(np.mean(aup)), float(np.mean(rates)), len(aur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cond200")
    ap.add_argument("--horizons", default="5,10,20,30")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--tag", default="n200")
    args = ap.parse_args()

    rows_all = build(HERE / args.data, hist=3, want_tokens=False)
    rng = np.random.default_rng(0)
    for r in rows_all:
        r["random"] = rng.standard_normal(4).astype(np.float32)
    n_ep = len({(r["suite"], r["task"], r["init"]) for r in rows_all})
    print(f"에피소드 {n_ep}   C1(t)=0 호출 샘플 {len(rows_all)}   guard k=0 고정")
    print(f"분할 {args.folds}-fold × {args.repeats}회 재추출, 1차 지표 = 미학습 task macro AUROC\n")

    out = {}
    for H in [int(x) for x in args.horizons.split(",")]:
        rows, y = target_horizon(rows_all, H, 0)
        groups = sorted({r["group"] for r in rows})
        gl = [r["group"] for r in rows]
        by = collections.defaultdict(list)
        for g, yy in zip(gl, y):
            by[g].append(yy)
        both = sum(1 for g in groups if 0 < sum(by[g]) < len(by[g]))
        print(f"=== H = {H} env step ===")
        print(f"  샘플 {len(rows)}   양성 {int(y.sum())} ({y.mean():.1%})   "
              f"양·음 모두 있는 task {both}/{len(groups)}")
        print(f"  {'feature':12s} {'macro AUROC':>18s} {'macro AUPRC':>18s} {'AUPRC 하한':>11s}")
        cell = {}
        for f in FEATURES:
            X = np.stack([r[f] for r in rows])
            aur, aup, rate = [], [], None
            for rep in range(args.repeats):
                r2 = np.random.default_rng(100 + rep)
                fold_of = {g: i % args.folds for i, g in enumerate(r2.permutation(groups))}
                sc = np.zeros(len(rows))
                for fold in range(args.folds):
                    te = np.array([fold_of[r["group"]] == fold for r in rows])
                    if te.sum() == 0 or y[~te].sum() < 2 or (1 - y[~te]).sum() < 2:
                        continue
                    sc[te] = fit_eval(X[~te], y[~te], X[te], seed=rep)
                a, p, rate, _ = macro_both(sc, y, gl)
                aur.append(a); aup.append(p)
            m = lambda v: (float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0)
            cell[f] = dict(auroc=m(aur), auprc=m(aup))
            print(f"  {f:12s} {cell[f]['auroc'][0]:11.3f} ± {cell[f]['auroc'][1]:.3f} "
                  f"{cell[f]['auprc'][0]:11.3f} ± {cell[f]['auprc'][1]:.3f} {rate:11.3f}")
        out[str(H)] = dict(n=len(rows), pos=int(y.sum()), pos_rate=float(y.mean()),
                           tasks_both=both, n_groups=len(groups), base_rate_macro=float(rate), cells=cell)
        print()

    print("  AUPRC 하한 = 평가에 쓰인 task 들의 평균 양성률. AUPRC 는 이 값과 비교해서 읽어야 한다.")
    print("  H 를 넓히면 질문의 범위와 양성 비율이 함께 바뀌므로, 두 열을 같이 봐야 한다.")
    path = HERE / "results" / f"c1_horizon_{args.tag}.json"
    path.write_text(json.dumps(dict(args=vars(args), n_episodes=n_ep, out=out), indent=2,
                               ensure_ascii=False, default=float))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
