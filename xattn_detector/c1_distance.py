"""How far ahead of C1 does the signal appear?

The horizon sweep cannot answer this on its own. Widening H adds distant samples but keeps the near ones, so
H=30 scoring 0.813 could mean "we see it 30 steps out" or "the 0-10 step cases are still carrying it".

Here each bin is scored on its own. Positives are the calls whose onset is that far away and no closer;
negatives are the same set every time -- calls with no C1 onset within `--clear` steps. So the only thing that
changes between rows is how distant the positives are, and a decline across rows is the signal fading rather
than the mix shifting.

Onset is the episode-global definition from c1_depth.build: the condition's debounce is judged once over the
whole episode and its distance measured from each call. Calls where C1 is already active are excluded, so a
positive is always a violation that has not happened yet.

    conda activate fasi
    python xattn_detector/c1_distance.py --data data/cond200 --repeats 20
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

from c1_depth import build  # noqa: E402
from c1_horizon import FEATURES, macro_both  # noqa: E402
from prospective_c1 import fit_eval  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cond200")
    ap.add_argument("--bins", default="0-10,10-20,20-30,30-40")
    ap.add_argument("--clear", type=int, default=40, help="음성 조건: 앞으로 이만큼 안에 onset 없음")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--tag", default="n200")
    args = ap.parse_args()

    rows_all = build(HERE / args.data, hist=3, want_tokens=False)
    rng = np.random.default_rng(0)
    for r in rows_all:
        r["random"] = rng.standard_normal(4).astype(np.float32)
    n_ep = len({(r["suite"], r["task"], r["init"]) for r in rows_all})
    neg = [r for r in rows_all if r["d_onset"] is None or r["d_onset"] > args.clear]
    neg_tasks = {r["group"] for r in neg}
    print(f"에피소드 {n_ep}   C1(t)=0 호출 샘플 {len(rows_all)}")
    print(f"음성: 앞으로 {args.clear} step 안에 onset 없는 호출 {len(neg)}개 "
          f"({len(neg_tasks)}개 task) — 모든 bin 에서 동일\n")

    out = {}
    for spec in args.bins.split(","):
        lo, hi = (int(x) for x in spec.split("-"))
        pos = [r for r in rows_all if r["d_onset"] is not None and lo < r["d_onset"] <= hi]
        rows = pos + neg
        y = np.array([1.0] * len(pos) + [0.0] * len(neg))
        groups = sorted({r["group"] for r in rows})
        gl = [r["group"] for r in rows]
        by = collections.defaultdict(list)
        for g, yy in zip(gl, y):
            by[g].append(yy)
        both = sum(1 for g in groups if 0 < sum(by[g]) < len(by[g]))
        print(f"=== onset 까지 {lo}–{hi} env step ===")
        print(f"  샘플 {len(rows)}   양성 {len(pos)} ({y.mean():.1%})   양·음 모두 있는 task {both}/{len(groups)}")
        if len(pos) < 15 or both < 3:
            print("  — 양성 또는 task 부족, 건너뜀\n")
            continue
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
        out[spec] = dict(n=len(rows), pos=len(pos), neg=len(neg), pos_rate=float(y.mean()),
                         tasks_both=both, n_groups=len(groups), base_rate_macro=float(rate), cells=cell)
        print()

    print("  음성 집합이 모든 bin 에서 같으므로, 행 사이의 차이는 양성이 얼마나 먼 미래인지뿐이다.")
    path = HERE / "results" / f"c1_distance_{args.tag}.json"
    path.write_text(json.dumps(dict(args=vars(args), n_episodes=n_ep, n_neg=len(neg), out=out),
                               indent=2, ensure_ascii=False, default=float))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
