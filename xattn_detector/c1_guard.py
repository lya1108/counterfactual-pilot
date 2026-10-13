"""Does the C1 signal survive a guard interval, or does it live in the last few steps before the violation?

H is fixed at 10. The guard k removes the samples whose onset is imminent -- and removes them, rather than
relabelling them negative: a call with the onset 2 steps away is neither a clean positive for "will it happen
between t+3 and t+10" nor a true negative, so it is dropped from the evaluation entirely.

    y_t = 1[ C1 onset in (t+k, t+10] ],  evaluated only on calls whose onset is not within (t, t+k]

Calls where C1 is already active are excluded throughout, as in every result since the canonical definition.

    python xattn_detector/c1_guard.py --data data/cond200 --repeats 20
"""
from __future__ import annotations

import argparse, collections, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "safe_detector"))

from c1_depth import build, target_horizon
from c1_horizon import macro_both
from prospective_c1 import fit_eval

FEATURES = ("suf_mean", "vlm_img", "state_hist", "state", "call_index")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cond200")
    ap.add_argument("--H", type=int, default=10)
    ap.add_argument("--guards", default="0,3,5")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--hist", type=int, default=3)
    ap.add_argument("--hist-stride", type=int, default=5)
    ap.add_argument("--tag", default="n200")
    args = ap.parse_args()

    rows_all = build(HERE / args.data, args.hist, want_tokens=True, hist_stride=args.hist_stride)
    n_ep = len({(r["suite"], r["task"], r["init"]) for r in rows_all})
    print(f"에피소드 {n_ep}   C1(t)=0 호출 샘플 {len(rows_all)}   H={args.H} 고정")
    print(f"state_hist = state_vector(t, t-{args.hist_stride}, t-{2*args.hist_stride}), 45차원")
    print(f"guard k: onset 이 (t, t+k] 안인 샘플은 음성으로 바꾸지 않고 평가에서 제외\n")

    out = {}
    for k in [int(x) for x in args.guards.split(",")]:
        rows, y = target_horizon(rows_all, args.H, k)
        dropped = len(rows_all) - len(rows)
        groups = sorted({r["group"] for r in rows}); gl = [r["group"] for r in rows]
        by = collections.defaultdict(list)
        for g, yy in zip(gl, y): by[g].append(yy)
        both = sum(1 for g in groups if 0 < sum(by[g]) < len(by[g]))
        print(f"=== guard k = {k} ===")
        print(f"  샘플 {len(rows)} (제외 {dropped})   양성 {int(y.sum())} ({y.mean():.1%})   "
              f"양·음 모두 있는 task {both}/{len(groups)}")
        print(f"  {'feature':12s} {'macro AUROC':>18s} {'macro AUPRC':>18s} {'하한':>7s}")
        cell = {}
        for f in FEATURES:
            X = np.stack([np.asarray(r[f]).ravel() for r in rows])
            aur, aup, rate = [], [], None
            for rep in range(args.repeats):
                r2 = np.random.default_rng(100 + rep)
                fold_of = {g: i % args.folds for i, g in enumerate(r2.permutation(groups))}
                sc = np.zeros(len(rows))
                for fold in range(args.folds):
                    te = np.array([fold_of[r["group"]] == fold for r in rows])
                    if te.sum() == 0 or y[~te].sum() < 2 or (1 - y[~te]).sum() < 2: continue
                    sc[te] = fit_eval(X[~te], y[~te], X[te], seed=rep)
                a, p, rate, _ = macro_both(sc, y, gl)
                aur.append(a); aup.append(p)
            m = lambda v: (float(np.nanmean(v)), float(np.nanstd(v, ddof=1)))
            cell[f] = dict(auroc=m(aur), auprc=m(aup))
            print(f"  {f:12s} {cell[f]['auroc'][0]:11.3f} ± {cell[f]['auroc'][1]:.3f} "
                  f"{cell[f]['auprc'][0]:11.3f} ± {cell[f]['auprc'][1]:.3f} {rate:7.3f}")
        out[str(k)] = dict(n=len(rows), dropped=dropped, pos=int(y.sum()), pos_rate=float(y.mean()),
                           tasks_both=both, base_rate_macro=float(rate), cells=cell)
        print()

    path = HERE / "results" / f"c1_guard_{args.tag}.json"
    path.write_text(json.dumps(dict(args=vars(args), n_episodes=n_ep, out=out), indent=2,
                               ensure_ascii=False, default=float))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
