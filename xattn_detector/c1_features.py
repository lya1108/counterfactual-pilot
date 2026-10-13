"""Where does the C1 signal live? Same target, same split, different representations.

Target and split are unchanged from the canonical setting: calls where C1 is not already active, positive if
the episode-global onset falls in (t, t+H], held-out-task macro AUROC over repeated task-group splits.

Representations, all already on disk -- nothing is re-collected:
  suf_mean    action expert tokens, mean-pooled                      1024
  suf_tokens  the same tokens, flattened                            51200
  vlm_img     VLM image tokens, mean-pooled (2 cameras, 8x8 each)     2048
  vlm_lang    VLM language tokens, mean-pooled over valid positions   2048
              -- these carry the discretised robot state as digits (processor_pi05.py:83)
  vlm_all     image and language concatenated                        4096
  state       15-D instantaneous state
  state_hist  the same 15-D at t, t-5, t-10                            45
  opening / call_index / random                                        baselines

suf_mean is a function of the action chunk pi0.5 is generating, which the state features are not. A gap
between them is therefore not by itself evidence that the latent "knows more about the world" -- it is
reported as a measurement, not an interpretation.

    python xattn_detector/c1_features.py --data data/cond200 --H 10
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

ORDER = ("suf_mean", "suf_tokens", "vlm_all", "vlm_img", "vlm_lang",
         "state_hist", "state", "opening", "call_index", "random")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cond200")
    ap.add_argument("--H", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--hist", type=int, default=3)
    ap.add_argument("--hist-stride", type=int, default=5)
    ap.add_argument("--tag", default="n200")
    args = ap.parse_args()

    rows_all = build(HERE / args.data, args.hist, want_tokens=True, hist_stride=args.hist_stride)
    rng = np.random.default_rng(0)
    for r in rows_all:
        r["random"] = rng.standard_normal(4).astype(np.float32)
    rows, y = target_horizon(rows_all, args.H, 0)
    groups = sorted({r["group"] for r in rows}); gl = [r["group"] for r in rows]
    by = collections.defaultdict(list)
    for g, yy in zip(gl, y): by[g].append(yy)
    both = sum(1 for g in groups if 0 < sum(by[g]) < len(by[g]))
    n_ep = len({(r["suite"], r["task"], r["init"]) for r in rows_all})
    feats = [f for f in ORDER if f in rows[0]]
    print(f"에피소드 {n_ep}   샘플 {len(rows)}   양성 {int(y.sum())} ({y.mean():.1%})   "
          f"양·음 모두 있는 task {both}/{len(groups)}   H={args.H}, guard=0")
    print(f"state_hist = state_vector(t, t-{args.hist_stride}, t-{2*args.hist_stride}), "
          f"{args.hist}개 시점 × 15차원\n")
    print(f"  {'feature':12s} {'차원':>6s} {'macro AUROC':>18s} {'macro AUPRC':>18s} {'하한':>7s}")
    out, per_task = {}, {}
    for f in feats:
        X = np.stack([np.asarray(r[f]).ravel() for r in rows])
        aur, aup, rate, last = [], [], None, {}
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
            bb = collections.defaultdict(lambda: ([], []))
            for s_, yy, g in zip(sc, y, gl): bb[g][0].append(s_); bb[g][1].append(yy)
            from train_detector import auroc as _au
            last = {g: _au(np.array(a2), np.array(b2)) for g, (a2, b2) in bb.items()
                    if 0 < sum(b2) < len(b2)}
        m = lambda v: (float(np.nanmean(v)), float(np.nanstd(v, ddof=1)))
        out[f] = dict(dim=int(X.shape[1]), auroc=m(aur), auprc=m(aup))
        per_task[f] = last
        print(f"  {f:12s} {X.shape[1]:6d} {out[f]['auroc'][0]:11.3f} ± {out[f]['auroc'][1]:.3f} "
              f"{out[f]['auprc'][0]:11.3f} ± {out[f]['auprc'][1]:.3f} {rate:7.3f}")

    print(f"\n=== task 별 AUROC (마지막 분할 기준) ===")
    show = [f for f in ("suf_mean", "vlm_all", "state_hist", "state") if f in per_task]
    print(f"  {'task':22s} {'양성률':>7s} " + " ".join(f"{f:>12s}" for f in show))
    for g in sorted(groups, key=lambda g: -per_task[show[0]].get(g, 0)):
        if g not in per_task[show[0]]: continue
        print(f"  {g:22s} {np.mean(by[g]):6.1%} " +
              " ".join(f"{per_task[f].get(g, float('nan')):12.3f}" for f in show))
    path = HERE / "results" / f"c1_features_{args.tag}.json"
    path.write_text(json.dumps(dict(args=vars(args), n_episodes=n_ep, n_rows=len(rows),
                                    n_pos=int(y.sum()), tasks_both=both, out=out,
                                    per_task=per_task), indent=2, ensure_ascii=False, default=float))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
