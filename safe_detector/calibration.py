"""Are the detectors' failure PROBABILITIES any good -- not just their ranking?

AUROC only reads the order of the scores. Brier reads the numbers: a detector that says 0.9 on every episode
and is right 60% of the time ranks perfectly and is still badly wrong as a probability.

On a binary outcome Brier and R² are the same quantity in different clothes:

    Brier = mean (p - y)^2            SS_tot / n = ȳ(1 - ȳ)
    R²    = 1 - Brier / (ȳ(1 - ȳ))

so R² here IS the Brier skill score against "always predict the base rate". Both are printed because both were
asked for, but the column that carries information beyond them is the Murphy decomposition:

    Brier = reliability - resolution + uncertainty

  reliability  how far the stated probabilities sit from the observed frequencies (0 is perfect; lower better)
  resolution   how much the episodes it separates actually differ in outcome (higher better)
  uncertainty  ȳ(1 - ȳ), fixed by the test set -- nothing a model does changes it

Two caveats this script measures rather than hides:
  - Training weights the two classes equally, so the output prior sits near 0.5 while the LIBERO test sets run
    ~60% failure. That is a reliability error, not a discrimination error, and the decomposition separates them.
  - `raw` columns are the model's own numbers; `platt` columns refit a scalar slope and intercept on the
    TRAINING episodes only and apply it to test, which fixes calibration without touching the ranking.

    conda activate fasi
    python safe_detector/calibration.py --seeds 0-7
    python safe_detector/calibration.py --seeds 0 --dump results/calib_episodes.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent

from multiseed import PREFIX_K, truncate  # noqa: E402
from multiseed import run_cell  # noqa: E402
from train_detector import auroc, load_episodes, split_episodes  # noqa: E402

CELLS = [("logreg", "vlm_mean"), ("lstm", "vlm_mean"), ("logreg", "act_mean"), ("lstm", "act_mean")]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def episode_prob(scores):
    """One failure probability per episode: the mean of its per-call probabilities.

    Not the max -- the max of several probabilities is not itself a probability, and with 8 calls per episode it
    would drift toward 1 for every episode regardless of outcome.
    """
    return np.array([sigmoid(s).mean() for s in scores], float)


def platt(train_logit, train_y, test_logit):
    """Fit p = sigmoid(a * z + b) on the training episodes; apply to test. Monotone, so AUROC is unchanged."""
    z = torch.tensor(train_logit, dtype=torch.float64)
    y = torch.tensor(train_y, dtype=torch.float64)
    a = torch.ones(1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(a * z + b, y)
        loss.backward()
        return loss

    opt.step(closure)
    return sigmoid(a.item() * np.asarray(test_logit) + b.item()), a.item(), b.item()


def brier_parts(p, y, bins=10):
    """Brier, R², and the Murphy decomposition on equal-width probability bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    base = y.mean()
    brier = float(np.mean((p - y) ** 2))
    unc = float(base * (1 - base))
    r2 = float("nan") if unc == 0 else 1.0 - brier / unc
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    rel = res = 0.0
    for k in range(bins):
        m = idx == k
        if not m.any():
            continue
        w = m.sum() / len(p)
        rel += w * (p[m].mean() - y[m].mean()) ** 2
        res += w * (y[m].mean() - base) ** 2
    return dict(brier=brier, r2=r2, reliability=float(rel), resolution=float(res),
                uncertainty=unc, mean_p=float(p.mean()), base_rate=float(base))


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data"])
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--splits", nargs="+", default=["init", "task"])
    ap.add_argument("--cond", default="prefix8", choices=["prefix8", "full"])
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--dump", default=None, help="write the per-episode table of the FIRST seed to this csv")
    ap.add_argument("--out", default="results/calibration.json")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    roots = []
    for d in args.data:
        p = Path(d)
        if not p.is_absolute():
            p = HERE / p
        roots += [p] if (p / "episodes.jsonl").exists() else sorted(q for q in p.iterdir() if (q / "episodes.jsonl").exists())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"cond={args.cond}  seeds={seeds}  splits={args.splits}  device={device}\n", flush=True)

    out, dumped = {}, []
    for feature in sorted({f for _, f in CELLS}):
        eps = load_episodes(roots, feature)
        view = truncate(eps, PREFIX_K) if args.cond == "prefix8" else eps
        for split in args.splits:
            for model, feat in CELLS:
                if feat != feature:
                    continue
                rows = []
                for seed in seeds:
                    tr, te, _ = split_episodes(view, split, args.holdout, seed)
                    _, te_s, tr_s = run_cell(tr, te, model, args.epochs, args.lr, seed, device, return_scores=True)
                    te_y = np.array([e["y"] for e in te], float)
                    tr_y = np.array([e["y"] for e in tr], float)
                    p_raw = episode_prob(te_s)
                    # Platt is fitted on the episode-level logit, i.e. the mean per-call logit
                    z_te = np.array([s.mean() for s in te_s], float)
                    z_tr = np.array([s.mean() for s in tr_s], float)
                    p_cal, slope, _ = platt(z_tr, tr_y, z_te)
                    # the same detector scored on its OWN training episodes: a large train-test gap is what makes
                    # a train-fitted recalibration hurt instead of help
                    p_tr = episode_prob(tr_s)
                    rows.append((brier_parts(p_raw, te_y, args.bins),
                                 brier_parts(p_cal, te_y, args.bins),
                                 auroc(p_raw, te_y), slope,
                                 brier_parts(p_tr, tr_y, args.bins)["brier"], auroc(p_tr, tr_y)))
                    if args.dump and seed == seeds[0]:
                        for e, pr, pc, yy in zip(te, p_raw, p_cal, te_y):
                            dumped.append(dict(split=split, model=model, feature=feature, seed=seed,
                                               suite=e["suite"], task=e["task"], init=e["init"],
                                               label="FAILURE" if yy else "SUCCESS",
                                               p_raw=round(float(pr), 4), p_platt=round(float(pc), 4)))
                key = f"{model}_{feature}_{split}"
                agg = lambda which, k: np.array([r[which][k] for r in rows], float)
                out[key] = {"n_seeds": len(rows),
                            "auroc": [float(np.mean([r[2] for r in rows])), float(np.std([r[2] for r in rows], ddof=1))],
                            "platt_slope": [float(np.mean([r[3] for r in rows])), float(np.std([r[3] for r in rows], ddof=1))],
                            "train_brier": float(np.mean([r[4] for r in rows])),
                            "train_auroc": float(np.mean([r[5] for r in rows])),
                            **{f"raw_{k}": [float(agg(0, k).mean()), float(agg(0, k).std(ddof=1))]
                               for k in ("brier", "r2", "reliability", "resolution", "uncertainty", "mean_p", "base_rate")},
                            **{f"platt_{k}": [float(agg(1, k).mean()), float(agg(1, k).std(ddof=1))]
                               for k in ("brier", "r2", "reliability", "resolution")}}
        del eps, view

    # ---------------------------------------------------------------- report
    for split in args.splits:
        rows = {k: v for k, v in out.items() if k.endswith("_" + split)}
        if not rows:
            continue
        base = list(rows.values())[0]["raw_base_rate"][0]
        unc = list(rows.values())[0]["raw_uncertainty"][0]
        print("=" * 100)
        print(f"{split}_{args.cond}   테스트 실패 기저율 ȳ = {base:.3f}   uncertainty ȳ(1-ȳ) = {unc:.4f}   "
              f"{len(seeds)}시드 평균 ± sd")
        print(f"  {'detector':20s} {'mean p':>8s} {'Brier':>14s} {'R²':>15s} {'rel':>8s} {'res':>8s} "
              f"{'Brier(platt)':>14s} {'R²(platt)':>10s}")
        for k, v in sorted(rows.items(), key=lambda kv: kv[1]["raw_brier"][0]):
            name = k[: -(len(split) + 1)]
            print(f"  {name:20s} {v['raw_mean_p'][0]:8.3f} "
                  f"{v['raw_brier'][0]:7.4f}±{v['raw_brier'][1]:.4f} "
                  f"{v['raw_r2'][0]:8.3f}±{v['raw_r2'][1]:.3f} "
                  f"{v['raw_reliability'][0]:8.4f} {v['raw_resolution'][0]:8.4f} "
                  f"{v['platt_brier'][0]:7.4f}±{v['platt_brier'][1]:.4f} "
                  f"{v['platt_r2'][0]:9.3f}")
        print(f"\n  {'detector':20s} {'train Brier':>12s} {'train AUROC':>12s} {'test AUROC':>11s} "
              f"{'gap(AUROC)':>11s} {'platt 기울기':>13s}")
        for k, v in sorted(rows.items(), key=lambda kv: kv[1]["raw_brier"][0]):
            name = k[: -(len(split) + 1)]
            print(f"  {name:20s} {v['train_brier']:12.4f} {v['train_auroc']:12.3f} {v['auroc'][0]:11.3f} "
                  f"{v['train_auroc'] - v['auroc'][0]:11.3f} {v['platt_slope'][0]:10.2f}")
        print()
    print("R² = 1 - Brier / ȳ(1-ȳ)  (기저율 대비 Brier skill score — Brier와 같은 양의 다른 표현)")
    print("Brier = reliability - resolution + uncertainty.  rel 낮을수록, res 높을수록 좋다.")
    print("platt 열은 학습 에피소드에서만 적합한 1차 보정을 테스트에 적용한 것 — 순위는 바뀌지 않는다.")

    path = HERE / args.out if not Path(args.out).is_absolute() else Path(args.out)
    path.write_text(json.dumps({"args": vars(args), "cells": out}, indent=2))
    print(f"\n-> {path}")
    if args.dump:
        import csv
        dpath = HERE / args.dump if not Path(args.dump).is_absolute() else Path(args.dump)
        with open(dpath, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dumped[0].keys()))
            w.writeheader()
            w.writerows(dumped)
        print(f"-> {dpath}  ({len(dumped)} rows, seed {seeds[0]})")


if __name__ == "__main__":
    main()
