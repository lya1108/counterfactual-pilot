"""Repeat every detector cell over several seeds and report mean +- sd.

One seed is one draw of the held-out groups, so a single run confounds "this detector generalises" with "this
draw happened to hold out easy tasks". LIBERO makes that acute: libero_goal t1 is 20/20 success and t3 and t7
are 0/20, so which tasks land in the holdout moves the score on its own. The seed also sets the model init and
the random baseline, so repeating it covers both sources at once.

Episodes are loaded once per feature and reused across every seed, split and condition -- running
train_detector.py 100-odd times would re-read 800 npz files each time.

    conda activate fasi
    python safe_detector/multiseed.py --seeds 0-4
    python safe_detector/multiseed.py --seeds 0-9 --conds prefix8      # the condition that matters, more draws
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent

from train_detector import LSTMDet, LogReg, auroc, load_episodes, pad_batch, split_episodes  # noqa: E402

# (model, feature) cells. `time` and `random` are the two shortcuts a latent detector has to beat.
CELLS = [
    ("time", "time"),
    ("random", "time"),  # the feature is irrelevant; it rides along on the cheapest load
    ("logreg", "vlm_mean"), ("lstm", "vlm_mean"),
    ("logreg", "act_mean"), ("lstm", "act_mean"),
    ("lstm", "concat"),
]
PREFIX_K = 8


def truncate(eps, k: int):
    out = []
    for e in eps:
        if len(e["x"]) < k:
            continue
        out.append({**e, "x": e["x"][:k], "n_calls_full": e["n_calls"], "n_calls": k})
    return out


def run_cell(tr, te, model: str, epochs: int, lr: float, seed: int, device: str, return_scores: bool = False):
    """Train one detector and return (per-call AUROC, per-episode AUROC, stratified AUROC).

    `return_scores` additionally yields the raw per-call logits for the test episodes and, so that a caller can
    recalibrate without retraining, for the training episodes too.
    """
    tr_scores = None
    if model == "random":
        rng = np.random.default_rng(seed)
        scores = [rng.standard_normal(len(e["x"])).astype(np.float32) for e in te]
        tr_scores = [rng.standard_normal(len(e["x"])).astype(np.float32) for e in tr]
    else:
        torch.manual_seed(seed)
        cat = np.concatenate([e["x"] for e in tr])
        mu, sd = cat.mean(0), cat.std(0) + 1e-6
        norm = lambda s: [{**e, "x": (e["x"] - mu) / sd} for e in s]
        trn, ten = norm(tr), norm(te)
        d = trn[0]["x"].shape[1]
        net = (LogReg(d) if model in ("logreg", "time") else LSTMDet(d)).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        x, m, y = pad_batch(trn, device)
        nf = sum(e["y"] for e in trn)
        w = m * torch.where(y[:, None] > 0, 1.0 / max(nf, 1), 1.0 / max(len(trn) - nf, 1))
        tgt = y[:, None].expand_as(m)
        for _ in range(epochs):
            opt.zero_grad()
            loss = (nn.functional.binary_cross_entropy_with_logits(net(x), tgt, reduction="none") * w).sum() / w.sum()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            xt, _, _ = pad_batch(ten, device)
            st = net(xt).cpu().numpy()
            scores = [st[i, : len(e["x"])].astype(np.float32) for i, e in enumerate(te)]
            if return_scores:
                sr = net(x).cpu().numpy()
                tr_scores = [sr[i, : len(e["x"])].astype(np.float32) for i, e in enumerate(trn)]

    flat_s = np.concatenate(scores)
    flat_y = np.concatenate([np.full(len(s), e["y"]) for s, e in zip(scores, te)])
    ep_s = np.array([s.max() for s in scores])
    ep_y = np.array([e["y"] for e in te])
    strat = []
    for i in range(max(len(s) for s in scores)):
        s = np.array([sc[i] for sc in scores if len(sc) > i])
        yy = np.array([e["y"] for sc, e in zip(scores, te) if len(sc) > i])
        if yy.sum() and (1 - yy).sum():
            strat.append(auroc(s, yy))
    metrics = (auroc(flat_s, flat_y), auroc(ep_s, ep_y), (float(np.mean(strat)) if strat else float("nan")))
    return (metrics, scores, tr_scores) if return_scores else metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data"])
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--splits", nargs="+", default=["init", "task"])
    ap.add_argument("--conds", nargs="+", default=["full", "prefix8"])
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these model names (e.g. --only time random), skipping the rest")
    ap.add_argument("--out", default="results/multiseed.json")
    args = ap.parse_args()

    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds += list(range(int(a), int(b) + 1))
        elif part:
            seeds.append(int(part))

    roots = []
    for d in args.data:
        p = Path(d)
        if not p.is_absolute():
            p = HERE / p
        roots += [p] if (p / "episodes.jsonl").exists() else sorted(q for q in p.iterdir() if (q / "episodes.jsonl").exists())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"seeds {seeds}  splits {args.splits}  conds {args.conds}  device {device}", flush=True)

    cells = [c for c in CELLS if args.only is None or c[0] in args.only]
    out: dict[str, dict] = {}
    for feature in sorted({f for _, f in cells}):
        eps = load_episodes(roots, feature)
        print(f"\nloaded {len(eps)} episodes for feature={feature} dim={eps[0]['x'].shape[1]}", flush=True)
        views = {"full": eps, "prefix8": truncate(eps, PREFIX_K)}
        for cond in args.conds:
            view = views[cond]
            for split in args.splits:
                for model, feat in cells:
                    if feat != feature:
                        continue
                    key = f"{model}_{feature if model not in ('time', 'random') else model}_{split}_{cond}"
                    vals = []
                    for seed in seeds:
                        tr, te, groups = split_episodes(view, split, args.holdout, seed)
                        if not tr or not te:
                            continue
                        vals.append(run_cell(tr, te, model, args.epochs, args.lr, seed, device))
                    if not vals:
                        continue
                    a = np.array(vals, float)
                    out[key] = {"n_seeds": len(vals),
                                "auroc_call": [a[:, 0].mean(), a[:, 0].std(ddof=1)],
                                "auroc_episode": [a[:, 1].mean(), a[:, 1].std(ddof=1)],
                                "strat": [a[:, 2].mean(), a[:, 2].std(ddof=1)],
                                "per_seed_episode": a[:, 1].tolist()}
                    print(f"  {key:34s} ep {a[:, 1].mean():.3f} ± {a[:, 1].std(ddof=1):.3f}   "
                          f"strat {a[:, 2].mean():.3f} ± {a[:, 2].std(ddof=1):.3f}   "
                          f"range [{a[:, 1].min():.3f}, {a[:, 1].max():.3f}]", flush=True)
        del eps, views

    path = HERE / args.out if not Path(args.out).is_absolute() else Path(args.out)
    path.write_text(json.dumps({"args": vars(args), "seeds": seeds, "cells": out}, indent=2))

    # final table, grouped by condition
    print("\n" + "=" * 92)
    print(f"mean ± sd over {len(seeds)} seeds   (each seed redraws the held-out groups AND the model init)")
    for cond in args.conds:
        for split in args.splits:
            rows = {k: v for k, v in out.items() if k.endswith(f"{split}_{cond}")}
            if not rows:
                continue
            print(f"\n=== {split}_{cond} ===")
            print(f"  {'detector':22s} {'ep AUROC':>16s} {'strat':>16s} {'seed spread':>16s}")
            for k, v in sorted(rows.items(), key=lambda kv: -kv[1]["auroc_episode"][0]):
                name = k[: -(len(split) + len(cond) + 2)]
                e, s = v["auroc_episode"], v["strat"]
                lo, hi = min(v["per_seed_episode"]), max(v["per_seed_episode"])
                print(f"  {name:22s} {e[0]:7.3f} ± {e[1]:.3f} {s[0]:7.3f} ± {s[1]:.3f} "
                      f"{lo:7.3f} – {hi:.3f}")
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
