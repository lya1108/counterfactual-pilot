"""One table over every detector cell produced by run_analysis.sh.

Each row is a trained detector; the columns that decide whether it means anything are `time` (the same cell with
the call index as its only feature) and the stratified AUROC, which compares successes and failures at the SAME
call index so that episode length cannot contribute.

    python safe_detector/summarize.py [--fpr 0.1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

from eval_leadtime import evaluate, load  # noqa: E402


def mean_strat(meta: dict, lo: int = 0, hi: int = 99) -> float:
    rows = [r["auroc"] for r in meta.get("auroc_by_call", []) if lo <= r["call"] <= hi]
    return float(np.mean(rows)) if rows else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpr", type=float, default=0.1)
    args = ap.parse_args()

    tags = sorted(p.stem[len("scores_"):] for p in RESULTS.glob("scores_*.json"))
    tags = [t for t in tags if t != "dryrun"]
    if not tags:
        raise SystemExit("no scores_*.json in results/ -- run run_analysis.sh first")

    groups: dict[str, list[str]] = {}
    for t in tags:
        for cond in ("init_full", "init_prefix8", "task_full", "task_prefix8"):
            if t.endswith(cond):
                groups.setdefault(cond, []).append(t)

    print(f"false-alarm budget {args.fpr:.0%} of successful episodes\n")
    hdr = (f"{'detector':26s} {'AUROC':>6s} {'ep':>5s} {'strat':>6s} {'strat t0-3':>10s} "
           f"{'detect':>7s} {'FPR':>5s} {'lead(steps)':>11s}")
    for cond, ts in groups.items():
        print(f"=== {cond} ===")
        print(hdr)
        for t in sorted(ts, key=lambda s: (not s.startswith(("time", "random")), s)):
            meta, eps = load(t)
            name = t[: -(len(cond) + 1)]
            try:
                r = evaluate(eps, args.fpr)
                det, fpr, lead = f"{r['detection_rate']:.0%}", f"{r['fpr_held']:.0%}", f"{r['lead_steps_median']:.0f}"
            except SystemExit:
                det = fpr = lead = "--"
            print(f"{name:26s} {meta['auroc_call']:6.3f} {meta['auroc_episode']:5.3f} "
                  f"{mean_strat(meta):6.3f} {mean_strat(meta, 0, 3):10.3f} {det:>7s} {fpr:>5s} {lead:>11s}")
        print()

    print("AUROC = per call, every call of a failed episode positive.  ep = per episode, max score.")
    print("strat = AUROC averaged over call indices where both classes are still alive (length carries no signal).")
    print("Compare each row with the `time` row of its block: that is what the latents add over the clock.")


if __name__ == "__main__":
    main()
