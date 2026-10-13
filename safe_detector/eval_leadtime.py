"""SAFE step 3: turn detector scores into a lead time -- how early a failure is flagged.

A detector is useful only if it can stay quiet on rollouts that succeed. So the threshold is set by a false-alarm
budget: the largest threshold at which at most `--fpr` of SUCCESSFUL episodes ever cross it. Successful test
episodes are halved -- one half calibrates the threshold, the other half reports the false-alarm rate -- so the
reported FPR is not the one that was optimised.

For a failed episode, lead time is (last call - first call whose score crosses the threshold), in VLA calls and in
environment steps (one call covers `exec` steps). The reference point is the END of the episode, which for LIBERO
failures is the timeout, not a labelled moment of breakage: read the number as "flagged this long before the
rollout gave up", not "this long before the mistake".

    python safe_detector/eval_leadtime.py --tag lstm_vlm_mean_init --fpr 0.1
    python safe_detector/eval_leadtime.py --tag lstm_vlm_mean_init --compare time_time_init
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def load(tag: str):
    meta = json.loads((RESULTS / f"scores_{tag}.json").read_text())
    z = np.load(RESULTS / f"scores_{tag}.npz")
    eps = []
    for i, e in enumerate(meta["episodes"]):
        eps.append({**e, "score": z[f"s{i}"], "y": int(not e["success"])})
    return meta, eps


def evaluate(eps, fpr_budget: float, seed: int = 0):
    succ = [e for e in eps if not e["y"]]
    fail = [e for e in eps if e["y"]]
    if not succ or not fail:
        raise SystemExit(f"need both classes in the test split (success {len(succ)}, failure {len(fail)})")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(succ))
    half = max(1, len(succ) // 2)
    cal = [succ[i] for i in order[:half]]
    held = [succ[i] for i in order[half:]] or cal

    # smallest threshold whose false-alarm rate on the calibration successes is within budget
    peaks = np.sort(np.array([e["score"].max() for e in cal]))[::-1]
    k = int(np.floor(fpr_budget * len(cal)))
    thr = float(peaks[k] + 1e-6) if k < len(peaks) else float(peaks[-1] - 1e-6)

    fired_held = [float(e["score"].max()) > thr for e in held]
    rows, detected = [], 0
    for e in fail:
        over = np.flatnonzero(e["score"] > thr)
        if len(over) == 0:
            rows.append({**{k: e[k] for k in ("suite", "task", "init", "n_calls", "t", "exec")},
                         "fired": False, "lead_calls": None, "lead_steps": None})
            continue
        detected += 1
        first = int(over[0])
        lead_calls = e.get("n_calls_full", e["n_calls"]) - 1 - first  # back from the real end, even on a prefix
        rows.append({**{k: e[k] for k in ("suite", "task", "init", "n_calls", "t", "exec")},
                     "fired": True, "first_call": first, "lead_calls": lead_calls,
                     "lead_steps": lead_calls * e["exec"]})
    leads = np.array([r["lead_calls"] for r in rows if r["fired"]], float)
    steps = np.array([r["lead_steps"] for r in rows if r["fired"]], float)
    return dict(
        threshold=thr, n_cal=len(cal), n_held=len(held), n_fail=len(fail),
        fpr_held=float(np.mean(fired_held)), detection_rate=detected / len(fail),
        lead_calls_median=float(np.median(leads)) if len(leads) else float("nan"),
        lead_calls_mean=float(leads.mean()) if len(leads) else float("nan"),
        lead_steps_median=float(np.median(steps)) if len(steps) else float("nan"),
        rows=rows,
    )


def line(tag: str, m: dict, r: dict) -> str:
    return (f"{tag:28s} AUROC call {m['auroc_call']:.3f} ep {m['auroc_episode']:.3f} | "
            f"detect {r['detection_rate']:.0%} at FPR {r['fpr_held']:.0%} | "
            f"lead {r['lead_calls_median']:.0f} calls = {r['lead_steps_median']:.0f} steps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--compare", nargs="*", default=[], help="other tags to print alongside (e.g. the time baseline)")
    ap.add_argument("--fpr", type=float, default=0.1, help="false-alarm budget on successful episodes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"false-alarm budget {args.fpr:.0%} of successful episodes; lead time is measured back from episode end\n")
    out = {}
    for tag in [args.tag, *args.compare]:
        meta, eps = load(tag)
        res = evaluate(eps, args.fpr, args.seed)
        out[tag] = {"auroc_call": meta["auroc_call"], "auroc_episode": meta["auroc_episode"],
                    **{k: v for k, v in res.items() if k != "rows"}}
        print(line(tag, meta, res))
        if tag == args.tag:
            fired = [r for r in res["rows"] if r["fired"]]
            missed = [r for r in res["rows"] if not r["fired"]]
            print(f"\n  threshold {res['threshold']:.3f} (calibrated on {res['n_cal']} successes, "
                  f"FPR reported on the other {res['n_held']})")
            print(f"  detected {len(fired)}/{res['n_fail']} failures; missed {len(missed)}")
            if fired:
                ls = np.array([r["lead_steps"] for r in fired], float)
                qs = np.percentile(ls, [10, 50, 90])
                print(f"  lead steps  p10 {qs[0]:.0f}  median {qs[1]:.0f}  p90 {qs[2]:.0f}  (max {ls.max():.0f})")

    (RESULTS / f"leadtime_{args.tag}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n-> {RESULTS / f'leadtime_{args.tag}.json'}")


if __name__ == "__main__":
    main()
