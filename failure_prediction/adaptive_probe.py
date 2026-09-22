#!/usr/bin/env python3
"""Can the model choose the temporal evidence it needs, instead of being handed a fixed window?

Stage 3 left a split prescription: `lateral` is readable from one frame, `gripper_closed_early`
needs the right *camera* (wrist 6-8/12 vs agentview 2/12), `close_too_high` needs a *window*
(10/120 -> 36/120 with a 3-frame chunk, p=2.4e-5). And a fixed window is not free -- on
`gripper_closed_early` at t-1 the same chunk broke 14 frames and fixed 2 (p=0.0042), because there
the arm is still moving 7.8 cm over the window and the motion reads as a normal approach. So the
question is whether the model can ask for the frame it needs.

Candidate observations (the selection is one-shot over a fixed menu, not free generation):

    current_only   the lead-point frame alone
    add_t5         + the frame 5 steps earlier
    add_t10        + the frame 10 steps earlier
    add_t20        + the frame 20 steps earlier
    fixed3         + both t-10 and t-20        (not a candidate; the arm-C baseline)

Modes, in the order they should be run:

  --mode self     GATE 2, cheapest. Current frame only, asked whether one frame is enough. No
                  ground truth exists for that question; what matters is whether the answer varies
                  by fault. If `close_too_high` at t-1 comes back "enough" -- a frame we have
                  measured to be unjudgeable -- then there is no self-uncertainty signal for a
                  selector to act on and the direction stops here. Two phrasings, because a probe
                  that collapses to a constant has done that twice in this pilot.

  --mode oracle   GATE 1. Every frame answered under all four candidate settings. Gives (a) the
                  ceiling any selector could reach, (b) arms A and B for free, and (c) the per-frame
                  label of *which* candidate was right -- without which only end accuracy is
                  measurable and selection quality is not. If oracle@4 is no better than
                  current_only, there is nothing to select.

  --mode adaptive ARM D. Free-form reason -> constrained choice over the four candidates -> the
                  outcome question with the chosen frames. Three calls.

  --mode placebo  ARM E, the control that decides whether the result means anything. Identical in
                  call count, turn count and prompt structure to --mode adaptive; the reason call
                  still happens and its output is still placed in context. Only the choice is
                  replaced by a seeded random pick from the same menu. If adaptive does not beat
                  this, the gain is from having an extra frame, not from choosing it.

  --mode fixed    ARM C: one call with `fixed3`. (Arms A and B come out of --mode oracle.)

Offsets clipped by the episode start are not silently swapped for a nearer frame: the candidate is
marked unavailable for that record, excluded from the oracle ceiling, and if the model selects it
the selection is recorded as UNAVAILABLE and falls back to current_only. 51 of the 320 records are
affected, all near t=0.

    python adaptive_probe.py --mode self   --dry-run
    python adaptive_probe.py --mode self
    python adaptive_probe.py --mode oracle
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
import zlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from leadtime_probe import CHOICES, SYSTEM_PROMPT, data_url, load_frames

OFFSETS = {"current_only": (0,), "add_t5": (-5, 0), "add_t10": (-10, 0), "add_t20": (-20, 0),
           "fixed3": (-20, -10, 0)}
CANDIDATES = ["current_only", "add_t5", "add_t10", "add_t20"]
SEL_LABELS = {"current_only": "CURRENT_ONLY", "add_t5": "ADD_T5",
              "add_t10": "ADD_T10", "add_t20": "ADD_T20"}
LABEL_TO_KEY = {v: k for k, v in SEL_LABELS.items()}

SCENE_ONE = ("This image is the most recent frame from a Franka Panda robot attempting to grasp the "
             "small cube on the table and lift it off the table surface.\n\n"
             "Viewpoint: a fixed camera in front of the table, looking down at it.\n\n")
SCENE_MANY = ("You are shown {n} frames from a Franka Panda robot attempting to grasp the small cube "
              "on the table and lift it off the table surface, in time order: FRAME 1 is the oldest "
              "and FRAME {n} is the most recent.\n\n"
              "Viewpoint: a fixed camera in front of the table, looking down at it. The camera does "
              "not move, so anything that changes between the frames is the robot moving.\n\n")
OUTCOME_Q = "Question: will this grasp attempt succeed or fail?"

# GATE 2. Bare binary, no menu in the text: restating options inside the prompt measurably biases
# this server toward the first one listed (stage-1 ablation).
SELF_CHOICES = ["ENOUGH", "NEED_EARLIER"]
SELF_VARIANTS = {
    "canonical": "Question: is this single frame enough to judge the outcome, or do you need to see "
                 "an earlier frame of the same attempt?",
    "swapped": "Question: do you need to see an earlier frame of the same attempt to judge the "
               "outcome, or is this single frame enough?",
}

# ARM D stage 1. The menu has to be named here because the model is choosing from it; the wording
# matches the label spellings so the answer vocabulary and the choice list agree (a mismatch made a
# perfectly-reasoned text probe score 0/12 earlier in this stage).
SELECT_Q = (
    "Question: you must predict whether this attempt will succeed or fail, and you may ask to see "
    "one earlier frame of the same attempt first. Your options are CURRENT_ONLY (judge from this "
    "frame alone), ADD_T5 (also see the frame 5 steps earlier), ADD_T10 (10 steps earlier), or "
    "ADD_T20 (20 steps earlier).\n\n"
    "In at most three sentences: what, if anything, is unclear from this frame alone, and which "
    "option resolves it?"
)
SELECT_COMMIT = "Now state your choice as exactly one label."


def obs_files(f: dict, key: str) -> tuple[list[str] | None, list[int]]:
    """(chunk file names in time order, realised offsets) for one observation setting.

    Returns (None, []) when the setting needs a frame the episode is too short to have.
    """
    by_off = {ts - f["timestep"]: fn for ts, fn in zip(f["chunk_timesteps"], f["chunk_files"])}
    want = OFFSETS[key]
    if any(o not in by_off for o in want):
        return None, []
    return [by_off[o] for o in want], list(want)


def outcome_prompt(n: int) -> str:
    return (SCENE_ONE if n == 1 else SCENE_MANY.format(n=n)) + OUTCOME_Q


def ask(client, model: str, urls: list[str], prompt: str, choices: list[str],
        max_tokens: int = 12, temperature: float = 0.0, history: list | None = None) -> str:
    msgs = history or [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": prompt}]
                                    + [{"type": "image_url", "image_url": {"url": u}} for u in urls]},
    ]
    r = client.chat.completions.create(
        model=model, messages=msgs, max_tokens=max_tokens, temperature=temperature,
        extra_body={"structured_outputs": {"choice": choices}},
    )
    return (r.choices[0].message.content or "").strip()


def free_form(client, model: str, urls: list[str], prompt: str, max_tokens: int) -> tuple[str, list]:
    """Unconstrained reply plus the message list to continue from."""
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": prompt}]
                                    + [{"type": "image_url", "image_url": {"url": u}} for u in urls]},
    ]
    r = client.chat.completions.create(model=model, messages=msgs, max_tokens=max_tokens,
                                       temperature=0.2)
    txt = (r.choices[0].message.content or "").strip()
    return txt, msgs + [{"role": "assistant", "content": txt}]


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["self", "oracle", "adaptive", "placebo", "fixed"])
    ap.add_argument("--images-dir", default=str(here / "images_leadtime_adaptive"))
    ap.add_argument("--results-dir", default=str(here / "results"))
    ap.add_argument("--faults", default=None)
    ap.add_argument("--leads", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0, help="--mode placebo: the random choice seed")
    ap.add_argument("--base-url", default=os.environ.get("GEMMA_BASE_URL", "http://localhost:18007/v1"))
    ap.add_argument("--model", default=os.environ.get("GEMMA_MODEL", "gemma4"))
    ap.add_argument("--api-key", default=os.environ.get("GEMMA_API_KEY", "EMPTY"))
    ap.add_argument("--max-reason-tokens", type=int, default=300)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    frames = load_frames(images_dir)
    if args.faults:
        want = {x.strip() for x in args.faults.split(",") if x.strip()}
        pairs = {f["pair_id"] for f in frames if f["kind"] == "failure" and f["failure_type"] in want}
        frames = [f for f in frames if f["pair_id"] in pairs]
    if args.leads:
        keep = {int(x) for x in args.leads.split(",") if x.strip()}
        frames = [f for f in frames if f["lead_time"] in keep]
    if args.limit:
        frames = frames[: args.limit]
    if not frames:
        raise SystemExit("no frames after filtering")

    avail = {c: sum(1 for f in frames if obs_files(f, c)[0]) for c in CANDIDATES + ["fixed3"]}
    per_call = {"self": 1 * len(SELF_VARIANTS), "oracle": len(CANDIDATES),
                "adaptive": 3, "placebo": 3, "fixed": 1}[args.mode]
    if args.dry_run:
        print(f"[dry-run] mode       : {args.mode}")
        print(f"[dry-run] images_dir : {images_dir}")
        nf = sum(1 for f in frames if f['kind'] == 'failure')
        print(f"[dry-run] frames     : {len(frames)}  ({nf} FAILURE / {len(frames) - nf} SUCCESS)")
        print(f"[dry-run] 후보 가용   : " + "  ".join(f"{c} {avail[c]}/{len(frames)}" for c in CANDIDATES))
        print(f"[dry-run]   (모자란 것은 에피소드 시작에 걸린 기록; 오라클 천장에서 제외된다)")
        print(f"[dry-run] API calls  : {len(frames) * per_call}")
        if args.mode == "self":
            print("[dry-run] 문구 2종:")
            for k, v in SELF_VARIANTS.items():
                print(f"    [{k}] {v}")
        elif args.mode in ("adaptive", "placebo"):
            print("[dry-run] 선택 프롬프트:")
            for line in (SCENE_ONE + SELECT_Q).splitlines():
                print(f"    {line}")
        print("[dry-run] OK — nothing sent.")
        return

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=600)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"run {run_id}: mode={args.mode} frames={len(frames)} -> {len(frames)*per_call} calls")
    rng = random.Random(args.seed)
    rows = []

    for i, f in enumerate(frames, 1):
        base = {k: v for k, v in f.items() if k != "path"}
        rec = {**base, "mode": args.mode, "run_id": run_id, "model_name": args.model}
        t0 = time.time()
        try:
            if args.mode == "self":
                for vname, q in SELF_VARIANTS.items():
                    a = ask(client, args.model, [data_url(f["path"])], SCENE_ONE + q, SELF_CHOICES)
                    rec[f"self_{vname}"] = a if a in SELF_CHOICES else "INVALID"
            elif args.mode in ("oracle", "fixed"):
                keys = CANDIDATES if args.mode == "oracle" else ["fixed3"]
                for c in keys:
                    fl, offs = obs_files(f, c)
                    if fl is None:
                        rec[f"ans_{c}"] = "UNAVAILABLE"
                        continue
                    urls = [data_url(images_dir / fn) for fn in fl]
                    a = ask(client, args.model, urls, outcome_prompt(len(urls)), CHOICES)
                    rec[f"ans_{c}"] = a if a in CHOICES else "INVALID"
                    rec[f"offsets_{c}"] = offs
            else:  # adaptive / placebo
                reason, hist = free_form(client, args.model, [data_url(f["path"])],
                                         SCENE_ONE + SELECT_Q, args.max_reason_tokens)
                rec["reason"] = reason
                picked = ask(client, args.model, [], SELECT_COMMIT, list(SEL_LABELS.values()),
                             history=hist + [{"role": "user", "content": SELECT_COMMIT}])
                rec["model_choice"] = picked
                if args.mode == "placebo":
                    chosen = rng.choice(CANDIDATES)
                    rec["placebo_choice"] = SEL_LABELS[chosen]
                else:
                    chosen = LABEL_TO_KEY.get(picked)
                rec["used"] = chosen
                if chosen is None:
                    rec["used"], rec["fallback"] = "current_only", "INVALID_CHOICE"
                    chosen = "current_only"
                fl, offs = obs_files(f, chosen)
                if fl is None:
                    rec["fallback"] = "UNAVAILABLE"
                    rec["used"] = "current_only"
                    fl, offs = obs_files(f, "current_only")
                urls = [data_url(images_dir / fn) for fn in fl]
                a = ask(client, args.model, urls, outcome_prompt(len(urls)), CHOICES)
                rec["answer"] = a if a in CHOICES else "INVALID"
                rec["offsets"] = offs
            rec["error"] = None
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["latency_s"] = round(time.time() - t0, 1)
        rows.append(rec)
        if i % 20 == 0 or rec["error"]:
            shown = rec.get("answer") or rec.get("ans_current_only") or rec.get("self_canonical")
            print(f"  [{i:3d}/{len(frames)}] {f['image_file'][:42]:44s} truth={f['truth']:7s} "
                  f"-> {str(shown):13s}" + (f"  ERR {rec['error'][:60]}" if rec["error"] else ""))

    out = Path(args.results_dir); out.mkdir(parents=True, exist_ok=True)
    stem = f"adaptive_{args.mode}_{run_id}"
    with (out / f"{stem}.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    keys = sorted({k for r in rows for k in r} - {"reason"})
    with (out / f"{stem}.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(r[k], ensure_ascii=False) if isinstance(r.get(k), list)
                            else r.get(k)) for k in keys})
    (out / f"{stem}.args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    summarize(rows, args.mode)
    print(f"\nresults -> {out / (stem + '.jsonl')}")


def summarize(rows: list[dict], mode: str) -> None:
    ok = [r for r in rows if not r.get("error")]
    if not ok:
        print("\n!! every call failed — check the tunnel"); return
    print(f"\n=== {mode}: {len(ok)}/{len(rows)} 성공한 기록 ===")

    if mode == "self":
        for vname in SELF_VARIANTS:
            d = defaultdict(int)
            for r in ok:
                d[r.get(f"self_{vname}")] += 1
            print(f"  [{vname}] 전체 분포 {dict(d)}")
        print(f"\n  {'결함':24s} " + "".join(f"{v[:12]+' NEED':>18s}" for v in SELF_VARIANTS))
        for ft in sorted({r["failure_type"] for r in ok}):
            cells = []
            for vname in SELF_VARIANTS:
                rs = [r for r in ok if r["failure_type"] == ft]
                n = sum(1 for r in rs if r.get(f"self_{vname}") == "NEED_EARLIER")
                cells.append(f"{n}/{len(rs)}")
            print(f"  {ft:24s} " + "".join(f"{c:>18s}" for c in cells))
        print(f"\n  {'lead':>6s} " + "".join(f"{v[:12]+' NEED':>18s}" for v in SELF_VARIANTS))
        for lead in sorted({r["lead_time"] for r in ok}, reverse=True):
            cells = []
            for vname in SELF_VARIANTS:
                rs = [r for r in ok if r["lead_time"] == lead]
                n = sum(1 for r in rs if r.get(f"self_{vname}") == "NEED_EARLIER")
                cells.append(f"{n}/{len(rs)}")
            print(f"  t-{lead:<4d} " + "".join(f"{c:>18s}" for c in cells))
        agree = sum(1 for r in ok if r.get("self_canonical") == r.get("self_swapped"))
        print(f"\n  두 문구 일치: {agree}/{len(ok)}  (낮으면 문구가 답을 지배한다)")
        return

    if mode in ("oracle", "fixed"):
        keys = CANDIDATES if mode == "oracle" else ["fixed3"]
        print(f"  {'설정':14s} {'정확도':>8s} {'recall':>10s} {'FPR':>10s} {'불가':>6s}")
        for c in keys:
            rs = [r for r in ok if r.get(f"ans_{c}") in CHOICES]
            un = sum(1 for r in ok if r.get(f"ans_{c}") == "UNAVAILABLE")
            fa = [r for r in rs if r["truth"] == "FAILURE"]; su = [r for r in rs if r["truth"] == "SUCCESS"]
            tp = sum(1 for r in fa if r[f"ans_{c}"] == "FAILURE")
            fp = sum(1 for r in su if r[f"ans_{c}"] == "FAILURE")
            acc = (tp + len(su) - fp) / len(rs) if rs else 0
            print(f"  {c:14s} {acc:>7.0%} {f'{tp}/{len(fa)}':>10s} {f'{fp}/{len(su)}':>10s} {un:>6d}")
        if mode == "oracle":
            any_ok = 0; best = defaultdict(int)
            for r in ok:
                hits = [c for c in CANDIDATES if r.get(f"ans_{c}") == r["truth"]]
                if hits:
                    any_ok += 1; best[hits[0]] += 1
            print(f"\n  Oracle@4 (네 후보 중 하나라도 맞음): {any_ok}/{len(ok)} ({any_ok/len(ok):.0%})")
            print(f"  current_only 로만: "
                  f"{sum(1 for r in ok if r.get('ans_current_only')==r['truth'])}/{len(ok)}")
            print(f"  맞힌 후보 중 가장 이른 것의 분포: {dict(best)}")
            print(f"\n  {'결함':24s} {'current':>10s} {'oracle@4':>10s}")
            for ft in sorted({r["failure_type"] for r in ok}):
                rs = [r for r in ok if r["failure_type"] == ft]
                c1 = sum(1 for r in rs if r.get("ans_current_only") == r["truth"])
                o4 = sum(1 for r in rs if any(r.get(f"ans_{c}") == r["truth"] for c in CANDIDATES))
                print(f"  {ft:24s} {f'{c1}/{len(rs)}':>10s} {f'{o4}/{len(rs)}':>10s}")
        return

    # adaptive / placebo
    d = defaultdict(int)
    for r in ok:
        d[r.get("used")] += 1
    print(f"  선택 분포: {dict(d)}")
    fb = [r for r in ok if r.get("fallback")]
    if fb:
        print(f"  대체된 기록: {len(fb)}  ({defaultdict(int, {r['fallback']: 1 for r in fb})})")
    rs = [r for r in ok if r.get("answer") in CHOICES]
    fa = [r for r in rs if r["truth"] == "FAILURE"]; su = [r for r in rs if r["truth"] == "SUCCESS"]
    tp = sum(1 for r in fa if r["answer"] == "FAILURE"); fp = sum(1 for r in su if r["answer"] == "FAILURE")
    print(f"  정확도 {(tp+len(su)-fp)/len(rs):.0%}  recall {tp}/{len(fa)}  FPR {fp}/{len(su)}")
    print(f"\n  {'결함':24s} {'선택 분포':>40s}")
    for ft in sorted({r["failure_type"] for r in ok}):
        dd = defaultdict(int)
        for r in ok:
            if r["failure_type"] == ft:
                dd[r.get("used")] += 1
        print(f"  {ft:24s} {str(dict(dd)):>40s}")


if __name__ == "__main__":
    main()
