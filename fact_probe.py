#!/usr/bin/env python3
"""Is the vertical/gripper blindness a physics gap, a vision gap, or a recruitment gap?

The lead-time run (2026-09-18) predicts outcomes from lateral misalignment and nothing else:
`close_too_high` 0/36 across all three conditions, `gripper_closed_early` 5/36, the two lateral
faults 41/48. The thinking traces state the wrong fact rather than hedge -- "the gripper is very
close to the cube, ready to close" on a frame with an 8.5 cm gap, "the fingers are open" on a frame
with the fingers pressed together.

Three explanations survive that, and they need different fixes:

  A. physics   -- the model does not know that closing 8 cm too high, or descending already closed,
                  fails. Then no amount of better perception helps.
  B. vision    -- the gap and the finger state are not legible in this fixed 256 px view.
  C. recruitment -- both are legible, and the outcome question does not consult them.

`gemma_action_choice.py --task height` and `--task gripper_state` separate B from (A or C): they ask
the geometry alone, scored against the recorded eef/cube positions and gripper command. This file
covers the other two arms:

  --probe text   the six items of prompts.TEXT_ITEMS. The geometry is stated in words and there is
                 no image at all, so a wrong answer here is A and nothing else. Each fault has a
                 matched control differing in one clause, because "says FAILURE to everything" and
                 "understands the fault" look identical on the failure item alone.
  --probe fact   the lead-time outcome question with one true sentence about the state prepended,
                 built from the recorded numbers by prompts.state_fact(). It never names the
                 outcome. If this recovers the blind faults, the gap is C, and the prescription is
                 the one the lateral decomposition already pointed at: ask the state first, then
                 the outcome. Pass --baseline to get the paired comparison against the lead-time
                 run on the identical frames.

Read the per-fault recall row of --probe fact, not its accuracy. The stated height only separates
the classes at the commit moment: at t_commit-20 a failing frame sits 7.6 cm up and its matched
success frame 6.7 cm up, both mid-descent, so the fact is true and useless there. Worse, the
`gripper_closed_early` pairs are matched to nominal frames still in hover (10.6 cm up), because
those episodes run 234 steps against the others' ~130 and progress-matching pulls the partner
early. That confound moves the false-positive rate and the accuracy; it cannot touch the failure
recall, which is computed on failure frames alone. So the recall row is the test and the accuracy
column is not.

    python fact_probe.py --probe text --dry-run
    python fact_probe.py --probe text
    python fact_probe.py --probe fact --baseline results/leadtime_20260918_135728.jsonl
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import zlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import prompts as P
from leadtime_probe import CONDITIONS, data_url, load_frames, run_one


def mcnemar(pairs: list[tuple[bool, bool]]) -> tuple[int, int, float]:
    """(a_wrong & b_right, a_right & b_wrong, two-sided exact p) over paired items."""
    b01 = sum(1 for a, b in pairs if not a and b)
    b10 = sum(1 for a, b in pairs if a and not b)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    p = 2 * sum(math.comb(n, i) for i in range(min(b01, b10) + 1)) / 2**n
    return b01, b10, min(p, 1.0)


def ask_text(client, model: str, condition: str, text: str, max_think: int) -> dict:
    """The two-stage / direct machinery of leadtime_probe, with no image attached."""
    base = [
        {"role": "system", "content": P.OUTCOME_SYSTEM},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]
    import time
    t0 = time.time()
    think, think_tokens, truncated = None, 0, False
    if condition != "direct":
        kw = {"chat_template_kwargs": {"enable_thinking": True}} if condition == "twostage_think" else {}
        r1 = client.chat.completions.create(model=model, messages=base, max_tokens=max_think,
                                            temperature=0.2, extra_body=kw or None)
        think = r1.choices[0].message.content or ""
        think_tokens = r1.usage.completion_tokens
        truncated = r1.choices[0].finish_reason == "length"
        msgs = base + [{"role": "assistant", "content": think},
                       {"role": "user", "content": "Now give your final answer as exactly one label."}]
    else:
        msgs = base
    r2 = client.chat.completions.create(
        model=model, messages=msgs, max_tokens=12, temperature=0.0,
        extra_body={"structured_outputs": {"choice": P.OUTCOME_CHOICES}},
    )
    a = (r2.choices[0].message.content or "").strip()
    return {"answer": a if a in P.OUTCOME_CHOICES else "INVALID", "raw_answer": a,
            "think_tokens": think_tokens, "think_truncated": truncated, "thinking": think,
            "latency_s": round(time.time() - t0, 1)}


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", default="text", choices=["text", "fact"])
    ap.add_argument("--images-dir", default=str(here / "images_leadtime"))
    ap.add_argument("--results-dir", default=str(here / "results"))
    ap.add_argument("--conditions", default="direct,twostage_think")
    ap.add_argument("--baseline", default=None,
                    help="--probe fact: a leadtime_probe jsonl, for the paired comparison on the "
                         "identical frames")
    ap.add_argument("--repeats", type=int, default=1,
                    help="--probe text: repeat each item (temperature is 0, so >1 only catches "
                         "server-side nondeterminism)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--base-url", default=os.environ.get("GEMMA_BASE_URL", "http://localhost:18007/v1"))
    ap.add_argument("--model", default=os.environ.get("GEMMA_MODEL", "gemma4"))
    ap.add_argument("--api-key", default=os.environ.get("GEMMA_API_KEY", "EMPTY"))
    ap.add_argument("--max-think-tokens", type=int, default=1200)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    bad = [c for c in conds if c not in CONDITIONS]
    if bad:
        ap.error(f"unknown condition(s) {bad}; available: {list(CONDITIONS)}")

    # ------------------------------------------------------------------ build the work list
    if args.probe == "text":
        items = [{"item_id": i, "truth": t, "text": x} for i, t, x in P.TEXT_ITEMS]
        if args.limit:
            items = items[: args.limit]
        units = [dict(it, rep=r) for it in items for r in range(args.repeats)]
        n_calls = len(units) * sum(2 if c != "direct" else 1 for c in conds)
    else:
        frames = load_frames(Path(args.images_dir))
        meta = {}
        for line in (Path(args.images_dir) / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                m = json.loads(line)
                meta[m["image_file"]] = m
        units = []
        for f in frames:
            m = meta[f["image_file"]]
            h = float(m["eef_pos"][2]) - float(m["cube_pos"][2])
            units.append(dict(f, height_m=round(h, 4),
                              finger_gap_m=m["finger_gap_m"],
                              fact=P.state_fact(h, float(m["finger_gap_m"]))))
        if args.limit:
            units = units[: args.limit]
        n_calls = len(units) * sum(2 if c != "direct" else 1 for c in conds)

    if args.dry_run:
        print(f"[dry-run] probe      : {args.probe}")
        print(f"[dry-run] conditions : {conds}")
        print(f"[dry-run] units      : {len(units)}")
        print(f"[dry-run] API calls  : {n_calls}")
        if args.probe == "text":
            for u in units[: len(P.TEXT_ITEMS)]:
                print(f"    [{u['truth']:7s}] {u['item_id']}")
                print(f"              {u['text']}")
        else:
            print("[dry-run] example prompts (one per fault):")
            seen = set()
            for u in units:
                if u["failure_type"] in seen:
                    continue
                seen.add(u["failure_type"])
                print(f"    --- {u['image_file']}  truth={u['truth']}  h={u['height_m']}")
                print("        " + P.FACT_OUTCOME_PROMPT.format(fact=u["fact"]).replace("\n", "\n        "))
        print("[dry-run] OK — nothing sent.")
        return

    client = None
    if not args.mock:
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=600)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"run {run_id}: probe={args.probe} units={len(units)} conditions={conds} "
          f"-> {n_calls} calls  model={'MOCK' if args.mock else args.model}")

    rows = []
    for cond in conds:
        print(f"\n--- {cond} ---")
        for i, u in enumerate(units, 1):
            if args.mock:
                h = zlib.crc32(f"{cond}|{u.get('item_id') or u.get('image_file')}|{u.get('rep', 0)}".encode())
                r = {"answer": "INVALID" if h % 41 == 0 else P.OUTCOME_CHOICES[h % 2],
                     "raw_answer": None, "think_tokens": 0, "think_truncated": False,
                     "thinking": None, "latency_s": 0.0}
                err = None
            else:
                try:
                    if args.probe == "text":
                        r = ask_text(client, args.model, cond, u["text"], args.max_think_tokens)
                    else:
                        r = run_one(client, args.model, data_url(u["path"]), cond,
                                    P.FACT_OUTCOME_PROMPT.format(fact=u["fact"]),
                                    args.max_think_tokens)
                    err = None
                except Exception as e:
                    r = {"answer": "REQUEST_ERROR", "raw_answer": None, "think_tokens": 0,
                         "think_truncated": False, "thinking": None, "latency_s": None}
                    err = f"{type(e).__name__}: {e}"
            ok = r["answer"] == u["truth"]
            rows.append({**{k: v for k, v in u.items() if k not in ("path", "text")},
                         "probe": args.probe, "condition": cond, "correct": ok, "error": err,
                         "run_id": run_id, "prompt_version": P.OUTCOME_VERSION,
                         "model_name": "MOCK" if args.mock else args.model, **r})
            if args.probe == "text":
                print(f"  [{u['item_id']:14s}] truth={u['truth']:7s} -> {r['answer']:13s} "
                      f"{'O' if ok else 'X'}  {r['think_tokens']:4d}tok"
                      + (f"  ERR {err[:60]}" if err else ""))
            elif i % 20 == 0 or err:
                print(f"  [{i:2d}/{len(units)}] {u['image_file'][:44]:46s} truth={u['truth']:7s} "
                      f"-> {r['answer']:13s} {'O' if ok else 'X'}" + (f"  ERR {err[:60]}" if err else ""))

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"fact_{args.probe}_{run_id}"
    jp = out / f"{stem}.jsonl"
    with jp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    cp = out / f"{stem}.csv"
    with cp.open("w", encoding="utf-8", newline="") as fh:
        keep = [k for k in rows[0] if k != "thinking"]
        w = csv.DictWriter(fh, fieldnames=keep)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(r[k], ensure_ascii=False) if isinstance(r[k], list) else r[k])
                        for k in keep})
    (out / f"{stem}.args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=1),
                                           encoding="utf-8")

    if args.probe == "text":
        summarize_text(rows, conds)
    else:
        summarize_fact(rows, conds, args.baseline)
    print(f"\nresults -> {jp}\n            {cp}")


def summarize_text(rows: list[dict], conds: list[str]) -> None:
    print("\n=== stated geometry, no image (a miss here is a physics gap, not a vision gap) ===")
    ids = [i for i, _, _ in P.TEXT_ITEMS]
    print(f"{'condition':18s}" + "".join(f"{i[:14]:>16s}" for i in ids))
    for c in conds:
        cells = []
        for i in ids:
            rs = [r for r in rows if r["condition"] == c and r["item_id"] == i]
            if not rs:
                cells.append("-")
                continue
            hit = sum(r["correct"] for r in rs)
            cells.append(f"{rs[0]['answer'][:7]} {hit}/{len(rs)}")
        print(f"{c:18s}" + "".join(f"{x:>16s}" for x in cells))
    print("\n  fault/control pairs -- both halves must be right for the model to have the concept:")
    for c in conds:
        for base in ("too_high", "closed", "lateral"):
            f = [r for r in rows if r["condition"] == c and r["item_id"] == f"{base}_fail"]
            o = [r for r in rows if r["condition"] == c and r["item_id"] == f"{base}_ok"]
            if not f or not o:
                continue
            both = all(r["correct"] for r in f) and all(r["correct"] for r in o)
            print(f"    {c:18s} {base:10s} fail->{f[0]['answer']:8s} control->{o[0]['answer']:8s} "
                  f"{'BOTH OK' if both else 'not both'}")


def summarize_fact(rows: list[dict], conds: list[str], baseline: str | None) -> None:
    ok = lambda rs: [r for r in rs if r["answer"] in P.OUTCOME_CHOICES]
    print("\n=== outcome with the true state handed over (positive = FAILURE) ===")
    print(f"{'condition':18s} {'acc':>7s} {'recall':>8s} {'FPR':>7s} {'bad':>5s}  answers")
    for c in conds:
        rs = [r for r in rows if r["condition"] == c]
        u = ok(rs)
        fails = [r for r in u if r["truth"] == "FAILURE"]
        succs = [r for r in u if r["truth"] == "SUCCESS"]
        tp = sum(1 for r in fails if r["answer"] == "FAILURE")
        fp = sum(1 for r in succs if r["answer"] == "FAILURE")
        dist = defaultdict(int)
        for r in rs:
            dist[r["answer"]] += 1
        acc = (tp + len(succs) - fp) / len(u) if u else 0
        print(f"{c:18s} {acc:>6.0%} {tp}/{len(fails):<6} "
              f"{fp}/{len(succs):<5} {len(rs) - len(u):>5d}  {dict(dist)}")

    print("\n=== failure recall by injected fault (this is the row that has to move) ===")
    ftypes = sorted({r["failure_type"] for r in rows if r["kind"] == "failure"})
    print(f"{'condition':18s}" + "".join(f"{t[:20]:>22s}" for t in ftypes))
    for c in conds:
        cells = []
        for ft in ftypes:
            u = ok([r for r in rows if r["condition"] == c and r["failure_type"] == ft])
            hit = sum(1 for r in u if r["answer"] == "FAILURE")
            cells.append(f"{hit}/{len(u)}" if u else "-")
        print(f"{c:18s}" + "".join(f"{x:>22s}" for x in cells))

    if not baseline:
        print("\n(no --baseline given, so no paired comparison against the lead-time run)")
        return
    base = defaultdict(dict)
    for line in Path(baseline).read_text(encoding="utf-8").splitlines():
        if line.strip():
            b = json.loads(line)
            base[b["condition"]][b["image_file"]] = b
    print(f"\n=== paired vs {Path(baseline).name} (same frames, same conditions) ===")
    for c in conds:
        if c not in base:
            print(f"  {c}: not in the baseline file")
            continue
        pairs, moved = [], defaultdict(lambda: [0, 0])
        for r in rows:
            if r["condition"] != c or r["image_file"] not in base[c]:
                continue
            b = base[c][r["image_file"]]
            pairs.append((bool(b["correct"]), bool(r["correct"])))
            if b["truth"] == "FAILURE":
                key = r["failure_type"]
                moved[key][0] += int(b["answer"] == "FAILURE")
                moved[key][1] += int(r["answer"] == "FAILURE")
        b01, b10, p = mcnemar(pairs)
        print(f"  {c:18s} n={len(pairs)}  {b01} fixed / {b10} broken  p={p:.3f}")
        for ft, (was, now) in sorted(moved.items()):
            print(f"      {ft:24s} recall {was} -> {now}")


if __name__ == "__main__":
    main()
