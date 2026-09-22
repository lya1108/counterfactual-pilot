#!/usr/bin/env python3
"""How early can Gemma tell a Lift attempt is going to fail? (binary outcome prediction only)

One image in, one of SUCCESS / FAILURE out. No failure cause, no correction: this probe is only
about whether the outcome is readable, and how that changes with distance from the failure.

The frame set is `images_leadtime/` from `collect_leadtime.py`: 40 failure frames at t_commit-20,
-10, -5, -1 and 40 success frames matched by progress through the episode. The set is balanced 40/40
overall and 10/10 at every lead time, so a model that stops looking at the image and answers a
constant lands on exactly 50% -- the collapse this pilot has already been caught by twice (the
gripper probe's constant OPEN, the `labels_omitted` degenerate run). Read the answer distribution
before reading the accuracy.

Conditions:
  direct            one constrained call
  twostage_nothink  two calls, no thinking -- the CONTROL. Two calls add a turn, an assistant
                    message and a different final question, all of which change the answer on their
                    own; without this arm an improvement cannot be attributed to reasoning
  twostage_think    two calls, thinking enabled on the first
`structured_outputs` and thinking do not coexist on this server (a single call with both returns a
valid label and no thinking at all), which is why reasoning needs the two-stage shape.

Prompt wording is deliberately option-free: this pilot measured that enumerating the labels inside
the prompt text biases the answer toward the first one listed, while the structured-output
constraint already guarantees the format. The wording does match the label vocabulary
(succeed / fail), because phrasings that do not match stopped discriminating in the gripper probe.
`--robustness` runs three phrasings so a single-phrasing result cannot be mistaken for a property
of the model.

    python leadtime_probe.py --dry-run
    python leadtime_probe.py --mock          # no server: validates paths, prompts, parsing, metrics
    python leadtime_probe.py
    python leadtime_probe.py --robustness
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import time
import zlib
from datetime import datetime
from pathlib import Path

CHOICES: list[str] = ["SUCCESS", "FAILURE"]
CONDITIONS = ("direct", "twostage_nothink", "twostage_think")
PROMPT_VERSION = "leadtime_v1"

SYSTEM_PROMPT = (
    "You are analysing a single frame from a robot manipulation attempt. "
    "You answer with exactly one label and nothing else."
)

SCENE = (
    "This image is a single frame from a Franka Panda robot attempting to grasp the small cube on "
    "the table and lift it off the table surface.\n\n"
    "Viewpoint: a fixed camera in front of the table, looking down at it.\n\n"
)

# Chunk framing. The 2026-09-21 fact-injection run showed `close_too_high` is not a function of one
# frame's state -- the failing frame sits 8.5 cm above the cube and its matched success frame sits
# higher, at 10.5 cm -- so what is missing is whether the arm is still descending. This preamble
# names the frames in time order; the server was checked first and reads a 9 cm change correctly in
# 4/4 order-flipped probes, both as separate images and as one montage, while a 2.3 cm change is at
# chance. Frame count is filled in per record because clipping at the episode start can shorten it.
CHUNK_SCENE = (
    "You are shown {n} frames from a Franka Panda robot attempting to grasp the small cube on the "
    "table and lift it off the table surface. They are consecutive samples from the same attempt, "
    "in time order: FRAME 1 is the oldest and FRAME {n} is the most recent.\n\n"
    "Viewpoint: a fixed camera in front of the table, looking down at it. The camera does not "
    "move, so anything that changes between the frames is the robot moving.\n\n"
)

# The three phrasings. `canonical` and `reversed` differ only in the order the two outcomes are
# named, which is the permutation that changed 5 of 10 answers in the stage-1 action task.
VARIANTS: dict[str, str] = {
    "canonical": SCENE + "Question: will this grasp attempt succeed or fail?",
    "reversed": SCENE + "Question: will this grasp attempt fail or succeed?",
    "paraphrase": SCENE + "Question: judge the eventual outcome of the attempt shown here. "
                          "Is it a success or a failure?",
}

FINAL_TURN = "Now give your final answer as exactly one label."

# The question clause of each variant, reused so the chunk prompts differ from the single-frame
# prompts only in the scene preamble and nothing else.
QUESTIONS: dict[str, str] = {k: v.split("\n\n")[-1] for k, v in VARIANTS.items()}


def chunk_prompt(variant: str, n: int) -> str:
    return CHUNK_SCENE.format(n=n) + QUESTIONS[variant]


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def load_frames(images_dir: Path) -> list[dict]:
    meta = images_dir / "metadata.jsonl"
    if not meta.is_file():
        raise SystemExit(f"{meta} not found — run collect_leadtime.py first")
    out = []
    for line in meta.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        out.append({
            "image_file": m["image_file"],
            "path": images_dir / m["image_file"],
            "truth": m["label"],
            "kind": m["kind"],
            "lead_time": m["lead_time"],
            "failure_type": m["failure_type"],
            "pair_id": m["pair_id"],
            "episode_id": m["episode_id"],
            "reset_seed": m["reset_seed"],
            "timestep": m["timestep"],
            "progress": m["progress"],
            "phase": m["phase"],
            "gripper_closed_cmd": m["gripper_closed_cmd"],
            "chunk_files": m.get("chunk_files") or [m["image_file"]],
            "chunk_timesteps": m.get("chunk_timesteps") or [m["timestep"]],
            "chunk_span": m.get("chunk_span", 0),
            "chunk_len": m.get("chunk_len", 1),
        })
    return out


def mock_reply(key: str) -> str:
    """Deterministic stand-in, uncorrelated with the truth on purpose.

    An alternating mock silently scored 100% here, because the metadata itself alternates
    failure/success — the mock has to be independent of the label or it validates nothing. Every
    37th key returns an unusable label so the `n_bad` path is exercised too.
    """
    h = zlib.crc32(key.encode())
    if h % 37 == 0:
        return "MAYBE"
    return CHOICES[h % len(CHOICES)]


def run_one(client, model: str, img, condition: str, user_prompt: str,
            max_think: int) -> dict:
    """``img`` is one data URL or a list of them in time order (the chunk case)."""
    urls = [img] if isinstance(img, str) else list(img)
    base = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]
                                    + [{"type": "image_url", "image_url": {"url": u}} for u in urls]},
    ]
    t0 = time.time()
    think, think_tokens, truncated = None, 0, False

    if condition != "direct":
        kw = {}
        if condition == "twostage_think":
            kw["chat_template_kwargs"] = {"enable_thinking": True}
        r1 = client.chat.completions.create(
            model=model, messages=base, max_tokens=max_think, temperature=0.2,
            extra_body=kw or None,
        )
        think = r1.choices[0].message.content or ""
        think_tokens = r1.usage.completion_tokens
        truncated = r1.choices[0].finish_reason == "length"
        msgs = base + [
            {"role": "assistant", "content": think},
            {"role": "user", "content": FINAL_TURN},
        ]
    else:
        msgs = base

    r2 = client.chat.completions.create(
        model=model, messages=msgs, max_tokens=12, temperature=0.0,
        extra_body={"structured_outputs": {"choice": CHOICES}},
    )
    answer = (r2.choices[0].message.content or "").strip()
    return {
        "answer": answer if answer in CHOICES else "INVALID",
        "raw_answer": answer,
        "think_tokens": think_tokens,
        "think_truncated": truncated,
        "thinking": think,
        "latency_s": round(time.time() - t0, 1),
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", default=str(here / "images_leadtime"))
    ap.add_argument("--results-dir", default=str(here / "results"))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--variants", default="canonical")
    ap.add_argument("--robustness", action="store_true", help="run all three phrasings")
    ap.add_argument("--chunk", action="store_true",
                    help="send every frame of each record's chunk, in time order, instead of the "
                         "single lead-point frame. Needs an images dir collected with "
                         "collect_leadtime.py --chunk K")
    ap.add_argument("--base-url", default=os.environ.get("GEMMA_BASE_URL", "http://localhost:18007/v1"))
    ap.add_argument("--model", default=os.environ.get("GEMMA_MODEL", "gemma4"))
    ap.add_argument("--api-key", default=os.environ.get("GEMMA_API_KEY", "EMPTY"))
    ap.add_argument("--max-think-tokens", type=int, default=1200)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--faults", default=None,
                    help="comma-separated failure_type values to keep; the matched success frames "
                         "of the same pairs are kept with them so the slice stays balanced")
    ap.add_argument("--leads", default=None, help="comma-separated lead times to keep")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N frames per condition, keeping the lead times and "
                         "both classes balanced. For checking the server answers at all before "
                         "committing to the full set; the rates it prints are not a result.")
    ap.add_argument("--mock", action="store_true", help="no server call; validates the whole pipeline")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown_c = [c for c in conds if c not in CONDITIONS]
    if unknown_c:
        ap.error(f"unknown condition(s) {unknown_c}; available: {list(CONDITIONS)}")
    vnames = list(VARIANTS) if args.robustness else [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown_v = [v for v in vnames if v not in VARIANTS]
    if unknown_v:
        ap.error(f"unknown variant(s) {unknown_v}; available: {sorted(VARIANTS)}")

    frames = load_frames(images_dir)
    if args.faults:
        want = {x.strip() for x in args.faults.split(",") if x.strip()}
        # keep the failure frames of those faults AND the success frames paired with them, so the
        # slice is still 1:1 and a constant answer still scores 50%
        pairs = {f["pair_id"] for f in frames if f["kind"] == "failure" and f["failure_type"] in want}
        frames = [f for f in frames if f["pair_id"] in pairs]
    if args.leads:
        keep = {int(x) for x in args.leads.split(",") if x.strip()}
        frames = [f for f in frames if f["lead_time"] in keep]
    if not frames:
        raise SystemExit("no frames left after --faults/--leads")
    if args.limit:
        # take frames round-robin over (lead time, class) so a small --limit stays balanced
        # instead of being all-failure or all-one-lead-time
        groups: dict[tuple, list[dict]] = {}
        for f in frames:
            groups.setdefault((f["lead_time"], f["kind"]), []).append(f)
        order = sorted(groups, key=lambda k: (-k[0], k[1]))
        picked, i = [], 0
        while len(picked) < min(args.limit, len(frames)):
            g = groups[order[i % len(order)]]
            j = i // len(order)
            if j < len(g):
                picked.append(g[j])
            i += 1
        frames = picked
    n_f = sum(1 for f in frames if f["kind"] == "failure")
    api_calls = len(frames) * len(vnames) * sum(2 if c != "direct" else 1 for c in conds)

    if args.dry_run:
        print(f"[dry-run] images_dir : {images_dir}")
        print(f"[dry-run] frames     : {len(frames)}  ({n_f} FAILURE / {len(frames) - n_f} SUCCESS)")
        by_lead = {}
        for f in frames:
            k = (f["lead_time"], f["kind"])
            by_lead[k] = by_lead.get(k, 0) + 1
        for lead in sorted({f["lead_time"] for f in frames}, reverse=True):
            print(f"             lead-{lead:<3d} failure {by_lead.get((lead, 'failure'), 0)} / "
                  f"success {by_lead.get((lead, 'success'), 0)}")
        print(f"[dry-run] choices    : {CHOICES}   (balanced set -> a constant answer scores 50%)")
        print(f"[dry-run] conditions : {conds}")
        print(f"[dry-run] variants   : {vnames}")
        print(f"[dry-run] API calls  : {api_calls}")
        if args.chunk:
            lens = sorted({f["chunk_len"] for f in frames})
            spans = sorted({f["chunk_span"] for f in frames})
            print(f"[dry-run] chunk      : frames per record {lens}, spans {spans} steps "
                  f"({sum(f['chunk_len'] for f in frames)} images per condition-variant)")
            if lens == [1]:
                print("[dry-run] !! this images dir has no chunks — collect with --chunk K first")
        print("[dry-run] prompt (canonical):")
        text = chunk_prompt("canonical", max(f["chunk_len"] for f in frames)) if args.chunk \
            else VARIANTS["canonical"]
        for line in text.splitlines():
            print(f"    {line}")
        print("[dry-run] OK — nothing sent.")
        return

    client = None
    if not args.mock:
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=600)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"run {run_id}: {len(frames)} frames x {len(conds)} conditions x {len(vnames)} variants "
          f"= {api_calls} API calls  model={'MOCK' if args.mock else args.model}")
    rows = []
    k = 0
    for vname in vnames:
        for cond in conds:
            print(f"\n--- {cond} / {vname} ---")
            for i, f in enumerate(frames, 1):
                k += 1
                if args.mock:
                    r = {"answer": mock_reply(f"{vname}|{cond}|{f['image_file']}"),
                         "raw_answer": None, "think_tokens": 0,
                         "think_truncated": False, "thinking": None, "latency_s": 0.0}
                    err = None
                else:
                    try:
                        if args.chunk:
                            urls = [data_url(images_dir / fn) for fn in f["chunk_files"]]
                            prompt = chunk_prompt(vname, len(urls))
                        else:
                            urls, prompt = data_url(f["path"]), VARIANTS[vname]
                        r = run_one(client, args.model, urls, cond, prompt,
                                    args.max_think_tokens)
                        err = None
                    except Exception as e:  # one bad call must not lose the run
                        r = {"answer": "REQUEST_ERROR", "raw_answer": None, "think_tokens": 0,
                             "think_truncated": False, "thinking": None, "latency_s": None}
                        err = f"{type(e).__name__}: {e}"
                ok = r["answer"] == f["truth"]
                rows.append({**{kk: vv for kk, vv in f.items() if kk != "path"},
                             "condition": cond, "prompt_variant": vname,
                             "prompt_version": PROMPT_VERSION + ("-chunk" if args.chunk else ""),
                             "input_mode": "chunk" if args.chunk else "single", "correct": ok,
                             "error": err, "run_id": run_id,
                             "model_name": "MOCK" if args.mock else args.model, **r})
                if i % 10 == 0 or err or r["answer"] == "INVALID":
                    print(f"  [{i:2d}/{len(frames)}] {f['image_file'][:44]:46s} "
                          f"truth={f['truth']:7s} -> {r['answer']:13s} {'O' if ok else 'X'}"
                          + (f"  ERR {err[:70]}" if err else ""))

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    jp = out / f"leadtime_{run_id}.jsonl"
    with jp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    cp = out / f"leadtime_{run_id}.csv"
    with cp.open("w", encoding="utf-8", newline="") as fh:
        keep = [kk for kk in rows[0] if kk != "thinking"]  # thinking stays in the jsonl only
        w = csv.DictWriter(fh, fieldnames=keep)
        w.writeheader()
        for r in rows:
            w.writerow({kk: r[kk] for kk in keep})

    metrics = summarize(rows, conds, vnames)
    mp = out / f"leadtime_{run_id}.metrics.json"
    mp.write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    ap_json = out / f"leadtime_{run_id}.args.json"
    ap_json.write_text(json.dumps(vars(args), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nresults -> {jp}\n            {cp}\n            {mp}")


def rates(rs: list[dict]) -> dict:
    """Accuracy / failure recall / false-positive rate over one slice.

    FAILURE is the positive class: recall is over the failure frames, the false-positive rate over
    the success frames. Calls that errored or returned an unusable label are counted in `n_bad` and
    excluded from the rates, so a dead tunnel cannot look like a change in behaviour.
    """
    usable = [r for r in rs if r["answer"] in CHOICES]
    bad = len(rs) - len(usable)
    fails = [r for r in usable if r["truth"] == "FAILURE"]
    succs = [r for r in usable if r["truth"] == "SUCCESS"]
    tp = sum(1 for r in fails if r["answer"] == "FAILURE")
    fp = sum(1 for r in succs if r["answer"] == "FAILURE")
    dist: dict[str, int] = {}
    for r in rs:
        dist[r["answer"]] = dist.get(r["answer"], 0) + 1
    return {
        "n": len(rs), "n_usable": len(usable), "n_bad": bad,
        "n_failure": len(fails), "n_success": len(succs),
        "accuracy": round((tp + (len(succs) - fp)) / len(usable), 4) if usable else None,
        "failure_recall": round(tp / len(fails), 4) if fails else None,
        "false_positive_rate": round(fp / len(succs), 4) if succs else None,
        "tp": tp, "fn": len(fails) - tp, "fp": fp, "tn": len(succs) - fp,
        "answer_distribution": dict(sorted(dist.items(), key=lambda kv: -kv[1])),
    }


def summarize(rows: list[dict], conds: list[str], vnames: list[str]) -> dict:
    leads = sorted({r["lead_time"] for r in rows}, reverse=True)
    ftypes = sorted({r["failure_type"] for r in rows if r["kind"] == "failure"})
    metrics: dict = {"overall": {}, "by_lead_time": {}, "by_failure_type": {}}

    print("\n=== overall (FAILURE is the positive class) ===")
    print(f"{'condition/variant':32s} {'acc':>9s} {'recall':>9s} {'FPR':>9s} {'bad':>5s} "
          f"{'tok':>6s} {'s':>6s}  answers")
    for vn in vnames:
        for c in conds:
            rs = [r for r in rows if r["condition"] == c and r["prompt_variant"] == vn]
            if not rs:
                continue
            m = rates(rs)
            lat = [r["latency_s"] for r in rs if r["latency_s"] is not None]
            tok = [r["think_tokens"] for r in rs]
            m["mean_latency_s"] = round(sum(lat) / len(lat), 1) if lat else None
            m["median_think_tokens"] = sorted(tok)[len(tok) // 2] if tok else 0
            metrics["overall"][f"{c}/{vn}"] = m
            fmt = lambda x: "  -  " if x is None else f"{x:.0%}"
            print(f"{c + '/' + vn:32s} {fmt(m['accuracy']):>9s} {fmt(m['failure_recall']):>9s} "
                  f"{fmt(m['false_positive_rate']):>9s} {m['n_bad']:>5d} "
                  f"{m['median_think_tokens']:>6d} {str(m['mean_latency_s']):>6s}  "
                  f"{m['answer_distribution']}")

    cell_n = {}
    for r in rows:
        key = (r["lead_time"], r["kind"])
        cell_n[key] = cell_n.get(key, 0)
    per_cell = {l: (sum(1 for r in rows if r["lead_time"] == l and r["kind"] == "failure"
                        and r["condition"] == conds[0] and r["prompt_variant"] == vnames[0]),
                    sum(1 for r in rows if r["lead_time"] == l and r["kind"] == "success"
                        and r["condition"] == conds[0] and r["prompt_variant"] == vnames[0]))
                for l in leads}
    shape = " / ".join(f"t-{l}: {a}F+{b}S" for l, (a, b) in per_cell.items())
    print(f"\n=== accuracy by steps remaining to failure ({shape}) ===")
    header = f"{'condition/variant':32s}" + "".join(f"{'t-' + str(l):>18s}" for l in leads)
    print(header)
    for vn in vnames:
        for c in conds:
            cells = []
            for lead in leads:
                rs = [r for r in rows if r["condition"] == c and r["prompt_variant"] == vn
                      and r["lead_time"] == lead]
                if not rs:
                    cells.append("-")
                    continue
                m = rates(rs)
                metrics["by_lead_time"].setdefault(f"{c}/{vn}", {})[f"t-{lead}"] = m
                acc = "-" if m["accuracy"] is None else f"{m['accuracy']:.0%}"
                rec = "-" if m["failure_recall"] is None else f"{m['failure_recall']:.0%}"
                fpr = "-" if m["false_positive_rate"] is None else f"{m['false_positive_rate']:.0%}"
                cells.append(f"{acc} r{rec} f{fpr}")
            print(f"{c + '/' + vn:32s}" + "".join(f"{x:>18s}" for x in cells))
    print("  (acc = accuracy on that cell, r = failure recall, f = false-positive rate)")

    print("\n=== failure recall by injected fault (failure frames only, all lead times) ===")
    print(f"{'condition/variant':32s}" + "".join(f"{t[:20]:>22s}" for t in ftypes))
    for vn in vnames:
        for c in conds:
            cells = []
            for ft in ftypes:
                rs = [r for r in rows if r["condition"] == c and r["prompt_variant"] == vn
                      and r["failure_type"] == ft]
                usable = [r for r in rs if r["answer"] in CHOICES]
                hit = sum(1 for r in usable if r["answer"] == "FAILURE")
                metrics["by_failure_type"].setdefault(f"{c}/{vn}", {})[ft] = {
                    "n": len(rs), "n_usable": len(usable), "recall_hits": hit,
                    "recall": round(hit / len(usable), 4) if usable else None,
                }
                cells.append(f"{hit}/{len(usable)}" if usable else "-")
            print(f"{c + '/' + vn:32s}" + "".join(f"{x:>22s}" for x in cells))

    bad = [r for r in rows if r["answer"] not in CHOICES]
    if bad:
        kinds: dict[str, int] = {}
        for r in bad:
            kinds[r["answer"]] = kinds.get(r["answer"], 0) + 1
        print(f"\n!! {len(bad)} unusable call(s): {kinds} — these are excluded from every rate above")
    trunc = [r for r in rows if r.get("think_truncated")]
    if trunc:
        print(f"!! thinking hit the token cap on {len(trunc)} call(s) — raise --max-think-tokens")
    return metrics


if __name__ == "__main__":
    main()
