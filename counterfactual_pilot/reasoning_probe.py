#!/usr/bin/env python3
"""Does turning reasoning on fix the left/right asymmetry, and does it cost anything where the
model was already right?

Background. The 8-way task is 72% correct when the answer is MOVE_RIGHT and 98% when it is
MOVE_LEFT, and the decomposed probes showed the information is there: perception 99%, direction
100%, both symmetric. So the loss is in choosing the lateral axis among eight options — exactly the
step a reasoning pass could carry.

Two things make this awkward on this server:

* `structured_outputs` and thinking do not coexist. Sending both returns one valid label with no
  thinking at all, and the answer changes (a probe answered MOVE_LEFT constrained-only and
  OPEN_GRIPPER with thinking also requested). So reasoning needs two calls: one free-form to think,
  one constrained to commit.
* Two calls change more than "reasoning" — an extra turn, an assistant message in context, a
  different final question. `twostage_nothink` is the control that isolates the thinking itself.

Frames are the balanced lateral set: 12 with the gripper on the image-left (answer MOVE_RIGHT) and
12 on the right (answer MOVE_LEFT). Both halves are run, because a treatment measured only on the
frames it was meant to fix can look perfect while being net-neutral — the same trap as scoring a
recovery method without checking what it breaks.

    python reasoning_probe.py --dry-run
    python reasoning_probe.py
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import time
from datetime import datetime
from pathlib import Path

import prompts as P

CONDITIONS = ("direct", "twostage_nothink", "twostage_think")
FRAME_SETS = ("images_ypos", "images_yneg", "images_ypos_mirror", "images_yneg_mirror")


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def load_frames(here: Path, sets: list[str]) -> list[dict]:
    out = []
    for s in sets:
        d = here / s
        for line in (d / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            m = json.loads(line)
            truth = m.get("reference_correction")
            if truth not in ("MOVE_LEFT", "MOVE_RIGHT"):
                continue  # this probe is about the lateral axis only
            out.append({
                "set": s, "image_file": m["image_file"], "path": d / m["image_file"],
                "truth": truth,
                # reference_correction is the fix, so the gripper sits on the opposite side
                "gripper_side": "right" if truth == "MOVE_LEFT" else "left",
                "failure_type": m["failure_type"], "episode_id": m["episode_id"],
            })
    return out


def base_messages(img: str, user_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": P.SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": img}},
        ]},
    ]


def run_one(client, model: str, img: str, condition: str, max_think: int,
            user_prompt: str) -> dict:
    base = base_messages(img, user_prompt)
    t0 = time.time()
    think, think_tokens = None, 0

    if condition != "direct":
        kw = {}
        if condition == "twostage_think":
            kw["chat_template_kwargs"] = {"enable_thinking": True}
        r1 = client.chat.completions.create(
            model=model, messages=base, max_tokens=max_think, temperature=0.2, extra_body=kw or None,
        )
        think = r1.choices[0].message.content or ""
        think_tokens = r1.usage.completion_tokens
        msgs = base + [
            {"role": "assistant", "content": think},
            {"role": "user", "content": "Now give your final answer as exactly one label."},
        ]
        truncated = r1.choices[0].finish_reason == "length"
    else:
        msgs, truncated = base, False

    r2 = client.chat.completions.create(
        model=model, messages=msgs, max_tokens=12, temperature=0.0,
        extra_body={"structured_outputs": {"choice": P.ACTIONS}},
    )
    answer = (r2.choices[0].message.content or "").strip()
    return {
        "answer": answer if answer in P.ACTIONS else "INVALID",
        "raw_answer": answer,
        "think_tokens": think_tokens,
        "think_truncated": truncated,
        "thinking": think,
        "latency_s": round(time.time() - t0, 1),
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default=",".join(FRAME_SETS))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--variants", default="canonical",
                    help="prompt variants from prompts.action_prompt_variants(); the direct "
                         "condition is most wrong under labels_reversed and paraphrase, so those "
                         "are the informative ones for asking whether reasoning suppresses it")
    ap.add_argument("--results-dir", default=str(here / "results"))
    ap.add_argument("--base-url", default=os.environ.get("GEMMA_BASE_URL", "http://localhost:18007/v1"))
    ap.add_argument("--model", default=os.environ.get("GEMMA_MODEL", "gemma4"))
    ap.add_argument("--api-key", default=os.environ.get("GEMMA_API_KEY", "EMPTY"))
    ap.add_argument("--max-think-tokens", type=int, default=1200,
                    help="600 truncated one frame's thinking in the 4-frame pilot")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    allv = dict(P.action_prompt_variants())
    vnames = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in vnames if v not in allv]
    if unknown:
        ap.error(f"unknown variant(s) {unknown}; available: {sorted(allv)}")
    frames = load_frames(here, sets)
    sides = {s: sum(1 for f in frames if f["gripper_side"] == s) for s in ("left", "right")}

    if args.dry_run:
        print(f"[dry-run] frames     : {len(frames)}  (gripper left {sides['left']}, right {sides['right']})")
        print(f"[dry-run] conditions : {conds}")
        print(f"[dry-run] variants   : {vnames}")
        print(f"[dry-run] calls      : {len(frames) * len(conds) * len(vnames)} frame-condition-variants, "
              f"{len(frames) * len(vnames) * sum(2 if c != 'direct' else 1 for c in conds)} API calls")
        print("[dry-run] OK — nothing sent.")
        return

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=600)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for vname in vnames:
      for cond in conds:
        print(f"\n--- {cond} / {vname} ---")
        for i, f in enumerate(frames, 1):
            try:
                r = run_one(client, args.model, data_url(f["path"]), cond,
                            args.max_think_tokens, allv[vname])
                err = None
            except Exception as e:  # one bad call must not lose the run
                r = {"answer": "REQUEST_ERROR", "raw_answer": None, "think_tokens": 0,
                     "think_truncated": False, "thinking": None, "latency_s": None}
                err = f"{type(e).__name__}: {e}"
            ok = r["answer"] == f["truth"]
            rows.append({**{k: v for k, v in f.items() if k != "path"},
                         "condition": cond, "prompt_variant": vname, "correct": ok,
                         "error": err, "run_id": run_id, **r})
            print(f"  [{i:2d}/{len(frames)}] {f['image_file'][:38]:40s} {f['gripper_side']:5s} "
                  f"-> {r['answer']:14s} {'O' if ok else 'X'}  {r['think_tokens']:4d}tok "
                  f"{r['latency_s']}s" + (f"  ERR {err[:60]}" if err else ""))

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    jp = out / f"reasoning_{run_id}.jsonl"
    with jp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    cp = out / f"reasoning_{run_id}.csv"
    with cp.open("w", encoding="utf-8", newline="") as fh:
        keep = [k for k in rows[0] if k != "thinking"]  # thinking stays in the jsonl only
        w = csv.DictWriter(fh, fieldnames=keep)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keep})

    summarize(rows, conds, vnames)
    print(f"\nresults -> {jp}\n            {cp}")


def summarize(rows: list[dict], conds: list[str], vnames: list[str]) -> None:
    print("\n--- accuracy vs the verified lateral answer ---")
    print(f"{'condition/variant':34s} {'gripper left':>14s} {'gripper right':>14s} {'overall':>10s} {'mean s':>8s}")
    for c, vn in [(c, v) for v in vnames for c in conds]:
        rs = [r for r in rows if r["condition"] == c and r["prompt_variant"] == vn]
        if not rs:
            continue
        cells = []
        for side in ("left", "right"):
            v = [r for r in rs if r["gripper_side"] == side]
            h = sum(r["correct"] for r in v)
            cells.append(f"{h}/{len(v)} ({h / len(v):.0%})" if v else "-")
        h = sum(r["correct"] for r in rs)
        lat = [r["latency_s"] for r in rs if r["latency_s"] is not None]
        print(f"{c + '/' + vn:34s} {cells[0]:>14s} {cells[1]:>14s} {f'{h}/{len(rs)}':>10s} "
              f"{(sum(lat) / len(lat) if lat else 0):>8.1f}")

    print("\n--- answer distribution (where the truth is MOVE_RIGHT, i.e. the hard side) ---")
    for c in conds:
        d: dict[str, int] = {}
        for r in rows:
            if r["condition"] == c and r["truth"] == "MOVE_RIGHT":
                d[r["answer"]] = d.get(r["answer"], 0) + 1
        print(f"  {c:18s} {dict(sorted(d.items(), key=lambda kv: -kv[1]))}")

    trunc = [r for r in rows if r.get("think_truncated")]
    if trunc:
        print(f"\n!! thinking hit the token cap on {len(trunc)} call(s) — raise --max-think-tokens")


if __name__ == "__main__":
    main()
