#!/usr/bin/env python3
"""Stage-1 pilot: show each pre-failure frame to Gemma and record the one action it proposes.

One image per request, identical prompt every time (prompts.py). The model is given no hint about
the failure type, the original action, or the reference correction.

Output is constrained server-side to the 8 labels via vLLM structured outputs. Verified against
the remote server on 2026-09-10 (vLLM 0.23.0, model id `gemma4` = google/gemma-4-31b-it):

    {"structured_outputs": {"choice": [...]}}  -> returns exactly one label, finish_reason=stop
    {"guided_choice": [...]}                   -> SILENTLY IGNORED, returns free-form prose

The old `guided_choice` spelling does not error, it just does nothing, so this script always uses
`structured_outputs` and still validates the reply against ACTIONS rather than trusting it.

Connection settings come from CLI args or environment variables, never hardcoded:
    GEMMA_BASE_URL (default http://localhost:18007/v1)
    GEMMA_MODEL    (default gemma4)
    GEMMA_API_KEY  (default EMPTY — vLLM does not authenticate)

Run (conda fasi):
    python gemma_action_choice.py --mock          # no server, no GPU
    python gemma_action_choice.py --limit 10
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
from datetime import datetime
from pathlib import Path

from collections import Counter

from prompts import (
    ACTIONS,
    ACTIONS_6,
    action_prompt,
    GRIPPER_CHOICES,
    GRIPPER_PROBE_PROMPT,
    GRIPPER_PROBE_VERSION,
    GRIPPER_SYSTEM_PROMPT,
    HEIGHT_ABOVE_MIN,
    HEIGHT_CHOICES,
    HEIGHT_LEVEL_MAX,
    HEIGHT_PROMPT,
    HEIGHT_VERSION,
    LATERAL_CHOICES,
    LATERAL_PROMPT,
    LATERAL_VERSION,
    PROMPT_VERSION,
    SIDE_CHOICES,
    SIDE_PROMPT,
    SIDE_VERSION,
    SYSTEM_PROMPT,
    USER_PROMPT,
    action_prompt_variants,
    build_messages,
    gripper_prompt_variants,
    height_prompt_variants,
    lateral_prompt_variants,
    side_prompt_variants,
)

# Two tasks over the same frames: the counterfactual action choice, and a factual perception
# probe used to tell "cannot see the gripper state" apart from "sees it but does not use it".
TASKS: dict[str, dict] = {
    "action": {"choices": ACTIONS, "prompt": USER_PROMPT, "system": SYSTEM_PROMPT,
               "version": PROMPT_VERSION, "variants": action_prompt_variants},
    "gripper_state": {"choices": GRIPPER_CHOICES, "prompt": GRIPPER_PROBE_PROMPT,
                      "system": GRIPPER_SYSTEM_PROMPT, "version": GRIPPER_PROBE_VERSION,
                      "variants": gripper_prompt_variants, "truth": lambda m: gripper_truth(m)},
    # decomposed probes: perception only, then direction with the axis already chosen
    "side": {"choices": SIDE_CHOICES, "prompt": SIDE_PROMPT, "system": GRIPPER_SYSTEM_PROMPT,
             "version": SIDE_VERSION, "variants": side_prompt_variants,
             "truth": lambda m: side_truth(m)},
    "lateral": {"choices": LATERAL_CHOICES, "prompt": LATERAL_PROMPT, "system": SYSTEM_PROMPT,
                "version": LATERAL_VERSION, "variants": lateral_prompt_variants,
                "truth": lambda m: lateral_truth(m)},
    # the vertical axis, added after the lead-time run came back blind on close_too_high
    "height": {"choices": HEIGHT_CHOICES, "prompt": HEIGHT_PROMPT, "system": GRIPPER_SYSTEM_PROMPT,
               "version": HEIGHT_VERSION, "variants": height_prompt_variants,
               "truth": lambda m: height_truth(m)},
}

PARSE_OK = "OK"
PARSE_INVALID = "INVALID"          # replied, but not one of the 8 labels
PARSE_ERROR = "PARSE_ERROR"        # empty / missing content
REQUEST_ERROR = "REQUEST_ERROR"    # the call itself failed


def image_data_url(path: Path) -> str:
    """Local image -> data URL, the image_url form the server accepts (same as fasi/llm/client.py:38-45)."""
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def load_metadata(images_dir: Path) -> dict[str, dict]:
    """image_file -> metadata record written by collect_failures.py (absent is fine)."""
    path = images_dir / "metadata.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["image_file"]] = r
    return out


def classify(raw: str | None, choices: list[str]) -> tuple[str, str]:
    """(answer, parse_status). A reply outside the candidate list is never quietly accepted."""
    if raw is None or not raw.strip():
        return PARSE_ERROR, PARSE_ERROR
    token = raw.strip().strip(".\"'` \n").upper()
    if token in choices:
        return token, PARSE_OK
    return PARSE_INVALID, PARSE_INVALID


def consensus(answers: list[str]) -> tuple[str, float]:
    """(modal answer, fraction of variants agreeing with it). 1.0 means every phrasing agreed."""
    if not answers:
        return PARSE_ERROR, 0.0
    mode, count = Counter(answers).most_common(1)[0]
    return mode, count / len(answers)


def gripper_truth(m: dict) -> str | None:
    """OPEN/CLOSED at the captured step, from the recorded gripper command (action[6]: +1 closes).

    This is a real ground truth — it is what the simulator was commanded — so the perception probe
    can be scored. The counterfactual task has no such truth and is never scored here.

    Two metadata schemas are accepted: `original_action` from collect_failures.py, and
    `gripper_closed_cmd` from collect_leadtime.py, whose records hold one frame out of a whole
    episode rather than a single captured commit frame.
    """
    # Prefer the measured finger gap where it was recorded: the probe asks what the fingers look
    # like, and the command disagrees with that on every successful grasp (the cube holds the
    # fingers apart while the command says close). Scoring those frames against the command marked
    # a correct OPEN answer wrong -- see collect_leadtime.frame_record for the measured numbers.
    if "fingers_pressed" in m:
        return "CLOSED" if m["fingers_pressed"] else "OPEN"
    a = m.get("original_action")
    if a and len(a) >= 7:
        return "CLOSED" if float(a[6]) > 0 else "OPEN"
    if "gripper_closed_cmd" in m:
        return "CLOSED" if m["gripper_closed_cmd"] else "OPEN"
    return None


def height_truth(m: dict) -> str | None:
    """LEVEL/ABOVE from the recorded end-effector and cube positions.

    Deliberately unscored in the band between the two thresholds (see prompts.HEIGHT_*): a frame
    caught mid-descent has no defensible answer, and grading it on an arbitrary cut would put noise
    into the one number this probe exists to produce.
    """
    eef, cube = m.get("eef_pos"), m.get("cube_pos")
    if not eef or not cube:
        return None
    h = float(eef[2]) - float(cube[2])
    if h <= HEIGHT_LEVEL_MAX:
        return "LEVEL"
    if h >= HEIGHT_ABOVE_MIN:
        return "ABOVE"
    return None


def side_truth(m: dict) -> str | None:
    """Which side of the cube the gripper is on, from the recorded offset. `reference_correction`
    is the direction that fixes it, so the gripper sits on the opposite side."""
    return {"MOVE_LEFT": "RIGHT", "MOVE_RIGHT": "LEFT"}.get(m.get("reference_correction"))


def lateral_truth(m: dict) -> str | None:
    ref = m.get("reference_correction")
    return ref if ref in ("MOVE_LEFT", "MOVE_RIGHT") else None


def mock_reply(index: int, choices: list[str]) -> str:
    """Deterministic stand-ins that exercise every branch of `classify`, including bad ones."""
    canned = [
        *choices[:3],
        f"  {choices[0].lower()} \n",                         # needs normalising -> OK
        "The robot should have moved down before closing.",   # free-form -> INVALID
        "",                                                   # empty -> PARSE_ERROR
    ]
    return canned[index % len(canned)]


def ask(client, model: str, data_url: str, temperature: float, max_tokens: int,
        structured: bool, task: dict, prompt: str):
    kw = {}
    if structured:
        # NB: `structured_outputs`, not `guided_choice` — see module docstring.
        kw["extra_body"] = {"structured_outputs": {"choice": task["choices"]}}
    resp = client.chat.completions.create(
        model=model,
        messages=build_messages(data_url, prompt, task["system"]),
        temperature=temperature,
        max_tokens=max_tokens,
        **kw,
    )
    choice = resp.choices[0]
    return choice.message.content, {
        "finish_reason": choice.finish_reason,
        "served_model": resp.model,
        "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
        "completion_tokens": getattr(resp.usage, "completion_tokens", None),
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", default=str(here / "images"))
    ap.add_argument("--results-dir", default=str(here / "results"))
    ap.add_argument("--task", default="action", choices=sorted(TASKS),
                    help="action: the counterfactual choice among 8 labels. "
                         "gripper_state: a factual OPEN/CLOSED perception probe, scored against "
                         "the gripper command recorded in metadata. side/lateral/height: the "
                         "decomposed perception probes, scored against the recorded geometry")
    ap.add_argument("--base-url", default=os.environ.get("GEMMA_BASE_URL", "http://localhost:18007/v1"))
    ap.add_argument("--model", default=os.environ.get("GEMMA_MODEL", "gemma4"))
    ap.add_argument("--api-key", default=os.environ.get("GEMMA_API_KEY", "EMPTY"))
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=12)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--actions", default=None,
                    help='restrict the candidate set (task=action): "6" drops MOVE_FORWARD/'
                         'MOVE_BACKWARD, or pass a comma-separated subset')
    ap.add_argument("--robustness", action="store_true",
                    help="ask each image under several prompt variants and report how often they "
                         "agree, instead of trusting a single phrasing")
    ap.add_argument("--free-form", action="store_true",
                    help="drop the structured-output constraint, to see how often unconstrained "
                         "replies fall outside the 8 labels")
    ap.add_argument("--mock", action="store_true", help="no server call; validates paths, prompt, parsing, output")
    ap.add_argument("--run-id", default=None, help="default: timestamp, so runs never overwrite each other")
    args = ap.parse_args()

    images_dir, results_dir = Path(args.images_dir), Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    task = dict(TASKS[args.task])
    if args.actions:
        if args.task != "action":
            ap.error("--actions only applies to --task action")
        sel = ACTIONS_6 if args.actions == "6" else [a.strip().upper() for a in args.actions.split(",")]
        bad = [a for a in sel if a not in ACTIONS]
        if bad:
            ap.error(f"unknown action label(s) {bad}; available: {ACTIONS}")
        task["choices"] = sel
        task["prompt"] = action_prompt(sel)
        task["version"] = f"{PROMPT_VERSION}-{len(sel)}label"
        task["variants"] = lambda s=sel: action_prompt_variants(actions=s)

    images = sorted(p for p in images_dir.glob("*.png") if not p.name.endswith("_simstate.npz"))[: args.limit]
    meta = load_metadata(images_dir)
    print(f"run_id={run_id}  task={args.task}  images_dir={images_dir}  found={len(images)}  "
          f"metadata_rows={len(meta)}")
    print(f"model={args.model}  base_url={args.base_url}  structured={not args.free_form}  "
          f"mock={args.mock}  choices={task['choices']}")
    if not images:
        print("no images found — run collect_failures.py first (or point --images-dir elsewhere)")
        return

    client = None
    if not args.mock:
        from openai import OpenAI

        client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)

    variants = task["variants"]() if args.robustness else [("canonical", task["prompt"])]
    if args.robustness:
        print(f"prompt variants: {[v for v, _ in variants]}  ->  {len(images) * len(variants)} calls")

    rows = []
    for i, img in enumerate(images):
        m = meta.get(img.name, {})
        data_url = None if args.mock else image_data_url(img)
        per_image: list[str] = []
        for j, (variant_id, prompt_text) in enumerate(variants):
            info: dict = {}
            if args.mock:
                raw, err = mock_reply(i * len(variants) + j, task["choices"]), None
            else:
                try:
                    raw, info = ask(client, args.model, data_url, args.temperature,
                                    args.max_tokens, not args.free_form, task, prompt_text)
                    err = None
                except Exception as e:  # keep going; one bad call must not lose the whole run
                    raw, err = None, f"{type(e).__name__}: {e}"

            if err is not None:
                action, status = REQUEST_ERROR, REQUEST_ERROR
            else:
                action, status = classify(raw, task["choices"])
            per_image.append(action)

            rows.append({
                "image_path": str(img),
                "image_file": img.name,
                "prompt_variant": variant_id,
                "gemma_action": action,
                "raw_output": raw,
                "parse_status": status,
                "error": err,
                "model_name": args.model if not args.mock else "MOCK",
                "served_model": info.get("served_model"),
                "task": args.task,
                "prompt_version": task["version"],
                "structured_output": not args.free_form,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "run_id": run_id,
                # joined from collect_failures.py metadata (blank if the image has none)
                "episode_id": m.get("episode_id"),
                "timestep": m.get("timestep"),
                "failure_type": m.get("failure_type"),
                "failure_reason": m.get("failure_reason"),
                "original_action": m.get("original_action"),
                "original_action_discrete": m.get("original_action_discrete"),
                "reference_correction": m.get("reference_correction"),
                # ground truth for the perception probe only: gripper command at the captured step
                "gripper_state_truth": gripper_truth(m),
                "truth": task["truth"](m) if task.get("truth") else None,
                "finish_reason": info.get("finish_reason"),
                "prompt_tokens": info.get("prompt_tokens"),
                "completion_tokens": info.get("completion_tokens"),
            })
            if not args.robustness:
                print(f"  [{i+1}/{len(images)}] {img.name:44s} -> {action:14s} ({status})"
                      + (f"  raw={raw!r}" if status != PARSE_OK else ""))

        if args.robustness:
            mode, agree = consensus(per_image)
            mark = "=" if agree == 1.0 else " "
            print(f"  [{i+1}/{len(images)}] {img.name[:40]:40s} {mode:14s} agree={agree:.2f}{mark} "
                  f"{per_image}")

    stem = f"run_{run_id}_{args.task}"
    jsonl_path = results_dir / f"{stem}.jsonl"
    csv_path = results_dir / f"{stem}.csv"
    args_path = results_dir / f"{stem}.args.json"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v)
                        for k, v in r.items()})
    args_path.write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    summarize(rows, args.task, task["choices"])
    print(f"\nresults -> {jsonl_path}\n            {csv_path}\n            {args_path}")


def summarize(rows: list[dict], task_name: str, choices: list[str]) -> None:
    """Stage-1 looks at output *patterns*, not accuracy — so: distribution, invalid rate, and
    whether the choice tracks the image at all."""
    n = len(rows)
    dist: dict[str, int] = {}
    for r in rows:
        dist[r["gemma_action"]] = dist.get(r["gemma_action"], 0) + 1
    bad = sum(1 for r in rows if r["parse_status"] != PARSE_OK)
    echoed = sum(1 for r in rows
                 if r["parse_status"] == PARSE_OK and r["gemma_action"] == r.get("original_action_discrete"))

    print(f"\n--- {task_name} distribution ---")
    for a, c in sorted(dist.items(), key=lambda kv: -kv[1]):
        print(f"  {a:14s} {c:2d}/{n}  {'#' * c}")
    distinct_valid = len({r["gemma_action"] for r in rows if r["parse_status"] == PARSE_OK})
    print(f"distinct valid answers  : {distinct_valid} of {len(choices)}  (1 = always the same answer)")
    print(f"invalid / error         : {bad}/{n} ({bad / n:.0%})")
    if task_name == "action":
        print(f"echoed original action  : {echoed}/{n}  (high = not proposing an alternative)")

    scored = [r for r in rows if r["parse_status"] == PARSE_OK and r.get("truth")]
    if scored:
        # Scored only where the recorded state gives a real answer. Splitting by the true label
        # matters: on a balanced set a probe that always says one thing lands on exactly 50%, which
        # is how a degenerate constant answer is caught rather than read as "half right".
        hit = sum(1 for r in scored if r["gemma_action"] == r["truth"])
        print(f"correct vs recorded     : {hit}/{len(scored)} ({hit / len(scored):.0%})")
        per: dict[str, list[int]] = {}
        for r in scored:
            per.setdefault(r["truth"], []).append(int(r["gemma_action"] == r["truth"]))
        for truth, v in sorted(per.items()):
            print(f"  when truth is {truth:12s}: {sum(v):3d}/{len(v):<3d} ({sum(v) / len(v):.0%})")

    variant_ids = sorted({r["prompt_variant"] for r in rows})
    by_image: dict[str, list[dict]] = {}
    for r in rows:
        by_image.setdefault(r["image_file"], []).append(r)

    if len(variant_ids) > 1:
        print(f"\n--- prompt robustness over {len(variant_ids)} variants ---")
        agreements, stable = [], []
        for fn, rs in by_image.items():
            mode, agree = consensus([r["gemma_action"] for r in rs])
            agreements.append(agree)
            if agree == 1.0:
                stable.append((fn, mode))
        mean_agree = sum(agreements) / len(agreements)
        print(f"mean agreement          : {mean_agree:.2f}  (1.00 = every phrasing gave the same answer)")
        print(f"unanimous images        : {len(stable)}/{len(by_image)}")
        print("  " + ("none" if not stable else
                      "\n  ".join(f"{fn[:44]:44s} {a}" for fn, a in sorted(stable))))
        print("per-variant distribution (an outlier row means that phrasing drives the answer):")
        for vid in variant_ids:
            d = Counter(r["gemma_action"] for r in rows if r["prompt_variant"] == vid)
            print(f"  {vid:16s} {dict(d)}")

    # By failure type: with variants, count each image once by its modal answer, so a single
    # unstable image cannot look like a trend.
    by_type: dict[str, Counter] = {}
    refs: dict[str, str | None] = {}
    for fn, rs in by_image.items():
        ft = rs[0].get("failure_type") or "unknown"
        mode, agree = consensus([r["gemma_action"] for r in rs])
        by_type.setdefault(ft, Counter())[mode] += 1
        refs[ft] = rs[0].get("reference_correction")
    if len(by_type) > 1:
        label = "modal choice" if len(variant_ids) > 1 else "choice"
        print(f"\n--- {label} by failure_type (does the image change the answer?) ---")
        for ft, d in sorted(by_type.items()):
            print(f"  {ft:24s} {dict(d)}   [reference: {refs[ft]}]")


if __name__ == "__main__":
    main()
