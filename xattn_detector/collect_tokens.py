"""Re-run the first 8 VLA calls of every rollout in safe_detector/data and keep the tokens.

Why only the first 8: the headline evaluation (`prefix8`) uses exactly that window, and storing every call's
prefix would cost 51 GB. 784 of the 800 rollouts reach 8 calls.

Why re-running is safe: the pipeline is deterministic end to end -- `LiberoTask.reset` seeds on `init_id`, the
start noise is `manual_seed(1000*tid + 17*init + t)`, and the policy is frozen. So the first 80 environment
steps replay exactly the episode that was already labelled, and the label is read from the existing
episodes.jsonl rather than re-derived.

That is a claim, so it is checked rather than assumed: safe_detector stored the end-effector position at every
call, and this script compares its own against it. An episode whose eef drifts past --tol is written out but
flagged `repro_ok: false`, and the final report counts them.

    conda activate fasi
    python xattn_detector/collect_tokens.py --out data/tokens            # pooled 2x2, ~6 GB, ~25 min
    python xattn_detector/collect_tokens.py --out data/tokens_full --pool 1 --limit 40   # for the pooling study
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SAFE = HERE.parent / "safe_detector"
sys.path.insert(0, str(SAFE))
sys.path.insert(0, str(HERE))

import jb  # noqa: E402

jb.enter()

from fasi.envs.libero_env import LiberoTask  # noqa: E402
from fasi.vla.pi05_steer import Pi05Steer  # noqa: E402

from token_tap import TokenTap  # noqa: E402

PREFIX_K = 8
CKPT = "outputs/vla/pi05_libero_finetuned"


def load_index():
    """Every episode safe_detector recorded, with the path to its npz (for the reproducibility check)."""
    rows = []
    for idx in sorted((SAFE / "data").glob("*/episodes.jsonl")):
        for line in idx.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["_dir"] = idx.parent
                rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tokens")
    ap.add_argument("--pool", type=int, default=2, help="average-pool the 16x16 image token grid by this factor")
    ap.add_argument("--calls", type=int, default=PREFIX_K)
    ap.add_argument("--exec", type=int, default=10)
    ap.add_argument("--tol", type=float, default=1e-4, help="eef agreement tolerance, metres")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many episodes (0 = all)")
    ap.add_argument("--suites", nargs="*", default=None)
    ap.add_argument("--refix", action="store_true",
                    help="re-collect only the episodes already marked repro_ok=false, replaying every earlier "
                         "init of their task at FULL length first. LiberoTask.reset() leaves residual simulator "
                         "state, so an episode's path depends on how long the previous one ran; the plain pass "
                         "truncates at --calls and that difference, normally 6e-8 m, occasionally amplifies. "
                         "Verified on libero_spatial t2 i17: 1.8e-2 m without the warmup, 5.3e-8 m with it.")
    args = ap.parse_args()

    out = HERE / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "episodes.jsonl"
    done = set()
    if index_path.exists():
        for line in index_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["suite"], r["task"], r["init"]))
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    src = load_index()
    by_key = {(r["suite"], r["task"], r["init"]): r for r in src}
    if args.refix:
        prev = [json.loads(l) for l in index_path.read_text().splitlines() if l.strip()]
        flagged = [p for p in prev if not p["repro_ok"]]
        todo = [by_key[(p["suite"], p["task"], p["init"])] for p in flagged]
        print(f"refix: 재현 불일치 {len(todo)}건을 선행 init 전체 재생 후 다시 수집", flush=True)
        if not todo:
            return
    else:
        rows = [r for r in src if r["n_calls"] >= args.calls]
        if args.suites:
            rows = [r for r in rows if r["suite"] in args.suites]
        rows.sort(key=lambda r: (r["suite"], r["task"], r["init"]))
        todo = [r for r in rows if (r["suite"], r["task"], r["init"]) not in done]
        print(f"{len(rows)} 에피소드가 {args.calls}호출 이상 · 남은 작업 {len(todo)} · pool={args.pool}", flush=True)
        if not todo:
            return

    vla = Pi05Steer(CKPT)
    tap = TokenTap(vla.policy, pool=args.pool)

    task = None
    cur = None
    n_ok = n_bad = 0
    t_start = time.time()
    for i, rec in enumerate(todo):
        key = (rec["suite"], rec["task"])
        if key != cur:
            if task is not None:
                task.close()
            task = LiberoTask(rec["suite"], rec["task"])
            cur = key
        init, tid = rec["init"], rec["task"]
        if args.refix:
            # the original collection ran inits 0..19 in order, each to its own end; replay that history so the
            # residual simulator state entering this episode matches
            for w in range(init):
                task.reset(w)
                tw = 0
                while tw < task.max_steps:
                    ag = task.obs["agentview_image"][::-1, ::-1].copy()
                    wr = task.obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
                    bw = vla.batch_for(ag, wr, task.state8(), task.language)
                    gw = torch.Generator().manual_seed(1000 * tid + 17 * w + tw)
                    aw = vla.denoise(bw, torch.randn((1, vla.C, vla.A), generator=gw), [], None)[0]
                    hit = False
                    for j in range(args.exec):
                        task.step(aw[j])
                        tw += 1
                        if task.success():
                            hit = True
                            break
                    if hit:
                        break
        task.reset(init)

        t = 0
        steps, eefs, openings, goals = [], [], [], []
        per_call = []
        for _ in range(args.calls):
            ag = task.obs["agentview_image"][::-1, ::-1].copy()
            wr = task.obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
            b = vla.batch_for(ag, wr, task.state8(), task.language)
            g = torch.Generator().manual_seed(1000 * tid + 17 * init + t)
            tap.reset()
            a = vla.denoise(b, torch.randn((1, vla.C, vla.A), generator=g), [], None)[0]
            got = tap.pop()
            if got is None:
                raise SystemExit("token tap captured nothing")
            steps.append(t)
            eefs.append(task.eef())
            openings.append(task.opening())
            goals.append([task.predicate(p) for p in task.goal()])
            per_call.append({k: v.float().cpu().numpy().astype(np.float16) for k, v in got.items()})
            for j in range(args.exec):
                task.step(a[j])
                t += 1
                if task.success():
                    break

        # ---- reproducibility against the run that produced the labels
        ref = np.load(rec["_dir"] / rec["npz"])
        d_eef = float(np.abs(np.asarray(eefs) - ref["eef"][: args.calls]).max())
        d_step = int(np.abs(np.asarray(steps) - ref["step"][: args.calls]).max())
        ok = d_eef <= args.tol and d_step == 0
        n_ok += ok
        n_bad += not ok

        # ---- pack: language token count varies per call (the prompt carries the discretised state)
        n_lang = [c["pre_lang"].shape[0] for c in per_call]
        L = max(n_lang)
        D = per_call[0]["pre_lang"].shape[1]
        pre_lang = np.zeros((len(per_call), L, D), np.float16)
        lang_mask = np.zeros((len(per_call), L), bool)
        for k, c in enumerate(per_call):
            pre_lang[k, : n_lang[k]] = c["pre_lang"]
            lang_mask[k, : n_lang[k]] = True

        name = f"tok_{rec['suite']}_t{tid}_i{init}"
        np.savez_compressed(
            out / f"{name}.npz",
            pre_img=np.stack([c["pre_img"] for c in per_call]),
            pre_lang=pre_lang, lang_mask=lang_mask,
            suf_last=np.stack([c["suf_last"] for c in per_call]),
            suf_mean=np.stack([c["suf_mean"] for c in per_call]),
            step=np.asarray(steps, np.int32), eef=np.asarray(eefs, np.float32),
            opening=np.asarray(openings, np.float32), goal=np.asarray(goals, bool),
        )
        out_rec = dict(suite=rec["suite"], task=tid, init=init, success=rec["success"],
                       n_calls_full=rec["n_calls"], t_full=rec["t"], language=rec["language"],
                       goal_names=rec["goal_names"], calls=args.calls, pool=args.pool,
                       repro_ok=bool(ok), d_eef=d_eef, d_step=d_step,
                       n_img_tok=int(per_call[0]["pre_img"].shape[0]), n_lang_tok=n_lang, npz=f"{name}.npz")
        out_rec["refixed"] = bool(args.refix)
        if args.refix:
            kept = [json.loads(l) for l in index_path.read_text().splitlines() if l.strip()]
            kept = [k for k in kept if (k["suite"], k["task"], k["init"]) != (rec["suite"], tid, init)]
            kept.append(out_rec)
            kept.sort(key=lambda k: (k["suite"], k["task"], k["init"]))
            index_path.write_text("\n".join(json.dumps(k, ensure_ascii=False) for k in kept) + "\n")
        else:
            with open(index_path, "a") as fh:
                fh.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

        if (i + 1) % 25 == 0 or not ok:
            el = time.time() - t_start
            eta = el / (i + 1) * (len(todo) - i - 1)
            flag = "" if ok else f"   ** 재현 불일치 d_eef={d_eef:.2e} d_step={d_step}"
            print(f"[{i + 1}/{len(todo)}] {rec['suite']} t{tid} i{init} "
                  f"{'S' if rec['success'] else 'F'}  d_eef={d_eef:.2e}  "
                  f"({el / 60:.1f}분 경과, 남은 {eta / 60:.0f}분){flag}", flush=True)

    if task is not None:
        task.close()
    tap.close()
    size = sum(f.stat().st_size for f in out.glob("*.npz")) / 1024 ** 3
    print(f"\n재현 일치 {n_ok} / 불일치 {n_bad}   ({args.tol} m 허용)")
    print(f"저장 {size:.2f} GB -> {out}")


if __name__ == "__main__":
    main()
