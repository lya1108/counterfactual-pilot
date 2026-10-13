"""SAFE step 1: run pi0.5 on LIBERO and record a latent feature per action chunk, with the episode outcome.

The rollout loop is the one in ~/바탕화면/jungbeom/scripts/libero_base_eval.py:44-59 (same sampler, same seeds,
same 10-of-50 chunk execution); `LiberoTask` and `Pi05Steer` are imported from there unchanged. What is added is
`LatentTap`, which reads pi0.5's hidden states out of the forward pass, and a per-call record of oracle state and
BDDL goal predicates so that failure onset can be defined after the fact rather than fixed to the timeout.

One VLA call covers `--exec` environment steps, so latents arrive every 10 steps: lead time is measured in chunks.

    conda activate fasi
    python safe_detector/collect_rollouts.py --suite libero_spatial --tasks 0-9 --inits 0-4 --out data/spatial

Re-running the same --out resumes: episodes already in episodes.jsonl are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

import jb

HERE = jb.HERE
jb.enter()

from fasi.envs.libero_env import LiberoTask  # noqa: E402
from fasi.vla.pi05_steer import Pi05Steer  # noqa: E402

from latent_hook import FEATURES, LatentTap  # noqa: E402


def parse_ids(s: str):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def observe(task: LiberoTask, view: str):
    if view == "rot180":
        return (task.obs["agentview_image"][::-1, ::-1].copy(),
                task.obs["robot0_eye_in_hand_image"][::-1, ::-1].copy())
    return task.images()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/vla/pi05_libero_finetuned")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--inits", default="0-4")
    ap.add_argument("--exec", type=int, default=10)
    # the LIBERO fine-tune needs the raw frame rotated 180 (libero_base_eval.py:33). Measured on
    # libero_spatial t0: rot180 2/2 success in ~76 steps, upright 0/2 (both ran to the 280-step timeout).
    ap.add_argument("--view", default="rot180", choices=["upright", "rot180"])
    ap.add_argument("--out", required=True, help="directory, relative to safe_detector/ unless absolute")
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = HERE / out
    out.mkdir(parents=True, exist_ok=True)
    index = out / "episodes.jsonl"
    done = set()
    if index.exists():
        for line in index.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["suite"], r["task"], r["init"], r["view"]))
    (out / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))

    vla = Pi05Steer(args.ckpt)
    tap = LatentTap(vla.policy)
    print(f"ckpt={args.ckpt} view={args.view} K={vla.K} chunk={vla.C} exec={args.exec}", flush=True)

    for tid in parse_ids(args.tasks):
        todo = [i for i in parse_ids(args.inits) if (args.suite, tid, i, args.view) not in done]
        if not todo:
            print(f"task {tid}: all inits already collected", flush=True)
            continue
        task = LiberoTask(args.suite, tid)
        goal_names = [" ".join(p) for p in task.goal()]
        for init in todo:
            task.reset(init)
            t, ok, t0 = 0, False, time.time()
            feats = {k: [] for k in FEATURES}
            steps, eefs, openings, goals = [], [], [], []
            while t < task.max_steps:
                ag, wr = observe(task, args.view)
                b = vla.batch_for(ag, wr, task.state8(), task.language)
                g = torch.Generator().manual_seed(1000 * tid + 17 * init + t)
                tap.reset()
                a = vla.denoise(b, torch.randn((1, vla.C, vla.A), generator=g), [], None)[0]
                f = tap.pop()
                if f is None:
                    raise SystemExit("latent tap captured nothing -- the hook did not fire")
                for k in FEATURES:
                    feats[k].append(f[k])
                steps.append(t)
                eefs.append(task.eef())
                openings.append(task.opening())
                goals.append([task.predicate(p) for p in task.goal()])
                for j in range(args.exec):
                    task.step(a[j])
                    t += 1
                    if task.success():
                        ok = True
                        break
                if ok:
                    break

            name = f"ep_{args.suite}_t{tid}_i{init}_{args.view}"
            np.savez_compressed(
                out / f"{name}.npz",
                step=np.asarray(steps, np.int32),
                eef=np.asarray(eefs, np.float32),
                opening=np.asarray(openings, np.float32),
                goal=np.asarray(goals, bool),
                **{k: np.stack(v).astype(np.float16) for k, v in feats.items()},
            )
            rec = dict(suite=args.suite, task=tid, init=init, view=args.view, ckpt=args.ckpt,
                       language=task.language, success=ok, t=t, n_calls=len(steps),
                       exec=args.exec, sec=round(time.time() - t0, 1), goal_names=goal_names, npz=f"{name}.npz")
            with open(index, "a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"task {tid} init {init} success={ok} t={t} calls={len(steps)} ({rec['sec']}s) | {task.language}",
                  flush=True)
        task.close()
    tap.close()
    print(f"\nwrote -> {out}", flush=True)


if __name__ == "__main__":
    main()
