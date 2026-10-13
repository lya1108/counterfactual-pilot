"""Smoke test before collecting anything: does the tap fire, and is this checkpoint usable as a SAFE subject?

SAFE needs a policy whose rollouts contain BOTH classes. Base pi0.5 on LIBERO is 0/30 in
~/바탕화면/jungbeom/outputs/libero/base_l10/episodes.jsonl, so this run exists to check whether the LIBERO
fine-tune (outputs/vla/pi05_libero_finetuned) succeeds often enough through the same sampler, and which image
convention it wants (libero_base_eval.py:33 offers upright vs raw-rotated-180).

    conda activate fasi && python safe_detector/smoke_latent.py --suite libero_spatial --tasks 0 --inits 0,1
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

import jb

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/vla/pi05_libero_finetuned")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="0")
    ap.add_argument("--inits", default="0,1")
    ap.add_argument("--exec", type=int, default=10)
    ap.add_argument("--views", default="upright,rot180", help="image conventions to compare")
    args = ap.parse_args()

    vla = Pi05Steer(args.ckpt)
    cfg = vla.policy.config
    print(f"ckpt={args.ckpt}  K={vla.K} chunk={vla.C} action_dim={vla.A}")
    print(f"image_features={list(cfg.image_features)}  -> batch_for fills {min(2, len(vla.feats))} of them")

    tap = LatentTap(vla.policy)
    shapes_reported = False
    for view in args.views.split(","):
        for tid in parse_ids(args.tasks):
            task = LiberoTask(args.suite, tid)
            for init in parse_ids(args.inits):
                task.reset(init)
                t, ok, t0, n_calls = 0, False, time.time(), 0
                while t < task.max_steps:
                    if view == "rot180":
                        ag = task.obs["agentview_image"][::-1, ::-1].copy()
                        wr = task.obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
                    else:
                        ag, wr = task.images()
                    b = vla.batch_for(ag, wr, task.state8(), task.language)
                    g = torch.Generator().manual_seed(1000 * tid + 17 * init + t)
                    tap.reset()
                    a = vla.denoise(b, torch.randn((1, vla.C, vla.A), generator=g), [], None)[0]
                    feat = tap.pop()
                    n_calls += 1
                    if feat is None:
                        raise SystemExit("tap captured nothing -- the hook did not fire")
                    if not shapes_reported:
                        for k in FEATURES:
                            v = feat[k]
                            print(f"  latent {k:9s} shape={v.shape} dtype={v.dtype} |v|={np.linalg.norm(v):.1f}")
                        shapes_reported = True
                    for j in range(args.exec):
                        task.step(a[j])
                        t += 1
                        if task.success():
                            ok = True
                            break
                    if ok:
                        break
                print(f"[{view}] task {tid} init {init}  success={ok}  t={t}  calls={n_calls}  "
                      f"({time.time() - t0:.0f}s)  | {task.language}", flush=True)
            task.close()
    tap.close()


if __name__ == "__main__":
    main()
