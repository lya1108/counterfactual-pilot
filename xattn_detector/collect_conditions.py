"""Collect rollouts with everything the temporal analysis needs, from one execution.

The condition pilot logged environment state but neither the detector's input nor where each VLA call began,
so a score trajectory could not be placed on the environment-step axis. This records all of it together:

  per environment step   the raw state StateLogger already captured (object poses, contacts, opening, ...)
  per VLA call           suf_mean, and `call_step` -- the environment step at which that call was issued
  around each onset      agentview frames, so the automatic conditions can be checked against what happened

Latents and labels come from the same run on purpose. Replaying an earlier run does not reproduce it bit for
bit (measured: identical for 97% of episodes, up to 1.8 cm apart for the rest), so pairing a new state log with
old latents would silently mix two trajectories.

Frames are held in memory for the episode and only the windows around onsets are written, because onsets are
known only once the episode has ended.

    conda activate fasi
    python xattn_detector/collect_conditions.py --suites libero_spatial libero_object libero_goal libero_10 \
        --tasks 0,2,4,6,9 --inits 30,31 --out data/cond40
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
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "safe_detector"))

import jb  # noqa: E402

jb.enter()

from fasi.envs.libero_env import LiberoTask  # noqa: E402
from fasi.vla.pi05_steer import Pi05Steer  # noqa: E402

from conditions import CONDITIONS, StateLogger, earliest_onset, evaluate  # noqa: E402
from token_tap import TokenTap  # noqa: E402


def parse_ids(s):
    out = []
    for part in str(s).split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/vla/pi05_libero_finetuned")
    ap.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    ap.add_argument("--tasks", default="0,2,4,6,9")
    ap.add_argument("--inits", default="30,31")
    ap.add_argument("--exec", type=int, default=10)
    ap.add_argument("--frame-window", type=int, default=12, help="onset 전후로 저장할 env step 범위")
    ap.add_argument("--frame-stride", type=int, default=3)
    ap.add_argument("--store-tokens", action="store_true",
                    help="action suffix 토큰과 VLM prefix 도 저장 (호출당 약 0.85 MB)")
    ap.add_argument("--out", default="data/cond40")
    args = ap.parse_args()

    out = HERE / args.out
    (out / "frames").mkdir(parents=True, exist_ok=True)
    index = out / "episodes.jsonl"
    done = set()
    if index.exists():
        for line in index.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["suite"], r["task"], r["init"]))

    vla = Pi05Steer(args.ckpt)
    tap = TokenTap(vla.policy, pool=2)
    t_start = time.time()
    n = 0
    for suite in args.suites:
        for tid in parse_ids(args.tasks):
            task = LiberoTask(suite, tid)
            for init in parse_ids(args.inits):
                if (suite, tid, init) in done:
                    continue
                task.reset(init)
                log = StateLogger(task)
                suf, call_step, frames = [], [], {}
                # item 6 needs the pieces separately, and re-collecting later would cost another pass
                suf_tok, pre_img, pre_lang = [], [], []
                t, ok = 0, False
                while t < task.max_steps:
                    ag = task.obs["agentview_image"][::-1, ::-1].copy()
                    wr = task.obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
                    b = vla.batch_for(ag, wr, task.state8(), task.language)
                    g = torch.Generator().manual_seed(1000 * tid + 17 * init + t)
                    tap.reset()
                    a = vla.denoise(b, torch.randn((1, vla.C, vla.A), generator=g), [], None)[0]
                    got = tap.pop(want_steps=("mean",))
                    st = got["suf_mean"].float().cpu().numpy()          # (50, 1024) action tokens
                    suf.append(st.mean(0).astype(np.float32))           # the pooled vector used so far
                    if args.store_tokens:
                        suf_tok.append(st.astype(np.float16))
                        pre_img.append(got["pre_img"].float().cpu().numpy().astype(np.float16))
                        pre_lang.append(got["pre_lang"].float().cpu().numpy().astype(np.float16))
                    call_step.append(t)  # the environment step this call was issued at
                    for j in range(args.exec):
                        task.step(a[j])
                        t += 1
                        log.record(t)
                        if t % args.frame_stride == 0:
                            frames[t] = task.obs["agentview_image"][::-1].copy()
                        if task.success():
                            ok = True
                            break
                    if ok:
                        break

                packed = log.pack()
                _, onsets = evaluate(packed)
                steps = packed["step"]
                onset_steps = {c: (int(steps[v]) if v is not None else None) for c, v in onsets.items()}

                name = f"{suite}_t{tid}_i{init}"
                extra = {}
                if args.store_tokens and suf_tok:
                    L = max(x.shape[0] for x in pre_lang)
                    D = pre_lang[0].shape[1]
                    pl = np.zeros((len(pre_lang), L, D), np.float16)
                    lm = np.zeros((len(pre_lang), L), bool)
                    for i, x in enumerate(pre_lang):
                        pl[i, : x.shape[0]] = x
                        lm[i, : x.shape[0]] = True
                    extra = dict(suf_tokens=np.stack(suf_tok), pre_img=np.stack(pre_img),
                                 pre_lang=pl, lang_mask=lm)
                np.savez_compressed(out / f"{name}.npz", suf_mean=np.stack(suf),
                                    call_step=np.asarray(call_step, np.int32), **packed, **extra)

                saved = []
                for c, s0 in onset_steps.items():
                    if s0 is None:
                        continue
                    for s in sorted(frames):
                        if abs(s - s0) <= args.frame_window:
                            p = out / "frames" / f"{name}__{c.split('_')[0]}_on{s0}__s{s:04d}.png"
                            if not p.exists():
                                Image.fromarray(frames[s]).save(p)
                            saved.append(p.name)

                rec = dict(suite=suite, task=tid, init=init, success=ok, n_steps=t, n_calls=len(suf),
                           language=task.language, goal_names=[str(x) for x in packed["goal_names"]],
                           object_names=[str(x) for x in packed["object_names"]],
                           goal_objects=[str(x) for x in packed["goal_objects"]],
                           condition_onset_step=onset_steps, n_frames=len(saved), npz=f"{name}.npz")
                with open(index, "a") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                k, i = earliest_onset(onsets)
                print(f"  [{n}] {name} {'S' if ok else 'F'} steps={t} calls={len(suf)} "
                      f"onset={ {c: v for c, v in onset_steps.items() if v is not None} } "
                      f"earliest={k.split('_')[0] + '@' + str(steps[i]) if k else '없음'} "
                      f"({time.time() - t_start:.0f}s)", flush=True)
            task.close()
    tap.close()
    print(f"\n{n} 에피소드 -> {out}   프레임 -> {out / 'frames'}")


if __name__ == "__main__":
    main()
