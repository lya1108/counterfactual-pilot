"""Deterministically replay the eight pre-selected episodes and record what the stored run could not.

The eight come from c1_cases.py's rule -- the two episodes nearest each category's median -- and are replayed
as given. None is swapped out: an episode whose replay drifts from the stored trajectory is reported as
diverged, not replaced, because replacing it would quietly turn a fixed selection into a filtered one.

Agreement is measured where it can be: the stored run logged the end-effector at every environment step, so the
replay's end-effector is compared against it at every VLA call step and over the whole episode. Replay across
processes is not bit-exact here -- measured earlier as agreeing to ~6e-8 m for 97% of episodes and drifting up
to 1.8 cm for the rest -- so the per-episode maximum is what decides whether the frames can be trusted to show
the rollout the detector scored.

Both cameras are written, the whole rollout goes to video, and for episodes with a C1 onset the window from 40
steps before to 10 after is kept frame by frame.

    conda activate fasi
    python xattn_detector/replay_cases.py --cases results/c1_cases_n200.json
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "safe_detector"))

import imageio.v2 as imageio  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

for _f in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",):
    if Path(_f).exists():
        font_manager.fontManager.addfont(_f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

import jb  # noqa: E402

jb.enter()

from fasi.envs.libero_env import LiberoTask  # noqa: E402
from fasi.vla.pi05_steer import Pi05Steer  # noqa: E402

DIVERGE_M = 1e-3  # above this the replayed frames no longer describe the scored trajectory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="results/c1_cases_n200.json")
    ap.add_argument("--data", default="data/cond200")
    ap.add_argument("--ckpt", default="outputs/vla/pi05_libero_finetuned")
    ap.add_argument("--exec", type=int, default=10)
    ap.add_argument("--pre", type=int, default=40, help="onset 기준 몇 step 전부터 프레임 보관")
    ap.add_argument("--post", type=int, default=10)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", default="results/replay")
    args = ap.parse_args()

    cases = json.loads((HERE / args.cases).read_text())
    tau = cases["tau"]
    out = HERE / args.out
    (out / "frames").mkdir(parents=True, exist_ok=True)
    data = HERE / args.data

    todo = []
    for cat, items in cases["chosen"].items():
        for e in items:
            suite, rest = e["name"].rsplit("_t", 1)
            tid, init = rest.split("_i")
            todo.append(dict(cat=cat, name=e["name"], suite=suite, task=int(tid), init=int(init),
                             onset=e["onset"], steps=np.array(e["steps"]), score=np.array(e["score"]),
                             success=e["success"]))
    print(f"replay 대상 {len(todo)}개 (사전 선정 그대로, 교체 없음)   τ={tau:.4f}\n")

    vla = Pi05Steer(args.ckpt)
    report = []
    cur, task = None, None
    for n, c in enumerate(sorted(todo, key=lambda x: (x["suite"], x["task"], x["init"])), 1):
        if (c["suite"], c["task"]) != cur:
            if task is not None:
                task.close()
            task = LiberoTask(c["suite"], c["task"])
            cur = (c["suite"], c["task"])
        t0 = time.time()
        task.reset(c["init"])
        t, ok = 0, False
        eef_rep, ag_all, wr_all, call_steps = [], [], [], []
        while t < task.max_steps:
            ag = task.obs["agentview_image"][::-1, ::-1].copy()
            wr = task.obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
            b = vla.batch_for(ag, wr, task.state8(), task.language)
            g = torch.Generator().manual_seed(1000 * c["task"] + 17 * c["init"] + t)
            a = vla.denoise(b, torch.randn((1, vla.C, vla.A), generator=g), [], None)[0]
            call_steps.append(t)
            for j in range(args.exec):
                task.step(a[j])
                t += 1
                eef_rep.append(task.eef().copy())
                ag_all.append(task.obs["agentview_image"][::-1].copy())
                wr_all.append(task.obs["robot0_eye_in_hand_image"][::-1].copy())
                if task.success():
                    ok = True
                    break
            if ok:
                break

        # ---- agreement with the stored run
        z = np.load(data / f"{c['name']}.npz")
        s_stored, eef_stored = z["step"].astype(int), z["eef"]
        eef_rep = np.asarray(eef_rep)
        n_cmp = min(len(eef_rep), len(eef_stored))
        d_all = np.abs(eef_rep[:n_cmp] - eef_stored[:n_cmp]).max(1)
        idx_calls = [s - 1 for s in c["steps"] if 0 < s <= n_cmp]
        d_calls = d_all[idx_calls] if idx_calls else np.array([0.0])
        diverged = float(d_all.max()) > DIVERGE_M
        report.append(dict(cat=c["cat"], name=c["name"], onset=c["onset"],
                           steps_stored=int(len(s_stored)), steps_replay=int(len(eef_rep)),
                           same_length=len(s_stored) == len(eef_rep), same_outcome=bool(ok) == c["success"],
                           d_max_all=float(d_all.max()), d_max_calls=float(d_calls.max()),
                           d_median_all=float(np.median(d_all)), diverged=diverged))
        print(f"  [{n}/{len(todo)}] {c['cat']:3s} {c['name']:24s} "
              f"길이 {len(eef_rep)}/{len(s_stored)}  결과 {'일치' if bool(ok) == c['success'] else '불일치'}  "
              f"|Δeef| 최대 {d_all.max():.2e} (호출시점 {d_calls.max():.2e})"
              f"{'   ** DIVERGED' if diverged else ''}  ({time.time() - t0:.0f}s)", flush=True)

        # ---- video over the whole rollout
        both = [np.concatenate([a_, w_], 1) for a_, w_ in zip(ag_all, wr_all)]
        imageio.mimsave(out / f"{c['cat']}_{c['name']}.mp4", both, fps=args.fps, macro_block_size=1)

        # ---- frames in the window of interest
        lo, hi = (c["onset"] - args.pre, c["onset"] + args.post) if c["onset"] is not None else (1, len(ag_all))
        kept = []
        for s in range(max(1, lo), min(len(ag_all), hi) + 1):
            if s in c["steps"] or s % 5 == 0:
                for tag, img in (("ag", ag_all[s - 1]), ("wr", wr_all[s - 1])):
                    p = out / "frames" / f"{c['cat']}_{c['name']}__{tag}_s{s:04d}.png"
                    imageio.imwrite(p, img)
                kept.append(s)

        # ---- one figure: the score curve above the frames at those call steps
        calls = [s for s in c["steps"] if (c["onset"] is None or lo <= s <= hi) and s <= len(ag_all)]
        calls = calls[-8:] if len(calls) > 8 else calls
        fig = plt.figure(figsize=(max(9, 1.5 * len(calls)), 6.4), constrained_layout=True)
        gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1, 1])
        ax = fig.add_subplot(gs[0])
        ax.plot(c["steps"], c["score"], color="#0C6E78", lw=1.8, marker="o", ms=4)
        ax.axhline(tau, color="#B00020", ls="--", lw=1.2)
        if c["onset"] is not None:
            ax.axvline(c["onset"], color="#A8521B", lw=2)
            ax.text(c["onset"], 1.02, " C1 onset", color="#A8521B", fontsize=9, va="bottom")
        for s in calls:
            ax.axvline(s, color="#5A6A74", lw=0.6, alpha=0.5)
        ax.set_ylim(0, 1.05); ax.set_ylabel("out-of-fold score")
        ax.set_title(f"[{c['cat']}] {c['name']}   최종 {'성공' if c['success'] else '실패'}   "
                     f"|Δeef| 최대 {d_all.max():.1e} m{'  ** DIVERGED' if diverged else ''}", fontsize=11)
        ax.grid(alpha=0.25)
        for row, tag, lab in ((1, "ag", "agentview"), (2, "wr", "wrist")):
            axf = fig.add_subplot(gs[row]); axf.axis("off")
            ims = [np.asarray(imageio.imread(out / "frames" / f"{c['cat']}_{c['name']}__{tag}_s{s:04d}.png"))
                   for s in calls]
            if ims:
                axf.imshow(np.concatenate([np.asarray(thumb(i)) for i in ims], 1))
                axf.set_xticks([64 + 128 * i for i in range(len(calls))])
                onset = c["onset"]
                labels = [f"{s}" if onset is None else f"{s}\n{onset - s:+d}" for s in calls]
                axf.set_xticklabels(labels, fontsize=7)
                axf.tick_params(axis="x", length=0, pad=1)
            axf.set_ylabel(lab, fontsize=8)
        fig.savefig(out / f"panel_{c['cat']}_{c['name']}.png", dpi=140)
        plt.close(fig)

    if task is not None:
        task.close()
    print(f"\n=== 저장된 trajectory 와의 일치 ===")
    print(f"  {'cat':4s} {'episode':24s} {'길이':>9s} {'결과':>5s} {'|Δeef| 최대':>12s} {'호출시점':>11s} {'판정':>10s}")
    for r in report:
        print(f"  {r['cat']:4s} {r['name']:24s} {r['steps_replay']}/{r['steps_stored']:<4d} "
              f"{'일치' if r['same_outcome'] else '불일치':>5s} {r['d_max_all']:12.2e} "
              f"{r['d_max_calls']:11.2e} {'DIVERGED' if r['diverged'] else 'OK':>10s}")
    nd = sum(r["diverged"] for r in report)
    print(f"\n  기준 {DIVERGE_M} m 초과 = divergence.  {len(report) - nd}/{len(report)} 일치, {nd} 건 divergence")
    (HERE / "results" / "replay_check.json").write_text(
        json.dumps(dict(tau=tau, diverge_threshold=DIVERGE_M, episodes=report), indent=2,
                   ensure_ascii=False, default=float))
    print(f"\n-> {out}/*.mp4   패널 {out}/panel_*.png   프레임 {out}/frames/")
    print(f"-> {HERE / 'results' / 'replay_check.json'}")


def thumb(img, size=128):
    from PIL import Image
    return Image.fromarray(img).resize((size, size))


if __name__ == "__main__":
    main()
