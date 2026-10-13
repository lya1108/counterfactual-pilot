"""Replay video with the C1 score drawn alongside it, synchronised step by step.

Nothing is recomputed. Frames come from the deterministic replay (which agreed with the stored trajectory to
1.5e-08 m for this episode) and scores from the out-of-fold predictions already on disk.

Synchronisation: mp4 frame i is environment step i+1, and the call governing step s is the one issued at
10*floor((s-1)/10) -- the call that produced the actions being executed. The curve only ever draws up to the
current step, so a viewer never sees a score the robot had not produced yet.

Three calls of this episode carry no out-of-fold score and are shown as "no score" rather than filled in:
step 0 (the state log starts at step 1) and steps 60 and 90 (C1 was already active, so they were never
prediction samples). Carrying the previous value across them would invent a prediction that was never made.

The readout is labelled "C1 Risk Score". It is a detector output that has not been calibrated as a
probability, so calling it a failure probability would overstate what was measured.

    conda activate fasi
    python xattn_detector/make_sync_video.py --episode libero_object_t6_i39 --cat TP
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "safe_detector"))

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for _f in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",):
    if Path(_f).exists():
        font_manager.fontManager.addfont(_f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="libero_object_t6_i39")
    ap.add_argument("--cat", default="TP")
    ap.add_argument("--cases", default="results/c1_cases_n200.json")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--exec", type=int, default=10)
    ap.add_argument("--out", default="results/replay")
    args = ap.parse_args()

    C = json.loads((HERE / args.cases).read_text())
    tau = C["tau"]
    e = [x for x in C["chosen"][args.cat] if x["name"] == args.episode][0]
    call_steps = np.array(e["steps"], int)
    scores = np.array(e["score"], float)
    onset = e["onset"]
    score_of = dict(zip(call_steps.tolist(), scores.tolist()))

    src = HERE / args.out / f"{args.cat}_{args.episode}.mp4"
    frames = imageio.mimread(src, memtest=False)
    n = len(frames)
    print(f"{args.episode}: 프레임 {n}, 점수 있는 호출 {len(call_steps)}, onset {onset}")
    print(f"점수 없는 호출: {sorted(set(range(0, n, args.exec)) - set(call_steps.tolist()))}")

    dst = HERE / args.out / f"sync_{args.cat}_{args.episode}.mp4"
    writer = imageio.get_writer(dst, fps=args.fps, macro_block_size=1)
    for i in range(n):
        s = i + 1                                   # env step shown in this frame
        cs = args.exec * ((s - 1) // args.exec)     # the call whose actions are being executed
        val = score_of.get(cs)

        fig = plt.figure(figsize=(8.0, 6.4), dpi=110, constrained_layout=True)
        gs = fig.add_gridspec(2, 1, height_ratios=[2.1, 1])
        ax = fig.add_subplot(gs[0]); ax.axis("off")
        ax.imshow(frames[i])
        ax.text(0.01, 0.97, "agentview", transform=ax.transAxes, fontsize=9, color="w",
                va="top", bbox=dict(fc="k", alpha=0.45, pad=2))
        ax.text(0.51, 0.97, "wrist", transform=ax.transAxes, fontsize=9, color="w",
                va="top", bbox=dict(fc="k", alpha=0.45, pad=2))
        txt = "점수 없음" if val is None else f"{val:.3f}"
        col = "#5A6A74" if val is None else ("#B00020" if val > tau else "#0C6E78")
        ax.text(0.5, 0.03, f"env step {s}   ·   C1 Risk Score (call {cs}) : {txt}",
                transform=ax.transAxes, ha="center", fontsize=12, color="w",
                bbox=dict(fc=col, alpha=0.85, pad=4))
        if onset is not None and s >= onset:
            ax.text(0.5, 0.90, f"C1 onset (step {onset}) 도달", transform=ax.transAxes,
                    ha="center", fontsize=12, color="w",
                    bbox=dict(fc="#A8521B", alpha=0.9, pad=4))

        axc = fig.add_subplot(gs[1])
        past = call_steps <= cs                     # only what has already been produced
        axc.plot(call_steps[past], scores[past], color="#0C6E78", lw=1.8, marker="o", ms=4)
        if val is not None:
            axc.scatter([cs], [val], s=110, color="#B00020" if val > tau else "#0C6E78", zorder=5)
        axc.axhline(tau, color="#B00020", ls="--", lw=1.0)
        axc.text(n * 0.995, tau, f" τ={tau:.2f}", fontsize=7, color="#B00020", va="bottom", ha="right")
        if onset is not None:
            axc.axvline(onset, color="#A8521B", lw=2)
            if s >= onset:
                axc.text(onset, 1.02, " C1 onset", color="#A8521B", fontsize=9, va="bottom")
        axc.axvline(s, color="#111111", lw=1.2, alpha=0.8)
        axc.set_xlim(0, n); axc.set_ylim(0, 1.1)
        axc.set_xlabel("env step"); axc.set_ylabel("C1 Risk Score")
        axc.grid(alpha=0.25)
        axc.set_title("현재 step 까지의 out-of-fold 예측만 표시 (보정된 확률 아님)", fontsize=9)

        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n}", flush=True)
    writer.close()
    print(f"\n-> {dst}  ({dst.stat().st_size / 1024:.0f} KB, {n} 프레임, {args.fps} fps)")


if __name__ == "__main__":
    main()
