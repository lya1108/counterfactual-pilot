"""One-slide summary of Near-Future C1 prediction (lab meeting): panels A-D, read straight from results/.

  A  what is predicted     at a step t where C1 is not active: does C1 start within the next H = 10 env steps?
  B  from what             held-out-task macro AUROC per representation      results/c1_features_n200.json
  C  how early             AUROC by distance to onset, plus the guard inset  results/c1_distance_n200.json,
                                                                              results/c1_guard_n200.json
  D  one rollout           out-of-fold suf_mean scores of libero_object_t6_i39  results/c1_cases_n200.json
                           frames from its deterministic replay              results/replay/frames/
                           replay agreement                                  results/replay_check.json

Nothing is trained or rolled out here. Every plotted number is read from those files and checked against the
values quoted on the slide; the build stops if they disagree.

Notes that decide what the figure may say:
  * The D score is sigmoid(linear probe) trained with class-balanced weights, so it is a ranking score, not a
    calibrated probability -- it is labelled "C1 Risk Score" and never printed as a percentage.
  * No frame exists at exactly step 57: the replay kept steps on a 5-step grid around the onset. The two frames
    used are steps 55 and 60, labelled with their true offsets from the onset.
  * Panel C also draws the Call Index reference from the same distance file. It is not flat: far from the onset
    the call index alone already reaches 0.62-0.64, so the far bins are read against it, not against chance.
    Set SHOW_CALL_INDEX_IN_C = False to drop it.

    ~/miniconda3/envs/fasi/bin/python xattn_detector/figures/make_c1_summary.py   # -> c1_summary.png / .pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
FRAMES = RES / "replay" / "frames"
CASE = "libero_object_t6_i39"
FRAME_STEPS = (55, 60)
SHOW_CALL_INDEX_IN_C = True

# one colour per concept, used in every panel
SUF = "#1F5FA8"        # Action Expert (suf_mean) and the C1 Risk Score it produces
SUF_LIGHT = "#8FB0D8"  # the other pi0.5 representation (vlm_img)
STATE = "#A6A6A6"      # non-pi0.5 baselines
REF = "#C8C8C8"        # call index / reference
C1 = "#C62828"         # C1 onset and the H-step window before it
C1_BG = "#FBE3E3"
INK, GREY = "#1A1A1A", "#555555"

plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "axes.edgecolor": INK,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
})


# ============================================================================ data, read and checked
def load() -> dict:
    j = lambda name: json.loads((RES / name).read_text())
    feat, dist, guard = j("c1_features_n200.json"), j("c1_distance_n200.json"), j("c1_guard_n200.json")
    cases, replay = j("c1_cases_n200.json"), j("replay_check.json")

    assert feat["n_episodes"] == 200 and feat["args"]["H"] == 10
    B = [(k, *feat["out"][k]["auroc"]) for k in ("suf_mean", "vlm_img", "state_hist", "state", "call_index")]
    want_b = {"suf_mean": (0.891, 0.010), "vlm_img": (0.874, 0.006), "state_hist": (0.796, 0.009),
              "state": (0.766, 0.007), "call_index": (0.492, None)}
    for k, m, s in B:
        wm, ws = want_b[k]
        assert round(m, 3) == wm and (ws is None or round(s, 3) == ws), (k, m, s)

    bins = ("30-40", "20-30", "10-20", "0-10")  # far -> near, left -> right
    C = [(b, *dist["out"][b]["cells"]["suf_mean"]["auroc"]) for b in bins]
    C_ref = [dist["out"][b]["cells"]["call_index"]["auroc"][0] for b in bins]
    want_c = {"0-10": (0.891, 0.010), "10-20": (0.840, 0.006), "20-30": (0.793, 0.007), "30-40": (0.765, 0.009)}
    for b, m, s in C:
        assert (round(m, 3), round(s, 3)) == want_c[b], (b, m, s)

    G = [(int(k), guard["out"][k]["cells"]["suf_mean"]["auroc"][0], guard["out"][k]["dropped"]) for k in ("0", "3", "5")]
    assert [round(g[1], 3) for g in G] == [0.891, 0.888, 0.886]

    case = next(c for c in cases["chosen"]["TP"] if c["name"] == CASE)
    assert case["onset"] == 57 and case["success"] is True
    sc = dict(zip(case["steps"], case["score"]))
    assert [round(sc[s], 2) for s in (20, 30, 40, 50)] == [0.21, 0.33, 0.57, 0.88]
    rep = next(e for e in replay["episodes"] if e["name"] == CASE)
    assert not rep["diverged"] and rep["same_length"] and rep["same_outcome"]
    frames = {}
    for s in FRAME_STEPS:
        for cam in ("ag", "wr"):
            p = FRAMES / f"TP_{CASE}__{cam}_s{s:04d}.png"
            assert p.exists(), p
            frames[s, cam] = np.asarray(Image.open(p).convert("RGB"))
    ep = next(json.loads(l) for l in (HERE.parent / "data" / "cond200" / "episodes.jsonl").read_text().splitlines()
              if l.strip() and json.loads(l)["suite"] == "libero_object" and json.loads(l)["task"] == 6
              and json.loads(l)["init"] == 39)
    assert ep["condition_onset_step"]["C1_grasp_empty_close"] == 57
    return dict(B=B, C=C, C_ref=C_ref, bins=bins, G=G, case=case, frames=frames, language=ep["language"],
                n_steps=ep["n_steps"], d_max=rep["d_max_all"])


# ============================================================================ panels
def label(ax, letter, title, sub=None, x=-0.02):
    ax.text(x, 1.13, f"({letter})  {title}", transform=ax.transAxes, fontsize=19, fontweight="bold", va="bottom",
            ha="left")
    if sub:
        ax.text(x, 1.065, sub, transform=ax.transAxes, fontsize=14, color=GREY, va="bottom", ha="left",
                style="italic")


def panel_a(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    label(ax, "A", "Near-future C1 prediction", x=0.0)
    y, t, w = 5.0, 3.4, 7.9  # timeline height, current step, end of the 10-step window
    onset = 6.0
    ax.add_patch(Rectangle((t, y - 0.75), w - t, 1.5, facecolor=C1_BG, edgecolor="none", zorder=0))
    ax.plot([0.4, t], [y, y], color=REF, lw=5, solid_capstyle="round", zorder=1)
    ax.add_patch(FancyArrowPatch((t, y), (9.7, y), arrowstyle="-|>", mutation_scale=22, lw=2.2, color=INK, zorder=1))
    ax.text(1.9, y + 0.45, "past rollout", ha="center", fontsize=14, color=GREY)
    # bracket over the window
    by = y + 1.15
    ax.plot([t, t, w, w], [by - 0.2, by, by, by - 0.2], color=C1, lw=1.8)
    ax.text((t + w) / 2, by + 0.3, "next 10 env steps", ha="center", fontsize=15, color=C1, fontweight="bold")
    # current step and onset
    ax.plot([t], [y], "o", ms=14, color=SUF, zorder=3)
    ax.text(t, y - 1.35, "current step t", ha="center", fontsize=15, fontweight="bold", color=SUF)
    ax.text(t, y - 1.95, "(C1 not active)", ha="center", fontsize=13, color=GREY)
    ax.plot([onset], [y], "D", ms=13, color=C1, zorder=3)
    ax.text(onset, y - 1.35, "C1 onset?", ha="center", fontsize=15, fontweight="bold", color=C1)
    ax.plot([w, w], [y - 0.75, y + 0.75], color=C1, lw=1.5, ls=(0, (3, 2)))
    ax.text(w, y - 1.35, "t + 10", ha="center", fontsize=14, color=GREY)
    # the question asked at t
    ax.add_patch(FancyArrowPatch((t, y + 0.35), (t, 7.55), arrowstyle="-|>", mutation_scale=18, lw=2, color=SUF))
    ax.text(t - 0.2, 6.75, "C1 Risk Score", ha="right", fontsize=15, color=SUF, fontweight="bold")
    ax.text(0.25, 8.75, "At step t, from π0.5 features, predict:", ha="left", fontsize=15, color=INK)
    ax.text(0.25, 7.95, "Will C1 start within the next 10 env steps?", ha="left", fontsize=16, color=SUF,
            fontweight="bold")
    # definitions
    ax.text(0.25, 1.75, "C1  =  Empty Close / Missed Grasp  (fingers close with nothing between them)",
            fontsize=13, color=GREY)
    ax.text(0.25, 1.05, "C1 onset  =  the env step at which C1 starts", fontsize=13, color=GREY)


def panel_b(ax, B):
    names = {"suf_mean": "Action Expert (suf_mean)", "vlm_img": "π0.5 Image (vlm_img)",
             "state_hist": "State History", "state": "State", "call_index": "Call Index"}
    colors = {"suf_mean": SUF, "vlm_img": SUF_LIGHT, "state_hist": STATE, "state": STATE, "call_index": REF}
    lo = 0.4
    ys = np.arange(len(B))[::-1]
    for yv, (k, m, s) in zip(ys, B):
        ax.barh(yv, m - lo, left=lo, height=0.62, color=colors[k], zorder=2)
        if s > 0:
            ax.errorbar(m, yv, xerr=s, color=INK, lw=1.4, capsize=4, zorder=3)
        txt = f"{m:.3f}" + (f" ± {s:.3f}" if s > 0 else "")
        ax.text(m + max(s, 0) + 0.008, yv, txt, va="center", fontsize=15,
                fontweight="bold" if k == "suf_mean" else "normal", color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([names[k] for k, *_ in B], fontsize=15)
    for tl, (k, *_) in zip(ax.get_yticklabels(), B):
        if k == "suf_mean":
            tl.set_fontweight("bold")
            tl.set_color(SUF)
    ax.axvline(0.5, color=INK, lw=1.4, ls=(0, (4, 3)), zorder=1)
    ax.text(0.507, len(B) - 0.5, "chance", ha="left", fontsize=13, color=GREY)
    ax.set_xlim(lo, 1.13)
    ax.spines["bottom"].set_bounds(lo, 1.0)
    ax.set_ylim(-0.6, len(B) - 0.3)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("Task-macro AUROC  (held-out tasks)", fontsize=15)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    label(ax, "B", "Near-future C1 is predictable from π0.5 internal representations",
          sub="200 rollouts · H = 10 env steps · linear probe per representation", x=-0.40)


def panel_c(ax, ins, C, C_ref, bins, G):
    xs = np.arange(len(C))
    m = np.array([c[1] for c in C])
    s = np.array([c[2] for c in C])
    if SHOW_CALL_INDEX_IN_C:
        ax.plot(xs, C_ref, color=REF, lw=2, ls=(0, (5, 3)), marker="s", ms=7, zorder=2)
        ax.text(-0.25, 0.668, "Call Index (reference)", fontsize=13, color=GREY, ha="left")
    ax.errorbar(xs, m, yerr=s, color=SUF, lw=3, marker="o", ms=11, capsize=5, zorder=3)
    for x, v in zip(xs, m):
        if x == len(C) - 1:  # nearest bin: above, clear of the onset line
            ax.text(x, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=15, fontweight="bold", color=SUF)
        else:
            ax.text(x + 0.08, v - 0.03, f"{v:.3f}", ha="left", va="top", fontsize=15, fontweight="bold", color=SUF)
    # onset at the right end
    xo = len(C) - 0.35
    ax.axvline(xo, color=C1, lw=2.2, ls=(0, (5, 3)))
    ax.text(xo + 0.07, 0.72, "C1 onset", color=C1, fontsize=15, fontweight="bold", ha="left", va="center",
            rotation=90)
    ax.axhline(0.5, color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.text(-0.38, 0.508, "chance", fontsize=13, color=GREY)
    ax.set_xticks(xs)
    ax.set_xticklabels([b.replace("-", "–") for b in bins])
    ax.set_xlim(-0.45, len(C) - 0.05)
    ax.set_ylim(0.47, 0.96)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel("Steps before C1 onset", fontsize=15)
    ax.set_ylabel("Task-macro AUROC", fontsize=15)
    ax.text(-0.38, 0.94, "signal also observed\n30–40 steps before onset", fontsize=13.5, color=SUF,
            ha="left", va="top", style="italic")
    label(ax, "C", "Predictive signal before C1 onset", sub="Action Expert (suf_mean) · AUROC per distance bin",
          x=-0.12)

    # guard inset: drop samples within k steps of the onset
    g = [x[0] for x in G]
    gv = [x[1] for x in G]
    ins.plot(range(3), gv, color=SUF, lw=2.2, marker="o", ms=8)
    for i, v in enumerate(gv):
        ins.text(i, v + 0.025 if i != 1 else v - 0.045, f"{v:.3f}", ha="center", fontsize=12.5, color=SUF)
    ins.set_xticks(range(3))
    ins.set_xticklabels([str(k) for k in g], fontsize=13)
    ins.set_ylim(0.47, 0.96)
    ins.set_yticks([0.5, 0.7, 0.9])
    ins.tick_params(axis="y", labelsize=12)
    ins.axhline(0.5, color=INK, lw=1.0, ls=(0, (4, 3)))
    ins.set_xlim(-0.7, 2.5)
    ins.set_xlabel("guard k (steps)", fontsize=13, labelpad=2)
    ins.set_title("Guard: drop the last k\nsteps before onset", fontsize=13, color=INK, pad=6)
    for sp in ("top", "right"):
        ins.spines[sp].set_visible(False)


def panel_d(fig, cell, d):
    sub = cell.subgridspec(1, 2, width_ratios=[2.15, 1.0], wspace=0.08)
    ax = fig.add_subplot(sub[0])
    case = d["case"]
    steps, score = np.array(case["steps"]), np.array(case["score"])
    onset = case["onset"]
    pre = steps < onset
    ax.axvspan(onset - 10, onset, color=C1_BG, zorder=0, lw=0)
    # post-onset: grey, broken where no score exists (C1 active at that call)
    post_s, post_v = steps[~pre], score[~pre]
    seg_start = 0
    for i in range(1, len(post_s) + 1):
        if i == len(post_s) or post_s[i] - post_s[i - 1] > 10:
            ax.plot(post_s[seg_start:i], post_v[seg_start:i], color=STATE, lw=2, marker="o", ms=5, zorder=2)
            seg_start = i
    ax.plot(steps[pre], score[pre], color=SUF, lw=3.2, marker="o", ms=10, zorder=3)
    for s_, v in zip(steps[pre], score[pre]):
        if s_ in (20, 30):
            ax.text(s_ + 3.5, v - 0.07, f"{v:.2f}", ha="left", fontsize=15, fontweight="bold", color=SUF)
        elif s_ == 40:
            ax.text(s_ - 3.5, v + 0.03, f"{v:.2f}", ha="right", fontsize=15, fontweight="bold", color=SUF)
        elif s_ == 50:
            ax.text(s_ - 3.5, v + 0.04, f"{v:.2f}", ha="right", fontsize=15, fontweight="bold", color=SUF)
    ax.axvline(onset, color=C1, lw=2.4, ls=(0, (5, 3)), zorder=1)
    ax.text(onset + 3, 0.03, f"C1 onset\n(step {onset})", color=C1, fontsize=14, fontweight="bold", va="bottom")
    ax.text(225, 0.97, "Local C1 occurred → recovered\n→ final task success", ha="right", va="top", fontsize=13,
            color=GREY)
    ax.set_xlim(0, d["n_steps"] + 4)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Env step", fontsize=15)
    ax.set_ylabel("C1 Risk Score", fontsize=15)
    label(ax, "D", f"Example rollout: {CASE}",
          sub=f"“{d['language']}” · out-of-fold score (task held out)", x=-0.14)

    # frames from the replay of this exact episode
    fr = sub[1].subgridspec(2, 2, hspace=0.42, wspace=0.05)
    for r, s_ in enumerate(FRAME_STEPS):
        off = s_ - onset
        cap = f"step {s_}  ({-off} before onset)" if off < 0 else f"step {s_}  ({off} after onset)"
        for c, cam in enumerate(("ag", "wr")):
            a = fig.add_subplot(fr[r, c])
            a.imshow(d["frames"][s_, cam])
            a.set_xticks([])
            a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(True)
                sp.set_color(C1 if off >= 0 else SUF)
                sp.set_linewidth(2.5)
            if c == 0:
                a.text(0.0, 1.05, cap, transform=a.transAxes, fontsize=12.5, fontweight="bold",
                       color=C1 if off >= 0 else SUF, va="bottom")
            if r == 1:
                a.text(0.5, -0.1, "agentview" if cam == "ag" else "wrist", transform=a.transAxes, fontsize=12,
                       color=GREY, ha="center", va="top")


def main():
    d = load()
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, left=0.055, right=0.985, top=0.885, bottom=0.075, wspace=0.42, hspace=0.62,
                          width_ratios=[1, 1.08])
    panel_a(fig.add_subplot(gs[0, 0]))
    panel_b(fig.add_subplot(gs[0, 1]), d["B"])
    cg = gs[1, 0].subgridspec(1, 2, width_ratios=[3.0, 1.0], wspace=0.32)
    panel_c(fig.add_subplot(cg[0]), fig.add_subplot(cg[1]), d["C"], d["C_ref"], d["bins"], d["G"])
    panel_d(fig, gs[1, 1], d)
    out = HERE / "c1_summary"
    fig.savefig(out.with_suffix(".png"), dpi=240, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    print(out.with_suffix(".png"), out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
