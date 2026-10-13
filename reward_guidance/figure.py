"""One figure for the whole chain: what the head knows, what it can steer, and what training fixes.

Left    Stage A -- the same head scored two ways. Between states it is strong; within a state it is not
        separable from a scalar read straight off the action chunk, and a state feature is pinned at 0.5.
Middle  Stage C -- how far each guided arm moved the head's own score against how far it moved the world.
        The random arms are the ruler: a point only means something if it sits off their line.
Right   Stage D -- heads trained directly on the within-state question. The gripper line is the thing to
        beat; `resid` has that one direction regressed out, so its height is what phi adds on its own.

    python reward_guidance/figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
fp = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if Path(fp).exists():
    fm.fontManager.addfont(fp)
    plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
plt.rcParams["axes.unicode_minus"] = False

S = json.load(open(HERE / "results/sensitivity_main.json"))
C = json.load(open(HERE / "results/stagec_main.json"))
R = json.load(open(HERE / "results/ranking_main.json"))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19.5, 5.4))

# ---------------------------------------------------------------- left
rows = [("상태 간\n(head, macro)", S["between_macro"], None, "#2b6cb0"),
        ("상태 내\n(head)", S["within_auroc_boot"][0], S["within_auroc_boot"][1:], "#2b6cb0"),
        ("상태 내\n(그리퍼 명령 평균)", *[[S["ablation"]["그리퍼 명령 평균"][1], S["ablation"]["그리퍼 명령 평균"][2:]][i] for i in (0, 1)], "#dd6b20"),
        ("상태 내\n(상태 특징)", 0.5, None, "#a0aec0")]
for i, (lab, v, ci, col) in enumerate(rows):
    ax1.bar(i, v, 0.62, color=col, zorder=3)
    if ci:
        ax1.errorbar(i, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor="#1a202c", capsize=5, lw=1.4, zorder=4)
    ax1.text(i, v + 0.025, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
ax1.text(3, 0.455, "정의상 고정", ha="center", fontsize=9.5, color="#4a5568")
ax1.axhline(0.5, color="#718096", ls="--", lw=1)
ax1.set_xticks(range(4)); ax1.set_xticklabels([r[0] for r in rows], fontsize=10)
ax1.set_ylim(0.35, 1.0); ax1.set_ylabel("AUROC")
ax1.set_title("같은 head, 두 가지 질문", fontsize=13, fontweight="bold")
ax1.grid(axis="y", alpha=0.3, zorder=0)

# ---------------------------------------------------------------- right
order = ["위험↓ λ10", "위험↑ λ10", "무작위 λ10", "위험↓ λ30", "무작위 λ30"]
mk = {"위험↓": "v", "위험↑": "^", "무작위": "o"}
for name in order:
    a = C["arms"][name]
    kind = name.split()[0]
    col = {"위험↓": "#2f855a", "위험↑": "#c53030", "무작위": "#718096"}[kind]
    x, y = a["dlogit"], a["drate"][0] * 100
    lo, hi = a["drate"][1] * 100, a["drate"][2] * 100
    ax2.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt=mk[kind], ms=11, color=col,
                 capsize=5, lw=1.5, mec="white", mew=1.2, zorder=3,
                 label=name if name.endswith("λ10") else None)
    ax2.annotate(name, (x, y), textcoords="offset points", xytext=(8, 9), fontsize=9.5, color=col)
ax2.axhline(0, color="#1a202c", lw=1)
ax2.axvline(0, color="#718096", ls="--", lw=1)
ax2.set_xlabel("head 자신의 C1 점수 변화 (logit)")
ax2.set_ylabel("실제 C1 발생률 변화 (%p, 무개입 대비)")
ax2.set_title("점수를 움직인 만큼 세계가 움직이는가", fontsize=13, fontweight="bold")
ax2.grid(alpha=0.3, zorder=0)
ax2.text(0.02, 0.04, f"844 호출 · 짝지은 비교 · 95% CI 는 에피소드 부트스트랩\n"
                     f"무개입 C1 발생률 {C['arms']['무개입']['rate']:.1%}",
         transform=ax2.transAxes, fontsize=9, color="#4a5568", va="bottom")

# ---------------------------------------------------------------- right
st = R["stat"]
grip = st["그리퍼 명령 평균"]["auc"]
frz = st["현재 C1 head (frozen)"]["auc"]
ax3.axhspan(grip[1], grip[2], color="#dd6b20", alpha=0.13, zorder=0)
ax3.axhline(grip[0], color="#dd6b20", ls="-", lw=1.8, zorder=2)
ax3.axhline(frz[0], color="#2b6cb0", ls=":", lw=1.6, zorder=2)
ax3.axhline(0.5, color="#718096", ls="--", lw=1, zorder=2)

variants = [("raw", "raw\n(phi 전체)"), ("pca", "pca\n(상태 내 PC)"), ("resid", "resid\n(그리퍼 제거)")]
splits = [("과제", "#2f855a", -0.19), ("에피소드", "#9ae6b4", 0.19)]
for xi, (v, lab) in enumerate(variants):
    for tag, col, off in splits:
        a = st[f"Ranking {v} [{tag}]"]["auc"]
        ax3.bar(xi + off, a[0], 0.34, color=col, zorder=3, edgecolor="#22543d", lw=0.6,
                label=f"{tag} LOGO" if xi == 0 else None)
        ax3.errorbar(xi + off, a[0], yerr=[[a[0] - a[1]], [a[2] - a[0]]], fmt="none",
                     ecolor="#1a202c", capsize=4, lw=1.3, zorder=4)
        ax3.text(xi + off, a[0] + 0.012, f"{a[0]:.3f}", ha="center", fontsize=9, fontweight="bold")
ax3.text(2.42, grip[0] + 0.008, f"그리퍼 명령 평균 {grip[0]:.3f}", ha="right", fontsize=9, color="#9c4221")
ax3.text(2.42, frz[0] - 0.028, f"frozen head {frz[0]:.3f}", ha="right", fontsize=9, color="#2b6cb0")
ax3.text(-0.50, 0.506, "무작위", fontsize=9, color="#4a5568",
         bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
ax3.set_xticks(range(3)); ax3.set_xticklabels([l for _, l in variants], fontsize=10)
ax3.set_xlim(-0.55, 2.55); ax3.set_ylim(0.45, 0.92)
ax3.set_ylabel("상태 내 AUROC (폴드 외 예측)")
ax3.set_title("상태 내 질문으로 직접 학습하면", fontsize=13, fontweight="bold")
ax3.legend(fontsize=9, loc="upper left", framealpha=0.9)
ax3.grid(axis="y", alpha=0.3, zorder=0)
ax3.set_xlabel(f"혼합 상태 {R['n_mixed']}개 · 쌍 {R['n_pairs']}개 · 에피소드 군집 "
               f"{R['n_episodes_with_mixed']}/40 · 주황 띠는 그리퍼 baseline 의 95% CI",
               fontsize=9, color="#4a5568", labelpad=8)

fig.suptitle("C1 탐지기를 보상으로: 감시는 되지만 조종은 되지 않는다", fontsize=15, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = HERE / "results/reward_guidance_main.png"
fig.savefig(out, dpi=150)
print("저장:", out)
