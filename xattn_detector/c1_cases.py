"""Rollout-level look at the C1 detector, on out-of-fold predictions only.

Every score here comes from a model whose training folds excluded that rollout's task. The 20 repeated
task-group splits each produce one out-of-fold prediction per sample; the curve is their mean. No rollout is
ever scored by a model that saw its task.

The threshold tau is the 90th percentile of NEGATIVE calls (calls with no C1 onset within H). It exists only to
sort episodes into TP / FP / FN / TN for this figure. It is NOT an operating point, was not selected to
optimise anything, and no performance number is reported at it.

Cases are picked by a rule stated before looking: within each category, the two episodes whose episode score is
closest to that category's MEDIAN -- not the highest, not the clearest. The category sizes and the full score
distributions are printed alongside, so the chosen cases can be seen in context.

Frames come from what was already stored: collect_conditions.py saved agentview frames within +-12 environment
steps of an onset, at a stride of 3. Nothing earlier than that exists yet and nothing is replayed here.

    conda activate fasi
    python xattn_detector/c1_cases.py --data data/cond200 --repeats 20
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "safe_detector"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image  # noqa: E402

for _f in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",):
    if Path(_f).exists():
        font_manager.fontManager.addfont(_f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

from c1_depth import build, target_horizon  # noqa: E402
from calibration import sigmoid  # noqa: E402
from prospective_c1 import fit_eval  # noqa: E402

BUCKETS = ((30, 40), (20, 30), (10, 20), (0, 10))


def out_of_fold_scores(rows, y, groups, folds, repeats):
    """Mean out-of-fold score per sample. A sample is only ever scored by a fold that held out its task."""
    X = np.stack([r["suf_mean"] for r in rows])
    acc = np.zeros(len(rows))
    cnt = np.zeros(len(rows))
    for rep in range(repeats):
        rng = np.random.default_rng(100 + rep)
        fold_of = {g: i % folds for i, g in enumerate(rng.permutation(groups))}
        for fold in range(folds):
            te = np.array([fold_of[r["group"]] == fold for r in rows])
            if te.sum() == 0 or y[~te].sum() < 2 or (1 - y[~te]).sum() < 2:
                continue
            acc[te] += sigmoid(fit_eval(X[~te], y[~te], X[te], seed=rep))
            cnt[te] += 1
    return acc / np.maximum(cnt, 1), cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cond200")
    ap.add_argument("--H", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--pct", type=float, default=90.0, help="tau = 음성 호출의 이 백분위")
    ap.add_argument("--per-cat", type=int, default=2)
    ap.add_argument("--tag", default="n200")
    args = ap.parse_args()

    root = HERE / args.data
    rows_all = build(root, hist=3, want_tokens=False)
    rows, y = target_horizon(rows_all, args.H, 0)
    groups = sorted({r["group"] for r in rows})
    print(f"샘플 {len(rows)}  양성 {int(y.sum())}  task {len(groups)}  "
          f"— out-of-fold 점수 계산 ({args.repeats}회 분할 × {args.folds}-fold)", flush=True)
    score, cnt = out_of_fold_scores(rows, y, groups, args.folds, args.repeats)
    print(f"샘플당 out-of-fold 예측 횟수: 중앙값 {np.median(cnt):.0f}, 최소 {cnt.min():.0f}\n")

    tau = float(np.percentile(score[y == 0], args.pct))
    print(f"τ = 음성 호출 {args.pct:.0f} 백분위 = {tau:.4f}")
    print("  ※ TP/FP/FN/TN 분류용일 뿐 detector 의 operating threshold 가 아니며,")
    print("     이 값에서의 성능 수치는 보고하지 않습니다.\n")

    # ---- episode level
    by_ep = collections.defaultdict(list)
    for r, s in zip(rows, score):
        by_ep[(r["suite"], r["task"], r["init"])].append((r, s))
    index = {}
    for line in (root / "episodes.jsonl").read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            index[(rec["suite"], rec["task"], rec["init"])] = rec

    eps = []
    for key, items in by_ep.items():
        items.sort(key=lambda x: x[0]["call_step"])
        rec = index[key]
        onset = rec["condition_onset_step"].get("C1_grasp_empty_close")
        steps = np.array([r["call_step"] for r, _ in items])
        sc = np.array([s for _, s in items])
        if onset is not None:
            m = (steps < onset) & (steps >= onset - args.H)
            ep_score = float(sc[m].max()) if m.any() else float("nan")
        else:
            ep_score = float(sc.max())
        eps.append(dict(key=key, name=f"{key[0]}_t{key[1]}_i{key[2]}", onset=onset,
                        steps=steps, score=sc, ep_score=ep_score,
                        success=rec["success"], n_steps=rec["n_steps"]))
    eps = [e for e in eps if e["ep_score"] == e["ep_score"]]

    cats = {"TP": [], "FP": [], "FN": [], "TN": []}
    for e in eps:
        has = e["onset"] is not None
        hit = e["ep_score"] > tau
        cats["TP" if (has and hit) else "FN" if has else "FP" if hit else "TN"].append(e)

    print("=== 범주별 에피소드 수 (C1 onset 유무 × τ 초과 여부) ===")
    print(f"  {'범주':6s} {'정의':34s} {'개수':>5s} {'ep score 중앙값':>15s}")
    defs = {"TP": "onset 있음 · τ 초과", "FP": "onset 없음 · τ 초과",
            "FN": "onset 있음 · τ 미만", "TN": "onset 없음 · τ 미만"}
    for c in ("TP", "FP", "FN", "TN"):
        v = [e["ep_score"] for e in cats[c]]
        print(f"  {c:6s} {defs[c]:34s} {len(v):5d} {np.median(v) if v else float('nan'):15.4f}")
    print()

    chosen = {}
    for c in ("TP", "FP", "FN", "TN"):
        v = cats[c]
        if not v:
            chosen[c] = []
            continue
        med = float(np.median([e["ep_score"] for e in v]))
        chosen[c] = sorted(v, key=lambda e: abs(e["ep_score"] - med))[: args.per_cat]
        print(f"  {c} 중앙값 {med:.4f} 에 가장 가까운 {len(chosen[c])}개: "
              + ", ".join(f"{e['name']}({e['ep_score']:.3f})" for e in chosen[c]))
    print()

    # ---- bucket table for onset episodes
    print("=== onset 까지 남은 거리별 out-of-fold score (onset 있는 에피소드) ===")
    print(f"  {'episode':26s} {'onset':>6s} " + " ".join(f"{f'{a}-{b}':>9s}" for a, b in BUCKETS[::-1]))
    bucket_rows = []
    for c in ("TP", "FN"):
        for e in chosen[c]:
            cells = []
            for a, b in BUCKETS[::-1]:
                m = (e["steps"] < e["onset"] - a) & (e["steps"] >= e["onset"] - b)
                cells.append(float(e["score"][m].mean()) if m.any() else float("nan"))
            bucket_rows.append(dict(name=e["name"], cat=c, onset=e["onset"], buckets=cells))
            print(f"  [{c}] {e['name']:21s} {e['onset']:6d} "
                  + " ".join(f"{v:9.3f}" if v == v else f"{'-':>9s}" for v in cells))
    print()

    # ---- figures
    res = HERE / "results"
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    data = [[e["ep_score"] for e in cats[c]] for c in ("TP", "FN", "FP", "TN")]
    bp = ax.boxplot(data, tick_labels=[f"{c}\n(n={len(cats[c])})" for c in ("TP", "FN", "FP", "TN")],
                    widths=0.55, showfliers=False, patch_artist=True)
    for p, col in zip(bp["boxes"], ("#0C6E78", "#A8521B", "#6A4C93", "#5A6A74")):
        p.set_facecolor(col); p.set_alpha(0.25); p.set_edgecolor(col)
    for i, c in enumerate(("TP", "FN", "FP", "TN"), start=1):
        v = [e["ep_score"] for e in cats[c]]
        ax.scatter(np.random.default_rng(0).normal(i, 0.05, len(v)), v, s=8, color="#5A6A74", alpha=0.35)
        for e in chosen[c]:
            ax.scatter([i], [e["ep_score"]], s=90, marker="*", color="#B00020", zorder=5)
    ax.axhline(tau, color="#B00020", ls="--", lw=1.2)
    ax.text(4.45, tau, f" τ={tau:.3f}", color="#B00020", fontsize=9, va="center")
    ax.set_ylabel("episode score (out-of-fold)")
    ax.set_title("범주별 episode score 분포 — 별표는 선정된 중앙값 사례")
    fig.savefig(res / f"c1_cases_overview_{args.tag}.png", dpi=150)

    frames = sorted(glob.glob(str(root / "frames" / "*__C1_on*__s*.png")))
    by_name = collections.defaultdict(list)
    for f in frames:
        by_name[Path(f).name.split("__")[0]].append(f)

    for c in ("TP", "FP", "FN", "TN"):
        for e in chosen[c]:
            fs = sorted(by_name.get(e["name"], []))
            nrow = 2 if fs else 1
            fig = plt.figure(figsize=(11, 5.2 if fs else 3.4), constrained_layout=True)
            gs = fig.add_gridspec(nrow, 1, height_ratios=[2, 1] if fs else [1])
            ax = fig.add_subplot(gs[0])
            ax.plot(e["steps"], e["score"], color="#0C6E78", lw=1.8, marker="o", ms=4,
                    label="out-of-fold score")
            for s in e["steps"]:
                ax.axvline(s, color="#5A6A74", lw=0.4, alpha=0.3)
            ax.axhline(tau, color="#B00020", ls="--", lw=1.2, label=f"τ={tau:.3f} (분류용)")
            if e["onset"] is not None:
                ax.axvline(e["onset"], color="#A8521B", lw=2)
                ax.text(e["onset"], 1.0, " C1 onset", color="#A8521B", fontsize=9, va="top")
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("env step   (세로 옅은 선 = VLA 호출 시점)")
            ax.set_ylabel("score")
            ax.set_title(f"[{c}] {e['name']}   최종 {'성공' if e['success'] else '실패'}   "
                         f"{e['n_steps']} step   episode score {e['ep_score']:.3f}", fontsize=11)
            ax.legend(frameon=False, fontsize=8, loc="upper left")
            ax.grid(alpha=0.25)
            if fs:
                pick = [fs[i] for i in np.linspace(0, len(fs) - 1, min(8, len(fs))).astype(int)]
                axf = fig.add_subplot(gs[1]); axf.axis("off")
                ims = [np.asarray(Image.open(p).convert("RGB").resize((128, 128))) for p in pick]
                axf.imshow(np.concatenate(ims, 1))
                ticks = [64 + 128 * i for i in range(len(pick))]
                labs = [re.search(r"__s(\d+)", p).group(1).lstrip("0") for p in pick]
                axf.set_xticks(ticks); axf.set_xticklabels(labs, fontsize=8)
                axf.tick_params(axis="x", length=0, pad=2)
                axf.set_title("저장된 프레임 (onset ±12 step, stride 3) — env step", fontsize=9)
            fig.savefig(res / f"c1_case_{c}_{e['name']}.png", dpi=140)
            plt.close(fig)

    (res / f"c1_cases_{args.tag}.json").write_text(json.dumps(
        dict(args=vars(args), tau=tau,
             counts={c: len(cats[c]) for c in cats},
             medians={c: float(np.median([e["ep_score"] for e in cats[c]])) if cats[c] else None
                      for c in cats},
             chosen={c: [dict(name=e["name"], ep_score=e["ep_score"], onset=e["onset"],
                              success=e["success"], n_steps=e["n_steps"],
                              steps=e["steps"].tolist(),
                              score=[round(float(v), 4) for v in e["score"]])
                         for e in chosen[c]] for c in cats},
             buckets=bucket_rows), indent=2, ensure_ascii=False, default=float))
    print(f"-> {res / f'c1_cases_overview_{args.tag}.png'}")
    print(f"-> {res}/c1_case_*.png   ({sum(len(v) for v in chosen.values())}개)")
    print(f"-> {res / f'c1_cases_{args.tag}.json'}")


if __name__ == "__main__":
    main()
