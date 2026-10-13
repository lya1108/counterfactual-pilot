"""Per-task breakdown of C1 prospective prediction, on out-of-fold predictions.

Setting is the canonical one and nothing about it changes here: calls where C1 is not already active, positive
if the episode-global onset falls in (t, t+10], guard 0. Scores are out-of-fold -- each sample is scored only by
folds that held its task out -- averaged over the repeated task-group splits.

A per-task AUROC rests on that task's own positives. Where those are few the number moves a lot for reasons
that have nothing to do with the detector, so the positive count is printed next to every row and tasks under
`--min-pos` are marked; the summary counts them separately rather than folding them in silently.

    conda activate fasi
    python xattn_detector/c1_pertask.py --data data/cond200 --repeats 20
"""
from __future__ import annotations

import argparse, collections, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "safe_detector"))

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

from c1_depth import build, target_horizon
from c1_horizon import average_precision
from calibration import sigmoid
from prospective_c1 import fit_eval
from train_detector import auroc

FEATURES = ("suf_mean", "vlm_img", "state_hist", "state", "call_index")


def oof(rows, y, groups, feat, folds, repeats):
    X = np.stack([np.asarray(r[feat]).ravel() for r in rows])
    acc, cnt = np.zeros(len(rows)), np.zeros(len(rows))
    for rep in range(repeats):
        rng = np.random.default_rng(100 + rep)
        fold_of = {g: i % folds for i, g in enumerate(rng.permutation(groups))}
        for f in range(folds):
            te = np.array([fold_of[r["group"]] == f for r in rows])
            if te.sum() == 0 or y[~te].sum() < 2 or (1 - y[~te]).sum() < 2: continue
            acc[te] += sigmoid(fit_eval(X[~te], y[~te], X[te], seed=rep)); cnt[te] += 1
    return acc / np.maximum(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cond200")
    ap.add_argument("--H", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--min-pos", type=int, default=10, help="이보다 적으면 불안정으로 표시")
    ap.add_argument("--tag", default="n200")
    args = ap.parse_args()

    rows_all = build(HERE / args.data, hist=3, want_tokens=True, hist_stride=5)
    rows, y = target_horizon(rows_all, args.H, 0)
    groups = sorted({r["group"] for r in rows}); gl = np.array([r["group"] for r in rows])
    print(f"샘플 {len(rows)}  양성 {int(y.sum())}  task {len(groups)}  H={args.H}, guard=0  "
          f"— out-of-fold ({args.repeats}회 × {args.folds}-fold)", flush=True)
    S = {f: oof(rows, y, groups, f, args.folds, args.repeats) for f in FEATURES}

    per = {}
    for g in groups:
        m = gl == g
        yy = y[m]
        d = dict(n=int(m.sum()), pos=int(yy.sum()), neg=int((1 - yy).sum()),
                 rate=float(yy.mean()), usable=bool(0 < yy.sum() < len(yy)))
        for f in FEATURES:
            d[f] = (auroc(S[f][m], yy), average_precision(S[f][m], yy)) if d["usable"] else (float("nan"),) * 2
        per[g] = d

    print(f"\n{'task':22s} {'n':>5s} {'pos':>4s} {'neg':>5s} {'rate':>6s} "
          f"{'suf AUROC':>10s} {'suf AUPRC':>10s} {'vlm_img':>8s} {'st_hist':>8s} {'state':>7s} {'call':>6s}")
    for g in sorted(groups, key=lambda g: -(per[g]["suf_mean"][0] if per[g]["usable"] else -1)):
        d = per[g]
        flag = "  *" if d["pos"] < args.min_pos else ""
        if not d["usable"]:
            print(f"{g:22s} {d['n']:5d} {d['pos']:4d} {d['neg']:5d} {d['rate']:6.1%}   단일 클래스 — 계산 불가")
            continue
        print(f"{g:22s} {d['n']:5d} {d['pos']:4d} {d['neg']:5d} {d['rate']:6.1%} "
              f"{d['suf_mean'][0]:10.3f} {d['suf_mean'][1]:10.3f} {d['vlm_img'][0]:8.3f} "
              f"{d['state_hist'][0]:8.3f} {d['state'][0]:7.3f} {d['call_index'][0]:6.3f}{flag}")
    print(f"  * = 양성 {args.min_pos}개 미만 — AUROC/AUPRC 가 소수 샘플에 좌우되므로 깊게 해석하지 않음")

    ok = [g for g in groups if per[g]["usable"]]
    a = np.array([per[g]["suf_mean"][0] for g in ok])
    print(f"\n=== 집계 (계산 가능한 task {len(ok)}/{len(groups)}) ===")
    for lab, m in ((">0.8", a > 0.8), ("0.7–0.8", (a > 0.7) & (a <= 0.8)),
                   ("0.6–0.7", (a > 0.6) & (a <= 0.7)), ("<=0.6", a <= 0.6)):
        names = [g for g, k in zip(ok, m) if k]
        print(f"  suf_mean AUROC {lab:8s} {int(m.sum()):2d} task" + (f"   {', '.join(names)}" if m.sum() else ""))
    for other in ("vlm_img", "state_hist"):
        w = sum(per[g]["suf_mean"][0] > per[g][other][0] for g in ok)
        dif = np.array([per[g]["suf_mean"][0] - per[g][other][0] for g in ok])
        print(f"  suf_mean > {other:11s} {w:2d}/{len(ok)} task   차이 중앙값 {np.median(dif):+.3f} "
              f"(최소 {dif.min():+.3f}, 최대 {dif.max():+.3f})")
    few = [g for g in ok if per[g]["pos"] < args.min_pos]
    print(f"  양성 {args.min_pos}개 미만인 task: {len(few)}개" + (f"  ({', '.join(few)})" if few else ""))

    order = sorted(ok, key=lambda g: per[g]["suf_mean"][0])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), sharey=True, constrained_layout=True)
    yy = np.arange(len(order))
    ax = axes[0]
    ax.barh(yy, [per[g]["suf_mean"][0] for g in order], color="#0C6E78", height=0.6)
    ax.scatter([per[g]["vlm_img"][0] for g in order], yy, s=26, color="#A8521B", label="vlm_img", zorder=3)
    ax.scatter([per[g]["state_hist"][0] for g in order], yy, s=26, color="#6A4C93", marker="s",
               label="state_hist", zorder=3)
    ax.axvline(0.5, color="#5A6A74", ls="--", lw=1)
    ax.set_yticks(yy); ax.set_yticklabels([f"{g}  (p={per[g]['pos']})" for g in order], fontsize=8)
    ax.set_xlim(0.3, 1.0); ax.set_xlabel("AUROC"); ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("task별 AUROC — 막대 suf_mean", fontsize=11)
    ax = axes[1]
    ax.barh(yy, [per[g]["suf_mean"][1] for g in order], color="#0C6E78", height=0.6)
    ax.scatter([per[g]["rate"] for g in order], yy, s=26, color="#B00020", marker="|",
               label="양성률 (AUPRC 하한)", zorder=3)
    ax.set_xlabel("AUPRC"); ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("task별 AUPRC — 빨간 선은 그 task 의 양성률", fontsize=11)
    fig.suptitle(f"C1 prospective (H={args.H}, guard=0, out-of-fold)   괄호 안 p = 양성 수", fontsize=12)
    png = HERE / "results" / f"c1_pertask_{args.tag}.png"
    fig.savefig(png, dpi=140)

    (HERE / "results" / f"c1_pertask_{args.tag}.json").write_text(json.dumps(
        dict(args=vars(args), per_task={g: {k: (list(v) if isinstance(v, tuple) else v)
                                            for k, v in per[g].items()} for g in groups}),
        indent=2, ensure_ascii=False, default=float))
    print(f"\n-> {png}")


if __name__ == "__main__":
    main()
