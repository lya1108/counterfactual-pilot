"""Is prefix8 simply too early, or does suf_mean not carry within-task failure information at all?

The learning curve left task-macro AUROC at 0.478 +- 0.036 -- chance. Two explanations survive that number:
the first 8 VLA calls (80 environment steps) may be before anything has gone wrong, or the mean-pooled
action-expert vector may not separate failing from succeeding rollouts of the same task at any moment. This
script separates them by scoring every call of a full rollout.

Setup, chosen so nothing is scored by a detector that trained on it:
  train      the 454 offline rollouts (inits 0-19), 8 calls each, 8 seeds
  evaluate   the 240 online rollouts (inits 20-29), which the detector never saw, over their whole length

Both sets come from the same 24 tasks, so this is the easier "unseen rollouts of seen tasks" question. If the
curve is flat even here, timing is not the problem.

Survival bias, which must be read alongside every late point: every failure runs to the timeout (n_calls >= 28)
while successes stop when solved (median 18). From call 8 to 27 the failure set is therefore fixed at 146 and
only successes drain away, so the successes still alive late are the slow, struggling ones -- the hardest to
tell from failures. A rise in AUROC at late calls is not by itself evidence of a growing failure signal, and a
fall is not by itself evidence of a shrinking one. The counts are printed and plotted next to the curve for
that reason, and points with fewer than `--min-tasks` evaluable tasks are drawn faded.

A second view aligns episodes by calls REMAINING. For failures that counts down to the timeout, not to the
moment the rollout went wrong, which LIBERO does not label -- read it as "N calls before the attempt was
abandoned".

    conda activate fasi
    python xattn_detector/score_timeline.py --seeds 8
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "safe_detector"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# DejaVu has no Hangul, so every Korean label would render as boxes
for _f in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",):
    if Path(_f).exists():
        font_manager.fontManager.addfont(_f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

from buffer import Buffer  # noqa: E402
from calibration import sigmoid  # noqa: E402
from online_runner import LogRegDetector, collect_split, episode_probs, fit_logreg, macro_auroc  # noqa: E402
from train_detector import auroc  # noqa: E402


def per_call_scores(net, mu, sd, feats, device):
    with torch.no_grad():
        x = torch.from_numpy((feats - mu) / sd).to(device).float()
        return sigmoid(net(x).cpu().numpy())


def stat_at(scores, ys, keys, min_per_task=1):
    """macro (mean of per-task AUROCs), pooled AUROC, and how much data each rested on."""
    by = collections.defaultdict(lambda: ([], []))
    for s, y, k in zip(scores, ys, keys):
        by[k][0].append(s)
        by[k][1].append(y)
    per = {}
    for k, (ss, yy) in by.items():
        yy = np.array(yy)
        if yy.sum() >= min_per_task and (1 - yy).sum() >= min_per_task:
            per[k] = auroc(np.array(ss), yy)
    ys = np.array(ys)
    pooled = auroc(np.array(scores), ys) if 0 < ys.sum() < len(ys) else float("nan")
    return (float(np.mean(list(per.values()))) if per else float("nan")), pooled, len(per), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", default="data/online")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--calls", type=int, default=8, help="학습에 쓰는 앞쪽 호출 수")
    ap.add_argument("--max-call", type=int, default=40)
    ap.add_argument("--min-tasks", type=int, default=5, help="이보다 적으면 신뢰할 수 없는 점으로 표시")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=100.0)
    ap.add_argument("--select", default="macro")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    buf = Buffer.load(HERE / args.buffer)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- train on the offline rollouts only
    Xtr, Mtr, ytr, rtr = collect_split(buf, "train", calls=args.calls)
    keep = [i for i, r in enumerate(rtr) if r.get("source") != "online"]
    Xva, Mva, yva, rva = collect_split(buf, "val", calls=args.calls)
    print(f"학습: offline {len(keep)}개 (앞 {args.calls}호출)   val {len(yva)}")

    # ---- evaluation set: the online rollouts, whole length
    online = [r for r in buf.records if r.get("source") == "online"]
    feats = {r["rollout_id"]: buf.fetch(r, ("suf_mean",))["suf_mean"].astype(np.float32) for r in online}
    ys = {r["rollout_id"]: r["y"] for r in online}
    keys = {r["rollout_id"]: f"{r['suite']}/t{r['task']}" for r in online}
    ncalls = {r["rollout_id"]: r["n_calls"] for r in online}
    print(f"평가: online {len(online)}개 (성공 {sum(1 for r in online if r['y'] == 0)} / "
          f"실패 {sum(r['y'] for r in online)}), task {len(set(keys.values()))}개\n")

    nets = []
    for seed in range(args.seeds):
        net, mu, sd, _ = fit_logreg(Xtr[keep], Mtr[keep], ytr[keep], Xva, Mva, yva, rva, args, seed, device)
        nets.append((net, mu, sd))

    # ---- by call index
    rows_call = []
    print(f"  {'call':>5s} {'생존':>5s} {'성공':>5s} {'실패':>5s} {'task':>5s} {'macro':>14s} {'pooled':>14s}")
    for t in range(args.max_call + 1):
        alive = [rid for rid in feats if ncalls[rid] > t]
        if not alive:
            break
        yy = [ys[rid] for rid in alive]
        kk = [keys[rid] for rid in alive]
        F = np.stack([feats[rid][t] for rid in alive])
        ms, ps, nts = [], [], []
        for net, mu, sd in nets:
            sc = per_call_scores(net, mu, sd, F, device)
            m, p, nt, _ = stat_at(sc, yy, kk)
            ms.append(m); ps.append(p); nts.append(nt)
        row = dict(call=t, alive=len(alive), succ=int(sum(1 for v in yy if v == 0)), fail=int(sum(yy)),
                   tasks=int(np.mean(nts)),
                   macro=[float(np.nanmean(ms)), float(np.nanstd(ms, ddof=1))],
                   pooled=[float(np.nanmean(ps)), float(np.nanstd(ps, ddof=1))])
        rows_call.append(row)
        if t % 2 == 0 or t < 10:
            flag = "" if row["tasks"] >= args.min_tasks else "  (task 부족)"
            print(f"  {t:5d} {row['alive']:5d} {row['succ']:5d} {row['fail']:5d} {row['tasks']:5d} "
                  f"{row['macro'][0]:7.3f}±{row['macro'][1]:.3f} {row['pooled'][0]:7.3f}±{row['pooled'][1]:.3f}{flag}")

    # ---- by calls remaining until the episode ended
    rows_rem = []
    for r in range(0, args.max_call + 1):
        alive = [rid for rid in feats if ncalls[rid] > r]
        if not alive:
            break
        yy = [ys[rid] for rid in alive]
        kk = [keys[rid] for rid in alive]
        F = np.stack([feats[rid][ncalls[rid] - 1 - r] for rid in alive])
        ms, ps, nts = [], [], []
        for net, mu, sd in nets:
            sc = per_call_scores(net, mu, sd, F, device)
            m, p, nt, _ = stat_at(sc, yy, kk)
            ms.append(m); ps.append(p); nts.append(nt)
        rows_rem.append(dict(remaining=r, alive=len(alive),
                             succ=int(sum(1 for v in yy if v == 0)), fail=int(sum(yy)),
                             tasks=int(np.mean(nts)),
                             macro=[float(np.nanmean(ms)), float(np.nanstd(ms, ddof=1))],
                             pooled=[float(np.nanmean(ps)), float(np.nanstd(ps, ddof=1))]))

    # ---- the same statistic on tasks the detector never trained on. Those episodes are the offline test
    # rollouts, which exist only for the first `--calls` calls, so the line stops there -- but that is exactly
    # the range where the seen-task curve peaks, so the comparison lands where it matters.
    Xte, Mte, yte, rte = collect_split(buf, "test", calls=args.calls)
    rows_unseen = []
    for t in range(Xte.shape[1]):
        yy = list(yte)
        kk = [f"{r['suite']}/t{r['task']}" for r in rte]
        F = Xte[:, t]
        ms, ps, nts = [], [], []
        for net, mu, sd in nets:
            sc = per_call_scores(net, mu, sd, F, device)
            m, pl, nt, _ = stat_at(sc, yy, kk)
            ms.append(m); ps.append(pl); nts.append(nt)
        rows_unseen.append(dict(call=t, alive=len(yy), succ=int(sum(1 for v in yy if v == 0)),
                                fail=int(sum(yy)), tasks=int(np.mean(nts)),
                                macro=[float(np.nanmean(ms)), float(np.nanstd(ms, ddof=1))],
                                pooled=[float(np.nanmean(ps)), float(np.nanstd(ps, ddof=1))]))
    print(f"\n  미학습 task(test {len(rte)}개, 앞 {args.calls}호출만 존재) macro: "
          + "  ".join(f"c{r['call']}={r['macro'][0]:.3f}" for r in rows_unseen))

    # ---- figure
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6), height_ratios=[2.2, 1],
                             sharex="col", constrained_layout=True)
    for col, (rows, xk, title, xlabel) in enumerate([
            (rows_call, "call", "VLA 호출 순서대로", "call index (1 call = 10 env steps)"),
            (rows_rem, "remaining", "에피소드 종료까지 남은 호출", "calls remaining (0 = 마지막 호출)")]):
        x = np.array([r[xk] for r in rows])
        ok = np.array([r["tasks"] >= args.min_tasks for r in rows])
        ax = axes[0][col]
        for name, color in (("macro", "#0C6E78"), ("pooled", "#A8521B")):
            m = np.array([r[name][0] for r in rows])
            s = np.array([r[name][1] for r in rows])
            ax.plot(x[ok], m[ok], color=color, lw=2, label=name)
            ax.fill_between(x[ok], (m - s)[ok], (m + s)[ok], color=color, alpha=0.18, lw=0)
            ax.plot(x, m, color=color, lw=1, alpha=0.3, ls=":")
        if col == 0:  # unseen tasks, same statistic, over the calls where their data exists
            xu = np.array([r["call"] for r in rows_unseen])
            mu_ = np.array([r["macro"][0] for r in rows_unseen])
            su_ = np.array([r["macro"][1] for r in rows_unseen])
            ax.plot(xu, mu_, color="#6A4C93", lw=2, ls="--", label="macro (미학습 task)")
            ax.fill_between(xu, mu_ - su_, mu_ + su_, color="#6A4C93", alpha=0.15, lw=0)
        ax.axhline(0.5, color="#5A6A74", lw=1, ls="--")
        ax.text(x.max(), 0.503, " 우연 0.50", color="#5A6A74", fontsize=8, va="bottom", ha="right")
        ax.set_ylim(0.2, 0.9)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("AUROC (8 seeds, ±sd)")
        if col == 0:
            ax.legend(frameon=False, fontsize=9, loc="upper left")
        ax.grid(alpha=0.25, lw=0.6)

        ax2 = axes[1][col]
        ax2.plot(x, [r["succ"] for r in rows], color="#2E7D32", lw=1.8, label="성공 생존")
        ax2.plot(x, [r["fail"] for r in rows], color="#A8521B", lw=1.8, label="실패 생존")
        ax2.plot(x, [r["tasks"] for r in rows], color="#5A6A74", lw=1.4, ls="--", label="평가 가능 task")
        ax2.axhline(args.min_tasks, color="#5A6A74", lw=0.8, ls=":")
        ax2.set_yscale("symlog", linthresh=10)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("개수")
        ax2.grid(alpha=0.25, lw=0.6)
        if col == 0:
            ax2.legend(frameon=False, fontsize=8, ncol=3)
    fig.suptitle("호출 시점별 task-macro AUROC   (왼쪽만 해석 가능 — 오른쪽 정렬은 길이 정보가 되살아남)",
                 fontsize=12)
    out_png = HERE / "results" / f"score_timeline_{args.tag}.png"
    fig.savefig(out_png, dpi=150)
    print(f"\n-> {out_png}")

    out = HERE / "results" / f"score_timeline_{args.tag}.json"
    out.write_text(json.dumps(dict(seeds=args.seeds, train_calls=args.calls, n_train=len(keep),
                                   n_eval=len(online), by_call=rows_call, by_remaining=rows_rem,
                                   by_call_unseen_tasks=rows_unseen),
                              indent=2, ensure_ascii=False))
    print(f"-> {out}")

    # ---- plain-language read of the two hypotheses
    early = [r for r in rows_call if r["call"] < args.calls and r["tasks"] >= args.min_tasks]
    late = [r for r in rows_call if r["call"] >= args.calls and r["tasks"] >= args.min_tasks]
    if early and late:
        e = float(np.mean([r["macro"][0] for r in early]))
        l = float(np.mean([r["macro"][0] for r in late]))
        print(f"\n  macro 평균:  학습 구간(call < {args.calls}) {e:.3f}   "
              f"그 이후(신뢰 가능 구간만) {l:.3f}   차이 {l - e:+.3f}")
        print("  해석 주의: 늦은 호출일수록 살아남은 성공 에피소드가 느린 축에 치우치므로,")
        print("            위 차이는 실패 신호의 변화와 표본 구성 변화가 섞인 값이다.")


if __name__ == "__main__":
    main()
