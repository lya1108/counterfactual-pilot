"""Failure-score trajectories against condition onsets, on the environment-step axis.

The detector is the one every earlier result used: a LogReg over suf_mean, trained on the 454 offline rollouts
(8 calls each, 8 seeds). Here it scores the 40 freshly collected rollouts over their whole length.

Time alignment is literal. A call issued at environment step `call_step[k]` contributes its score at exactly
that step and nowhere else -- no interpolation between calls, no shifting a score to the step it "really"
describes. Alarms are therefore only ever evaluated at call steps, which is also all a running system would
have.

A condition onset is not a failure verdict. Episodes are reported in four groups, and the one that matters for
reading false alarms is `onset & final success`: the robot closed on nothing, or tipped something, and still
finished the task. Those steps are NOT counted as false alarms -- the condition really did happen there.

A false alarm is defined per call step, against the horizon: step t is a negative for condition c when c has no
onset in (t, t+H], and an alarm there is false. Episode outcome does not enter that definition.

No single threshold is chosen. These 40 episodes are the only data here, so picking a threshold on them and
reporting the same 40 would be reporting a fit. The whole trade-off is printed instead.

    conda activate fasi
    python xattn_detector/temporal_analysis.py --seeds 8
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

for _f in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",):
    if Path(_f).exists():
        font_manager.fontManager.addfont(_f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

from buffer import Buffer  # noqa: E402
from calibration import sigmoid  # noqa: E402
from online_runner import collect_split, fit_logreg  # noqa: E402

TRACKED = ("C1_grasp_empty_close", "C3_nontarget_contact", "C4_object_tipped")
COMPOSITE = "composite_C1C3C4"


def load_episodes(root: Path):
    eps = []
    for line in (root / "episodes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        z = np.load(root / r["npz"])
        on = {c: r["condition_onset_step"].get(c) for c in TRACKED}
        live = [v for v in on.values() if v is not None]
        on[COMPOSITE] = min(live) if live else None
        eps.append(dict(name=f"{r['suite']}_t{r['task']}_i{r['init']}", suite=r["suite"], task=r["task"],
                        init=r["init"], success=r["success"], n_steps=r["n_steps"],
                        call_step=z["call_step"].astype(int), suf=z["suf_mean"].astype(np.float32),
                        onset=on))
    return eps


def score_all(eps, nets, device):
    for e in eps:
        s = []
        for net, mu, sd in nets:
            with torch.no_grad():
                x = torch.from_numpy((e["suf"] - mu) / sd).to(device).float()
                s.append(sigmoid(net(x).cpu().numpy()))
        e["score"] = np.mean(s, 0)  # mean over the 8 seeds, one value per VLA call
        e["score_sd"] = np.std(s, 0)


def trade_off(eps, cond, taus, H):
    """For each threshold: detection rate and lead time on episodes with an onset, false alarms elsewhere."""
    rows = []
    for tau in taus:
        leads, detected, with_onset = [], 0, 0
        fa, neg = 0, 0
        n_pre = []   # call steps available before each onset -- the chances a random alarm would get
        for e in eps:
            on = e["onset"][cond]
            cs, sc = e["call_step"], e["score"]
            if on is not None:
                with_onset += 1
                n_pre.append(int((cs < on).sum()))
                before = [t for t, v in zip(cs, sc) if t < on and v > tau]
                if before:
                    detected += 1
                    leads.append(on - before[0])
            for t, v in zip(cs, sc):
                if on is not None and t < on <= t + H:
                    continue          # positive window -- an alarm here is correct, not a false alarm
                if on is not None and t >= on:
                    continue          # after onset the question no longer applies
                neg += 1
                fa += v > tau
        fa_rate = fa / max(neg, 1)
        # what an alarm that fires at random with the SAME per-step rate would reach, episode-wise
        null = float(np.mean([1.0 - (1.0 - fa_rate) ** n for n in n_pre])) if n_pre else float("nan")
        rows.append(dict(tau=float(tau), n_onset=with_onset, detected=detected,
                         detect_rate=detected / max(with_onset, 1), detect_rate_null=null,
                         n_pre_median=float(np.median(n_pre)) if n_pre else float("nan"),
                         lead_median=float(np.median(leads)) if leads else float("nan"),
                         lead_p25=float(np.percentile(leads, 25)) if leads else float("nan"),
                         lead_p75=float(np.percentile(leads, 75)) if leads else float("nan"),
                         fa_steps=fa, neg_steps=neg, fa_rate=fa_rate))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", default="data/online")
    ap.add_argument("--data", default="data/cond40")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--calls", type=int, default=8)
    ap.add_argument("--H", type=int, default=10, help="예측 지평 (env step)")
    ap.add_argument("--epochs", type=int, default=400); ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-2); ap.add_argument("--weight-decay", type=float, default=100.0)
    ap.add_argument("--select", default="macro")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    buf = Buffer.load(HERE / args.buffer)
    Xtr, Mtr, ytr, rtr = collect_split(buf, "train", calls=args.calls)
    keep = [i for i, r in enumerate(rtr) if r.get("source") != "online"]
    Xva, Mva, yva, rva = collect_split(buf, "val", calls=args.calls)
    nets = []
    for seed in range(args.seeds):
        net, mu, sd, _ = fit_logreg(Xtr[keep], Mtr[keep], ytr[keep], Xva, Mva, yva, rva, args, seed, device)
        nets.append((net, mu, sd))
    print(f"detector: offline {len(keep)}개로 학습한 LogReg × {args.seeds}시드 (점수는 시드 평균)")

    eps = load_episodes(HERE / args.data)
    score_all(eps, nets, device)
    split_of = {e["name"]: buf.split_for(e["suite"], e["task"], e["init"]) for e in eps}
    print(f"평가 롤아웃 {len(eps)}개   (detector 학습 split 기준: "
          + ", ".join(f"{k}={v}" for k, v in collections.Counter(split_of.values()).items()) + ")\n")

    # ---------------- 1. episode categories
    print("=== 에피소드 분류 (composite C1/C3/C4 기준) ===")
    cat = collections.Counter()
    for e in eps:
        has = e["onset"][COMPOSITE] is not None
        cat[("onset 있음" if has else "onset 없음", "최종 성공" if e["success"] else "최종 실패")] += 1
    for k in sorted(cat):
        label = "  조건 발생 후 회복하여 성공" if k == ("onset 있음", "최종 성공") else ""
        print(f"  {k[0]:10s} × {k[1]:8s} : {cat[k]:3d}{label}")
    print()

    # ---------------- 2. score trajectories, success vs failure
    print("=== 호출 인덱스별 평균 failure score ===")
    print(f"  {'call':>5s} {'성공 n':>7s} {'성공 평균':>9s} {'실패 n':>7s} {'실패 평균':>9s}")
    traj = []
    for k in range(0, 30):
        s = [e["score"][k] for e in eps if e["success"] and len(e["score"]) > k]
        f = [e["score"][k] for e in eps if not e["success"] and len(e["score"]) > k]
        if not s and not f:
            break
        traj.append(dict(call=k, n_succ=len(s), n_fail=len(f),
                         succ=float(np.mean(s)) if s else float("nan"),
                         fail=float(np.mean(f)) if f else float("nan")))
        if k < 12 or k % 4 == 0:
            print(f"  {k:5d} {len(s):7d} {traj[-1]['succ']:9.3f} {len(f):7d} {traj[-1]['fail']:9.3f}")
    print()

    # ---------------- 3-5. trade-off per condition
    allscores = np.concatenate([e["score"] for e in eps])
    taus = np.unique(np.percentile(allscores, np.arange(50, 100, 2.5)).round(4))
    out = {}
    for cond in (*TRACKED, COMPOSITE):
        rows = trade_off(eps, cond, taus, args.H)
        out[cond] = rows
        n_on = rows[0]["n_onset"]
        print(f"=== {cond}   onset 보유 에피소드 {n_on}/{len(eps)} ===")
        print(f"  {'threshold':>9s} {'탐지율':>7s} {'무작위기준':>9s} {'차이':>7s} {'lead 중앙값':>11s} "
              f"{'lead p25-p75':>13s} {'오경보율':>8s}")
        for r in rows:
            lm = f"{r['lead_median']:.0f}" if r["lead_median"] == r["lead_median"] else "-"
            lq = (f"{r['lead_p25']:.0f}–{r['lead_p75']:.0f}"
                  if r["lead_p25"] == r["lead_p25"] else "-")
            print(f"  {r['tau']:9.3f} {r['detect_rate']:6.0%} {r['detect_rate_null']:8.0%} "
                  f"{r['detect_rate'] - r['detect_rate_null']:+6.0%} {lm:>11s} {lq:>13s} {r['fa_rate']:7.1%}")
        print()

    print("  threshold 는 고르지 않았습니다. 이 40개에서 고른 값으로 같은 40개를 평가하면 적합한 값을")
    print("  성능으로 보고하는 셈이 되므로, 선택이 필요하면 별도의 에피소드 집합이 있어야 합니다.")

    # ---------------- figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax = axes[0][0]
    x = [t["call"] for t in traj]
    ax.plot(x, [t["succ"] for t in traj], color="#2E7D32", lw=2, label="최종 성공")
    ax.plot(x, [t["fail"] for t in traj], color="#A8521B", lw=2, label="최종 실패")
    ax.set_xlabel("call index"); ax.set_ylabel("failure score (8시드 평균)")
    ax.set_title("성공/실패 롤아웃의 score 추이"); ax.legend(frameon=False); ax.grid(alpha=0.3)

    ax = axes[0][1]
    for cond, color in zip((*TRACKED, COMPOSITE), ("#0C6E78", "#A8521B", "#6A4C93", "#111111")):
        r = out[cond]
        ax.plot([q["fa_rate"] for q in r], [q["detect_rate"] for q in r], "o-", ms=3,
                color=color, lw=1.6, label=cond.split("_")[0])
        ax.plot([q["fa_rate"] for q in r], [q["detect_rate_null"] for q in r], ls=":", lw=1.2,
                color=color, alpha=0.7)
    ax.set_xlabel(f"오경보율 (H={args.H} env step 안에 onset 없는 호출)")
    ax.set_ylabel("onset 이전 탐지율")
    ax.set_title("threshold trade-off  (점선 = 같은 발화율의 무작위 경보)")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3)

    show = [e for e in eps if e["onset"][COMPOSITE] is not None][:2]
    for ax, e in zip((axes[1][0], axes[1][1]), show):
        ax.plot(e["call_step"], e["score"], color="#0C6E78", lw=1.8, marker="o", ms=3, label="failure score")
        ax.fill_between(e["call_step"], e["score"] - e["score_sd"], e["score"] + e["score_sd"],
                        color="#0C6E78", alpha=0.15, lw=0)
        for cond, color in zip(TRACKED, ("#A8521B", "#6A4C93", "#B08900")):
            v = e["onset"][cond]
            if v is not None:
                ax.axvline(v, color=color, ls="--", lw=1.6)
                ax.text(v, 0.97, " " + cond.split("_")[0], color=color, fontsize=8,
                        rotation=90, va="top", ha="left")
        ax.set_ylim(0, 1); ax.set_xlabel("env step"); ax.set_ylabel("failure score")
        ax.set_title(f"{e['name']}  ({'성공' if e['success'] else '실패'})", fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("Failure score 와 condition onset — score 는 해당 호출이 발행된 env step 에만 표시 (보간 없음)",
                 fontsize=12)
    png = HERE / "results" / f"temporal_{args.tag}.png"
    fig.savefig(png, dpi=150)

    js = HERE / "results" / f"temporal_{args.tag}.json"
    js.write_text(json.dumps(dict(H=args.H, seeds=args.seeds, n_episodes=len(eps),
                                  categories={f"{k[0]}|{k[1]}": v for k, v in cat.items()},
                                  trajectory=traj, trade_off=out,
                                  episodes=[dict(name=e["name"], success=e["success"],
                                                 split=split_of[e["name"]], onset=e["onset"],
                                                 call_step=e["call_step"].tolist(),
                                                 score=[round(float(v), 4) for v in e["score"]])
                                            for e in eps]),
                             indent=2, ensure_ascii=False, default=float))
    print(f"\n-> {png}\n-> {js}")


if __name__ == "__main__":
    main()
