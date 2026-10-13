"""Read the branch runs and answer the one question: does the head rank actions, or only states?

Every number here is paired. A call contributes to the within-state statistics only if its N candidates did
not all share one outcome, because a call where every candidate causes C1 (or none does) carries no ranking
information no matter how confident the head is -- and most calls are of that kind, which is the point.

The counterfactual is explicit: "oracle" is what an infallible selector would have achieved from these very
candidates, "random" is what picking one uniformly would have, and "head" is what the frozen detector picks.
Reading the head's number without the other two would make any value look like progress.

    python reward_guidance/analyze_sensitivity.py --data data/sens
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "safe_detector"))

from train_detector import auroc  # noqa: E402


def rel(p):
    p = Path(p)
    return p if p.is_absolute() else HERE / p


def load(root: Path):
    eps = []
    for line in (root / "episodes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        z = np.load(root / r["npz"])
        if z["scores"].shape[0] == 0:
            continue
        eps.append(dict(**r, scores=z["scores"].astype(np.float64), labels=z["labels"].astype(int),
                        d_onset=z["d_onset"].astype(int), call_step=z["call_step"],
                        act_dist=z["act_dist"], dup=z["dup_label"].astype(int),
                        open_min=z["open_min"], replay_err=z["replay_err"],
                        chunks=z["chunks"].astype(np.float32), phi=z["phi"].astype(np.float32)))
    return eps


def boot_ci(vals, w=None, n=2000, seed=0):
    """Episode-level bootstrap. `vals` is one value per episode; `w` the count it summarises."""
    v = np.asarray(vals, float)
    w = np.ones_like(v) if w is None else np.asarray(w, float)
    ok = w > 0
    v, w = v[ok], w[ok]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (n, len(v)))
    b = (v[idx] * w[idx]).sum(1) / w[idx].sum(1)
    return float((v * w).sum() / w.sum()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data/sens"))
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    eps = load(rel(args.data))
    N = eps[0]["scores"].shape[1]
    S = np.concatenate([e["scores"] for e in eps])
    Y = np.concatenate([e["labels"] for e in eps])
    ep_of = np.concatenate([[i] * len(e["scores"]) for i, e in enumerate(eps)])
    grp = np.array([eps[i]["group"] for i in ep_of])
    n_call = len(S)
    tot = Y.sum(1)
    mixed = (tot > 0) & (tot < N)
    rep = np.concatenate([e["replay_err"] for e in eps]) if any(len(e["replay_err"]) for e in eps) else np.zeros((1, 2))
    dup = np.concatenate([e["dup"] for e in eps])

    print(f"에피소드 {len(eps)}   채점된 호출 {n_call}   후보 {N}개/호출   "
          f"branch 총 {n_call * N}회")
    print(f"분기 0 vs 실제 실행 qpos 최대 차이 {rep[:, 0].max():.2e}   "
          f"분기 0 vs 그 복제 {rep[:, 1].max():.2e}   (물리는 정확히 재현된다)")
    nf = float((dup != Y[:, 0]).mean())
    print(f"라벨 잡음 바닥: 같은 chunk 를 다시 실행했을 때 라벨이 뒤집힌 비율 {nf:.1%} "
          f"({int((dup != Y[:, 0]).sum())}/{n_call})")
    print("  (물리는 비트 단위로 같고, robosuite 가 돌려주는 관측값만 복원 후 미세하게 달라서,")
    print("   opening 이 0.04 m 문턱 바로 위아래일 때만 뒤집힌다)")
    print(f"후보 간 행동 거리 평균 {np.concatenate([e['act_dist'] for e in eps]).mean():.3f} "
          f"(정규화 전 env 단위, 앞 10 step L2)\n")

    # ---------------------------------------------------------------- 1. is there anything to choose between
    print("=== 1. 후보들이 서로 다른 결과를 내는가 ===")
    print(f"  모든 후보가 C1 (tot=N)        {int((tot == N).sum()):5d}  ({(tot == N).mean():5.1%})")
    print(f"  모든 후보가 안전 (tot=0)      {int((tot == 0).sum()):5d}  ({(tot == 0).mean():5.1%})")
    print(f"  갈리는 호출                   {int(mixed.sum()):5d}  ({mixed.mean():5.1%})  ← 고를 여지가 있는 전부")
    pe = [float(((e["labels"].sum(1) > 0) & (e["labels"].sum(1) < N)).mean()) for e in eps]
    we = [len(e["scores"]) for e in eps]
    m, lo, hi = boot_ci(pe, we)
    print(f"  갈리는 비율 (에피소드 부트스트랩)  {m:.3f}  [{lo:.3f}, {hi:.3f}]")
    print(f"  잡음 바닥 {nf:.1%} 와 비교 →  {'행동 차이가 잡음보다 크다' if mixed.mean() > 3 * nf else '잡음과 구분되지 않는다'}\n")

    # ---------------------------------------------------------------- 2. within-state ranking
    print("=== 2. 같은 상태 안에서 후보를 줄 세우는가 (reward guidance 가 실제로 쓰는 양) ===")
    per_call = np.array([auroc(S[i], Y[i]) for i in np.where(mixed)[0]])
    m, lo, hi = boot_ci([float(np.nanmean([auroc(e["scores"][j], e["labels"][j])
                                           for j in range(len(e["scores"]))
                                           if 0 < e["labels"][j].sum() < N] or [np.nan]))
                         for e in eps],
                        [int(((e["labels"].sum(1) > 0) & (e["labels"].sum(1) < N)).sum()) for e in eps])
    print(f"  상태 내 AUROC (갈리는 호출 {len(per_call)}개 평균)   {np.nanmean(per_call):.3f}")
    print(f"  에피소드 부트스트랩                        {m:.3f}  [{lo:.3f}, {hi:.3f}]   (무정보 = 0.500)")

    # the familiar between-state number, from candidate 0 only -- the unmodified rollout
    per_g = {}
    for g in sorted(set(grp)):
        k = grp == g
        if 0 < Y[k, 0].sum() < k.sum():
            per_g[g] = auroc(S[k, 0], Y[k, 0])
    print(f"  (대조) 상태 간 macro AUROC, 후보 0만        {np.mean(list(per_g.values())):.3f}   "
          f"{ {g: round(v, 3) for g, v in per_g.items()} }\n")

    # ---------------------------------------------------------------- 2b. is the latent doing the work
    print("\n=== 2b. 같은 상태 안에서라면, 그 신호는 '그리퍼를 닫으라는 명령' 아닌가 ===")
    print("  상태 고정이므로 상태 특징은 후보끼리 완전히 같고 AUROC 가 정의상 0.500 이다.")
    print("  따라서 대조군은 행동 자체에서 공짜로 읽히는 양이어야 한다 (chunk 의 그리퍼 채널 등).")
    CH = np.concatenate([e["chunks"] for e in eps])        # (calls, N, 16, 7)
    EX = 10                                                 # the steps actually executed
    feats = {
        "C1 head (suf_mean)":      S,
        "그리퍼 명령 평균":         CH[:, :, :EX, 6].mean(2),
        "그리퍼 닫힘 명령 비율":    (CH[:, :, :EX, 6] > 0).mean(2),
        "마지막 스텝 그리퍼 명령":  CH[:, :, EX - 1, 6],
        "수직 이동 명령 평균":      -CH[:, :, :EX, 2].mean(2),
        "병진 명령 크기":           np.linalg.norm(CH[:, :, :EX, :3], axis=(2, 3)),
    }
    idx = np.where(mixed)[0]
    abl = {}
    for name, F in feats.items():
        per = np.array([auroc(F[i], Y[i]) for i in idx])
        pe2 = [float(np.nanmean([auroc(F[j], Y[j]) for j in np.where(ep_of == i)[0] if mixed[j]] or [np.nan]))
               for i in range(len(eps))]
        we2 = [int(sum(mixed[j] for j in np.where(ep_of == i)[0])) for i in range(len(eps))]
        mm, ll, hh = boot_ci(pe2, we2)
        abl[name] = [float(np.nanmean(per)), mm, ll, hh]
        print(f"  {name:24s} 상태 내 AUROC {np.nanmean(per):.3f}   부트스트랩 [{ll:.3f}, {hh:.3f}]")
    # is the head's edge over the best action-only feature real?
    best_alt = max((k for k in abl if k != "C1 head (suf_mean)"), key=lambda k: abl[k][0])
    d = np.array([auroc(S[i], Y[i]) - auroc(feats[best_alt][i], Y[i]) for i in idx])
    rng = np.random.default_rng(0)
    bd = d[rng.integers(0, len(d), (4000, len(d)))].mean(1)
    print(f"  head − {best_alt} (호출별 차이 평균) {d.mean():+.3f}  "
          f"[{np.percentile(bd, 2.5):+.3f}, {np.percentile(bd, 97.5):+.3f}]")

    # ---------------------------------------------------------------- 3. what selection would buy
    print("=== 3. best-of-N 선택의 이득 — oracle / head / random ===")
    sel = S.argmin(1)
    y0, ysel = Y[:, 0], Y[np.arange(n_call), sel]
    yoracle = (Y.min(1) == 1).astype(int)          # oracle avoids C1 whenever any candidate does
    prand = Y.mean(1)                              # expected C1 rate of a uniformly random candidate
    print(f"  후보 0 (무개입)의 C1 발생률       {y0.mean():6.1%}")
    print(f"  head 가 고른 후보                 {ysel.mean():6.1%}")
    print(f"  무작위 후보 (기대값)              {prand.mean():6.1%}")
    print(f"  oracle (가능한 최선)              {yoracle.mean():6.1%}")
    risky = y0 == 1
    print(f"\n  후보 0이 C1 을 일으킨 {int(risky.sum())} 호출 중")
    if risky.sum():
        print(f"    안전한 후보가 존재한 경우        {int((Y[risky].min(1) == 0).sum()):4d}  "
              f"({(Y[risky].min(1) == 0).mean():5.1%})  ← 구제 가능 상한")
        print(f"    head 가 실제로 구제한 경우       {int((ysel[risky] == 0).sum()):4d}  "
              f"({(ysel[risky] == 0).mean():5.1%})")
        print(f"    무작위가 구제했을 기대값         {(1 - prand[risky]).mean():5.1%}")
    safe = y0 == 0
    if safe.sum():
        print(f"  후보 0이 안전했던 {int(safe.sum())} 호출 중 head 가 C1 을 새로 만든 경우  "
              f"{int((ysel[safe] == 1).sum()):4d} ({(ysel[safe] == 1).mean():5.1%})  "
              f"(무작위 기대 {prand[safe].mean():5.1%})")

    # ---------------------------------------------------------------- 4. where the variance lives
    print("\n=== 4. 점수의 분산은 상태 사이에 있는가, 상태 안에 있는가 ===")
    within = S.std(1, ddof=1).mean()
    between = S.mean(1).std(ddof=1)
    print(f"  상태 내 표준편차 평균 {within:.3f}   상태 간 표준편차 {between:.3f}   "
          f"비율 {within / between:.3f}")

    out = dict(n_eps=len(eps), n_calls=int(n_call), N=N,
               mixed_rate=[m if False else float(mixed.mean())], mixed_boot=[*boot_ci(pe, we)],
               within_auroc=float(np.nanmean(per_call)), within_auroc_boot=[m, lo, hi],
               between_macro=float(np.mean(list(per_g.values()))), between_per_group=per_g,
               rate_cand0=float(y0.mean()), rate_head=float(ysel.mean()),
               rate_random=float(prand.mean()), rate_oracle=float(yoracle.mean()),
               n_risky=int(risky.sum()),
               rescue_possible=float((Y[risky].min(1) == 0).mean()) if risky.sum() else None,
               rescue_head=float((ysel[risky] == 0).mean()) if risky.sum() else None,
               rescue_random=float((1 - prand[risky]).mean()) if risky.sum() else None,
               harm_head=float((ysel[safe] == 1).mean()) if safe.sum() else None,
               within_sd=float(within), between_sd=float(between), ablation=abl,
               head_minus_best_alt=[float(d.mean()), float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
               best_alt=best_alt,
               label_noise_floor=nf, replay_qpos_max=float(rep[:, 0].max()),
               replay_dup_qpos_max=float(rep[:, 1].max()))
    json.dump(out, open(HERE / f"results/sensitivity_{args.tag}.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n저장: results/sensitivity_{args.tag}.json")


if __name__ == "__main__":
    main()
