"""Guidance moves the detector's score. Does it move the robot?

Everything is paired at the call: the arms share a state, a start noise and a prefix, so a difference between
them is the guidance term and nothing else. The paired difference is bootstrapped over EPISODES, not calls,
because calls within an episode are not independent.

The three numbers that decide it, per arm: how far the chunk moved, how far the C1 logit moved, and how far
the C1 rate moved. A reward that is doing its job moves the third in step with the second.

    python reward_guidance/analyze_stagec.py --data data/stagec
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# kept in step with stage_c.ARMS / stage_c.LAM by hand: importing stage_c would boot robosuite and the policy
ARM_NAMES = ("무개입", "위험↓ λ10", "위험↑ λ10", "무작위 λ10", "위험↓ λ30", "무작위 λ30")
LAM = (0.0, 10.0, 10.0, 10.0, 30.0, 30.0)


def boot_paired(per_ep, n=4000, seed=0):
    """per_ep: list of (sum, count) per episode -> mean and 95% CI, resampling episodes."""
    a = np.array([p[0] for p in per_ep], float)
    c = np.array([p[1] for p in per_ep], float)
    k = c > 0
    a, c = a[k], c[k]
    if not len(a):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(a), (n, len(a)))
    b = a[i].sum(1) / c[i].sum(1)
    return float(a.sum() / c.sum()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stagec")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    root = Path(args.data) if Path(args.data).is_absolute() else HERE / args.data

    eps, L, S, D, O = [], [], [], [], []
    for line in (root / "episodes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        z = np.load(root / r["npz"])
        if z["labels"].shape[0] == 0:
            continue
        eps.append((r, z))
        L.append(z["labels"]); S.append(z["scores"]); D.append(z["disp"]); O.append(z["oob"])
    L, S, D, O = (np.concatenate(x) for x in (L, S, D, O))
    A = L.shape[1]
    print(f"에피소드 {len(eps)}   채점된 호출 {len(L)}   arm {A}개\n")

    print(f"  {'arm':14s} {'변위':>8s} {'범위밖':>7s} {'C1 logit':>10s} {'Δlogit':>9s} "
          f"{'C1 발생률':>10s} {'Δ발생률 (95% CI)':>24s}")
    out = {}
    for k in range(A):
        per_ep_d = [( (z["labels"][:, k] - z["labels"][:, 0]).sum(), len(z["labels"]) ) for _, z in eps]
        m, lo, hi = boot_paired(per_ep_d)
        flip_up = int(((L[:, 0] == 0) & (L[:, k] == 1)).sum())
        flip_dn = int(((L[:, 0] == 1) & (L[:, k] == 0)).sum())
        out[ARM_NAMES[k]] = dict(lam=LAM[k], disp=float(D[:, k].mean()), oob=float(O[:, k].mean()),
                                 logit=float(S[:, k].mean()), dlogit=float((S[:, k] - S[:, 0]).mean()),
                                 rate=float(L[:, k].mean()), drate=[m, lo, hi],
                                 flip_to_C1=flip_up, flip_from_C1=flip_dn)
        print(f"  {ARM_NAMES[k]:14s} {D[:, k].mean():8.3f} {O[:, k].mean():7.1%} {S[:, k].mean():10.3f} "
              f"{(S[:, k] - S[:, 0]).mean():+9.3f} {L[:, k].mean():10.1%} "
              f"{m:+9.3%} [{lo:+.3%}, {hi:+.3%}]")

    print(f"\n  라벨이 뒤집힌 호출 (무개입 대비):")
    for k in range(1, A):
        o = out[ARM_NAMES[k]]
        print(f"    {ARM_NAMES[k]:14s} 안전→C1 {o['flip_to_C1']:3d}   C1→안전 {o['flip_from_C1']:3d}   "
              f"(총 {len(L)} 호출 중 {o['flip_to_C1'] + o['flip_from_C1']:3d})")

    mixed = (L.sum(1) > 0) & (L.sum(1) < A)
    print(f"\n  arm 끼리 결과가 갈린 호출 {int(mixed.sum())} / {len(L)} ({mixed.mean():.1%})")
    risky = L[:, 0] == 1
    print(f"  무개입이 C1 을 일으킨 {int(risky.sum())} 호출 중 어느 arm 이든 피한 경우 "
          f"{int((L[risky].min(1) == 0).sum())} ({(L[risky].min(1) == 0).mean():.1%})")

    # guidance vs. a step of the same size in a scrambled direction -- the contrast that decides whether the
    # head supplied a direction or merely an excuse to move
    print("\n  같은 보폭에서 '기울기 방향' vs '뒤섞은 방향' (짝지은 차이):")
    for a, b, tag in ((1, 3, "λ10"), (4, 5, "λ30")):
        per = [((z["labels"][:, a] - z["labels"][:, b]).sum(), len(z["labels"])) for _, z in eps]
        m, lo, hi = boot_paired(per)
        out[f"guided_minus_random_{tag}"] = [m, lo, hi]
        print(f"    {tag}:  {ARM_NAMES[a]} − {ARM_NAMES[b]}   {m:+.3%} [{lo:+.3%}, {hi:+.3%}]"
              f"   {'기울기가 유의하게 낫다' if hi < 0 else '구분되지 않는다'}")

    # the sign test: the same magnitude, the same machinery, only the direction of travel reversed. A random
    # step cannot produce this contrast, so it isolates what the gradient DIRECTION is worth.
    per = [((z["labels"][:, 1] - z["labels"][:, 2]).sum(), len(z["labels"])) for _, z in eps]
    m, lo, hi = boot_paired(per)
    out["down_minus_up_lam10"] = [m, lo, hi]
    print(f"\n  부호 뒤집기 (λ10):  위험↓ − 위험↑  {m:+.3%} [{lo:+.3%}, {hi:+.3%}]"
          f"   {'기울기의 부호가 결과를 바꾼다' if hi < 0 else '부호가 결과를 바꾸지 않는다'}")

    # and only where it could matter: the calls the unguided chunk actually broke
    print("\n  무개입이 C1 을 일으킨 호출만 (구제율, 짝지은 부트스트랩):")
    for k in range(1, A):
        per = [(int(((z["labels"][:, 0] == 1) & (z["labels"][:, k] == 0)).sum()),
                int((z["labels"][:, 0] == 1).sum())) for _, z in eps]
        m, lo, hi = boot_paired(per)
        out[ARM_NAMES[k]]["rescue"] = [m, lo, hi]
        print(f"    {ARM_NAMES[k]:14s} {m:6.1%} [{lo:.1%}, {hi:.1%}]")

    # the headline: score moved a lot, rate moved how much
    k10 = ARM_NAMES[1]
    print(f"\n  요약: 위험 점수를 {abs(out[k10]['dlogit']):.2f} logit 낮췄을 때 C1 발생률 변화 "
          f"{out[k10]['drate'][0]:+.2%} [{out[k10]['drate'][1]:+.2%}, {out[k10]['drate'][2]:+.2%}]")
    json.dump(dict(n_eps=len(eps), n_calls=int(len(L)), arms=out,
                   mixed_rate=float(mixed.mean()), n_risky=int(risky.sum())),
              open(HERE / f"results/stagec_{args.tag}.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n저장: results/stagec_{args.tag}.json")


if __name__ == "__main__":
    main()
