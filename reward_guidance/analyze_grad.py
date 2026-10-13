"""Can the frozen head be differentiated usefully, and does its gradient point at safety?

Three questions, in the order that matters:

  1. Does the frozen head mean the same thing on `phi_eval` as on `suf_mean`? It was fitted on the latter.
     If the two disagree, no gradient taken through phi_eval is a gradient of the detector that scored 0.891,
     and any guided rollout would be steering by a different function that merely shares its weights.

  2. Where does the gradient live? Split its energy across the chunk's channels. The action-feature ablation
     already showed the head's within-state signal is the gripper command; if the gradient is also almost all
     gripper, reward guidance by this head is "open the gripper", written expensively.

  3. Does it point at the safe candidates? At a state with both safe and unsafe candidates there is a measured
     direction, (mean safe - mean unsafe). The cosine between the gradient (descending risk) and that
     direction is what decides whether guidance moves the chunk somewhere better. Zero cosine means a larger
     guidance weight just moves further sideways.

    python reward_guidance/analyze_grad.py --data data/grad
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "safe_detector"))

from train_detector import auroc  # noqa: E402

CH = {"병진 xyz": slice(0, 3), "회전 rpy": slice(3, 6), "그리퍼": slice(6, 7)}
EX = 10  # executed steps


def boot(v, n=4000, seed=0):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if not len(v):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/grad")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    root = Path(args.data) if Path(args.data).is_absolute() else HERE / args.data

    Ss, Se, G, C, Y, grp = [], [], [], [], [], []
    for line in (root / "episodes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        z = np.load(root / r["npz"])
        if z["score_suf"].shape[0] == 0:
            continue
        Ss.append(z["score_suf"]); Se.append(z["score_eval"]); G.append(z["grad0"])
        C.append(z["chunks_norm"]); Y.append(z["labels"])
        grp += [r["group"]] * len(z["score_suf"])
    Ss, Se, G, C, Y = (np.concatenate(x) for x in (Ss, Se, G, C, Y))
    grp = np.array(grp)
    N = Ss.shape[1]
    tot = Y.sum(1)
    mixed = (tot > 0) & (tot < N)
    print(f"호출 {len(Ss)}   후보 {N}   갈리는 호출 {int(mixed.sum())}\n")

    # ---------------------------------------------------------------- 1
    print("=== 1. 미분 가능한 대역 phi_eval 을 head 가 같은 뜻으로 읽는가 ===")
    a, b = Ss[:, 0], Se[:, 0]
    r = float(np.corrcoef(a, b)[0, 1])
    print(f"  호출 간 상관 (후보 0)        r = {r:+.3f}    suf_mean 표준편차 {a.std():.3f} / phi_eval {b.std():.3f}")
    rin = np.array([np.corrcoef(Ss[i], Se[i])[0, 1] for i in range(len(Ss)) if Ss[i].std() > 0 and Se[i].std() > 0])
    m, lo, hi = boot(rin)
    print(f"  같은 상태 안 후보 간 상관     r = {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  (후보를 같은 순서로 줄 세우는가)")
    for name, F in (("suf_mean", Ss), ("phi_eval", Se)):
        per = np.array([auroc(F[i], Y[i]) for i in np.where(mixed)[0]])
        mm, ll, hh = boot(per)
        print(f"  {name:10s} 상태 내 AUROC {mm:.3f} [{ll:.3f}, {hh:.3f}]")
    pg = {g: auroc(Se[grp == g, 0], Y[grp == g, 0]) for g in sorted(set(grp))
          if 0 < Y[grp == g, 0].sum() < (grp == g).sum()}
    pgs = {g: auroc(Ss[grp == g, 0], Y[grp == g, 0]) for g in sorted(set(grp))
           if 0 < Y[grp == g, 0].sum() < (grp == g).sum()}
    print(f"  상태 간 macro AUROC   suf_mean {np.mean(list(pgs.values())):.3f}   "
          f"phi_eval {np.mean(list(pg.values())):.3f}")

    # ---------------------------------------------------------------- 2
    print("\n=== 2. 기울기는 어느 채널에 있는가 (정규화 행동 공간, 실행되는 10 step) ===")
    g = G[:, :EX, :]
    e = (g ** 2).sum(axis=(1, 2))
    print(f"  ‖∇‖ 평균 {np.sqrt(e).mean():.4f}   (방향만 쓰므로 크기는 참고값)")
    for name, sl in CH.items():
        frac = (g[:, :, sl] ** 2).sum(axis=(1, 2)) / np.maximum(e, 1e-20)
        mm, ll, hh = boot(frac)
        dims = sl.stop - sl.start
        print(f"  {name:8s} ({dims}/7 차원)  에너지 비중 {mm:5.1%} [{ll:.1%}, {hh:.1%}]   "
              f"균등 기대 {dims / 7:5.1%}")
    gi = -g[:, :, 6]   # descending risk: the gripper component of the direction guidance would move in
    mm, ll, hh = boot(gi.mean(1))
    print(f"  위험을 낮추는 방향의 그리퍼 성분 평균 {mm:+.4f} [{ll:+.4f}, {hh:+.4f}]  "
          f"(+ 는 '더 닫아라', − 는 '열어라')")

    # ---------------------------------------------------------------- 3
    print("\n=== 3. 기울기가 실제로 안전한 후보 쪽을 가리키는가 ===")
    idx = [i for i in np.where(mixed)[0] if Y[i, 0] == 1]
    print(f"  후보 0 이 C1 을 일으켰고 안전한 후보도 있는 호출 {len(idx)}개")
    cos_safe, cos_grip, cos_rand = [], [], []
    rng = np.random.default_rng(0)
    for i in idx:
        safe = C[i][Y[i] == 0][:, :EX, :].mean(0)
        uns = C[i][Y[i] == 1][:, :EX, :].mean(0)
        d = (safe - uns).ravel()
        gv = (-G[i, :EX, :]).ravel()
        n = np.linalg.norm(d) * np.linalg.norm(gv)
        if n < 1e-12:
            continue
        cos_safe.append(float(d @ gv / n))
        gr = np.zeros_like(G[i, :EX, :]); gr[:, 6] = -1.0      # pure "open the gripper"
        cos_grip.append(float(d @ gr.ravel() / (np.linalg.norm(d) * np.linalg.norm(gr))))
        rv = rng.normal(size=gv.shape)
        cos_rand.append(float(d @ rv / (np.linalg.norm(d) * np.linalg.norm(rv))))
    for name, v in (("head 기울기", cos_safe), ("'그리퍼를 열어라' 방향", cos_grip), ("무작위 방향", cos_rand)):
        mm, ll, hh = boot(v)
        print(f"  {name:22s} 안전 방향과의 cos  {mm:+.3f} [{ll:+.3f}, {hh:+.3f}]")

    json.dump(dict(n_calls=int(len(Ss)), n_mixed=int(mixed.sum()),
                   corr_between=r, corr_within=[m, lo, hi],
                   macro_suf=float(np.mean(list(pgs.values()))), macro_eval=float(np.mean(list(pg.values()))),
                   energy={k: list(boot((g[:, :, s] ** 2).sum(axis=(1, 2)) / np.maximum(e, 1e-20)))
                           for k, s in CH.items()},
                   grip_component=list(boot(gi.mean(1))),
                   cos_head=list(boot(cos_safe)), cos_gripdir=list(boot(cos_grip)),
                   cos_random=list(boot(cos_rand)), n_cos=len(cos_safe)),
              open(HERE / f"results/grad_{args.tag}.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n저장: results/grad_{args.tag}.json")


if __name__ == "__main__":
    main()
