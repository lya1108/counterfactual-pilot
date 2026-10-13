"""What `data/sens` can and cannot support, measured before any head is trained.

Stage D's design rests on four facts about the collected data, and a reader should be able to
reproduce each of them without rerunning the policy:

  1. how many states are mixed at all -- the only states that carry a within-state pair
  2. whether those mixed states are real action differences or just the 0.04 m threshold flickering
  3. where the duplicate-branch label noise sits (it is not spread evenly)
  4. how much a perfect ranker could buy, and how much of that the gripper scalar already takes

    python reward_guidance/inspect_sens.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EX, N_CAND, THRESH = 10, 8, 0.04


def load(d=HERE / "data/sens"):
    eps = [json.loads(l) for l in (d / "episodes.jsonl").read_text().splitlines() if l.strip()]
    out = {k: [] for k in ("y", "s", "om", "dup", "cs", "do", "ch", "phi")}
    ep, gr = [], []
    for e in eps:
        z = np.load(d / e["npz"])
        n = len(z["labels"])
        for k, f in (("y", "labels"), ("s", "scores"), ("om", "open_min"), ("dup", "dup_label"),
                     ("cs", "call_step"), ("do", "d_onset"), ("ch", "chunks"), ("phi", "phi")):
            out[k].append(z[f])
        ep += [f"{e['group']}/i{e['init']}"] * n
        gr += [e["group"]] * n
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["y"] = out["y"].astype(int)
    out["phi"] = out["phi"].astype(np.float32)
    out["ep"], out["gr"] = np.array(ep), np.array(gr)
    return out


def main():
    D = load()
    y, om, k = D["y"], D["om"], D["y"].sum(1)
    mix = (k > 0) & (k < N_CAND)
    N = len(y)

    print("=== 1. 유효 표본 ===")
    print(f"  호출(상태) {N}   후보 {N * N_CAND}   위험 후보 {int(y.sum())} ({y.mean() * 100:.1f}%)")
    for lab, m in (("모두 안전 k=0", k == 0), ("혼합 0<k<8", mix), ("모두 위험 k=8", k == N_CAND)):
        pr = int((k[m] * (N_CAND - k[m])).sum()) if lab.startswith("혼합") else 0
        print(f"    {lab:14s} {int(m.sum()):4d}  {m.mean() * 100:5.1f}%   쌍 {pr}")
    print(f"  k 분포  " + "  ".join(f"{i}:{int((k == i).sum())}" for i in range(N_CAND + 1)))
    print(f"  에피소드 군집 {len(set(D['ep'][mix]))}/{len(set(D['ep']))}   과제 군집 {len(set(D['gr']))}")

    print("\n=== 2. 혼합 상태는 문턱 잡음인가 ===")
    sep = np.array([om[i][y[i] == 0].min() - om[i][y[i] == 1].max() for i in np.where(mix)[0]])
    near = sum(bool(np.all(np.abs(om[i] - THRESH) < 0.002)) for i in np.where(mix)[0])
    print(f"  min(안전 open_min) − max(위험 open_min)  중앙 {np.median(sep):+.4f} m, "
          f"겹침(<0) {int((sep < 0).sum())}/{len(sep)}")
    print(f"  후보 전부가 문턱 ±0.002 m 안에 있는 혼합 상태: {near}  -> 문턱 잡음이 아니다")
    print(f"  혼합 상태 후보의 문턱 여유 중앙값 {np.median(om[mix] - THRESH):+.4f} m "
          f"(대부분 닫히고, 접촉·debounce 에서 갈린다)")

    print("\n=== 3. 라벨 잡음은 어디 있나 (중복 분기, 후보 0 재실행) ===")
    bad = D["dup"] != y[:, 0]
    pm = bad[mix].mean() if mix.any() else 0.0
    pn = bad[~mix].mean()
    print(f"  전체 불일치 {int(bad.sum())}/{N} = {bad.mean() * 100:.2f}%")
    print(f"  P(불일치 | 혼합)   {int(bad[mix].sum())}/{int(mix.sum())} = {pm * 100:.1f}%")
    print(f"  P(불일치 | 비혼합) {int(bad[~mix].sum())}/{int((~mix).sum())} = {pn * 100:.2f}%"
          f"   -> 약 {pm / max(pn, 1e-9):.0f}배 몰려 있다")
    print(f"  쌍의 ~{2 * pm * 100:.0f}% 가 동전던지기라면 상태 내 AUROC 천장은 대략 "
          f"{(0.784 - pm) / (1 - 2 * pm):.2f} 근처")

    print("\n=== 4. 선택 여유폭과 그리퍼 스칼라의 몫 ===")
    g = D["ch"][:, :, :EX, 6].mean(2)
    def sel(sc): return float(y[np.arange(N), sc.argmin(1)].mean())
    rand, orc = float(y.mean()), float((k == N_CAND).mean())
    print(f"{'선택 규칙':24s} {'전체 C1':>9s}")
    for lab, v in (("무작위 (기대값)", rand), ("현재 C1 head", sel(D["s"])),
                   ("그리퍼 명령 평균", sel(g)), ("오라클", orc), ("최악", float((k > 0).mean()))):
        print(f"  {lab:22s} {v * 100:7.1f}%")
    print(f"  여유폭 {rand * 100:.1f}% → {orc * 100:.1f}% = {(rand - orc) * 100:.1f}%p   "
          f"head 가 먹은 몫 {(rand - sel(D['s'])) / (rand - orc) * 100:.0f}%, "
          f"그리퍼 {(rand - sel(g)) / (rand - orc) * 100:.0f}%")

    print("\n=== 5. phi 의 상태 내 편차는 몇 차원인가 ===")
    W = (D["phi"] - D["phi"].mean(1, keepdims=True)).reshape(-1, D["phi"].shape[-1]).astype(np.float64)
    c = W - W.mean(0)
    val = np.linalg.eigvalsh(c.T @ c)[::-1]
    ev = val / val.sum()
    print(f"  설명분산  PC1 {ev[0] * 100:.1f}%   PC1-5 {ev[:5].sum() * 100:.1f}%   "
          f"PC1-20 {ev[:20].sum() * 100:.1f}%   PC1-50 {ev[:50].sum() * 100:.1f}%")
    print(f"  유효 랭크 (참여비) {np.exp(-(ev * np.log(ev + 1e-12)).sum()):.0f}")
    vt = np.linalg.eigh(c.T @ c)[1][:, ::-1][:, 0]
    gw = (g - g.mean(1, keepdims=True)).reshape(-1)
    print(f"  상태 내 PC1 vs 그리퍼 명령 편차  r = {np.corrcoef(c @ vt, gw)[0, 1]:+.3f}"
          f"   -> phi 의 주된 상태 내 변동은 그리퍼 명령이 아니다")


if __name__ == "__main__":
    main()
