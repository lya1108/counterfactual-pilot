"""How early, from which latent, and beyond what a state monitor already knows -- C1 prospective prediction.

Primary metric throughout is unseen-task macro AUROC: the mean of per-task AUROCs over tasks held out by the
task-group split. Pooled AUROC is printed beside it only as a reminder of how far apart they can be; an earlier
stage of this project read 0.73 pooled where the macro was 0.50, because pooled rewards knowing WHICH task is
hard rather than which rollout is going wrong.

  --mode horizon    y = 1[C1 onset in (t+k, t+H]] for several H and guard intervals k.
                    k > 0 also requires C1 to stay clear through t+k, so a sample cannot be a state that is
                    already one step from violating -- technically a prediction, too late to steer.

  --mode distance   among samples that DO have a future onset, bucket by how far away it is (0-10, 10-20,
                    20-30, 30-40 steps) and score each bucket against samples with no onset within 40 steps.
                    This is the "how early does the signal appear" curve.

  --mode features   the same target across representations: the action expert's pooled tokens, the VLM's image
                    tokens, its language tokens (which carry the discretised robot state), both, and state
                    baselines -- including one with history, so "just watch the gripper approach" is ruled out
                    properly rather than by a single-instant feature.

  --mode pertask    per-task AUROC, so a high macro cannot be a few easy tasks carrying the rest.

  --mode latency    what the detector adds to a VLA call.

    conda activate fasi
    python xattn_detector/c1_depth.py --data data/cond200 --mode horizon
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "safe_detector"))

from conditions import evaluate  # noqa: E402
from prospective_c1 import fit_eval, state_vector  # noqa: E402
from train_detector import auroc  # noqa: E402

COND = "C1_grasp_empty_close"
NO_ONSET_WINDOW = 40  # a negative for the distance analysis must be clear of C1 for at least this long


def onset_index(series, debounce=3):
    run = 0
    for i, b in enumerate(series):
        run = run + 1 if b else 0
        if run >= debounce:
            return i - debounce + 1
    return None


def build(root: Path, hist: int = 3, want_tokens: bool = False, hist_stride: int = 5):
    """One row per VLA call where C1 is not already active, with every feature family and the onset distance."""
    rows = []
    for line in (root / "episodes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        z = np.load(root / r["npz"])
        series, _ = evaluate({k: z[k] for k in z.files})
        c1 = series[COND]
        step = z["step"]
        pos_of = {int(s): i for i, s in enumerate(step)}
        names = [str(x) for x in z["object_names"]]
        goal_objs = {str(x) for x in z["goal_objects"]}
        goal_mask = np.array([n in goal_objs for n in names], bool)
        has_tok = want_tokens and "pre_img" in z.files

        for k, cs in enumerate(z["call_step"].astype(int)):
            i_now = pos_of.get(int(cs))
            if i_now is None or c1[i_now]:
                continue
            # distance in environment steps to the next onset that starts after this call
            nxt = onset_index(c1[i_now + 1:])
            d_onset = None if nxt is None else int(step[i_now + 1 + nxt] - cs)
            row = dict(group=f"{r['suite']}/t{r['task']}", suite=r["suite"], task=r["task"], init=r["init"],
                       call=k, call_step=int(cs), d_onset=d_onset,
                       clear_ahead=int(step[i_now + 1 + nxt] - cs) if nxt is not None else 10 ** 6,
                       suf_mean=z["suf_mean"][k].astype(np.float32),
                       state=state_vector(z, i_now, goal_mask),
                       # t, t-stride, t-2*stride ... spanning one VLA call interval by default
                       state_hist=np.concatenate([state_vector(z, max(i_now - j * hist_stride, 0), goal_mask)
                                                  for j in range(hist)]),
                       opening=np.array([z["opening"][i_now]], np.float32),
                       call_index=np.array([k, cs], np.float32))
            if has_tok:
                row["vlm_img"] = z["pre_img"][k].astype(np.float32).mean(0)
                m = z["lang_mask"][k]
                row["vlm_lang"] = z["pre_lang"][k][m].astype(np.float32).mean(0)
                row["vlm_all"] = np.concatenate([row["vlm_img"], row["vlm_lang"]])
            rows.append(row)
    return rows


def macro_pooled(scores, y, groups):
    by = collections.defaultdict(lambda: ([], []))
    for s, yy, g in zip(scores, y, groups):
        by[g][0].append(s)
        by[g][1].append(yy)
    per = {g: auroc(np.array(a), np.array(b)) for g, (a, b) in by.items() if 0 < sum(b) < len(b)}
    pooled = auroc(np.asarray(scores), np.asarray(y)) if 0 < sum(y) < len(y) else float("nan")
    return (float(np.mean(list(per.values()))) if per else float("nan")), pooled, per


def repeated_cv(rows, y, feat, groups, folds, repeats):
    """Macro/pooled AUROC over repeated task-group splits. Returns (macro mean, sd, pooled mean, per-task)."""
    X = np.stack([r[feat] for r in rows])
    gl = [r["group"] for r in rows]
    ms, ps, last_per = [], [], {}
    for rep in range(repeats):
        rng = np.random.default_rng(100 + rep)
        fold_of = {g: i % folds for i, g in enumerate(rng.permutation(groups))}
        sc = np.zeros(len(rows))
        for f in range(folds):
            te = np.array([fold_of[r["group"]] == f for r in rows])
            if te.sum() == 0 or y[~te].sum() < 2 or (1 - y[~te]).sum() < 2:
                continue
            sc[te] = fit_eval(X[~te], y[~te], X[te], seed=rep)
        m, p, per = macro_pooled(sc, y, gl)
        ms.append(m); ps.append(p); last_per = per
    return (float(np.nanmean(ms)), float(np.nanstd(ms, ddof=1)) if len(ms) > 1 else 0.0,
            float(np.nanmean(ps)), last_per)


def target_horizon(rows, H, k):
    """1 if the onset lands in (t+k, t+H]; samples that violate within the guard interval are dropped."""
    keep, y = [], []
    for r in rows:
        d = r["d_onset"]
        if k > 0 and d is not None and d <= k:
            continue                       # already inside the guard -- too late to be a useful warning
        keep.append(r)
        y.append(d is not None and k < d <= H)
    return keep, np.array(y, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cond200")
    ap.add_argument("--mode", default="horizon",
                    choices=["horizon", "distance", "features", "pertask", "latency", "all"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--hist", type=int, default=3)
    ap.add_argument("--feature", default="suf_mean")
    ap.add_argument("--H", type=int, default=10)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    want_tokens = args.mode in ("features", "all")
    rows_all = build(HERE / args.data, args.hist, want_tokens)
    groups = sorted({r["group"] for r in rows_all})
    n_ep = len({(r["suite"], r["task"], r["init"]) for r in rows_all})
    print(f"에피소드 {n_ep}   C1(t)=0 호출 샘플 {len(rows_all)}   task 그룹 {len(groups)}")
    print(f"분할 {args.folds}-fold × {args.repeats}회 재추출, 1차 지표 = 미학습 task macro AUROC\n")
    out = {}

    if args.mode in ("horizon", "all"):
        print("=== 2·3. 예측 지평 H 와 guard interval k ===")
        print(f"  {'H':>4s} {'k':>3s} {'샘플':>7s} {'양성':>8s} {'macro (suf_mean)':>19s} "
              f"{'macro (state_hist)':>20s} {'pooled':>8s}")
        for H in (5, 10, 20, 30):
            for k in (0, 3, 5):
                if k >= H:
                    continue
                rows, y = target_horizon(rows_all, H, k)
                if y.sum() < 15:
                    print(f"  {H:4d} {k:3d} {len(rows):7d} {int(y.sum()):4d} — 양성 부족")
                    continue
                g = sorted({r["group"] for r in rows})
                m1, s1, p1, _ = repeated_cv(rows, y, "suf_mean", g, args.folds, args.repeats)
                m2, s2, _, _ = repeated_cv(rows, y, "state_hist", g, args.folds, args.repeats)
                out[f"H{H}_k{k}"] = dict(n=len(rows), pos=int(y.sum()), suf=[m1, s1],
                                         state_hist=[m2, s2], pooled=p1)
                print(f"  {H:4d} {k:3d} {len(rows):7d} {int(y.sum()):4d} ({y.mean():4.1%}) "
                      f"{m1:12.3f}±{s1:.3f} {m2:13.3f}±{s2:.3f} {p1:8.3f}")
        print()

    if args.mode in ("distance", "all"):
        print("=== 2. onset 까지 남은 거리별 — 언제부터 신호가 보이는가 ===")
        print("  음성 = 앞으로 40 step 안에 onset 이 없는 호출")
        print(f"  {'남은 거리':>12s} {'양성':>6s} {'음성':>6s} {'macro (suf_mean)':>19s} "
              f"{'macro (state_hist)':>20s} {'pooled':>8s}")
        neg = [r for r in rows_all if r["d_onset"] is None or r["d_onset"] > NO_ONSET_WINDOW]
        for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 40)):
            pos = [r for r in rows_all if r["d_onset"] is not None and lo < r["d_onset"] <= hi]
            if len(pos) < 15:
                print(f"  {f'{lo}–{hi}':>12s} {len(pos):6d} {len(neg):6d}  — 양성 부족")
                continue
            rows = pos + neg
            y = np.array([1.0] * len(pos) + [0.0] * len(neg))
            g = sorted({r["group"] for r in rows})
            m1, s1, p1, _ = repeated_cv(rows, y, "suf_mean", g, args.folds, args.repeats)
            m2, s2, _, _ = repeated_cv(rows, y, "state_hist", g, args.folds, args.repeats)
            out[f"dist_{lo}_{hi}"] = dict(pos=len(pos), neg=len(neg), suf=[m1, s1], state_hist=[m2, s2])
            print(f"  {f'{lo}–{hi}':>12s} {len(pos):6d} {len(neg):6d} {m1:12.3f}±{s1:.3f} "
                  f"{m2:13.3f}±{s2:.3f} {p1:8.3f}")
        print()

    if args.mode in ("features", "all"):
        print(f"=== 5·6. 표현별 비교 (H={args.H}, k=0) ===")
        rows, y = target_horizon(rows_all, args.H, 0)
        g = sorted({r["group"] for r in rows})
        feats = [f for f in ("suf_mean", "vlm_all", "vlm_img", "vlm_lang",
                             "state_hist", "state", "opening", "call_index") if f in rows[0]]
        print(f"  {'feature':14s} {'차원':>6s} {'macro':>16s} {'pooled':>8s}")
        for f in feats:
            m, s, p, _ = repeated_cv(rows, y, f, g, args.folds, args.repeats)
            out[f"feat_{f}"] = dict(dim=int(np.asarray(rows[0][f]).size), macro=[m, s], pooled=p)
            print(f"  {f:14s} {np.asarray(rows[0][f]).size:6d} {m:9.3f}±{s:.3f} {p:8.3f}")
        print()

    if args.mode in ("pertask", "all"):
        print(f"=== 4. task 별 AUROC ({args.feature}, H={args.H}, k=0) ===")
        rows, y = target_horizon(rows_all, args.H, 0)
        g = sorted({r["group"] for r in rows})
        _, _, _, per = repeated_cv(rows, y, args.feature, g, args.folds, args.repeats)
        base = collections.defaultdict(list)
        for r, yy in zip(rows, y):
            base[r["group"]].append(yy)
        print(f"  {'task':22s} {'n':>5s} {'양성률':>7s} {'AUROC':>7s}")
        for k2 in sorted(per, key=lambda x: -per[x]):
            print(f"  {k2:22s} {len(base[k2]):5d} {np.mean(base[k2]):6.1%} {per[k2]:7.3f}")
        v = np.array(list(per.values()))
        print(f"  → {len(per)}개 task, macro {v.mean():.3f}, 중앙값 {np.median(v):.3f}, "
              f"0.5 이하인 task {(v <= 0.5).sum()}개")
        out["per_task"] = per
        print()

    if args.mode in ("latency", "all"):
        print("=== 8. detector 추론 시간 (VLA 호출에 더해지는 비용) ===")
        rows, y = target_horizon(rows_all, args.H, 0)
        X = np.stack([r["suf_mean"] for r in rows])
        mu, sd = X.mean(0), X.std(0) + 1e-6
        for dev in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            net = nn.Linear(X.shape[1], 1).to(dev).eval()
            x = torch.from_numpy((X[:1] - mu) / sd).to(dev).float()
            with torch.no_grad():
                for _ in range(50):
                    net(x)
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(2000):
                    net(x)
                if dev == "cuda":
                    torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 2000 * 1000
            out[f"latency_{dev}"] = ms
            print(f"  {dev:5s} {ms:.4f} ms / 호출   (π0.5 denoise 1회 411 ms 대비 {ms / 411:.2%})")
        print("  측정 범위: 표준화 + 선형 1층 forward, batch 1. latent 추출은 VLA forward 안에서 이미 일어난다.")

    path = HERE / "results" / f"c1_depth_{args.mode}_{args.tag}.json"
    path.write_text(json.dumps(dict(args=vars(args), n_episodes=n_ep, n_rows=len(rows_all),
                                    n_groups=len(groups), out=out), indent=2,
                               ensure_ascii=False, default=float))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
