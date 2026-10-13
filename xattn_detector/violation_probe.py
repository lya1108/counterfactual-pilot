"""Can pi0.5's latent predict a local condition violation 10 environment steps ahead?

Every earlier detector was trained on the episode's final outcome. This trains on the local target instead:
given the latent of the VLA call issued at step t, will C1 / C3 / C4 be violated in (t, t+H]? With H = 10 and
10 actions executed per call, that window is exactly the steps this call's action chunk drives, which is the
question a guidance signal would have to answer.

Two target definitions, because they differ by an order of magnitude in how many positives exist:
  onset   the condition's onset falls in the window -- at most one positive per episode per condition
  active  the condition holds anywhere in the window -- the state the robot is actually in

Baselines decide whether any number here means anything:
  call_index  the call number alone. C1 clusters around steps 45-130, so the clock predicts it without
              looking at anything; an earlier stage of this project had a clock baseline reach ep-AUROC 0.96
              on the outcome target, and it has to be ruled out again here.
  eef         the end-effector position. Physically informative and not part of any condition's definition.
  opening     the gripper gap. For C1 this is PARTLY DEFINITIONAL (C1 is "closed and touching nothing"), so it
              is an upper reference showing the target is predictable from something, not a fair competitor.
  random      the chance floor at this sample size.

Splits are grouped by (suite, task) -- the 40 episodes cover 20 tasks, 2 inits each -- so a task never appears
on both sides. AUROC is macro over held-out tasks that contain both classes, the statistic every other result
in this project now uses.

    conda activate fasi
    python xattn_detector/violation_probe.py --H 10
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "safe_detector"))

from conditions import evaluate  # noqa: E402
from train_detector import auroc  # noqa: E402

TRACKED = ("C1_grasp_empty_close", "C3_nontarget_contact", "C4_object_tipped")
COMPOSITE = "composite_C1C3C4"
FEATURES = ("suf_mean", "call_index", "eef", "opening", "random")


def build(root: Path, H: int):
    """One row per VLA call: features, the two target variants per condition, and its task group."""
    rows = []
    for line in (root / "episodes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        z = np.load(root / r["npz"])
        series, _ = evaluate({k: z[k] for k in z.files})
        step = z["step"]
        pos_of = {int(s): i for i, s in enumerate(step)}
        comp_series = np.zeros(len(step), bool)
        for c in TRACKED:
            comp_series |= series[c]
        series = {**{c: series[c] for c in TRACKED}, COMPOSITE: comp_series}
        onset_idx = {}
        for c, v in series.items():
            run, got = 0, None
            for i, b in enumerate(v):
                run = run + 1 if b else 0
                if run >= 3:
                    got = i - 2
                    break
            onset_idx[c] = got

        for k, cs in enumerate(z["call_step"].astype(int)):
            i0 = pos_of.get(cs + 1)
            if i0 is None:
                continue
            i1 = pos_of.get(cs + H, len(step) - 1)
            win = slice(i0, i1 + 1)
            row = dict(group=f"{r['suite']}/t{r['task']}", episode=f"{r['suite']}_t{r['task']}_i{r['init']}",
                       call=k, call_step=int(cs), success=r["success"],
                       suf_mean=z["suf_mean"][k].astype(np.float32),
                       eef=z["eef"][i0].astype(np.float32),
                       opening=np.array([z["opening"][i0]], np.float32),
                       call_index=np.array([k, cs], np.float32))
            for c, v in series.items():
                row[f"active__{c}"] = bool(v[win].any())
                o = onset_idx[c]
                row[f"onset__{c}"] = bool(o is not None and i0 <= o <= i1)
            rows.append(row)
    return rows


def fit_eval(Xtr, ytr, Xte, seed, epochs=300, lr=1e-2, wd=1.0, device="cpu"):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    net = nn.Linear(Xtr.shape[1], 1).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    xt = torch.from_numpy((Xtr - mu) / sd).to(device).float()
    yt = torch.from_numpy(ytr).to(device).float()[:, None]
    w = torch.where(yt > 0, 1.0 / max(ytr.sum(), 1), 1.0 / max((1 - ytr).sum(), 1))
    for _ in range(epochs):
        opt.zero_grad()
        loss = (nn.functional.binary_cross_entropy_with_logits(net(xt), yt, reduction="none") * w).sum() / w.sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return net(torch.from_numpy((Xte - mu) / sd).to(device).float()).cpu().numpy().ravel()


def macro(scores, y, groups):
    by = collections.defaultdict(lambda: ([], []))
    for s, yy, g in zip(scores, y, groups):
        by[g][0].append(s)
        by[g][1].append(yy)
    per = {g: auroc(np.array(a), np.array(b)) for g, (a, b) in by.items() if 0 < sum(b) < len(b)}
    return (float(np.mean(list(per.values()))) if per else float("nan")), len(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cond40")
    ap.add_argument("--H", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--targets", nargs="+", default=["active", "onset"])
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    rows = build(HERE / args.data, args.H)
    groups = sorted({r["group"] for r in rows})
    print(f"호출 샘플 {len(rows)}개, task 그룹 {len(groups)}개, H={args.H} env step "
          f"(= 해당 호출의 액션이 실행되는 구간)\n")

    print("=== 타깃별 양성 비율 ===")
    print(f"  {'조건':24s} {'active 양성':>14s} {'onset 양성':>13s}")
    for c in (*TRACKED, COMPOSITE):
        a = sum(r[f"active__{c}"] for r in rows)
        o = sum(r[f"onset__{c}"] for r in rows)
        print(f"  {c:24s} {a:6d} ({a / len(rows):5.1%}) {o:5d} ({o / len(rows):5.1%})")
    print()

    rng = np.random.default_rng(0)
    for r in rows:
        r["random"] = rng.standard_normal(4).astype(np.float32)
    fold_of = {g: i % args.folds for i, g in enumerate(rng.permutation(groups))}

    out = {}
    for target in args.targets:
        print(f"=== 타깃: {target}__  (task 그룹 {args.folds}-fold, {args.seeds}시드, macro AUROC) ===")
        print(f"  {'조건':24s} " + " ".join(f"{f:>14s}" for f in FEATURES))
        for c in (*TRACKED, COMPOSITE):
            y = np.array([r[f"{target}__{c}"] for r in rows], float)
            if y.sum() < 10:
                print(f"  {c:24s} 양성 {int(y.sum())}개 — 건너뜀")
                continue
            cells = []
            for feat in FEATURES:
                vals = []
                for seed in range(args.seeds):
                    sc = np.zeros(len(rows))
                    for f in range(args.folds):
                        te = np.array([fold_of[r["group"]] == f for r in rows])
                        if te.sum() == 0 or y[~te].sum() < 2:
                            continue
                        X = np.stack([r[feat] for r in rows])
                        sc[te] = fit_eval(X[~te], y[~te], X[te], seed)
                    m, n_tasks = macro(sc, y, [r["group"] for r in rows])
                    vals.append(m)
                cells.append((float(np.nanmean(vals)), float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0))
                out[f"{target}|{c}|{feat}"] = cells[-1]
            print(f"  {c:24s} " + " ".join(f"{m:7.3f}±{s:.3f}" for m, s in cells))
        print()

    print("  call_index 를 넘지 못하면 latent 가 시계 이상의 정보를 주지 못한다는 뜻이다.")
    print("  opening 은 C1 정의의 일부이므로 공정한 경쟁자가 아니라 '이 타깃이 무언가로는 예측된다'는 상한이다.")
    path = HERE / "results" / f"violation_probe_{args.tag}.json"
    path.write_text(json.dumps(dict(H=args.H, n_rows=len(rows), n_groups=len(groups),
                                    folds=args.folds, seeds=args.seeds, cells=out),
                               indent=2, ensure_ascii=False, default=float))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
