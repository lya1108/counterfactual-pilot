"""SUPERSEDED -- do not use for new results. Kept only because earlier numbers came from it.

Its target judged the debounce inside each window, so an onset whose 3 consecutive steps straddled the window
edge was scored as a negative: at H=10 on the 200-episode set it counted 300 positives where the episode-global
definition counts 353, and reported 0.914 where the correct figure is 0.891. c1_depth.build() defines the onset
once per episode and measures its distance from each call, which is what every result from here on uses.

Original docstring follows.

Pure prospective prediction of C1: given that the gripper is NOT yet closing on nothing, will it be in the
next 10 environment steps?

The previous probe let in calls where C1 was already true at the moment of the call. Scoring those is reading
the present, not predicting, and it is the obvious way to inflate 0.888. Here every sample with C1 active at
step t is dropped, so a positive can only be a violation that has not happened yet.

The comparison that matters is against the environment state at the same instant. If the current state already
says a bad grasp is coming, the latent has to beat that to be worth anything:

  state   opening and how fast it is changing, end-effector position, distance to the nearest object and to the
          nearest goal / non-goal object separately, where that object sits relative to the gripper, and the
          contact flags. Everything a hand-written monitor could read.
  opening the gripper gap alone -- half of C1's definition, so a reference rather than a competitor.
  call_index  the clock.
  random  the floor.

The task-level split is repeated, not fixed. One 5-fold assignment over 20 task groups gives four tasks per
fold, and the spread between assignments is larger than the spread between seeds within one -- the earlier
"+-0.000" came from holding the split still, not from precision.

40 episodes is a preliminary sample and the numbers here are read as such.

    conda activate fasi
    python xattn_detector/prospective_c1.py --repeats 20
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

COND = "C1_grasp_empty_close"
FEATURES = ("suf_mean", "state", "opening", "call_index", "random")


def state_vector(z, i, goal_mask):
    """What a hand-written monitor could read at this instant. Nothing from the future."""
    eef = z["eef"][i]
    pos = z["obj_pos"][i]
    d = np.linalg.norm(pos - eef[None, :], axis=1)
    j = int(np.argmin(d))
    dg = d[goal_mask].min() if goal_mask.any() else 9.0
    dn = d[~goal_mask].min() if (~goal_mask).any() else 9.0
    op = z["opening"]
    dop = float(op[i] - op[max(i - 3, 0)])  # how fast the gripper is closing
    touch, support = z["obj_touch"][i], z["obj_support"][i]
    return np.array([op[i], dop, *eef, d.min(), dg, dn, *(pos[j] - eef),
                     float(touch.any()), float((touch & goal_mask).any()), float(touch.sum()),
                     float(support.any())], np.float32)


def build(root: Path, H: int, debounce: int = 3):
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

        for k, cs in enumerate(z["call_step"].astype(int)):
            i_now = pos_of.get(int(cs))
            if i_now is None:
                continue                      # the very first call has no logged state yet
            if c1[i_now]:
                continue                      # already violating -- reading the present, not predicting
            w0, w1 = pos_of.get(int(cs) + 1), pos_of.get(int(cs) + H)
            if w0 is None:
                continue
            if w1 is None:
                w1 = len(step) - 1            # episode ended inside the window
            win = c1[w0:w1 + 1]
            run, onset = 0, False
            for b in win:
                run = run + 1 if b else 0
                if run >= debounce:
                    onset = True
                    break
            rows.append(dict(group=f"{r['suite']}/t{r['task']}",
                             episode=f"{r['suite']}_t{r['task']}_i{r['init']}", call=k, call_step=int(cs),
                             y_onset=onset, y_any=bool(win.any()),
                             suf_mean=z["suf_mean"][k].astype(np.float32),
                             state=state_vector(z, i_now, goal_mask),
                             opening=np.array([z["opening"][i_now]], np.float32),
                             call_index=np.array([k, cs], np.float32)))
    return rows


def fit_eval(Xtr, ytr, Xte, seed, epochs=300, lr=1e-2, wd=1.0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    net = nn.Linear(Xtr.shape[1], 1)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    xt = torch.from_numpy((Xtr - mu) / sd).float()
    yt = torch.from_numpy(ytr).float()[:, None]
    w = torch.where(yt > 0, 1.0 / max(ytr.sum(), 1), 1.0 / max((1 - ytr).sum(), 1))
    for _ in range(epochs):
        opt.zero_grad()
        loss = (nn.functional.binary_cross_entropy_with_logits(net(xt), yt, reduction="none") * w).sum() / w.sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return net(torch.from_numpy((Xte - mu) / sd).float()).numpy().ravel()


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
    ap.add_argument("--repeats", type=int, default=20, help="task 분할을 몇 번 다시 뽑을지")
    ap.add_argument("--target", default="y_onset", choices=["y_onset", "y_any"])
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    rows = build(HERE / args.data, args.H)
    groups = sorted({r["group"] for r in rows})
    y = np.array([r[args.target] for r in rows], float)
    gl = [r["group"] for r in rows]
    print(f"C1(t)=0 인 호출 샘플 {len(rows)}개   task 그룹 {len(groups)}개   H={args.H}")
    print(f"타깃 {args.target}: 양성 {int(y.sum())} ({y.mean():.1%})")
    per_task = collections.Counter(g for g, yy in zip(gl, y) if yy > 0)
    both = sum(1 for g in groups if 0 < per_task[g] < sum(1 for x in gl if x == g))
    print(f"양성을 가진 task {len(per_task)}/{len(groups)}, 양·음 모두 가진 task {both}\n")

    rng = np.random.default_rng(0)
    for r in rows:
        r["random"] = rng.standard_normal(4).astype(np.float32)
    feats = {f: np.stack([r[f] for r in rows]) for f in FEATURES}

    print(f"=== macro AUROC — task 분할을 {args.repeats}회 다시 뽑음 ({args.folds}-fold) ===")
    print(f"  {'feature':12s} {'평균 ± sd':>16s} {'최소–최대':>15s} {'평가 task 수':>12s}")
    out = {}
    for f in FEATURES:
        X = feats[f]
        vals, ntasks = [], []
        for rep in range(args.repeats):
            r2 = np.random.default_rng(100 + rep)
            fold_of = {g: i % args.folds for i, g in enumerate(r2.permutation(groups))}
            sc = np.zeros(len(rows))
            for fold in range(args.folds):
                te = np.array([fold_of[r["group"]] == fold for r in rows])
                if te.sum() == 0 or y[~te].sum() < 2 or (1 - y[~te]).sum() < 2:
                    sc[te] = 0.0
                    continue
                sc[te] = fit_eval(X[~te], y[~te], X[te], seed=rep)
            m, n = macro(sc, y, gl)
            vals.append(m)
            ntasks.append(n)
        v = np.array(vals, float)
        out[f] = dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v, ddof=1)),
                      lo=float(np.nanmin(v)), hi=float(np.nanmax(v)), n_tasks=float(np.mean(ntasks)))
        print(f"  {f:12s} {out[f]['mean']:9.3f} ± {out[f]['sd']:.3f} "
              f"{out[f]['lo']:7.3f}–{out[f]['hi']:.3f} {out[f]['n_tasks']:12.0f}")

    d = out["suf_mean"]["mean"] - out["state"]["mean"]
    se = np.sqrt(out["suf_mean"]["sd"] ** 2 + out["state"]["sd"] ** 2) / np.sqrt(args.repeats)
    print(f"\n  suf_mean − state = {d:+.3f}   (분할 간 변동 기준 표준오차 약 {se:.3f})")
    print("  40 에피소드 예비 결과입니다. task 그룹 20개를 5-fold로 나누면 fold당 4개뿐이라,")
    print("  위 sd 는 추정의 정밀도가 아니라 '어느 task가 테스트로 빠지느냐'의 변동을 보여줍니다.")

    path = HERE / "results" / f"prospective_c1_{args.tag}.json"
    path.write_text(json.dumps(dict(args=vars(args), n_rows=len(rows), n_pos=int(y.sum()),
                                    n_groups=len(groups), cells=out), indent=2,
                               ensure_ascii=False, default=float))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
