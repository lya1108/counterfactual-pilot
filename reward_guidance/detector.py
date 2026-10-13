"""Freeze one C1 head from cond200 so the guidance experiments all score with the SAME function.

Everything upstream refit the detector inside each cross-validation fold, which is right for measuring
generalisation and useless for steering: a rollout needs one fixed w. This trains on 16 task groups and
holds out the four the guidance experiments roll out in, so the frozen head has never seen those tasks.

The feature is exactly the one that reached 0.891 unseen-task macro AUROC: `suf_mean`, the action expert's
suffix hidden state averaged over the 10 denoising steps and the 50 action tokens (1024-D). The target is
the canonical prospective C1 label -- onset in (t, t+10], episode-global, calls where C1 is already active
dropped -- and 10 environment steps is exactly one chunk's execution interval, so the label is caused by the
very chunk being scored. That is what makes this head a candidate reward and not just a monitor.

    conda activate fasi
    python reward_guidance/detector.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
X = HERE.parent / "xattn_detector"
sys.path.insert(0, str(X))
sys.path.insert(0, str(HERE.parent / "safe_detector"))

from c1_depth import build, macro_pooled, target_horizon  # noqa: E402

# high C1-rate groups spanning three suites: enough positives that a held-out rollout meets the condition often
EVAL_GROUPS = ("libero_spatial/t9", "libero_object/t6", "libero_object/t0", "libero_goal/t2")
H, GUARD = 10, 0


def fit(Xtr, ytr, seed=0, epochs=300, lr=1e-2, wd=1.0):
    """Same optimiser, class weighting and standardisation as prospective_c1.fit_eval -- kept identical on purpose."""
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
    return dict(mu=mu.astype(np.float32), sd=sd.astype(np.float32),
                w=net.weight.detach().numpy().ravel().astype(np.float32),
                b=float(net.bias.detach().item()))


class C1Head:
    """logit of P(C1 onset within the next 10 environment steps). Higher = riskier."""

    def __init__(self, path):
        d = torch.load(path, map_location="cpu", weights_only=False)
        self.mu, self.sd, self.w, self.b = d["mu"], d["sd"], d["w"], d["b"]
        self.train_groups, self.eval_groups = d["train_groups"], d["eval_groups"]
        self.meta = d["meta"]

    def numpy(self, x):
        x = np.atleast_2d(np.asarray(x, np.float32))
        return ((x - self.mu) / self.sd) @ self.w + self.b

    def torch_(self, x, device=None, dtype=torch.float32):
        """Differentiable branch: x is (..., 1024) and may carry an autograd graph."""
        dev = device or x.device
        mu = torch.as_tensor(self.mu, device=dev, dtype=dtype)
        sd = torch.as_tensor(self.sd, device=dev, dtype=dtype)
        w = torch.as_tensor(self.w, device=dev, dtype=dtype)
        return ((x.to(dtype) - mu) / sd) @ w + self.b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(X / "data/cond200"))
    ap.add_argument("--out", default=str(HERE / "results/c1_head.pt"))
    args = ap.parse_args()

    rows_all = build(Path(args.data), hist=3, want_tokens=False)
    rows, y = target_horizon(rows_all, H, GUARD)
    groups = sorted({r["group"] for r in rows})
    ev = [g for g in EVAL_GROUPS if g in groups]
    tr_mask = np.array([r["group"] not in ev for r in rows])

    Xall = np.stack([r["suf_mean"] for r in rows])
    print(f"표본 {len(rows)}   양성 {int(y.sum())} ({y.mean():.1%})   task group {len(groups)}")
    print(f"학습 group {len(groups) - len(ev)}   평가 group {ev}")
    print(f"학습 {int(tr_mask.sum())} (양성 {int(y[tr_mask].sum())})   "
          f"평가 {int((~tr_mask).sum())} (양성 {int(y[~tr_mask].sum())})\n")

    head = fit(Xall[tr_mask], y[tr_mask])
    sc_te = ((Xall[~tr_mask] - head["mu"]) / head["sd"]) @ head["w"] + head["b"]
    gl = [r["group"] for r, m in zip(rows, ~tr_mask) if m]
    macro, pooled, per = macro_pooled(sc_te, y[~tr_mask], gl)
    print("미학습 task 성능 (이 고정 head 하나로):")
    for g, a in sorted(per.items()):
        n = sum(1 for x in gl if x == g)
        p = int(y[~tr_mask][np.array([x == g for x in gl])].sum())
        print(f"  {g:22s} n={n:4d} 양성={p:3d}  AUROC {a:.3f}")
    print(f"  {'macro':22s}                      {macro:.3f}   (pooled {pooled:.3f})")

    torch.save({**head, "train_groups": [g for g in groups if g not in ev], "eval_groups": ev,
                "meta": dict(H=H, guard=GUARD, feature="suf_mean", dim=int(Xall.shape[1]),
                             n_train=int(tr_mask.sum()), n_pos_train=int(y[tr_mask].sum()),
                             macro_heldout=macro, pooled_heldout=pooled, per_task=per)},
               args.out)
    json.dump(dict(macro_heldout=macro, pooled_heldout=pooled, per_task=per,
                   eval_groups=ev, n_train=int(tr_mask.sum()), n_pos_train=int(y[tr_mask].sum())),
              open(HERE / "results/c1_head.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
