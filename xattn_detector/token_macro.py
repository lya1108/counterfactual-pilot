"""Do the token-level and VLM latents carry within-task failure signal that suf_mean does not?

Everything so far used suf_mean, the action-expert tokens averaged. The collected token data also holds the 50
action tokens separately and the VLM prefix (image tokens pooled 2x2, language tokens whole, the latter
carrying the discretised robot state). train_xattn.py already compared models over those, but reported POOLED
AUROC under a seed-permutation split; pooled turned out to measure mostly the score offset between tasks, so
those numbers cannot answer this question.

This recomputes the comparison under the split the online buffer uses (a hash of the task key, so a task is on
one side only) and reports TASK-MACRO AUROC, which is the statistic the conclusion now rests on. No new data is
collected -- it reads xattn_detector/data/tokens, already on disk.

  logreg       suf_mean                     the incumbent
  attnpool     50 action tokens             learned weighting inside a call
  xattn        action tokens x VLM tokens   cross-attention, plus a residual copy of suf_mean
  vlm_logreg   VLM prefix, mean-pooled      the VLM latent on its own -- no action-expert input at all

vlm_logreg is the direct test of whether the VLM side holds anything: it has no access to the action expert, so
whatever it scores comes from image, language and state tokens alone.

    conda activate fasi
    python xattn_detector/token_macro.py --seeds 8
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

from buffer import _split_of  # noqa: E402  (same hash the online buffer uses)
from calibration import brier_parts, sigmoid  # noqa: E402
from models import build, n_params  # noqa: E402
from train_detector import auroc  # noqa: E402

CELLS = ("logreg", "attnpool", "xattn", "vlm_logreg")


class VlmLogReg(nn.Module):
    """Mean-pool the VLM prefix tokens, then one linear unit. No action-expert input."""

    def __init__(self, d_pre: int = 2048, dropout: float = 0.0, **_):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_pre, 1)

    def forward(self, suf, pre, pre_mask=None):
        if pre_mask is not None:
            m = pre_mask.unsqueeze(-1).float()
            pooled = (pre * m).sum(2) / m.sum(2).clamp_min(1)
        else:
            pooled = pre.mean(2)
        return self.head(self.drop(pooled)).squeeze(-1)


def load_tokens(tokens: Path, calls: int, want_pre: bool):
    rows = [json.loads(l) for l in (tokens / "episodes.jsonl").read_text().splitlines() if l.strip()]
    eps = []
    for r in rows:
        if not r.get("repro_ok"):
            continue
        z = np.load(tokens / r["npz"])
        e = dict(suite=r["suite"], task=r["task"], init=r["init"], y=int(not r["success"]),
                 split=_split_of(f"{r['suite']}/t{r['task']}"),
                 suf=z["suf_mean"][:calls].astype(np.float16))
        if want_pre:
            img, lang, lm = z["pre_img"][:calls], z["pre_lang"][:calls], z["lang_mask"][:calls]
            e["pre"] = np.concatenate([img, lang], 1).astype(np.float16)
            e["pre_mask"] = np.concatenate([np.ones(img.shape[:2], bool), lm], 1)
        eps.append(e)
    return eps


def collate(batch, device, need_pre):
    suf = torch.from_numpy(np.stack([e["suf"] for e in batch])).to(device).float()
    y = torch.tensor([e["y"] for e in batch], dtype=torch.float32, device=device)
    if not need_pre:
        return suf, None, None, y
    P = max(e["pre"].shape[1] for e in batch)
    B, T = suf.shape[0], suf.shape[1]
    pre = np.zeros((B, T, P, batch[0]["pre"].shape[2]), np.float16)
    msk = np.zeros((B, T, P), bool)
    for i, e in enumerate(batch):
        p = e["pre"].shape[1]
        pre[i, :, :p] = e["pre"]
        msk[i, :, :p] = e["pre_mask"]
    return suf, torch.from_numpy(pre).to(device).float(), torch.from_numpy(msk).to(device), y


def macro_of(p, y, keys):
    by = collections.defaultdict(lambda: ([], []))
    for pi, yi, k in zip(p, y, keys):
        by[k][0].append(pi)
        by[k][1].append(yi)
    per = {k: auroc(np.array(ss), np.array(yy)) for k, (ss, yy) in by.items()
           if 0 < sum(yy) < len(yy)}
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


@torch.no_grad()
def predict(net, eps, device, need_pre, batch):
    net.eval()
    out = []
    for s in range(0, len(eps), batch):
        suf, pre, msk, _ = collate(eps[s:s + batch], device, need_pre)
        out.append(sigmoid(net(suf, pre, msk).cpu().numpy()).mean(1))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="data/tokens")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--calls", type=int, default=8)
    ap.add_argument("--models", nargs="+", default=list(CELLS))
    ap.add_argument("--d", type=int, default=128); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    tokens = HERE / args.tokens
    device = "cuda" if torch.cuda.is_available() else "cpu"
    need_pre = any(m in ("xattn", "vlm_logreg") for m in args.models)
    eps = load_tokens(tokens, args.calls, need_pre)
    tr = [e for e in eps if e["split"] == "train"]
    va = [e for e in eps if e["split"] == "val"]
    te = [e for e in eps if e["split"] == "test"]
    kte = [f"{e['suite']}/t{e['task']}" for e in te]
    yte = np.array([e["y"] for e in te], float)
    ktr = [f"{e['suite']}/t{e['task']}" for e in tr]
    ytr_np = np.array([e["y"] for e in tr], float)
    print(f"버퍼와 동일한 해시 분할   train {len(tr)}  val {len(va)}  test {len(te)} "
          f"(미학습 task {len(set(kte))}개, 실패 {int(yte.sum())})")
    print(f"입력 — suf {tr[0]['suf'].shape}" + (f"   pre {tr[0]['pre'].shape}" if need_pre else "") + "\n")

    print(f"  {'model':12s} {'params':>9s} {'macro(미학습)':>16s} {'pooled':>14s} {'Brier':>9s} "
          f"{'train macro':>12s}")
    out = {}
    for name in args.models:
        np_ = None
        ms, ps, bs, tms = [], [], [], []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            net = (VlmLogReg(dropout=args.dropout) if name == "vlm_logreg"
                   else build(name, d=args.d, heads=args.heads, dropout=args.dropout, freeze_down=True)).to(device)
            np_ = np_ or n_params(net)
            opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad],
                                   lr=args.lr, weight_decay=args.weight_decay)
            rng = np.random.default_rng(seed)
            nf = sum(e["y"] for e in tr)
            best, best_state, bad = -1.0, None, 0
            for _ in range(args.epochs):
                net.train()
                for s in range(0, len(tr), args.batch):
                    chunk = [tr[i] for i in rng.permutation(len(tr))[s:s + args.batch]]
                    if not chunk:
                        continue
                    suf, pre, msk, y = collate(chunk, device, need_pre)
                    logit = net(suf, pre, msk)
                    w = torch.where(y[:, None] > 0, 1.0 / max(nf, 1), 1.0 / max(len(tr) - nf, 1)).expand_as(logit)
                    loss = (nn.functional.binary_cross_entropy_with_logits(
                        logit, y[:, None].expand_as(logit), reduction="none") * w).sum() / w.sum()
                    opt.zero_grad(); loss.backward(); opt.step()
                pv = predict(net, va, device, need_pre, args.batch)
                a = macro_of(pv, [e["y"] for e in va], [f"{e['suite']}/t{e['task']}" for e in va])[0]
                if a > best + 1e-4:
                    best, best_state, bad = a, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= args.patience:
                        break
            net.load_state_dict(best_state)
            p = predict(net, te, device, need_pre, args.batch)
            ms.append(macro_of(p, yte, kte)[0])
            ps.append(auroc(p, yte))
            bs.append(brier_parts(p, yte)["brier"])
            tms.append(macro_of(predict(net, tr, device, need_pre, args.batch), ytr_np, ktr)[0])
        f = lambda v: (float(np.nanmean(v)), float(np.nanstd(v, ddof=1)))
        out[name] = dict(n_params=np_, macro=f(ms), pooled=f(ps), brier=f(bs), train_macro=f(tms))
        o = out[name]
        print(f"  {name:12s} {np_:>9,} {o['macro'][0]:9.3f}±{o['macro'][1]:.3f} "
              f"{o['pooled'][0]:7.3f}±{o['pooled'][1]:.3f} {o['brier'][0]:9.4f} {o['train_macro'][0]:12.3f}",
              flush=True)

    path = HERE / "results" / f"token_macro_{args.tag}.json"
    path.write_text(json.dumps(dict(seeds=args.seeds, calls=args.calls, n_test=len(te),
                                    n_test_tasks=len(set(kte)), cells=out), indent=2,
                               ensure_ascii=False, default=float))
    print(f"\n-> {path}")
    print("  macro(미학습) = 해시 분할로 통째로 빠진 task들에서, task별 AUROC의 평균.")
    print("  vlm_logreg 는 action expert 입력이 전혀 없으므로 VLM latent 단독 성능이다.")


if __name__ == "__main__":
    main()
