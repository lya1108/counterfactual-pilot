"""Offline comparison of LogReg / LSTM / AttnPool / CrossAttn on the token data, in safe_detector's own split.

The four sit on two different axes (see models.py): LSTM is the only one that looks across the 8 VLA calls,
AttnPool and CrossAttn are the only ones that look inside a call's 50 action tokens. So LSTM vs AttnPool is not
two answers to one question -- they can both help, or neither.

The split comes from safe_detector/train_detector.py so the groups are literally the same ones the LogReg and
LSTM numbers were computed on -- `init` holds out fresh rollouts of trained tasks, `task` holds out whole tasks,
both shuffled by seed before the cut.

Two LSTM numbers appear in the output and they are not interchangeable:
  retrained  this run, same data, same split, same training protocol as the other three
  cited      safe_detector/results/multiseed.json, read from disk, NOT re-run. Same architecture and split,
             but full-batch 60 epochs with no validation slice, no early stopping and no weight decay, on
             pooled `act_mean` vectors rather than tokens. A difference between the two is a protocol
             difference, not a finding.

What is reported per cell, because AUROC alone hid the LSTM's problem last time:
  test AUROC / Brier / R²      how good the detector is
  train AUROC / Brier          the same detector on its own training episodes
  gap                          train minus test -- the LSTM scored 1.000 on train and memorised outright
  reliability / resolution     Murphy decomposition of the Brier score
  inference                    detector forward for one VLA call, batch 1 (see --time-runs for the scope)

Episodes whose replay did not reproduce the recorded trajectory are excluded by default: their latents belong
to a slightly different rollout than the label does. `--include-unverified` runs with them for comparison.

    conda activate fasi
    python xattn_detector/train_xattn.py --seeds 0-7
    python xattn_detector/train_xattn.py --seeds 0-7 --include-unverified --tag withbad
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "safe_detector"))
sys.path.insert(0, str(HERE))

from train_detector import auroc, split_episodes  # noqa: E402  (same split as the earlier experiment)

from models import build, n_params  # noqa: E402
sys.path.insert(0, str(HERE.parent / "safe_detector"))
from calibration import brier_parts, episode_prob, sigmoid  # noqa: E402

CELLS = ["logreg", "lstm", "attnpool", "xattn"]


# ---------------------------------------------------------------------------- data
def load(tokens: Path, need_pre: bool, include_unverified: bool, suffix_key: str):
    rows = [json.loads(l) for l in (tokens / "episodes.jsonl").read_text().splitlines() if l.strip()]
    kept, dropped = [], []
    for r in rows:
        trusted = r.get("repro_ok", False) or r.get("label_verified", False)
        (kept if (trusted or include_unverified) else dropped).append(r)
    eps = []
    for r in kept:
        z = np.load(tokens / r["npz"])
        e = dict(suite=r["suite"], task=r["task"], init=r["init"], success=r["success"],
                 y=int(not r["success"]), n_calls_full=r["n_calls_full"], repro_ok=r.get("repro_ok", False),
                 suf=z[suffix_key].astype(np.float16))
        if need_pre:
            img, lang, lm = z["pre_img"], z["pre_lang"], z["lang_mask"]
            e["pre"] = np.concatenate([img, lang], 1).astype(np.float16)
            e["pre_mask"] = np.concatenate([np.ones(img.shape[:2], bool), lm], 1)
        eps.append(e)
    return eps, dropped


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


# ---------------------------------------------------------------------------- train
def run(model_name, tr, va, te, device, need_pre, args, seed):
    torch.manual_seed(seed)
    net = build(model_name, d=args.d, heads=args.heads, dropout=args.dropout,
                freeze_down=not args.train_down).to(device)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=args.lr,
                           weight_decay=args.weight_decay)
    rng = np.random.default_rng(seed)
    nf = sum(e["y"] for e in tr)
    w_pos, w_neg = 1.0 / max(nf, 1), 1.0 / max(len(tr) - nf, 1)

    best, best_state, bad = -1.0, None, 0
    for ep in range(args.epochs):
        net.train()
        order = rng.permutation(len(tr))
        for s in range(0, len(tr), args.batch):
            chunk = [tr[i] for i in order[s:s + args.batch]]
            suf, pre, msk, y = collate(chunk, device, need_pre)
            logit = net(suf, pre, msk)
            w = torch.where(y[:, None] > 0, w_pos, w_neg).expand_as(logit)
            loss = (nn.functional.binary_cross_entropy_with_logits(
                logit, y[:, None].expand_as(logit), reduction="none") * w).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        s_va, y_va = predict(net, va, device, need_pre, args.batch)
        a = auroc(episode_prob(s_va), y_va)
        if a > best + 1e-4:
            best, best_state, bad = a, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net, best


@torch.no_grad()
def predict(net, eps, device, need_pre, batch):
    net.eval()
    out = []
    for s in range(0, len(eps), batch):
        chunk = eps[s:s + batch]
        suf, pre, msk, _ = collate(chunk, device, need_pre)
        out.append(net(suf, pre, msk).cpu().numpy())
    scores = np.concatenate(out)  # (N, T) logits
    return [scores[i] for i in range(len(eps))], np.array([e["y"] for e in eps], float)


def metrics(scores, y, bins=10):
    p = episode_prob(scores)
    flat_s = np.concatenate(scores)
    flat_y = np.concatenate([np.full(len(s), yy) for s, yy in zip(scores, y)])
    return dict(auroc_ep=auroc(p, y), auroc_call=auroc(flat_s, flat_y), **brier_parts(p, y, bins))


@torch.no_grad()
def time_one_call(net, eps, device, need_pre, runs=200):
    """Detector forward for ONE VLA call, batch 1. Excludes pi0.5 itself and the tap, which run anyway."""
    e = eps[0]
    suf = torch.from_numpy(e["suf"][:1][None]).to(device).float()
    pre = torch.from_numpy(e["pre"][:1][None]).to(device).float() if need_pre else None
    msk = torch.from_numpy(e["pre_mask"][:1][None]).to(device) if need_pre else None
    for _ in range(20):
        net(suf, pre, msk)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        net(suf, pre, msk)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="data/tokens")
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--splits", nargs="+", default=["init", "task"])
    ap.add_argument("--models", nargs="+", default=CELLS)
    ap.add_argument("--suffix", default="suf_mean", choices=["suf_mean", "suf_last"])
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--val-frac", type=float, default=0.2, help="slice of train used for early stopping")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--train-down", action="store_true", help="also train the 2048->d projection (+262k params)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--include-unverified", action="store_true")
    ap.add_argument("--time-runs", type=int, default=200)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds += list(range(int(a), int(b) + 1))
        elif part:
            seeds.append(int(part))

    tokens = HERE / args.tokens if not Path(args.tokens).is_absolute() else Path(args.tokens)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    need_pre = "xattn" in args.models
    eps, dropped = load(tokens, need_pre, args.include_unverified, args.suffix)
    nf = sum(e["y"] for e in eps)
    print(f"에피소드 {len(eps)} (실패 {nf} / 성공 {len(eps) - nf})   제외 {len(dropped)}건"
          f"{' — --include-unverified 로 포함됨' if args.include_unverified else ' (재현 미검증)'}")
    print(f"suffix={args.suffix}  device={device}  seeds={seeds}  d={args.d} heads={args.heads} "
          f"dropout={args.dropout} wd={args.weight_decay}\n", flush=True)

    out = {}
    for split in args.splits:
        for name in args.models:
            rows, tms = [], []
            for seed in seeds:
                tr_all, te, _ = split_episodes(eps, split, args.holdout, seed)
                rng = np.random.default_rng(1000 + seed)
                perm = rng.permutation(len(tr_all))
                n_va = max(1, int(len(tr_all) * args.val_frac))
                va = [tr_all[i] for i in perm[:n_va]]
                tr = [tr_all[i] for i in perm[n_va:]]
                np_ = n_params(build(name, d=args.d, heads=args.heads, dropout=args.dropout,
                                     freeze_down=not args.train_down))
                net, _ = run(name, tr, va, te, device, need_pre, args, seed)
                s_te, y_te = predict(net, te, device, need_pre, args.batch)
                s_tr, y_tr = predict(net, tr, device, need_pre, args.batch)
                m_te, m_tr = metrics(s_te, y_te), metrics(s_tr, y_tr)
                rows.append((m_te, m_tr))
                tms.append(time_one_call(net, te, device, need_pre, args.time_runs))
            key = f"{name}_{split}"
            g = lambda which, k: np.array([r[which][k] for r in rows], float)
            out[key] = dict(
                n_params=np_, n_seeds=len(rows), ms_per_call=[float(np.mean(tms)), float(np.std(tms, ddof=1))],
                **{f"test_{k}": [float(g(0, k).mean()), float(g(0, k).std(ddof=1))]
                   for k in ("auroc_ep", "auroc_call", "brier", "r2", "reliability", "resolution")},
                **{f"train_{k}": [float(g(1, k).mean()), float(g(1, k).std(ddof=1))]
                   for k in ("auroc_ep", "brier")})
            o = out[key]
            print(f"  {key:18s} params {np_:>8,}  test ep-AUROC {o['test_auroc_ep'][0]:.3f}±{o['test_auroc_ep'][1]:.3f}"
                  f"  Brier {o['test_brier'][0]:.4f}  | train AUROC {o['train_auroc_ep'][0]:.3f}"
                  f"  gap {o['train_auroc_ep'][0] - o['test_auroc_ep'][0]:+.3f}"
                  f"  | {o['ms_per_call'][0]:.3f} ms", flush=True)

    print("\n" + "=" * 104)
    for split in args.splits:
        print(f"\n=== {split}_prefix8   ({len(seeds)}시드 평균 ± sd) ===")
        print(f"  {'model':10s} {'params':>9s} {'test ep-AUROC':>15s} {'test Brier':>13s} {'R²':>8s} "
              f"{'train AUROC':>12s} {'gap':>7s} {'rel':>7s} {'res':>7s} {'ms/call':>8s}")
        for name in args.models:
            o = out[f"{name}_{split}"]
            print(f"  {name:10s} {o['n_params']:>9,} "
                  f"{o['test_auroc_ep'][0]:8.3f}±{o['test_auroc_ep'][1]:.3f} "
                  f"{o['test_brier'][0]:8.4f}±{o['test_brier'][1]:.3f} {o['test_r2'][0]:8.3f} "
                  f"{o['train_auroc_ep'][0]:12.3f} {o['train_auroc_ep'][0] - o['test_auroc_ep'][0]:+7.3f} "
                  f"{o['test_reliability'][0]:7.4f} {o['test_resolution'][0]:7.4f} {o['ms_per_call'][0]:8.3f}")
    # the earlier LSTM/LogReg run, read from disk rather than re-run -- a different training protocol
    cited = HERE.parent / "safe_detector" / "results" / "multiseed.json"
    if cited.exists():
        c = json.loads(cited.read_text())["cells"]
        print(f"\n=== 인용 (safe_detector/results/multiseed.json, 재학습 아님) ===")
        print("  같은 분할·같은 아키텍처이지만 풀배치 60epoch · early stopping 없음 · weight decay 없음 ·")
        print("  토큰이 아니라 pooled act_mean 입력. 위 표와의 차이는 프로토콜 차이이지 발견이 아니다.")
        print(f"  {'cell':28s} {'ep-AUROC':>15s} {'strat':>15s}")
        for k in ("logreg_act_mean", "lstm_act_mean", "lstm_vlm_mean"):
            for split in args.splits:
                kk = f"{k}_{split}_prefix8"
                if kk in c:
                    e, s = c[kk]["auroc_episode"], c[kk]["strat"]
                    print(f"  {kk:28s} {e[0]:8.3f}±{e[1]:.3f} {s[0]:8.3f}±{s[1]:.3f}")

    print("\n  gap = train ep-AUROC − test ep-AUROC.  이전 LSTM은 전체 에피소드에서 train 1.000 / gap +0.30 이었다.")
    print("  ms/call = detector forward 1회(batch 1, 특징은 이미 계산된 상태). π0.5 forward는 포함하지 않는다.")

    res = HERE / "results"
    res.mkdir(exist_ok=True)
    path = res / f"xattn_{args.tag}.json"
    path.write_text(json.dumps(dict(args=vars(args), n_episodes=len(eps), n_dropped=len(dropped),
                                    dropped=[{k: d[k] for k in ('suite', 'task', 'init')} for d in dropped],
                                    cells=out), indent=2, ensure_ascii=False))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
