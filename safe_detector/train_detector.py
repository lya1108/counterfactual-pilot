"""SAFE step 2: train a failure detector on the pi0.5 latents collected by collect_rollouts.py.

A detector reads one latent per VLA call and emits a failure score for that call. Labels are episode-level
(the rollout ended in failure), broadcast to every call of the episode, so the score is "this rollout is going
to fail", not "this step is wrong".

Two shortcuts have to be measured, not assumed away, because successful LIBERO episodes stop the moment the
task is solved while failures run to the timeout:
  `time`   -- a detector given ONLY the call index. Any gap between it and a latent detector is what the
              latents actually buy; a latent detector that fails to beat it has learned the clock.
  `random` -- scores drawn from noise, the chance floor for the same split.

    conda activate fasi
    python safe_detector/train_detector.py --data data/libero_spatial --feature vlm_mean --model lstm
    python safe_detector/train_detector.py --data data --split task --model time      # all suites, unseen tasks

sklearn is not installed in the `fasi` env (and that env is 중범's, so nothing is installed into it); the models
here are plain torch and the AUROC is the rank statistic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
FEATURES = ("vlm_mean", "vlm_last", "act_last", "act_mean")


# ---------------------------------------------------------------------------- data
def load_episodes(roots: list[Path], feature: str) -> list[dict]:
    """Every episode under the given directories, as {meta..., x: (T, D) float32, y: int}."""
    eps = []
    for root in roots:
        index = root / "episodes.jsonl"
        if not index.exists():
            continue
        for line in index.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            z = np.load(root / rec["npz"])
            if feature == "time":
                n = len(z["step"])
                x = np.stack([np.arange(n), np.asarray(z["step"], np.float32)], 1).astype(np.float32)
            elif feature == "concat":
                x = np.concatenate([z[k].astype(np.float32) for k in FEATURES], 1)
            else:
                x = z[feature].astype(np.float32)
            eps.append({**rec, "x": x, "y": int(not rec["success"]), "root": root.name})
    return eps


def split_episodes(eps: list[dict], how: str, holdout: float, seed: int = 0):
    """Grouped split so no test episode shares its group with a training one.

    The groups are shuffled before the cut. Taking them in sorted order would make the `task` split alphabetical,
    which puts whole suites on one side (libero_10 < libero_goal < libero_object < libero_spatial) and turns a
    held-out-task evaluation into a held-out-suite one.
    """
    if how == "init":
        key = lambda e: e["init"]
    elif how == "task":
        key = lambda e: (e["suite"], e["task"])
    else:
        raise ValueError(how)
    groups = sorted({key(e) for e in eps}, key=str)
    order = np.random.default_rng(seed).permutation(len(groups))
    groups = [groups[i] for i in order]
    n_test = max(1, int(round(len(groups) * holdout)))
    test_groups = set(groups[-n_test:])
    tr = [e for e in eps if key(e) not in test_groups]
    te = [e for e in eps if key(e) in test_groups]
    return tr, te, sorted(map(str, test_groups))


# ---------------------------------------------------------------------------- models
class LogReg(nn.Module):
    """Per-call linear probe: no memory, so its score depends on the current latent alone."""

    def __init__(self, d):
        super().__init__()
        self.f = nn.Linear(d, 1)

    def forward(self, x):  # (B, T, D) -> (B, T)
        return self.f(x).squeeze(-1)


class LSTMDet(nn.Module):
    """Causal recurrent detector: the score at call t may use calls 0..t and nothing later."""

    def __init__(self, d, hidden=128):
        super().__init__()
        self.rnn = nn.LSTM(d, hidden, batch_first=True)
        self.f = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.f(h).squeeze(-1)


# ---------------------------------------------------------------------------- metrics
def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = labels.sum(), (1 - labels).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


# ---------------------------------------------------------------------------- training
def pad_batch(eps, device):
    T = max(len(e["x"]) for e in eps)
    D = eps[0]["x"].shape[1]
    x = torch.zeros(len(eps), T, D)
    m = torch.zeros(len(eps), T)
    y = torch.zeros(len(eps))
    for i, e in enumerate(eps):
        t = len(e["x"])
        x[i, :t] = torch.from_numpy(e["x"])
        m[i, :t] = 1.0
        y[i] = e["y"]
    return x.to(device), m.to(device), y.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data"],
                    help="collection dirs (a parent dir is expanded to its subdirs holding episodes.jsonl)")
    ap.add_argument("--feature", default="vlm_mean", choices=[*FEATURES, "concat", "time"])
    ap.add_argument("--model", default="lstm", choices=["lstm", "logreg", "time", "random"])
    ap.add_argument("--max-calls", type=int, default=0,
                    help="truncate every episode to its first K calls and drop shorter ones. LIBERO successes stop "
                         "when solved (~10 calls) while failures always run to the timeout (28 calls), so on full "
                         "episodes the call index alone nearly separates the classes. With a common prefix every "
                         "episode has the same index sequence, the `time` baseline is exactly chance, and the "
                         "question becomes the one worth asking: is failure already readable this early?")
    ap.add_argument("--split", default="init", choices=["init", "task"])
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    roots = []
    for d in args.data:
        p = Path(d)
        if not p.is_absolute():
            p = HERE / p
        roots += [p] if (p / "episodes.jsonl").exists() else sorted(q for q in p.iterdir() if (q / "episodes.jsonl").exists())
    if not roots:
        raise SystemExit(f"no episodes.jsonl under {args.data}")

    feature = "time" if args.model == "time" else args.feature
    eps = load_episodes(roots, feature)
    if not eps:
        raise SystemExit("no episodes loaded")
    if args.max_calls:
        kept = [e for e in eps if len(e["x"]) >= args.max_calls]
        print(f"prefix of {args.max_calls} calls: kept {len(kept)}/{len(eps)} episodes "
              f"(dropped {len(eps) - len(kept)} shorter than that)")
        for e in kept:
            e["x"] = e["x"][: args.max_calls]
            e["n_calls_full"] = e["n_calls"]  # lead time is still measured back from the real episode end
            e["n_calls"] = args.max_calls
        eps = kept
        if not eps:
            raise SystemExit("--max-calls dropped every episode")
    tr, te, test_groups = split_episodes(eps, args.split, args.holdout, args.seed)
    nf = lambda s: sum(e["y"] for e in s)
    print(f"episodes {len(eps)}  failure {nf(eps)}  |  train {len(tr)} (fail {nf(tr)})  test {len(te)} (fail {nf(te)})")
    print(f"feature={feature} dim={eps[0]['x'].shape[1]}  model={args.model}  split={args.split} holdout={test_groups}")
    if nf(te) == 0 or nf(te) == len(te):
        print("WARNING: the test split has only one class -- AUROC is undefined")

    # standardise on training calls only
    cat = np.concatenate([e["x"] for e in tr])
    mu, sd = cat.mean(0), cat.std(0) + 1e-6
    for e in eps:
        e["x"] = (e["x"] - mu) / sd

    if args.model == "random":
        rng = np.random.default_rng(args.seed)
        for e in te:
            e["score"] = rng.standard_normal(len(e["x"])).astype(np.float32)
    else:
        d = eps[0]["x"].shape[1]
        net = (LogReg(d) if args.model == "logreg" else LSTMDet(d)).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        x, m, y = pad_batch(tr, device)
        w = m * torch.where(y[:, None] > 0, 1.0 / max(nf(tr), 1), 1.0 / max(len(tr) - nf(tr), 1))
        tgt = y[:, None].expand_as(m)
        for ep in range(args.epochs):
            opt.zero_grad()
            loss = (nn.functional.binary_cross_entropy_with_logits(net(x), tgt, reduction="none") * w).sum() / w.sum()
            loss.backward()
            opt.step()
            if (ep + 1) % 20 == 0:
                print(f"  epoch {ep + 1:3d}  loss {loss.item():.4f}")
        net.eval()
        with torch.no_grad():
            xt, mt, _ = pad_batch(te, device)
            st = net(xt).cpu().numpy()
        for i, e in enumerate(te):
            e["score"] = st[i, : len(e["x"])].astype(np.float32)

    # per-call AUROC (every call of a failed episode counts as positive) and per-episode (max score)
    flat_s = np.concatenate([e["score"] for e in te])
    flat_y = np.concatenate([np.full(len(e["score"]), e["y"]) for e in te])
    ep_s = np.array([e["score"].max() for e in te])
    ep_y = np.array([e["y"] for e in te])
    print(f"\nAUROC  per-call {auroc(flat_s, flat_y):.3f}   per-episode(max) {auroc(ep_s, ep_y):.3f}")

    # AUROC within a single call index: successes and failures compared at the same point in the rollout, so
    # episode length cannot contribute. Reported only where both classes still have episodes alive.
    longest = max(len(e["score"]) for e in te)
    strat = []
    for i in range(longest):
        s = np.array([e["score"][i] for e in te if len(e["score"]) > i])
        y = np.array([e["y"] for e in te if len(e["score"]) > i])
        if y.sum() and (1 - y).sum():
            strat.append((i, auroc(s, y), int((1 - y).sum()), int(y.sum())))
    if strat:
        print("  by call index (alive S/F):  " +
              "  ".join(f"t{i}={a:.2f}({ns}/{nf})" for i, a, ns, nf in strat[:12]))
    else:
        print("  by call index: no index has both classes alive -- length alone separates them")

    tag = args.tag or f"{args.model}_{feature}_{args.split}"
    outdir = HERE / "results"
    outdir.mkdir(exist_ok=True)
    np.savez_compressed(outdir / f"scores_{tag}.npz",
                        **{f"s{i}": e["score"] for i, e in enumerate(te)},
                        **{f"step{i}": np.arange(len(e["score"])) for i, e in enumerate(te)})
    meta = dict(args=vars(args), feature=feature, test_groups=test_groups,
                auroc_call=auroc(flat_s, flat_y), auroc_episode=auroc(ep_s, ep_y),
                auroc_by_call=[{"call": i, "auroc": a, "n_success": ns, "n_failure": nf} for i, a, ns, nf in strat],
                episodes=[{**{k: e[k] for k in ("suite", "task", "init", "success", "t", "n_calls", "exec")},
                           "n_calls_full": e.get("n_calls_full", e["n_calls"])} for e in te])
    (outdir / f"scores_{tag}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"scores -> {outdir / f'scores_{tag}.npz'}")


if __name__ == "__main__":
    main()
