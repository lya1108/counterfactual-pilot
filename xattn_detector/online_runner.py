"""Online loop: run pi0.5, bank the experience, retrain the detector offline, promote it only if it validates.

pi0.5 is frozen throughout -- it is loaded, run, and never optimised. The only trainable object is a LogReg
detector over the mean-pooled action-expert vector, which the offline comparison picked: on unseen tasks it
matched cross-attention (0.724 vs 0.724) at a seventh of the latency and 1/177 of the parameters.

Execution and learning are kept apart on purpose:

    collect   loads `detector/current.pt` and only runs it forward, scoring every call of the whole rollout.
              No optimiser is constructed in this mode.
    update    reads the buffer, trains a candidate, and promotes it to `current.pt` ONLY if it beats the
              incumbent on the validation split. A rollout in flight therefore never uses half-trained weights.

Leakage and duplication:
  * splits come from buffer.py's hash of the task key, so the same task cannot sit on both sides, and an
    episode cannot change split as the buffer grows or is reloaded.
  * `update` trains on the train split only; `eval` reports on the test split only.
  * new rollouts can only extend the TRAIN split -- collecting on a test task would poison the comparison, so
    the runner refuses it.
  * (suite, task, init) is unique in the buffer; re-collecting one is refused rather than silently duplicated.

Because training may only extend train-split tasks, the question this answers is "do more rollouts of tasks we
already know improve prediction on tasks we do not?" -- not "does seeing new tasks help", which needs its own run.

    conda activate fasi
    python xattn_detector/online_runner.py seed    --buffer data/online          # ingest the 770 offline rollouts
    python xattn_detector/online_runner.py update  --buffer data/online --tag base
    python xattn_detector/online_runner.py collect --buffer data/online --rollouts 20 --inits 20-21
    python xattn_detector/online_runner.py update  --buffer data/online --tag online1
    python xattn_detector/online_runner.py compare --buffer data/online --before base --after online1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "safe_detector"))

from buffer import Buffer  # noqa: E402
from calibration import brier_parts, sigmoid  # noqa: E402
from train_detector import auroc  # noqa: E402

D_SUF = 1024


# ---------------------------------------------------------------------------- detector
class LogRegDetector(nn.Module):
    """1024 -> 1. One score per VLA call; an episode's probability is the mean of its calls' probabilities."""

    def __init__(self, d: int = D_SUF):
        super().__init__()
        self.head = nn.Linear(d, 1)

    def forward(self, x):  # (..., d) -> (...)
        return self.head(x).squeeze(-1)


def save_detector(net, mu, sd, path: Path, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd, "meta": meta}, path)


def load_detector(path: Path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = LogRegDetector().to(device)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck["mu"], ck["sd"], ck.get("meta", {})


def episode_probs(net, mu, sd, X, M, device):
    """X (B, T, d), M (B, T) -> one probability per episode, averaging over that episode's real calls."""
    with torch.no_grad():
        x = torch.from_numpy((X - mu) / sd).to(device).float()
        logit = net(x).cpu().numpy()
    p = sigmoid(logit)
    return (p * M).sum(1) / np.maximum(M.sum(1), 1)


def collect_split(buf: Buffer, split: str, feature="suf_mean", calls: int = 8):
    """Episodes as (X, M, y, records), each truncated to its first `calls` VLA calls.

    The truncation is not cosmetic. Online rollouts are stored whole -- a success stops when the task is solved
    (9-21 calls here) while a failure runs to the timeout (28-52) -- so episode length alone separates the
    classes. Training on that teaches the detector to read the clock: in the earlier experiment a model given
    only the call index reached ep-AUROC 0.96. Cutting every episode to the same 8 calls removes the signal,
    and matches the offline rollouts seeded into the buffer, which are 8 calls by construction.
    """
    recs = [r for r in buf.query(split=split) if r["n_calls"] >= calls]
    if not recs:
        return None, None, None, []
    xs = [buf.fetch(r, (feature,))[feature][:calls].astype(np.float32) for r in recs]
    T = max(x.shape[0] for x in xs)
    X = np.zeros((len(xs), T, xs[0].shape[1]), np.float32)
    M = np.zeros((len(xs), T), np.float32)
    for i, x in enumerate(xs):
        X[i, : x.shape[0]] = x
        M[i, : x.shape[0]] = 1.0
    return X, M, np.array([r["y"] for r in recs], np.float32), recs


# ---------------------------------------------------------------------------- modes
def mode_seed(args):
    """Ingest the already-collected offline rollouts so online runs extend one buffer, one split scheme."""
    tok = HERE / args.tokens
    rows = [json.loads(l) for l in (tok / "episodes.jsonl").read_text().splitlines() if l.strip()]
    if not args.include_unverified:
        rows = [r for r in rows if r.get("repro_ok")]
    buf = Buffer(HERE / args.buffer, capacity_bytes=args.capacity_gb * 1024 ** 3, group_by=args.group_by)
    have = {(r["suite"], r["task"], r["init"]) for r in buf.records}
    added = skipped = 0
    for r in rows:
        key = (r["suite"], r["task"], r["init"])
        if key in have:
            skipped += 1
            continue
        z = np.load(tok / r["npz"])
        suf = z["suf_mean"].astype(np.float32).mean(1)  # (T, 50, 1024) -> (T, 1024)
        roll = buf.open_rollout(r["suite"], r["task"], r["init"], r["language"], r["goal_names"],
                                rollout_id=f"offline_{r['suite']}_t{r['task']}_i{r['init']}")
        for t in range(suf.shape[0]):
            roll.add_call(step=int(z["step"][t]), suf_mean=suf[t],
                          action_chunk=np.zeros((1, 1), np.float32),  # not recorded offline
                          action_exec=np.zeros((1, 1), np.float32),
                          eef=z["eef"][t], opening=np.float32(z["opening"][t]), score=np.float32(np.nan))
        roll.meta["source"] = "offline"
        buf.close_rollout(roll, success=r["success"], n_steps=int(z["step"][-1]))
        added += 1
    print(f"seed: 추가 {added}, 중복 건너뜀 {skipped}")
    buf.check_splits()
    print(json.dumps(buf.summary(), indent=2, ensure_ascii=False))


def mode_update(args):
    """Train a candidate on the train split; promote only if it beats the incumbent on validation."""
    buf = Buffer.load(HERE / args.buffer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Mtr, ytr, rtr = collect_split(buf, "train", calls=args.calls)
    Xva, Mva, yva, rva = collect_split(buf, "val", calls=args.calls)
    if Xtr is None or Xva is None:
        raise SystemExit("train 또는 val split이 비어 있습니다")
    print(f"train {len(rtr)} (실패 {int(ytr.sum())})  val {len(rva)} (실패 {int(yva.sum())})")

    net, mu, sd, best = fit_logreg(Xtr, Mtr, ytr, Xva, Mva, yva, rva, args, args.seed, device)
    print(f"후보 val {args.select} AUROC {best:.4f}")

    det_dir = HERE / args.buffer / "detector"
    cand = det_dir / f"{args.tag}.pt"
    meta = dict(tag=args.tag, val_auroc=float(best), n_train=len(rtr), n_val=len(rva),
                n_train_fail=int(ytr.sum()), trained=time.time(), seed=args.seed)
    save_detector(net, mu, sd, cand, meta)

    cur = det_dir / "current.pt"
    promote = True
    if cur.exists() and not args.force_promote:
        old_net, old_mu, old_sd, old_meta = load_detector(cur, device)
        pv = episode_probs(old_net, old_mu, old_sd, Xva, Mva, device)
        old_a = macro_auroc(pv, yva, rva)[0] if args.select == "macro" else auroc(pv, yva)
        promote = best > old_a + args.promote_margin
        print(f"현행 val AUROC {old_a:.4f} ({old_meta.get('tag', '?')})  ->  "
              f"{'교체' if promote else '유지'} (마진 {args.promote_margin})")
    if promote:
        save_detector(net, mu, sd, cur, meta)
        print(f"current.pt <- {args.tag}")
    print(f"-> {cand}")


def mode_collect(args):
    """Run rollouts with the CURRENT detector attached, inference only, and bank what they produce."""
    import jb
    jb.enter()
    from fasi.envs.libero_env import LiberoTask
    from fasi.vla.pi05_steer import Pi05Steer
    from token_tap import TokenTap

    buf = Buffer.load(HERE / args.buffer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    det_path = HERE / args.buffer / "detector" / "current.pt"
    net = mu = sd = None
    if det_path.exists():
        net, mu, sd, meta = load_detector(det_path, device)
        for p in net.parameters():
            p.requires_grad_(False)  # execution never trains
        print(f"detector: {meta.get('tag')} (val AUROC {meta.get('val_auroc'):.4f}) — 추론 전용")
    else:
        print("detector 없음 — 점수는 기록하지 않고 수집만 합니다")

    have = {(r["suite"], r["task"], r["init"]) for r in buf.records}
    targets = []
    for suite in args.suites:
        for task in parse_ids(args.tasks):
            if buf.split_for(suite, task, 0) != "train":
                continue  # collecting on a held-out task would poison the comparison
            for init in parse_ids(args.inits):
                if (suite, task, init) in have:
                    continue
                targets.append((suite, task, init))
    targets = targets[: args.rollouts] if args.rollouts else targets
    print(f"수집 대상 {len(targets)}개 (train split task만, 중복 제외)")
    if not targets:
        return

    vla = Pi05Steer(args.ckpt)
    tap = TokenTap(vla.policy, pool=2)
    cur_key, task_env = None, None
    t_start = time.time()
    for n, (suite, tid, init) in enumerate(targets):
        if (suite, tid) != cur_key:
            if task_env is not None:
                task_env.close()
            task_env = LiberoTask(suite, tid)
            cur_key = (suite, tid)
        task_env.reset(init)
        roll = buf.open_rollout(suite, tid, init, task_env.language,
                                [" ".join(p) for p in task_env.goal()])
        roll.meta["source"] = "online"
        t, ok = 0, False
        while t < task_env.max_steps:
            ag = task_env.obs["agentview_image"][::-1, ::-1].copy()
            wr = task_env.obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
            b = vla.batch_for(ag, wr, task_env.state8(), task_env.language)
            g = torch.Generator().manual_seed(1000 * tid + 17 * init + t)
            tap.reset()
            a = vla.denoise(b, torch.randn((1, vla.C, vla.A), generator=g), [], None)[0]
            got = tap.pop(want_steps=("mean",))
            suf_mean = got["suf_mean"].float().mean(0).cpu().numpy()  # (50,1024) -> (1024,)
            score = np.float32(np.nan)
            if net is not None:
                with torch.no_grad():
                    z = torch.from_numpy((suf_mean - mu) / sd).to(device).float()
                    score = np.float32(sigmoid(net(z).item()))
            exec_n = min(args.exec, a.shape[0])
            roll.add_call(step=t, suf_mean=suf_mean.astype(np.float32),
                          action_chunk=a.astype(np.float32), action_exec=a[:exec_n].astype(np.float32),
                          eef=task_env.eef().astype(np.float32),
                          opening=np.float32(task_env.opening()), score=score)
            for j in range(exec_n):
                task_env.step(a[j])
                t += 1
                if task_env.success():
                    ok = True
                    break
            if ok:
                break
        rec = buf.close_rollout(roll, success=ok, n_steps=t)
        el = time.time() - t_start
        print(f"[{n + 1}/{len(targets)}] {suite} t{tid} i{init}  {'S' if ok else 'F'}  "
              f"calls={rec['n_calls']}  split={rec['split']}  ({el / 60:.1f}분, "
              f"남은 {el / (n + 1) * (len(targets) - n - 1) / 60:.0f}분)", flush=True)
    if task_env is not None:
        task_env.close()
    tap.close()
    print(json.dumps(buf.summary(), indent=2, ensure_ascii=False))


def macro_auroc(p, y, recs):
    """Mean of the per-task AUROCs, over tasks that actually contain both outcomes.

    Pooling every task into one ranking asks the detector to place a failure of an easy task above a success of
    a hard one, so a pooled AUROC largely measures the score offset between tasks. Within a task that offset
    cancels, which is the question an online detector is actually for: of the rollouts of THIS task, which ones
    are going wrong.
    """
    keys = sorted({(r["suite"], r["task"]) for r in recs})
    per = {}
    for k in keys:
        m = np.array([(r["suite"], r["task"]) == k for r in recs])
        yy = y[m]
        if 0 < yy.sum() < len(yy):
            per[f"{k[0]}/t{k[1]}"] = auroc(p[m], yy)
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


def mode_compare(args):
    """Two detector checkpoints on the SAME held-out test split."""
    buf = Buffer.load(HERE / args.buffer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xte, Mte, yte, rte = collect_split(buf, "test", calls=args.calls)
    if Xte is None:
        raise SystemExit("test split이 비어 있습니다")
    tasks = sorted({(r["suite"], r["task"]) for r in rte})
    print(f"test {len(rte)} 에피소드 (실패 {int(yte.sum())}), task {len(tasks)}개 — 두 모델에 동일\n")
    Xtr, Mtr, ytr, rtr = collect_split(buf, "train", calls=args.calls)
    print(f"  {'detector':16s} {'n_train':>8s} {'macro':>7s} {'pooled':>7s} {'Brier':>8s} {'R²':>7s} "
          f"{'train':>7s} {'gap':>7s} {'val':>7s}")
    out, pers = {}, {}
    for tag in [args.before, args.after]:
        path = HERE / args.buffer / "detector" / f"{tag}.pt"
        if not path.exists():
            print(f"  {tag}: 체크포인트 없음 {path}")
            continue
        net, mu, sd, meta = load_detector(path, device)
        p = episode_probs(net, mu, sd, Xte, Mte, device)
        m = brier_parts(p, yte)
        macro, per = macro_auroc(p, yte, rte)
        tr_auroc = auroc(episode_probs(net, mu, sd, Xtr, Mtr, device), ytr) if Xtr is not None else float("nan")
        out[tag] = dict(auroc_pooled=auroc(p, yte), auroc_macro=macro, train_auroc=tr_auroc, **m, meta=meta)
        pers[tag] = per
        print(f"  {tag:16s} {meta.get('n_train', 0):8d} {macro:7.3f} {out[tag]['auroc_pooled']:7.3f} "
              f"{m['brier']:8.4f} {m['r2']:7.3f} {tr_auroc:7.3f} {tr_auroc - macro:+7.3f} "
              f"{meta.get('val_auroc', float('nan')):7.3f}")
    if len(out) == 2 and args.before != args.after:
        a, b = out[args.before], out[args.after]
        print(f"\n  변화  macro {b['auroc_macro'] - a['auroc_macro']:+.3f}   "
              f"pooled {b['auroc_pooled'] - a['auroc_pooled']:+.3f}   "
              f"Brier {b['brier'] - a['brier']:+.4f} (낮을수록 좋음)")
        print(f"  학습 에피소드 {a['meta'].get('n_train')} -> {b['meta'].get('n_train')}")
        print(f"\n  task별 macro 구성 (같은 test):")
        for k in sorted(pers[args.before]):
            d = pers[args.after][k] - pers[args.before][k]
            print(f"    {k:22s} {pers[args.before][k]:.3f} -> {pers[args.after][k]:.3f}  {d:+.3f}")
    elif len(out) == 1 or args.before == args.after:
        tag = args.before
        print(f"\n  task별 AUROC ({tag}):")
        for k, v in sorted(pers[tag].items(), key=lambda kv: -kv[1]):
            print(f"    {k:22s} {v:.3f}")
    print("\n  macro  = task별 AUROC의 평균 (두 클래스가 모두 있는 task만). task 간 점수 오프셋이 상쇄된다.")
    print("  pooled = test 전체를 한 줄로 세운 AUROC. task 간 오프셋도 함께 반영된다.")
    (HERE / "results" / f"online_compare_{args.before}_{args.after}.json").write_text(
        json.dumps({k: {kk: vv for kk, vv in v.items()} for k, v in out.items()}, indent=2, ensure_ascii=False,
                   default=float))


def mode_curve(args):
    """Hold the offline rollouts fixed and add online ones, nested, to see whether collecting more helps.

    Resampling the whole train split at each size would confound two things: how much data there is, and which
    episodes happened to be drawn. Here the offline set is always present in full and only the online rollouts
    accumulate, so consecutive points differ by exactly the episodes collected between them. Within a seed the
    online order is drawn once, so the k=60 subset is contained in k=120, which is contained in k=180: a point
    is the previous point plus new experience, which is what an online loop actually does.

    val and test are untouched at every point, so all sizes are scored on identical held-out episodes.
    """
    buf = Buffer.load(HERE / args.buffer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Mtr, ytr, rtr = collect_split(buf, "train", calls=args.calls)
    Xva, Mva, yva, rva = collect_split(buf, "val", calls=args.calls)
    Xte, Mte, yte, rte = collect_split(buf, "test", calls=args.calls)
    if Xtr is None or Xva is None or Xte is None:
        raise SystemExit("split 하나가 비어 있습니다")

    off = np.array([i for i, r in enumerate(rtr) if r.get("source") != "online"])
    onl = np.array([i for i, r in enumerate(rtr) if r.get("source") == "online"])
    adds = [k for k in parse_ids(args.adds) if k <= len(onl)]
    if len(onl) not in adds:
        adds.append(len(onl))
    adds = sorted(set(adds))

    va_macro_ok = macro_auroc(np.zeros(len(yva)), yva, rva)[1]
    print(f"train 고정 offline {len(off)} + online 누적 {adds}")
    print(f"val {len(yva)} (macro 계산 가능 task {len(va_macro_ok)}/{len({(r['suite'], r['task']) for r in rva})})"
          f"   test {len(yte)} (task {len({(r['suite'], r['task']) for r in rte})})")
    print(f"model selection = {args.select} validation AUROC\n")
    print(f"  {'online':>7s} {'n_train':>8s} {'macro':>14s} {'pooled':>14s} {'Brier':>15s} {'train':>14s}")

    rows, raw = {}, []
    for k in adds:
        ms, ps, bs, ts = [], [], [], []
        for seed in range(args.seeds):
            rng = np.random.default_rng(10_000 + seed)
            order = rng.permutation(len(onl))  # one order per seed -> nested prefixes
            idx = np.concatenate([off, onl[order[:k]]]).astype(int)
            net, mu, sd, sel = fit_logreg(Xtr[idx], Mtr[idx], ytr[idx], Xva, Mva, yva, rva,
                                          args, seed, device)
            p = episode_probs(net, mu, sd, Xte, Mte, device)
            macro, per = macro_auroc(p, yte, rte)
            pooled = auroc(p, yte)
            brier = brier_parts(p, yte)["brier"]
            tr_a = auroc(episode_probs(net, mu, sd, Xtr[idx], Mtr[idx], device), ytr[idx])
            ms.append(macro); ps.append(pooled); bs.append(brier); ts.append(tr_a)
            raw.append(dict(n_online=k, n_train=len(idx), seed=seed, macro=macro, pooled=pooled,
                            brier=brier, train_auroc=tr_a, val_selected=sel, per_task=per))
        f = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
        rows[k] = dict(n_train=len(off) + k, macro=f(ms), pooled=f(ps), brier=f(bs), train=f(ts))
        r = rows[k]
        print(f"  {k:7d} {r['n_train']:8d} {r['macro'][0]:7.3f}±{r['macro'][1]:.3f} "
              f"{r['pooled'][0]:7.3f}±{r['pooled'][1]:.3f} {r['brier'][0]:8.4f}±{r['brier'][1]:.4f} "
              f"{r['train'][0]:7.3f}±{r['train'][1]:.3f}", flush=True)

    first, last = rows[adds[0]], rows[adds[-1]]
    d = last["macro"][0] - first["macro"][0]
    pooled_sd = np.sqrt(first["macro"][1] ** 2 + last["macro"][1] ** 2) / np.sqrt(args.seeds)
    print(f"\n  online 0 -> {adds[-1]}:  macro {d:+.3f}   "
          f"(시드 평균의 표준오차 약 {pooled_sd:.3f})")
    print(f"                      pooled {last['pooled'][0] - first['pooled'][0]:+.3f}   "
          f"Brier {last['brier'][0] - first['brier'][0]:+.4f}")

    out = HERE / "results" / f"online_curve_{args.tag}.json"
    out.write_text(json.dumps(dict(args={k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
                                   n_offline=len(off), n_online_total=len(onl),
                                   adds=adds, rows={str(k): v for k, v in rows.items()}, raw=raw),
                              indent=2, ensure_ascii=False, default=float))
    print(f"\n-> {out}   (시드별 raw {len(raw)}건 포함)")


def fit_logreg(X, M, y, Xva, Mva, yva, rva, args, seed, device):
    """Shared by update and curve so both train identically.

    Model selection uses the SAME statistic the experiment is judged by. With `--select macro` that is the mean
    of the per-task validation AUROCs, which is computable here because all 6 validation tasks contain both
    outcomes -- though two of them hold only 2 and 4 successes, so the signal it stops on is coarse.
    """
    flat = X[M > 0]
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    torch.manual_seed(seed)
    net = LogRegDetector().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    xt = torch.from_numpy((X - mu) / sd).to(device).float()
    mt = torch.from_numpy(M).to(device).float()
    yt = torch.from_numpy(y).to(device).float()
    w = mt * torch.where(yt[:, None] > 0, 1.0 / max(y.sum(), 1), 1.0 / max((1 - y).sum(), 1))
    tgt = yt[:, None].expand_as(mt)
    best, best_state, bad = -1.0, None, 0
    for _ in range(args.epochs):
        net.train()
        opt.zero_grad()
        loss = (nn.functional.binary_cross_entropy_with_logits(net(xt), tgt, reduction="none") * w).sum() / w.sum()
        loss.backward()
        opt.step()
        net.eval()
        pv = episode_probs(net, mu, sd, Xva, Mva, device)
        a = macro_auroc(pv, yva, rva)[0] if args.select == "macro" else auroc(pv, yva)
        if a > best + 1e-4:
            best, best_state, bad = a, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, mu, sd, float(best)


def parse_ids(s: str):
    out = []
    for part in str(s).split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    common = dict(buffer=dict(default="data/online"), group_by=dict(default="task"),
                  capacity_gb=dict(type=float, default=8.0))

    p = sub.add_parser("seed")
    p.add_argument("--buffer", **common["buffer"]); p.add_argument("--tokens", default="data/tokens")
    p.add_argument("--group-by", **common["group_by"]); p.add_argument("--capacity-gb", **common["capacity_gb"])
    p.add_argument("--include-unverified", action="store_true")
    p.set_defaults(fn=mode_seed)

    p = sub.add_parser("update")
    p.add_argument("--buffer", **common["buffer"]); p.add_argument("--tag", required=True)
    p.add_argument("--epochs", type=int, default=400); p.add_argument("--patience", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-2); p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--promote-margin", type=float, default=0.0)
    p.add_argument("--calls", type=int, default=8, help="학습·평가에 쓰는 앞쪽 호출 수 (길이 지름길 차단)")
    p.add_argument("--select", default="macro", choices=["macro", "pooled"],
                   help="early stopping과 승격 기준으로 쓸 validation AUROC")
    p.add_argument("--force-promote", action="store_true")
    p.set_defaults(fn=mode_update)

    p = sub.add_parser("collect")
    p.add_argument("--buffer", **common["buffer"])
    p.add_argument("--ckpt", default="outputs/vla/pi05_libero_finetuned")
    p.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    p.add_argument("--tasks", default="0-9"); p.add_argument("--inits", default="20-29")
    p.add_argument("--rollouts", type=int, default=0, help="0 = 대상 전부")
    p.add_argument("--exec", type=int, default=10)
    p.set_defaults(fn=mode_collect)

    p = sub.add_parser("curve")
    p.add_argument("--buffer", **common["buffer"]); p.add_argument("--tag", default="main")
    p.add_argument("--adds", default="0,60,120,180", help="추가할 online 롤아웃 수 (누적, nested)")
    p.add_argument("--seeds", type=int, default=8); p.add_argument("--calls", type=int, default=8)
    p.add_argument("--select", default="macro", choices=["macro", "pooled"])
    p.add_argument("--epochs", type=int, default=400); p.add_argument("--patience", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-2); p.add_argument("--weight-decay", type=float, default=100.0)
    p.set_defaults(fn=mode_curve)

    p = sub.add_parser("compare")
    p.add_argument("--buffer", **common["buffer"])
    p.add_argument("--before", required=True); p.add_argument("--after", required=True)
    p.add_argument("--calls", type=int, default=8)
    p.set_defaults(fn=mode_compare)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
