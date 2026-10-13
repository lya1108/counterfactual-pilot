"""Within-state action ranking: can the same state's 8 candidates be ordered by causal C1 risk?

Stage A measured the frozen C1 head *within* a state (0.786) and found it indistinguishable from one
scalar read off the chunk -- the mean gripper command (0.794). That head was never trained for this
question; it was trained to separate states. This trains a head directly on the within-state question
and asks whether 1024 dimensions beat the scalar.

The model is linear on purpose. For f(phi) = w . phi,

    f(risky) - f(safe) = w . (phi_risky - phi_safe)

so the state's own offset cancels exactly and the head structurally *cannot* use between-state
information. A nonlinear head loses that property and would reintroduce the question already answered.

Three feature variants are compared under one cross-validation:

    raw      within-state deviations of phi, L2
    pca      projected on the top-k PCs of the training folds' within-state deviations
    resid    phi deviations with the gripper-command direction regressed out

`resid` is the decisive one: it asks whether phi carries anything the scalar does not.

Splits are by task (4 folds, the primary metric -- the same level as the unseen-task protocol) and by
episode (40 folds; the 9 episodes with no mixed state are still scored out-of-fold so the selection
rate covers all 841 calls). alpha and k are chosen by episode-clustered inner CV inside the training
folds only. All CIs resample the 40 episodes.

    python rank_head.py                      # -> results/ranking_main.json
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

NONCONV = [0, 0]   # [fits that hit max_iter, total fits] -- reported, never silently swallowed

HERE = Path(__file__).resolve().parent
EX = 10          # environment steps actually executed from a chunk -- the scored interval
N_CAND = 8
ALPHAS = np.logspace(-1, 4, 11)
PCA_K = (5, 10, 20, 40)
EPS = 1e-8


# ----------------------------------------------------------------------------- data
def load(dirname="data/sens"):
    d = HERE / dirname
    eps = [json.loads(l) for l in (d / "episodes.jsonl").read_text().splitlines() if l.strip()]
    phi, y, s, ch, ep, gr = [], [], [], [], [], []
    for e in eps:
        z = np.load(d / e["npz"])
        n = len(z["labels"])
        phi.append(z["phi"].astype(np.float32))
        y.append(z["labels"].astype(np.int64))
        s.append(z["scores"].astype(np.float64))
        ch.append(z["chunks"].astype(np.float64))
        ep += [f"{e['group']}/i{e['init']}"] * n
        gr += [e["group"]] * n
    return dict(phi=np.concatenate(phi), y=np.concatenate(y), s=np.concatenate(s),
                ch=np.concatenate(ch), ep=np.array(ep), gr=np.array(gr),
                episodes=np.array(sorted({f"{e['group']}/i{e['init']}" for e in eps})))


def pairs_of(y_state):
    """-> (risky index, safe index) for every ordered pair in one mixed state."""
    r = np.where(y_state == 1)[0]
    s = np.where(y_state == 0)[0]
    return [(a, b) for a in r for b in s]


def is_mixed(y):
    k = y.sum(1)
    return (k > 0) & (k < N_CAND)


# ----------------------------------------------------------------------------- features
class Transform:
    """Per-state centering + scaling, fit on the training folds only.

    The centering is what makes this a within-state model: every state's mean is removed before the
    head sees it, so the between-state axis is gone by construction rather than by regularisation.
    All training states contribute (not only the mixed ones) -- the subspace is estimated
    unsupervised, where data is plentiful. For `pca` the top max(PCA_K) components are kept once and
    sliced per k, which is exact because the components are ordered.
    """

    def __init__(self, variant, kmax=max(PCA_K)):
        self.variant, self.kmax = variant, kmax

    def fit(self, phi, ch):
        w = phi - phi.mean(1, keepdims=True)
        flat = w.reshape(-1, w.shape[-1]).astype(np.float64)
        self.sd = flat.std(0) + EPS
        flat = flat / self.sd
        if self.variant == "resid":
            gw = self._grip(ch).reshape(-1)
            self.b = (flat * gw[:, None]).sum(0) / max((gw ** 2).sum(), EPS)
        elif self.variant == "pca":
            c = flat - flat.mean(0)
            # eigh on the 1024x1024 Gram is exact here and far cheaper than a full SVD
            val, vec = np.linalg.eigh(c.T @ c)
            self.vt = vec[:, ::-1][:, : self.kmax].T
        return self

    @staticmethod
    def _grip(ch):
        g = ch[:, :, :EX, 6].mean(2)
        return g - g.mean(1, keepdims=True)

    def __call__(self, phi, ch):
        w = (phi - phi.mean(1, keepdims=True)).astype(np.float64) / self.sd
        if self.variant == "resid":
            w = w - self._grip(ch)[..., None] * self.b
        elif self.variant == "pca":
            w = w @ self.vt.T
        return w                                                  # (n, 8, d)


# ----------------------------------------------------------------------------- model
def build_pairs(z, y):
    """-> (n_pairs, d) difference vectors, risky minus safe, over the mixed states."""
    d = []
    for i in np.where(is_mixed(y))[0]:
        for a, b in pairs_of(y[i]):
            d.append(z[i, a] - z[i, b])
    return np.asarray(d) if d else None


def fit_rank(d, alpha):
    """Pairwise logistic with no intercept. The mirrored rows make the problem symmetric."""
    if d is None or len(d) == 0:
        return None
    x = np.concatenate([d, -d])
    t = np.concatenate([np.ones(len(d)), np.zeros(len(d))])
    m = LogisticRegression(C=1.0 / alpha, fit_intercept=False, max_iter=2000, tol=1e-6)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.fit(x, t)
        NONCONV[0] += any(issubclass(w.category, ConvergenceWarning) for w in caught)
    NONCONV[1] += 1
    return m.coef_[0]


# ----------------------------------------------------------------------------- metrics
def pair_auc(score, y, idx):
    """Pair-weighted concordance over the mixed states in `idx` -- the within-state AUROC."""
    won = tot = 0.0
    for i in idx:
        if not (0 < y[i].sum() < N_CAND):
            continue
        p = score[i][y[i] == 1][:, None] - score[i][y[i] == 0][None, :]
        won += (p > 0).sum() + 0.5 * (p == 0).sum()
        tot += p.size
    return won / tot if tot else np.nan


def state_auc(score, y, idx):
    a = []
    for i in idx:
        if not (0 < y[i].sum() < N_CAND):
            continue
        p = score[i][y[i] == 1][:, None] - score[i][y[i] == 0][None, :]
        a.append(((p > 0).sum() + 0.5 * (p == 0).sum()) / p.size)
    return float(np.mean(a)) if a else np.nan


def sel_rate(score, y, idx):
    """C1 rate after picking the candidate the head calls safest."""
    if len(idx) == 0:
        return np.nan
    j = score[idx].argmin(1)
    return float(y[idx][np.arange(len(idx)), j].mean())


# ----------------------------------------------------------------------------- cross-validation
def _grid(variant):
    ks = PCA_K if variant == "pca" else (None,)
    return [(a, k) for k in ks for a in ALPHAS]


def inner_select(phi, ch, y, ep, variant, n_inner=4):
    """Choose (alpha, k) by episode-clustered CV inside the training folds."""
    uniq = np.array(sorted(set(ep)))
    order = np.random.default_rng(0).permutation(len(uniq))
    fold = {uniq[j]: i % n_inner for i, j in enumerate(order)}
    f = np.array([fold[e] for e in ep])

    cache = []                       # one transform per inner fold, reused across the whole grid
    for v in range(n_inner):
        tr, va = f != v, f == v
        if not tr.any() or not va.any() or not is_mixed(y[tr]).any():
            continue
        t = Transform(variant).fit(phi[tr], ch[tr])
        cache.append((tr, va, t(phi[tr], ch[tr]), t(phi[va], ch[va])))
    if not cache:
        return ALPHAS[len(ALPHAS) // 2], (PCA_K[1] if variant == "pca" else None)

    best, best_s = None, -np.inf
    for a, k in _grid(variant):
        sc = np.full(y.shape, np.nan)
        for tr, va, ztr, zva in cache:
            zt = ztr[..., :k] if k else ztr
            zv = zva[..., :k] if k else zva
            w = fit_rank(build_pairs(zt, y[tr]), a)
            if w is None:
                continue
            sc[va] = zv @ w
        ok = np.where(~np.isnan(sc[:, 0]))[0]
        s = pair_auc(sc, y, ok) if len(ok) else np.nan
        if not np.isnan(s) and s > best_s:
            best, best_s = (a, k), s
    return best if best else (ALPHAS[len(ALPHAS) // 2], PCA_K[1] if variant == "pca" else None)


def run_cv(D, by, variant, verbose=True):
    """Leave-one-group-out over `by` ('gr' or 'ep'). -> out-of-fold scores (N, 8), per-fold info."""
    key, phi, ch, y, ep = D[by], D["phi"], D["ch"], D["y"], D["ep"]
    mixed = is_mixed(y)
    out = np.full(y.shape, np.nan)
    info = []
    for g in sorted(set(key)):
        te = key == g
        tr = ~te
        a, k = inner_select(phi[tr], ch[tr], y[tr], ep[tr], variant)
        t = Transform(variant).fit(phi[tr], ch[tr])
        ztr, zte = t(phi[tr], ch[tr]), t(phi[te], ch[te])
        if k:
            ztr, zte = ztr[..., :k], zte[..., :k]
        w = fit_rank(build_pairs(ztr, y[tr]), a)
        if w is None:
            continue
        out[te] = zte @ w
        npair = sum(len(pairs_of(y[i])) for i in np.where(tr & mixed)[0])
        info.append(dict(fold=str(g), n_test=int(te.sum()), n_test_mixed=int((te & mixed).sum()),
                         n_train_pairs=int(npair), alpha=float(a), k=(int(k) if k else None)))
        if verbose and by == "gr":
            print(f"    {g:24s} 학습쌍 {npair:4d}  alpha={a:<8.3g} k={k}  "
                  f"테스트 혼합 {int((te & mixed).sum()):2d}", flush=True)
    return out, info


# ----------------------------------------------------------------------------- bootstrap
def bootstrap(D, methods, n=2000, seed=0):
    """Resample the 40 episodes with replacement -> CIs that respect the clustering."""
    rng = np.random.default_rng(seed)
    ep, y = D["ep"], D["y"]
    idx_of = {e: np.where(ep == e)[0] for e in D["episodes"]}
    names = list(methods)
    acc = {m: {"auc": [], "auc_state": [], "sel": [], "sel_mixed": []} for m in names}
    for _ in range(n):
        pick = rng.choice(D["episodes"], size=len(D["episodes"]), replace=True)
        idx = np.concatenate([idx_of[e] for e in pick])
        mx = idx[is_mixed(y[idx])]
        for m in names:
            sc = methods[m]
            acc[m]["auc"].append(pair_auc(sc, y, mx))
            acc[m]["auc_state"].append(state_auc(sc, y, mx))
            acc[m]["sel"].append(sel_rate(sc, y, idx))
            acc[m]["sel_mixed"].append(sel_rate(sc, y, mx))
    stat = {}
    for m in names:
        stat[m] = {}
        for k, v in acc[m].items():
            v = np.asarray(v, float)
            v = v[~np.isnan(v)]
            stat[m][k] = [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return stat, acc


def paired(acc, a, b, metric):
    d = np.asarray(acc[a][metric], float) - np.asarray(acc[b][metric], float)
    d = d[~np.isnan(d)]
    return [float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="results/ranking_main.json")
    args = ap.parse_args()
    t0 = time.time()

    D = load()
    y = D["y"]
    k = y.sum(1)
    mixed = is_mixed(y)
    N = len(y)
    n_pairs = sum(len(pairs_of(y[i])) for i in np.where(mixed)[0])
    eps_mixed = len({e for e in D["ep"][mixed]})
    print(f"호출 {N}  후보 {N * N_CAND}  위험후보 {int(y.sum())} ({y.mean() * 100:.1f}%)")
    print(f"  k=0 {int((k == 0).sum())} ({(k == 0).mean() * 100:.1f}%)   "
          f"혼합 {int(mixed.sum())} ({mixed.mean() * 100:.1f}%)   "
          f"k=8 {int((k == N_CAND).sum())} ({(k == N_CAND).mean() * 100:.1f}%)")
    print(f"  유효 학습 쌍 {n_pairs}   혼합>0 에피소드 {eps_mixed}/40   과제 4\n")

    g_mean = D["ch"][:, :, :EX, 6].mean(2)
    g_shut = (D["ch"][:, :, :EX, 6] > 0).mean(2)
    methods = {
        "현재 C1 head (frozen)": D["s"],
        "그리퍼 명령 평균": g_mean,
        "그리퍼 닫힘 명령 비율": g_shut,
        "무작위": np.random.default_rng(1234).standard_normal(y.shape),
        "오라클": y.astype(float),
    }
    print("  in-sample 진단 (같은 쌍으로 학습하고 같은 쌍을 채점 -- 과적합 상한, 일반화 아님)")
    insample = {}
    for variant in ("raw", "pca", "resid"):
        t = Transform(variant).fit(D["phi"], D["ch"])
        z = t(D["phi"], D["ch"])
        if variant == "pca":
            z = z[..., :PCA_K[-1]]
        w = fit_rank(build_pairs(z, y), 1.0)
        a = pair_auc(z @ w, y, np.where(mixed)[0])
        insample[variant] = float(a)
        print(f"    {variant:6s} {a:.3f}")
    print()

    folds_info = {}
    for by, tag in (("gr", "과제"), ("ep", "에피소드")):
        for variant in ("raw", "pca", "resid"):
            name = f"Ranking {variant} [{tag}]"
            print(f"  {name}", flush=True)
            sc, info = run_cv(D, by, variant)
            sc = np.where(np.isnan(sc), 0.0, sc)
            methods[name] = sc
            folds_info[name] = info
            ks = [i["k"] for i in info]
            print(f"    -> within-state AUROC {pair_auc(sc, y, np.where(mixed)[0]):.3f}"
                  f"   선택 C1 {sel_rate(sc, y, np.arange(N)) * 100:.1f}%"
                  f"   k={sorted({str(x) for x in ks})}  ({time.time() - t0:.0f}s)\n", flush=True)

    print(f"부트스트랩 {args.boot}회 (에피소드 40개 복원추출) ...", flush=True)
    stat, acc = bootstrap(D, methods, n=args.boot)

    def fa(t):
        return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"

    def fp(t):
        return f"{t[0] * 100:4.1f}% [{t[1] * 100:4.1f}, {t[2] * 100:4.1f}]"

    print(f"\n{'방법':26s} {'within-state AUROC':>22s} {'선택 C1 (전체 841)':>22s} {'선택 C1 (혼합 50)':>22s}")
    for m in methods:
        print(f"{m:26s} {fa(stat[m]['auc']):>22s} {fp(stat[m]['sel']):>22s} {fp(stat[m]['sel_mixed']):>22s}")

    base = "그리퍼 명령 평균"
    print(f"\n상태 가중 within-state AUROC (상태당 1표):")
    for m in methods:
        print(f"  {m:26s} {fa(stat[m]['auc_state'])}")

    print(f"\n대조 (vs {base}, 동일 부트스트랩 표본):")
    deltas = {}
    for m in methods:
        if m in (base, "오라클", "무작위"):
            continue
        da, ds = paired(acc, m, base, "auc"), paired(acc, m, base, "sel")
        deltas[m] = dict(auc=da, sel=ds)
        mark = "유의" if (da[1] > 0 or da[2] < 0) else "구분되지 않는다"
        print(f"  {m:26s} ΔAUROC {da[0]:+.3f} [{da[1]:+.3f}, {da[2]:+.3f}]  {mark:14s}"
              f"  Δ선택C1 {ds[0] * 100:+.2f}%p [{ds[1] * 100:+.2f}, {ds[2] * 100:+.2f}]")

    out = HERE / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        n_calls=int(N), n_cand=int(N * N_CAND), n_pos=int(y.sum()),
        n_all_safe=int((k == 0).sum()), n_mixed=int(mixed.sum()), n_all_risky=int((k == N_CAND).sum()),
        n_pairs=int(n_pairs), n_episodes_with_mixed=eps_mixed, boot=args.boot,
        stat=stat, deltas=deltas, folds=folds_info, insample=insample,
        nonconverged=NONCONV[0], n_fits=NONCONV[1]), ensure_ascii=False, indent=1))
    print(f"\n수렴 실패 {NONCONV[0]}/{NONCONV[1]} 회")
    print(f"저장: {out}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
