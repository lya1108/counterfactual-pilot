"""Does the C1 head rank ACTIONS, or only states?  The gate every guidance idea has to pass.

0.891 unseen-task macro AUROC says: given two different moments, the head knows which one is about to close
an empty gripper. Reward guidance needs something strictly stronger -- that at ONE moment, with the
observation fixed, it ranks the policy's own candidate chunks by whether they cause C1. Nothing measured so
far distinguishes the two, exactly as pooled AUROC did not distinguish "knows which task is hard" from
"knows which rollout is going wrong" one level up.

So at every call where C1 is inactive, N chunks are sampled from pi0.5 (candidate 0 is the chunk the
unmodified rollout would execute, so the trajectory is untouched), each is scored by the frozen head, and
each is then actually executed from a snapshot of that exact state for 12 environment steps and labelled by
the simulator's own C1 predicate. The episode continues on candidate 0.

Labels are physics, not the detector: y_i = 1 if the C1 predicate holds for 3 consecutive steps starting
within the first 10 steps of candidate i -- the canonical prospective target, measured counterfactually.
(12 steps are rolled so a run starting at step 10 can complete. Steps 11-12 come from the same chunk, where
the real rollout would have re-planned; that is the closest attribution of the label to this chunk.)

Two controls ride along, because a branch comparison is worthless if branching itself is noisy:

  qpos replay   candidate 0's branch and the real continuation run the same chunk from the same snapshot, so
                their joint positions must agree exactly. Measured, not assumed. (They do: restore() is exact
                in `sim.data.qpos`. What it does NOT reproduce is robosuite's reported observation -- the
                same state comes back with `robot0_eef_pos` about 1.5e-4 m off, because the observable
                pipeline carries history the snapshot never took. Every quantity here that is read from an
                observation, `opening` among them, inherits that.)

  duplicate     one extra branch re-runs candidate 0's chunk. Its label must equal candidate 0's. Whatever
                fraction of calls it disagrees on is the noise floor, and no candidate disagreement rate
                below that floor means anything.

What comes out:
  disagreement rate    calls where the candidates do not all share one outcome. Nothing to steer without it.
  within-state AUROC   macro over those calls -- the quantity reward guidance actually consumes.
  oracle best-of-N     how often a safe candidate existed when candidate 0 was unsafe (the ceiling).
  realised best-of-N   how often picking the lowest-scoring candidate would have taken it, and how often
                       it would have caused C1 where candidate 0 did not.
  between-state AUROC  the familiar number, recomputed here, so the two are never confused again.

Every candidate's action chunk and suffix latent are stored, because the within-state number needs an ablation
that no between-state number does: at a fixed state, a feature that discriminates candidates is by definition
reading the ACTION, and the gripper channel of the chunk is an action feature anyone can compute for free.
If "the command closes the gripper" ranks candidates as well as the 1024-D latent, the latent has added
nothing here -- and a state-only feature, which is identical across candidates, is pinned at 0.500 by
construction, so it cannot serve as the control.

    conda activate fasi
    python reward_guidance/sensitivity.py --inits 30-39 --n-cand 8
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
X = HERE.parent / "xattn_detector"
for p in (str(HERE), str(X), str(HERE.parent / "safe_detector")):
    sys.path.insert(0, p)

import jb  # noqa: E402

jb.enter()

from fasi.envs.libero_env import LiberoTask  # noqa: E402

from conditions import DEFAULTS, StateLogger, _onset, evaluate  # noqa: E402
from detector import EVAL_GROUPS, C1Head  # noqa: E402
from tap import SuffixBatchTap  # noqa: E402
from vla_setup import make_vla  # noqa: E402
from train_detector import auroc  # noqa: E402

COND = "C1_grasp_empty_close"


def rel(p):
    """Resolve against this file, not the cwd -- jb.enter() chdirs into the jungbeom repo,
    and a relative --out there would write into a tree this project never writes to."""
    p = Path(p)
    return p if p.is_absolute() else HERE / p
BRANCH = 12   # environment steps rolled per candidate; 10 scored + 2 so a run starting at step 10 can finish
STORE_STEPS = 16  # of each candidate chunk kept on disk (the guidance sampler scores 16 too)
H = 10        # the canonical horizon, and exactly one chunk's execution interval


def parse_ids(s):
    out = []
    for part in str(s).split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def noise_for(vla, tid, init, t, i):
    """Candidate 0 reuses collect_conditions' seed, so the executed trajectory is the unmodified one."""
    seed = (1000 * tid + 17 * init + t) if i == 0 else (900_000 + 1000 * tid + 17 * init + 7 * t + i)
    g = torch.Generator().manual_seed(seed)
    return torch.randn((1, vla.C, vla.A), generator=g)


def branch(task, chunk, snap, at):
    """Execute one candidate from `snap` and label it with the simulator's C1 predicate.

    -> (y, d_onset, qpos after `at` steps, eef after `at` steps, lowest gripper opening seen)."""
    task.restore(snap)
    log = StateLogger(task)
    qp = None
    for j in range(BRANCH):
        task.step(chunk[j])
        log.record(j + 1)          # offsets are environment steps AFTER the call, matching c1_depth.build
        if j + 1 == at:
            qp = np.array(task.e.sim.data.qpos, copy=True)
    series, _ = evaluate(log.pack())
    idx = _onset(series[COND], DEFAULTS["debounce"])
    d = None if idx is None else idx + 1
    return ((d is not None and d <= H), d, qp,
            log.rows[at - 1]["eef"].copy(), float(min(r["opening"] for r in log.rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=list(EVAL_GROUPS))
    ap.add_argument("--inits", default="30-39")
    ap.add_argument("--n-cand", type=int, default=8)
    ap.add_argument("--exec", type=int, default=10)
    ap.add_argument("--head", default=str(HERE / "results/c1_head.pt"))
    ap.add_argument("--out", default=str(HERE / "data/sens"))
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    out = rel(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = out / "episodes.jsonl"
    done = set()
    if index.exists():
        for line in index.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["suite"], r["task"], r["init"]))

    head = C1Head(args.head)
    assert all(g in head.eval_groups for g in args.groups), \
        f"rolling out a task the head was TRAINED on: {set(args.groups) - set(head.eval_groups)}"
    print(f"고정 head: 학습 group {len(head.train_groups)}, 미학습 macro AUROC {head.meta['macro_heldout']:.3f}")
    print(f"평가 group {args.groups}   init {args.inits}   후보 {args.n_cand}개/호출\n")

    N = args.n_cand
    vla = make_vla(batch=N)   # one sub-batch, so the tap's K captures all belong to one candidate group
    tap = SuffixBatchTap(vla.policy)
    t_start = time.time()

    for group in args.groups:
        suite, tstr = group.split("/")
        tid = int(tstr[1:])
        task = LiberoTask(suite, tid)
        for init in parse_ids(args.inits):
            if (suite, tid, init) in done:
                continue
            t0 = time.time()
            task.reset(init)
            ep = StateLogger(task)
            ep.record(0)
            t, ok = 0, False
            scores, labels, dists, call_step, act_dist = [], [], [], [], []
            replay_err, dup_label, open_min = [], [], []
            chunk_store, phi_store = [], []
            while t < task.max_steps:
                last = ep.rows[-1]
                c1_now = bool(last["opening"] < DEFAULTS["close_gap"] and not last["touch"].any())
                ag = task.obs["agentview_image"][::-1, ::-1].copy()
                wr = task.obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
                b = vla.batch_for(ag, wr, task.state8(), task.language)

                if c1_now:              # the canonical target drops these calls; so does this
                    a = vla.denoise(b, noise_for(vla, tid, init, t, 0), [], None)[0]
                else:
                    tap.reset()
                    noise = torch.cat([noise_for(vla, tid, init, t, i) for i in range(N)])
                    chunks = vla.denoise(b, noise, [], None)             # (N, C, 7) env units
                    phi = tap.pop(vla.K, N).float().cpu().numpy()        # (N, 1024)
                    sc = head.numpy(phi)
                    snap = task.snapshot()
                    ys, ds, qps, om = [], [], [], []
                    for i in range(N):
                        y, d, qp, _, mo = branch(task, chunks[i], snap, args.exec)
                        ys.append(int(y)); ds.append(-1 if d is None else d); qps.append(qp); om.append(mo)
                    # the noise floor: the same chunk again, from the same snapshot
                    ydup, _, qdup, _, _ = branch(task, chunks[0], snap, args.exec)
                    dup_label.append(int(ydup))
                    task.restore(snap)
                    scores.append(sc.astype(np.float32)); labels.append(ys); dists.append(ds)
                    open_min.append(om); call_step.append(t)
                    chunk_store.append(chunks[:, :STORE_STEPS].astype(np.float32))
                    phi_store.append(phi.astype(np.float16))
                    act_dist.append(float(np.mean([np.linalg.norm(chunks[i][:args.exec] - chunks[0][:args.exec])
                                                   for i in range(1, N)])))
                    a = chunks[0]
                    qpos0, qpos_dup = qps[0], qdup

                for j in range(args.exec):
                    task.step(a[j])
                    t += 1
                    ep.record(t)
                    if task.success():
                        ok = True
                        break
                if not c1_now and not ok:
                    # candidate 0's branch, its duplicate and the real continuation ran the same chunk from
                    # the same snapshot, so their joint positions must agree -- measured, never assumed
                    qnow = np.asarray(task.e.sim.data.qpos)
                    replay_err.append([float(np.abs(qnow - qpos0).max()),
                                       float(np.abs(qpos_dup - qpos0).max())])
                if ok:
                    break

            packed = ep.pack()
            series, _ = evaluate(packed)
            name = f"{suite}_t{tid}_i{init}"
            np.savez_compressed(out / f"{name}.npz",
                                scores=np.stack(scores) if scores else np.zeros((0, N), np.float32),
                                labels=np.array(labels, np.int8).reshape(-1, N),
                                d_onset=np.array(dists, np.int32).reshape(-1, N),
                                call_step=np.asarray(call_step, np.int32),
                                act_dist=np.asarray(act_dist, np.float32),
                                open_min=np.array(open_min, np.float32).reshape(-1, N),
                                dup_label=np.asarray(dup_label, np.int8),
                                chunks=np.stack(chunk_store) if chunk_store else np.zeros((0, N, 1, 7), np.float32),
                                phi=np.stack(phi_store) if phi_store else np.zeros((0, N, 1), np.float16),
                                replay_err=np.array(replay_err, np.float32).reshape(-1, 2), **packed)
            rec = dict(suite=suite, task=tid, init=init, group=group, success=ok, n_steps=t,
                       n_scored_calls=len(scores), npz=f"{name}.npz",
                       c1_any=bool(series[COND].any()), secs=round(time.time() - t0, 1),
                       replay_qpos_max=float(np.max(np.array(replay_err)[:, 0])) if replay_err else 0.0,
                       dup_disagree=int(sum(int(a != b[0]) for a, b in zip(dup_label, labels))))
            with open(index, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            lab = np.array(labels).reshape(-1, N) if labels else np.zeros((0, N))
            dis = int(((lab.sum(1) > 0) & (lab.sum(1) < N)).sum()) if len(lab) else 0
            print(f"{name:26s} 성공={str(ok):5s} step={t:4d} 호출={len(scores):3d} "
                  f"후보불일치={dis:3d} C1={str(rec['c1_any']):5s} "
                  f"복제불일치={rec['dup_disagree']:2d} qpos재현={rec['replay_qpos_max']:.1e} "
                  f"{rec['secs']:6.1f}s")

    tap.close()
    print(f"\n총 {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
