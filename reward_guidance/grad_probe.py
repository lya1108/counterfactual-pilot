"""Where does the C1 head's gradient point, and can the frozen head even be differentiated?

Stage A showed the head ranks the policy's own candidates no better than the chunk's gripper channel does.
Best-of-N is confined to what the policy sampled, so the obvious reply is that a gradient can leave that set.
Before running a guided rollout, two things have to be true, and neither is obvious:

  1. The head has to be a function of the action at all. It was trained on `suf_mean`, the suffix hidden
     averaged over the whole 10-step denoising trajectory -- that is a function of the trajectory, not of the
     chunk, so it has no gradient w.r.t. a chunk. The differentiable stand-in is one extra action-expert pass
     at the final chunk: phi_eval(x1) = pooled suffix hidden from denoise_step(x_t = x1, t = 1/K). If the
     frozen head reads phi_eval differently from suf_mean, it cannot be used as a reward without retraining,
     and that is itself the answer.

  2. The gradient has to point somewhere useful. With labelled candidates at the same state, there is a
     measured direction to compare against: (mean safe chunk - mean unsafe chunk). A reward whose gradient is
     orthogonal to that will not steer, however large the guidance weight.

Both are measured on the same 40 held-out-task episodes as Stage A, with the branch labels recomputed in the
same run so nothing has to be aligned across runs.

    conda activate fasi
    python reward_guidance/grad_probe.py --inits 30-39 --n-cand 8
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

HERE = Path(__file__).resolve().parent
X = HERE.parent / "xattn_detector"
for p in (str(HERE), str(X), str(HERE.parent / "safe_detector")):
    sys.path.insert(0, p)

import jb  # noqa: E402

jb.enter()

from fasi.envs.libero_env import LiberoTask  # noqa: E402

from conditions import DEFAULTS, StateLogger  # noqa: E402
from detector import EVAL_GROUPS, C1Head  # noqa: E402
from sensitivity import STORE_STEPS, branch, noise_for, parse_ids, rel  # noqa: E402
from tap import SuffixBatchTap  # noqa: E402
from vla_setup import make_vla  # noqa: E402

D_ACT = 7


class EvalHead:
    """head(phi_eval(x1)): a differentiable risk for a clean action chunk under the current observation."""

    def __init__(self, vla, head, tap):
        self.vla, self.head, self.tap = vla, head, tap
        self.mm = vla.policy.model
        self.t_eval = 1.0 / vla.K          # the timestep the sampler's last step runs at

    def __call__(self, x1_norm, ppm, kv, grad=False):
        n = x1_norm.shape[0]
        tt = torch.tensor(self.t_eval, dtype=torch.float32, device="cuda").expand(n)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            x = x1_norm.clone().to("cuda", torch.float32)
            if grad:
                x.requires_grad_(True)
            self.tap.reset()
            self.mm.denoise_step(prefix_pad_masks=ppm, past_key_values=kv, x_t=x, timestep=tt)
            phi = self.tap.pop(1, n)
            r = self.head.torch_(phi)
            if not grad:
                return r.detach().float().cpu().numpy(), None
            (g,) = torch.autograd.grad(r.sum(), x)
            return r.detach().float().cpu().numpy(), g.detach().float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=list(EVAL_GROUPS))
    ap.add_argument("--inits", default="30-39")
    ap.add_argument("--n-cand", type=int, default=8)
    ap.add_argument("--exec", type=int, default=10)
    ap.add_argument("--head", default=str(HERE / "results/c1_head.pt"))
    ap.add_argument("--out", default="data/grad")
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
    assert all(g in head.eval_groups for g in args.groups)
    N = args.n_cand
    vla = make_vla(batch=N)
    tap = SuffixBatchTap(vla.policy, keep_graph=True)
    ev = EvalHead(vla, head, tap)
    print(f"phi_eval 시점 t = {ev.t_eval:.2f} (샘플러 마지막 스텝과 같은 시점), 후보 {N}개/호출\n")
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
            s_suf, s_ev, grads, labels, call_step, chunks_s = [], [], [], [], [], []
            while t < task.max_steps:
                last = ep.rows[-1]
                c1_now = bool(last["opening"] < DEFAULTS["close_gap"] and not last["touch"].any())
                agv = task.obs["agentview_image"][::-1, ::-1].copy()
                wr = task.obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
                b = vla.batch_for(agv, wr, task.state8(), task.language)

                noise = torch.cat([noise_for(vla, tid, init, t, i) for i in range(1 if c1_now else N)])
                tap.reset()
                env, xs = vla.denoise(b, noise, [], None, return_norm=True)
                suf = tap.pop(vla.K, noise.shape[0]).float().cpu().numpy()

                if not c1_now:
                    with torch.no_grad():
                        ppm, kv = vla._prefix(b, N)
                    r_ev, _ = ev(xs.to("cuda"), ppm, kv, grad=False)
                    with torch.no_grad():
                        ppm1, kv1 = vla._prefix(b, 1)
                    _, g0 = ev(xs[:1].to("cuda"), ppm1, kv1, grad=True)

                    snap = task.snapshot()
                    ys = [int(branch(task, env[i], snap, args.exec)[0]) for i in range(N)]
                    task.restore(snap)

                    s_suf.append(head.numpy(suf).astype(np.float32))
                    s_ev.append(r_ev.astype(np.float32))
                    grads.append(g0[0, :STORE_STEPS, :D_ACT].astype(np.float32))
                    chunks_s.append(xs[:, :STORE_STEPS, :D_ACT].numpy().astype(np.float32))
                    labels.append(ys)
                    call_step.append(t)

                a = env[0]
                for j in range(args.exec):
                    task.step(a[j])
                    t += 1
                    ep.record(t)
                    if task.success():
                        ok = True
                        break
                if ok:
                    break

            name = f"{suite}_t{tid}_i{init}"
            np.savez_compressed(out / f"{name}.npz",
                                score_suf=np.stack(s_suf) if s_suf else np.zeros((0, N), np.float32),
                                score_eval=np.stack(s_ev) if s_ev else np.zeros((0, N), np.float32),
                                grad0=np.stack(grads) if grads else np.zeros((0, STORE_STEPS, D_ACT), np.float32),
                                chunks_norm=np.stack(chunks_s) if chunks_s else np.zeros((0, N, STORE_STEPS, D_ACT), np.float32),
                                labels=np.array(labels, np.int8).reshape(-1, N),
                                call_step=np.asarray(call_step, np.int32))
            rec = dict(suite=suite, task=tid, init=init, group=group, success=ok, n_steps=t,
                       n_scored_calls=len(s_suf), npz=f"{name}.npz", secs=round(time.time() - t0, 1))
            with open(index, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{name:26s} 성공={str(ok):5s} step={t:4d} 호출={len(s_suf):3d} {rec['secs']:6.1f}s", flush=True)

    tap.close()
    print(f"\n총 {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
