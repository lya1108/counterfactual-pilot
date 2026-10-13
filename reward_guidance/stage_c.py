"""Does applying the head's gradient during denoising produce a safer chunk?  Paired, per call.

The obvious form of this experiment is a closed-loop sweep: run episodes with guidance on and off and compare
C1 rates. It is also the weak form. Guided and unguided rollouts diverge after the first call, so the
comparison is between two sets of episodes, and 40 episodes carry a few dozen C1 events in total.

Branching makes it paired instead. At every call the same start noise is denoised four ways -- unguided, with
the gradient descending the C1 logit, with it ascending, and along a fixed random direction of the same step
size -- and all four are executed from the same snapshot and labelled by the simulator. One call yields four
outcomes for the same state, and 40 episodes yield ~850 such quadruples. The rollout itself continues on the
unguided chunk, so every call is scored under the policy's own state distribution.

The random arm is not decoration. Any displacement of a chunk changes what happens; without it, "guidance
changed the C1 rate" cannot be told from "perturbing the action changed the C1 rate". The `up` arm is there
because grad_probe measured the gradient pointing AWAY from the safe candidates (cos -0.146), which predicts
the inverted result -- a prediction worth either confirming or losing.

    conda activate fasi
    python reward_guidance/stage_c.py --inits 30-39 --lam 2.0
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
from guided import guided_denoise  # noqa: E402
from sensitivity import STORE_STEPS, branch, noise_for, parse_ids, rel  # noqa: E402
from tap import SuffixBatchTap  # noqa: E402
from vla_setup import make_vla  # noqa: E402

# arm 0 is the chunk the rollout actually executes. Two step sizes, because one is not a result: LAM[1] is
# matched to how far pi0.5's own resampling moves the executed part of a chunk (median 0.757 in normalised
# units), LAM[4] is three times that -- far outside the cloud the policy samples from.
ARMS = ("none", "down", "up", "rand", "down", "rand")
LAM = (0.0, 10.0, 10.0, 10.0, 30.0, 30.0)
ARM_NAMES = ("무개입", "위험↓ λ10", "위험↑ λ10", "무작위 λ10", "위험↓ λ30", "무작위 λ30")
EX_DIMS = 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=list(EVAL_GROUPS))
    ap.add_argument("--inits", default="30-39")
    ap.add_argument("--exec", type=int, default=10)
    ap.add_argument("--head", default=str(HERE / "results/c1_head.pt"))
    ap.add_argument("--out", default="data/stagec")
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
    A = len(ARMS)
    vla = make_vla(batch=A)
    tap = SuffixBatchTap(vla.policy, keep_graph=True)
    print("arm:  " + "  ".join(f"{n}" for n in ARM_NAMES))
    print(f"한 호출에서 같은 시작 노이즈로 {A}갈래, 실행은 언제나 arm 0\n")
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
            labels, disp, call_step, scores, oob = [], [], [], [], []
            while t < task.max_steps:
                last = ep.rows[-1]
                c1_now = bool(last["opening"] < DEFAULTS["close_gap"] and not last["touch"].any())
                agv = task.obs["agentview_image"][::-1, ::-1].copy()
                wr = task.obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
                b = vla.batch_for(agv, wr, task.state8(), task.language)
                n0 = noise_for(vla, tid, init, t, 0).expand(A, -1, -1).contiguous()
                rng = np.random.default_rng(777_000 + 1000 * tid + 17 * init + t)
                lam = np.zeros(A, np.float32) if c1_now else np.asarray(LAM, np.float32)
                env, xs = guided_denoise(vla, head, tap, b, n0, ARMS, lam, rng)

                if not c1_now:
                    with torch.no_grad():
                        ppm, kv = vla._prefix(b, A)
                        tap.reset()
                        vla.policy.model.denoise_step(
                            prefix_pad_masks=ppm, past_key_values=kv, x_t=xs.to("cuda"),
                            timestep=torch.full((A,), 1.0 / vla.K, device="cuda"))
                        sc = head.numpy(tap.pop(1, A).float().cpu().numpy())
                    snap = task.snapshot()
                    ys = [int(branch(task, env[i], snap, args.exec)[0]) for i in range(A)]
                    task.restore(snap)
                    labels.append(ys)
                    scores.append(sc.astype(np.float32))
                    d0 = xs[:, :args.exec, :EX_DIMS].numpy()
                    disp.append(np.linalg.norm((d0 - d0[:1]).reshape(A, -1), axis=1).astype(np.float32))
                    # a large step can push the chunk outside the action range robosuite accepts, which would
                    # make "guidance did nothing" mean "the clip ate it"
                    oob.append((np.abs(env[:, :args.exec, :]) > 1.0).reshape(A, -1).mean(1).astype(np.float32))
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
                                labels=np.array(labels, np.int8).reshape(-1, A),
                                scores=np.array(scores, np.float32).reshape(-1, A),
                                disp=np.array(disp, np.float32).reshape(-1, A),
                                oob=np.array(oob, np.float32).reshape(-1, A),
                                call_step=np.asarray(call_step, np.int32))
            rec = dict(suite=suite, task=tid, init=init, group=group, success=ok, n_steps=t,
                       n_scored_calls=len(labels), npz=f"{name}.npz", lam=list(LAM),
                       secs=round(time.time() - t0, 1))
            with open(index, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            L = np.array(labels).reshape(-1, A) if labels else np.zeros((0, A))
            print(f"{name:26s} 성공={str(ok):5s} 호출={len(labels):3d} "
                  f"C1 {L.mean(0).round(3) if len(L) else '-'} {rec['secs']:6.1f}s", flush=True)

    tap.close()
    print(f"\n총 {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
