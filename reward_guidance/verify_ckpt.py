"""Is the re-downloaded checkpoint the one that produced cond200?

The Hub repo moved between collection and today (stale cache ref 8e174154, Hub now 7152f13e), and every result
in this project -- the 0.891 head included -- is tied to the weights that generated those rollouts. A head
fitted on one policy's latents is meaningless on another's.

The sharp test is call 0 of each episode: same reset, same observation, same noise seed, so `suf_mean` is a
deterministic function of the weights alone. Later calls also depend on the trajectory, which is known to
drift across processes (97% of episodes identical, up to 1.8 cm for the rest), so they are reported but not
used to decide.

The verdict is not "are the vectors bit-identical" -- bf16 kernels are not reproducible across GPUs, and this
machine's card changed too. It is "does the frozen head read them the same way": the recomputed and stored
latents are both pushed through the 0.891 head, and the logit shift is compared with the spread of logits the
head produces across cond200. A shift that is a per-cent of that spread changes no ranking; a different
fine-tune would move it by multiples.

    conda activate fasi
    python reward_guidance/verify_ckpt.py --inits 30-32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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

from detector import C1Head  # noqa: E402
from tap import SuffixBatchTap  # noqa: E402
from vla_setup import make_vla  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(X / "data/cond200"))
    ap.add_argument("--groups", nargs="+", default=["libero_object/t6", "libero_spatial/t9", "libero_goal/t2"])
    ap.add_argument("--inits", default="30,31,32")
    ap.add_argument("--calls", type=int, default=3, help="앞쪽 몇 호출까지 비교할지")
    ap.add_argument("--exec", type=int, default=10)
    ap.add_argument("--head", default=str(HERE / "results/c1_head.pt"))
    args = ap.parse_args()

    root = Path(args.data) if Path(args.data).is_absolute() else HERE / args.data
    head = C1Head(args.head)
    # the scale any logit shift has to be judged against
    import numpy as _np
    ref_logits = _np.concatenate([head.numpy(_np.load(f)['suf_mean'].astype(_np.float32))
                                  for f in sorted(root.glob('*.npz'))[:60]])
    spread = float(ref_logits.std())
    print(f'cond200 에서 head logit 표준편차 {spread:.3f}  (판정 기준 척도)\n')
    vla = make_vla(batch=1)
    tap = SuffixBatchTap(vla.policy)
    rows = []
    for group in args.groups:
        suite, tstr = group.split("/")
        tid = int(tstr[1:])
        task = LiberoTask(suite, tid)
        for init in [int(x) for x in args.inits.split(",")]:
            f = root / f"{suite}_t{tid}_i{init}.npz"
            if not f.exists():
                print(f"건너뜀 (저장된 에피소드 없음): {f.name}")
                continue
            z = np.load(f)
            ref = z["suf_mean"].astype(np.float32)
            task.reset(init)
            t = 0
            errs, cos, dlog = [], [], []
            for k in range(min(args.calls, len(ref))):
                ag = task.obs["agentview_image"][::-1, ::-1].copy()
                wr = task.obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
                b = vla.batch_for(ag, wr, task.state8(), task.language)
                g = torch.Generator().manual_seed(1000 * tid + 17 * init + t)
                tap.reset()
                a = vla.denoise(b, torch.randn((1, vla.C, vla.A), generator=g), [], None)
                new = tap.pop(vla.K, 1).float().cpu().numpy()[0]
                r = ref[k]
                errs.append(float(np.abs(new - r).max()))
                cos.append(float(new @ r / (np.linalg.norm(new) * np.linalg.norm(r) + 1e-12)))
                dlog.append(float(head.numpy(new)[0] - head.numpy(r)[0]))
                for j in range(args.exec):
                    task.step(a[0][j])
                    t += 1
                    if task.success():
                        break
            rows.append(dict(ep=f"{suite}_t{tid}_i{init}", call0_maxabs=errs[0], call0_cos=cos[0],
                             call0_dlogit=dlog[0], maxabs=errs, cos=cos, dlogit=dlog,
                             ref_scale=float(np.abs(ref[0]).max())))
            print(f"{rows[-1]['ep']:26s} call0 cos={cos[0]:.6f}  Δlogit={dlog[0]:+.4f}   "
                  f"이후 호출 Δlogit={['%+.3f' % d for d in dlog[1:]]}")
    tap.close()

    c0 = np.array([r["call0_cos"] for r in rows])
    d0 = np.array([r["call0_dlogit"] for r in rows])
    dall = np.array([d for r in rows for d in r["dlogit"]])
    print(f"\ncall 0 ({len(rows)} 에피소드): cos 최소 {c0.min():.6f}   |Δlogit| 최대 {np.abs(d0).max():.4f}")
    print(f"모든 호출 ({len(dall)}개):      |Δlogit| 최대 {np.abs(dall).max():.4f}  "
          f"평균 {np.abs(dall).mean():.4f}")
    frac = float(np.abs(d0).max() / spread)
    print(f"head logit 표준편차 대비 최대 이동: {frac:.1%}")
    same = bool(c0.min() > 0.999 and frac < 0.05)
    print("판정:", f"같은 정책이다 — 재계산된 잠재변수를 head 가 사실상 같게 읽는다 (최대 {frac:.1%} 이동)" if same
          else f"차이가 크다 — head 의 점수가 {frac:.1%} 만큼 흔들린다, 재수집 없이 재사용하면 안 된다")
    json.dump(dict(rows=rows, same=same, call0_cos_min=float(c0.min()),
                   call0_dlogit_absmax=float(np.abs(d0).max()), dlogit_absmax=float(np.abs(dall).max()),
                   dlogit_absmean=float(np.abs(dall).mean()), logit_spread_cond200=spread,
                   shift_vs_spread=frac),
              open(HERE / "results/verify_ckpt.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
