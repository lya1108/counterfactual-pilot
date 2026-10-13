"""Measure, before designing anything: what token-level tensors pi0.5 actually exposes, and at what cost.

Answers the questions the cross-attention design depends on, with numbers rather than guesses:
  1. prefix hidden  -- how many image tokens, how many language tokens, how many are padding
  2. suffix hidden  -- embed_suffix (modeling_pi05.py:695) emits only chunk_size action tokens, no state token
  3. storage        -- bytes per VLA call and for the existing 800-rollout set, token-level vs pooled
  4. gradient       -- does d(suffix hidden)/d(action chunk) exist? That decides whether a latent detector can
                       ever drive reward guidance, and whether a VLM-only detector is a dead end for it
  5. timing         -- one denoise call, and what an extra cross-attention head adds

Nothing is written outside xattn_detector/. The jungbeom repo and lerobot are read-only.

    conda activate fasi && python xattn_detector/shape_probe.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "safe_detector"))
import jb  # noqa: E402

jb.enter()

from fasi.envs.libero_env import LiberoTask  # noqa: E402
from fasi.vla.pi05_steer import Pi05Steer  # noqa: E402

CKPT = "outputs/vla/pi05_libero_finetuned"


class TokenTap:
    """Like safe_detector/latent_hook.py, but keeps every token -- and optionally the autograd graph."""

    def __init__(self, policy, keep_graph: bool = False):
        self.mod = policy.model.paligemma_with_expert
        self.keep_graph = keep_graph
        self.prefix = None
        self.prefix_pos = None
        self.suffix = []
        self._orig = self.mod.forward
        self.mod.forward = self._wrapped

    def _wrapped(self, *a, **kw):
        out = self._orig(*a, **kw)
        pre, suf = out[0]
        keep = (lambda t: t) if self.keep_graph else (lambda t: t.detach())
        if suf is None and pre is not None:
            self.prefix = keep(pre)
            self.prefix_pos = kw.get("position_ids")
        elif pre is None and suf is not None:
            self.suffix.append(keep(suf))
        return out

    def reset(self):
        self.prefix, self.prefix_pos, self.suffix = None, None, []

    def close(self):
        try:
            del self.mod.forward
        except AttributeError:
            self.mod.forward = self._orig


def mb(n_bytes):
    return n_bytes / 1024 ** 2


def main():
    print("=" * 78)
    print("1. 체크포인트와 설정")
    vla = Pi05Steer(CKPT)
    cfg = vla.policy.config
    print(f"  chunk_size C         {vla.C}")
    print(f"  num_inference_steps K {vla.K}")
    print(f"  max_action_dim A     {vla.A}")
    print(f"  image_features       {list(cfg.image_features)}")
    print(f"  paligemma variant    {cfg.paligemma_variant}   action expert {cfg.action_expert_variant}")

    task = LiberoTask("libero_spatial", 0)
    task.reset(0)
    ag = task.obs["agentview_image"][::-1, ::-1].copy()
    wr = task.obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
    batch = vla.batch_for(ag, wr, task.state8(), task.language)
    print(f"\n  프롬프트             {task.language!r}")
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"  batch[{k:38s}] {tuple(v.shape)} {v.dtype}")

    print("\n" + "=" * 78)
    print("2. token-level latent 실측")
    tap = TokenTap(vla.policy)
    g = torch.Generator().manual_seed(0)
    noise = torch.randn((1, vla.C, vla.A), generator=g)
    t0 = time.time()
    _ = vla.denoise(batch, noise, [], None)
    dt_denoise = time.time() - t0

    pre, pos, suf = tap.prefix, tap.prefix_pos, tap.suffix
    valid = None
    if pos is not None:
        c = pos.float() + 1.0
        mask = torch.cat([c[:, :1], c[:, 1:] - c[:, :-1]], 1).clamp(0, 1)[0]
        valid = int(mask.sum().item())
    print(f"  prefix hidden        {tuple(pre.shape)}  {pre.dtype}   유효 토큰 {valid}/{pre.shape[1]}")
    print(f"  suffix hidden        {tuple(suf[0].shape)}  {suf[0].dtype}   × flow step {len(suf)}개")
    print(f"  → suffix 토큰 수 {suf[0].shape[1]} == chunk_size {vla.C}: "
          f"{'예 (state/time 토큰 없음)' if suf[0].shape[1] == vla.C else '아니오'}")

    # how the prefix splits into image vs language
    n_img_feats = len(vla.feats)
    imgs, img_masks = vla.policy._preprocess_images(batch)
    per_img = vla.policy.model.paligemma_with_expert.embed_image(imgs[0].cuda()).shape[1]
    n_img_tok = per_img * len(imgs)
    n_lang_valid = int(batch["observation.language.attention_mask"].sum())
    n_img_valid = (valid or 0) - n_lang_valid
    print(f"  이미지 토큰          {per_img} × {len(imgs)}장 = {n_img_tok}  (유효 {n_img_valid} "
          f"= 실제 카메라 {n_img_valid // per_img}대, empty_camera_0는 마스크됨)")
    print(f"  언어 토큰            {pre.shape[1] - n_img_tok} 자리  (유효 {n_lang_valid})")

    print("\n" + "=" * 78)
    print("3. 저장량 — 지금 데이터(800 롤아웃)를 token-level로 다시 모으면")
    import json, glob
    calls = 0
    for f in glob.glob(str(Path.home() / "yuna_ws/safe_detector/data/*/episodes.jsonl")):
        for line in open(f):
            if line.strip():
                calls += json.loads(line)["n_calls"]
    P, Dp = pre.shape[1], pre.shape[2]
    S, Ds = suf[0].shape[1], suf[0].shape[2]
    b_suf = S * Ds * 2
    b_pre = P * Dp * 2
    b_pre_valid = (valid or P) * Dp * 2
    print(f"  전체 VLA 호출 수     {calls}")
    print(f"  호출당 suffix        {mb(b_suf):8.3f} MB  ({S}×{Ds} fp16)")
    print(f"  호출당 prefix(전체)  {mb(b_pre):8.3f} MB  ({P}×{Dp} fp16)")
    print(f"  호출당 prefix(유효)  {mb(b_pre_valid):8.3f} MB")
    print(f"  800 롤아웃 suffix만  {mb(b_suf * calls) / 1024:8.2f} GB")
    print(f"  800 롤아웃 suffix+prefix(유효) {mb((b_suf + b_pre_valid) * calls) / 1024:8.2f} GB")
    print(f"  참고: 현재 pooled 저장 = 241 MB")

    print("\n" + "=" * 78)
    print("4. gradient — suffix latent는 action chunk에 대해 미분 가능한가")
    tap.close()
    tap2 = TokenTap(vla.policy, keep_graph=True)
    # the prefix does not see the action, so it is a constant here -- building a graph through PaliGemma 2B for
    # it would only waste memory
    with torch.no_grad():
        ppm, kv = vla._prefix(batch, 1)
    tap2.reset()
    x = torch.randn(1, vla.C, vla.A, device="cuda", requires_grad=True)
    tt = torch.tensor(1.0, device="cuda").expand(1)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.enable_grad():
        v = vla.policy.model.denoise_step(prefix_pad_masks=ppm, past_key_values=kv, x_t=x, timestep=tt)
        h_suf = tap2.suffix[-1]  # (1, 50, 1024), graph intact
        loss = h_suf.float().pow(2).mean()
        (gx,) = torch.autograd.grad(loss, x, retain_graph=False)
    dt_grad = time.time() - t0
    print(f"  suffix hidden requires_grad   {h_suf.requires_grad}")
    print(f"  d(suffix)/d(action) 노름      {gx.norm().item():.6f}   (0이면 연결 없음)")
    print(f"  0이 아닌 원소                 {int((gx != 0).sum())}/{gx.numel()}")
    print(f"  prefix hidden은 action에 의존? {'아니오 — action 입력 전에 계산됨' }")
    print(f"  1회 forward+backward          {dt_grad * 1000:6.1f} ms, peak {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

    print("\n" + "=" * 78)
    print("5. 비용")
    print(f"  denoise 1회 (K={vla.K} step)   {dt_denoise * 1000:6.1f} ms")
    d_q, d_kv, heads = Ds, Dp, 4
    xattn_params = (d_q * d_q + d_q) + 2 * (d_kv * d_q + d_q) + (d_q * d_q + d_q) + (d_q + 1)
    attn_pool_params = (d_q * d_q + d_q) + (d_q + 1) + (d_q + 1)
    logreg_params = d_q + 1
    print(f"  파라미터 수 (학습되는 부분만, π0.5는 frozen)")
    print(f"    A  LogReg            {logreg_params:>10,}")
    print(f"    B  Attention pooling {attn_pool_params:>10,}")
    print(f"    C  Cross-Attention   {xattn_params:>10,}   (q/k/v/out proj + head, {heads}-head, d={d_q})")
    tap2.close()
    task.close()


if __name__ == "__main__":
    main()
