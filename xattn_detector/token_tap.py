"""Token-level tap on pi0.5, with the prefix reduced enough to fit on disk.

safe_detector/latent_hook.py pools each call to one vector. The cross-attention detector needs the tokens, so
this keeps them -- but the full prefix is 968x2048 per call (3.8 MB), which is 51 GB over the 800-rollout set.
What it keeps instead, measured on libero_spatial t0:

  suffix   (50, 1024)   every action token, exactly chunk_size (embed_suffix emits no state or time token)
  image    512 valid tokens = 2 cameras x 16x16 grid  ->  average-pooled by `pool` (2 => 8x8 = 64 per camera)
  language 57 valid tokens  ->  kept whole, never pooled

Language is kept whole on purpose: processor_pi05.py:83 builds the prompt as
`Task: {text}, State: {8 discretised ints};\nAction: `, so the robot's proprioceptive state lives in those
tokens as digits. Pooling them would average unrelated numbers together and destroy the state.

The third image feature (`empty_camera_0`) is masked out by pi0.5 itself -- its 256 tokens carry no information
and are dropped here rather than stored.

`keep_graph=True` leaves the autograd graph attached, which is what the action-gradient test needs.
"""
from __future__ import annotations

import numpy as np
import torch

IMG_GRID = 16  # SigLIP at 224px with patch 14 -> 16x16 = 256 tokens per image


def pad_mask_from_positions(pos: torch.Tensor) -> torch.Tensor:
    """(B, L) position_ids = cumsum(mask) - 1  ->  the (B, L) {0,1} mask that produced it."""
    c = pos.float() + 1.0
    return torch.cat([c[:, :1], c[:, 1:] - c[:, :-1]], dim=1).clamp(0, 1)


class TokenTap:
    """Captures one VLA call's tokens. Wraps the bound `forward` because both call sites bypass nn.Module.__call__."""

    def __init__(self, policy, pool: int = 2, keep_graph: bool = False):
        self.mod = policy.model.paligemma_with_expert
        self.chunk = int(policy.config.chunk_size)
        self.n_img_feats = len(list(policy.config.image_features))
        self.pool = int(pool)
        self.keep_graph = keep_graph
        self._orig = self.mod.forward
        self.mod.forward = self._wrapped
        self.reset()

    # ---------------------------------------------------------------- capture
    def _wrapped(self, *a, **kw):
        out = self._orig(*a, **kw)
        try:
            pre, suf = out[0]
            keep = (lambda t: t) if self.keep_graph else (lambda t: t.detach())
            if suf is None and pre is not None:
                self.prefix = keep(pre[0])
                self.prefix_pos = kw.get("position_ids")
            elif pre is None and suf is not None:
                self.suffix.append(keep(suf[0, -self.chunk :]))
        except Exception as e:  # a tap must never break the rollout it observes
            print(f"[TokenTap] capture failed: {e}")
        return out

    def reset(self):
        self.prefix, self.prefix_pos, self.suffix = None, None, []

    def close(self):
        try:
            del self.mod.forward
        except AttributeError:
            self.mod.forward = self._orig

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---------------------------------------------------------------- read out
    def split_prefix(self):
        """-> (image tokens (n_img, D), language tokens (n_lang, D)), valid only, image tokens pooled."""
        h, pos = self.prefix, self.prefix_pos
        if h is None:
            return None, None
        n_img_slots = IMG_GRID * IMG_GRID * self.n_img_feats
        mask = pad_mask_from_positions(pos)[0].bool() if pos is not None else torch.ones(h.shape[0], dtype=torch.bool, device=h.device)

        imgs = []
        for c in range(self.n_img_feats):
            lo = c * IMG_GRID * IMG_GRID
            hi = lo + IMG_GRID * IMG_GRID
            if not bool(mask[lo:hi].all()):  # pi0.5 masks a whole camera or none of it
                continue
            g = h[lo:hi]  # (256, D)
            if self.pool > 1:
                D = g.shape[1]
                k = IMG_GRID // self.pool
                g = g.reshape(IMG_GRID, IMG_GRID, D)
                g = g.reshape(k, self.pool, k, self.pool, D).mean(dim=(1, 3))
                g = g.reshape(k * k, D)
            imgs.append(g)
        img = torch.cat(imgs, 0) if imgs else h[:0]
        lang = h[n_img_slots:][mask[n_img_slots:]]
        return img, lang

    def pop(self, want_steps=("last", "mean")):
        """One call's tensors, or None if nothing ran. Suffix is returned per requested flow-step reduction."""
        if self.prefix is None or not self.suffix:
            self.reset()
            return None
        img, lang = self.split_prefix()
        out = {"pre_img": img, "pre_lang": lang}
        if "last" in want_steps:
            out["suf_last"] = self.suffix[-1]
        if "mean" in want_steps:
            out["suf_mean"] = torch.stack(self.suffix).mean(0)
        if "all" in want_steps:
            out["suf_all"] = torch.stack(self.suffix)
        self.reset()
        return out


def to_f16(d):
    return {k: v.float().cpu().numpy().astype(np.float16) for k, v in d.items()}
