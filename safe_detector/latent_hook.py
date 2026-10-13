"""Tap pi0.5's internal hidden states without editing lerobot or ~/바탕화면/jungbeom.

`PaliGemmaWithExpertModel.forward` (lerobot/src/lerobot/policies/pi05/modeling_pi05.py:460) branches on which
`inputs_embeds` slot is None and returns `([prefix_hidden, suffix_hidden], past_key_values)` (line 560):

  prefix pass  (Pi05Steer._prefix, pi05_steer.py:69)   inputs_embeds=[pe, None]   -> prefix_hidden (B, Lp, 2048), suffix None
  suffix pass  (denoise_step, modeling_pi05.py:896)    inputs_embeds=[None, se]   -> suffix_hidden (B, Ls, 1024), prefix None

so one tap on that method sees both halves. It has to be a wrapper around the bound method, NOT
`register_forward_hook`: both call sites invoke `.forward(...)` directly, which skips `nn.Module.__call__` and
therefore skips every registered hook. The wrapper lives on the instance and is removed by `close()`; no file in
either repo changes.

The suffix pass runs once per flow step (K = config.num_inference_steps = 10), so one chunk yields 1 prefix state
and K suffix states; `pop()` reduces them to one feature vector per VLA call.

Prefix padding: `_prefix` passes position_ids = cumsum(prefix_pad_mask) - 1, so the mask is recoverable as the
first difference of (position_ids + 1) -- no second tap and no change to the call sites.
"""
from __future__ import annotations

import numpy as np
import torch

# names of the vectors pop() returns, in a fixed order so collectors and the detector agree
FEATURES = ("vlm_mean", "vlm_last", "act_last", "act_mean")
DIMS = {"vlm_mean": 2048, "vlm_last": 2048, "act_last": 1024, "act_mean": 1024}


def _pad_mask_from_positions(pos: torch.Tensor) -> torch.Tensor:
    """(B, L) position_ids = cumsum(mask) - 1  ->  the (B, L) {0,1} mask that produced it."""
    c = pos.float() + 1.0  # == cumsum(mask)
    return torch.cat([c[:, :1], c[:, 1:] - c[:, :-1]], dim=1).clamp(0, 1)


class LatentTap:
    """Collects the hidden states of one `Pi05Steer.denoise` call; `pop()` returns float32 numpy vectors.

    Only the first row of the batch is kept -- the rollout collector samples a single action chunk per step.
    """

    def __init__(self, policy):
        self.mod = policy.model.paligemma_with_expert
        self.chunk_size = int(policy.config.chunk_size)
        self._prefix = None  # (Lp, 2048) float32, valid tokens only
        self._suffix = []  # per flow step: (chunk_size, 1024) float32
        self._orig = self.mod.forward
        self.mod.forward = self._wrapped

    # ------------------------------------------------------------------ tap
    def _wrapped(self, *args, **kwargs):
        out = self._orig(*args, **kwargs)
        try:
            self._capture(kwargs, out)
        except Exception as e:  # a tap must never break the rollout it is observing
            print(f"[LatentTap] capture failed: {e}")
        return out

    def _capture(self, kwargs, out):
        embeds = out[0] if isinstance(out, (tuple, list)) else None
        if not isinstance(embeds, (tuple, list)) or len(embeds) != 2:
            return
        pre, suf = embeds
        if suf is None and pre is not None:
            pos = kwargs.get("position_ids")
            h = pre[0].detach().float()
            if pos is not None:
                h = h[_pad_mask_from_positions(pos)[0].bool()]
            self._prefix = h.cpu()
        elif pre is None and suf is not None:
            self._suffix.append(suf[0, -self.chunk_size :].detach().float().cpu())

    # ------------------------------------------------------------------ use
    def reset(self):
        self._prefix, self._suffix = None, []

    def pop(self) -> dict[str, np.ndarray] | None:
        """One feature vector per name in FEATURES for the chunk just predicted, then clear. None if nothing ran."""
        if self._prefix is None or not self._suffix:
            self.reset()
            return None
        pre, suf = self._prefix, self._suffix
        out = {
            "vlm_mean": pre.mean(0).numpy(),  # (2048) pooled over image + language tokens
            "vlm_last": pre[-1].numpy(),  # (2048) last valid prefix token
            "act_last": suf[-1].mean(0).numpy(),  # (1024) action expert at the final flow step
            "act_mean": torch.stack(suf).mean((0, 1)).numpy(),  # (1024) averaged over all flow steps
        }
        self.reset()
        return {k: v.astype(np.float32) for k, v in out.items()}

    def close(self):
        try:
            del self.mod.forward  # drop the instance attribute, restoring the class method
        except AttributeError:
            self.mod.forward = self._orig

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
