"""Suffix tap that keeps the batch dimension, so N candidate action chunks are scored from one denoise call.

xattn_detector/token_tap.py keeps `suf[0]` only -- it was written for single-sample collection and silently
drops candidates 1..N-1. Rather than change it and invalidate the runs already made with it, this is a separate
class with the one difference that matters here.

`Pi05Steer.denoise` sub-batches by `self.batch`, so `vla.batch` must be >= N for the K captured tensors to
belong to one group of candidates; `pop` checks that and refuses rather than averaging two groups together.
"""
from __future__ import annotations

import torch


class SuffixBatchTap:
    """Captures the action expert's suffix hidden states for every candidate in the batch."""

    def __init__(self, policy, keep_graph: bool = False):
        self.mod = policy.model.paligemma_with_expert
        self.chunk = int(policy.config.chunk_size)
        self.keep_graph = keep_graph
        self._orig = self.mod.forward
        self.mod.forward = self._wrapped
        self.suffix = []

    def _wrapped(self, *a, **kw):
        out = self._orig(*a, **kw)
        try:
            pre, suf = out[0]
            if pre is None and suf is not None:
                t = suf[:, -self.chunk:]
                self.suffix.append(t if self.keep_graph else t.detach())
        except Exception as e:  # a tap must never break the rollout it observes
            print(f"[SuffixBatchTap] capture failed: {e}")
        return out

    def reset(self):
        self.suffix = []

    def close(self):
        try:
            del self.mod.forward
        except AttributeError:
            self.mod.forward = self._orig

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def pop(self, K: int, n: int):
        """-> (n, 1024) suf_mean, averaged over the K denoising steps and the 50 action tokens."""
        if len(self.suffix) != K:
            raise RuntimeError(f"denoise steps captured = {len(self.suffix)}, expected {K}; "
                               f"set vla.batch >= n so all candidates share one sub-batch")
        s = torch.stack(self.suffix)                 # (K, n, chunk, D)
        if s.shape[1] != n:
            raise RuntimeError(f"batch captured = {s.shape[1]}, expected {n}")
        self.reset()
        return s.mean(dim=(0, 2))                    # (n, D)
