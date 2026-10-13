"""Four failure detectors over pi0.5 tokens. pi0.5 itself is frozen; only these are trained.

Each call to the VLA produces 50 action tokens, and an episode's prefix8 window is 8 such calls. That gives two
separate axes, and the baselines sit on different ones -- which is the point of comparing them:

        within a call (50 action tokens)        across calls (8 VLA calls)
  A  LogReg      mean, no weighting                   none, each call scored alone
  L  LSTM        mean, no weighting                   recurrent over the 8 calls
  B  AttnPool    learned attention over tokens        none, each call scored alone
  C  CrossAttn   attention over tokens, conditioned   none, each call scored alone
                 on the VLM tokens

So A->L adds time and nothing else; A->B adds token selection and nothing else; A->C adds token selection
conditioned on what the VLM saw. L and B are not competing explanations of the same gain -- they can both help,
or neither.

  A  LogReg          mean-pool the 50 action tokens -> Linear. Reproduces safe_detector's `act_mean` probe,
                     so the new pipeline can be checked against a number that is already known.
  L  LSTM           same mean-pooled vector per call, then a causal LSTM over the 8 calls. Matches the model
                     safe_detector ran, so the two sets of numbers are about the same architecture.
  B  AttnPool        learn which action tokens matter, pool by those weights -> Linear.
  C  CrossAttn       action tokens query the VLM tokens (image + language, the latter carrying the discretised
                     robot state), then pool -> Linear.

On C and redundancy: pi0.5 already cross-attends. `denoise_step` (modeling_pi05.py:883) concatenates the prefix
KV cache into the attention mask, so every one of the action expert's layers reads the VLM. The suffix hidden
state is therefore ALREADY a cross-attended quantity, and C does not open a new information path -- it is a
second, failure-specific read of tokens the policy read for a different purpose. C keeps a residual path to the
same mean-pooled suffix vector A uses, so C is a strict extension of A and the comparison isolates what the
cross-attention head adds rather than what extra parameters add.

On size: the previous LSTM (~590k parameters) memorised 560 training episodes outright -- train AUROC 1.000,
train Brier 0.0018. So the prefix is projected to `d` before attention, dropout is on by default, and
`freeze_down=True` makes the 2048->d projection a fixed random map, which removes 262k trainable parameters.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D_SUF = 1024  # action expert width (gemma_300m)
D_PRE = 2048  # paligemma width (gemma_2b)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class LogRegDet(nn.Module):
    """A -- mean over action tokens, then one linear unit."""

    needs = ("suf",)

    def __init__(self, d_suf: int = D_SUF, dropout: float = 0.0, **_):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_suf, 1)

    def forward(self, suf, pre=None, pre_mask=None):
        # suf (B, T, S, d_suf) -> logit (B, T)
        return self.head(self.drop(suf.mean(2))).squeeze(-1)


class LSTMDet(nn.Module):
    """L -- A's per-call vector, then a causal recurrence over the 8 calls. The architecture safe_detector used.

    Its hidden state can count calls, which on full episodes let it score well without reading the latents at
    all (stratified AUROC 0.503 there). Inside prefix8 every episode has the same 8 calls, so counting is
    worthless and whatever it gains has to come from how the latents move.
    """

    needs = ("suf",)

    def __init__(self, d_suf: int = D_SUF, hidden: int = 128, dropout: float = 0.1, **_):
        super().__init__()
        self.rnn = nn.LSTM(d_suf, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, suf, pre=None, pre_mask=None):
        h, _ = self.rnn(suf.mean(2))  # (B, T, hidden)
        return self.head(self.drop(h)).squeeze(-1)


class AttnPoolDet(nn.Module):
    """B -- additive attention over the action tokens, then one linear unit."""

    needs = ("suf",)

    def __init__(self, d_suf: int = D_SUF, d: int = 128, dropout: float = 0.1, **_):
        super().__init__()
        self.proj = nn.Linear(d_suf, d)
        self.score = nn.Linear(d, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_suf, 1)

    def forward(self, suf, pre=None, pre_mask=None):
        a = self.score(torch.tanh(self.proj(suf))).softmax(dim=2)  # (B, T, S, 1)
        pooled = (a * suf).sum(2)
        return self.head(self.drop(pooled)).squeeze(-1)


class CrossAttnDet(nn.Module):
    """C -- action tokens (query) read the VLM tokens (key/value), plus a residual copy of A's input."""

    needs = ("suf", "pre")

    def __init__(self, d_suf: int = D_SUF, d_pre: int = D_PRE, d: int = 128, heads: int = 4,
                 dropout: float = 0.1, freeze_down: bool = True, **_):
        super().__init__()
        self.d, self.h = d, heads
        assert d % heads == 0
        self.down = nn.Linear(d_pre, d)  # VLM tokens 2048 -> d, before any attention
        if freeze_down:
            for p in self.down.parameters():
                p.requires_grad_(False)
        self.q = nn.Linear(d_suf, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d + d_suf, 1)  # [cross-attended read ; A's mean-pooled suffix]

    def forward(self, suf, pre, pre_mask=None):
        # suf (B, T, S, d_suf)   pre (B, T, P, d_pre)   pre_mask (B, T, P) bool
        B, T, S, _ = suf.shape
        P = pre.shape[2]
        q = self.q(suf).view(B * T, S, self.h, self.d // self.h).transpose(1, 2)
        kv = self.down(pre)
        k = self.k(kv).view(B * T, P, self.h, self.d // self.h).transpose(1, 2)
        v = self.v(kv).view(B * T, P, self.h, self.d // self.h).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.d // self.h)
        if pre_mask is not None:
            att = att.masked_fill(~pre_mask.reshape(B * T, 1, 1, P), float("-inf"))
        ctx = (att.softmax(-1) @ v).transpose(1, 2).reshape(B * T, S, self.d)
        ctx = self.out(ctx).view(B, T, S, self.d).mean(2)  # pool over the 50 queries
        z = torch.cat([self.drop(ctx), self.drop(suf.mean(2))], dim=-1)
        return self.head(z).squeeze(-1)


MODELS = {"logreg": LogRegDet, "lstm": LSTMDet, "attnpool": AttnPoolDet, "xattn": CrossAttnDet}


def build(name: str, **kw) -> nn.Module:
    return MODELS[name](**kw)


if __name__ == "__main__":  # shape and parameter report
    B, T, S, P = 2, 8, 50, 188
    suf = torch.randn(B, T, S, D_SUF)
    pre = torch.randn(B, T, P, D_PRE)
    msk = torch.ones(B, T, P, dtype=torch.bool)
    print(f"{'model':10s} {'params':>10s}  out shape")
    for name in MODELS:
        m = build(name)
        y = m(suf, pre, msk)
        print(f"{name:10s} {n_params(m):>10,}  {tuple(y.shape)}")
    m = build("xattn", freeze_down=False)
    print(f"{'xattn*':10s} {n_params(m):>10,}  (freeze_down=False)")
