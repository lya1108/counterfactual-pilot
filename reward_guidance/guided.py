"""Reward-gradient guided denoising, with the reward evaluated on the WHOLE normalised chunk.

`Pi05Steer.denoise` already implements guidance, but its hook hands the reward only `unnorm(x1[:, :16, :7])`
-- the first 16 steps of the 7 env-unit dimensions. A reward that has to run the action expert needs the full
normalised (50, 32) chunk, so the sampler is mirrored here instead. Everything else is kept identical to
pi05_steer.denoise: Euler flow t 1 -> 0 with dt = -1/K, the clean estimate x1 = x - (t+dt)*v, a gradient
normalised to unit length and applied as (lam/K) * direction after each step.

All arms are denoised in ONE batch from ONE start noise, so a comparison between them isolates the guidance
term: same observation, same prefix, same noise, same kernels. The direction and the step size are chosen per
sample, which is why several arms and several guidance weights cost barely more than one denoise.

The random arm is built by flipping the SIGN of each element of that arm's own gradient, with a pattern fixed
for the call. A direction drawn from an isotropic normal would be a weaker control: the gradient concentrates
its energy on the early steps and the 7 real action dimensions, so an isotropic step of the same length moves
the executed part of the chunk about half as far, and "guidance did more than noise" would then partly mean
"guidance moved further". Random signs on the gradient's own magnitudes keep the energy profile and destroy
the direction.
"""
from __future__ import annotations

import numpy as np
import torch

D_ACT = 7
MODES = ("down", "up", "rand", "none")   # down = descend the C1 logit, i.e. what guidance would actually do


def guided_denoise(vla, head, tap, batch, noise, modes, lam, rng=None):
    """-> (env-unit chunks (n, C, 7), normalised chunks (n, C, A)).

    `modes[i]` picks arm i's direction and `lam[i]` its total step size (a scalar lam applies to every arm).
    """
    mm = vla.policy.model
    n = noise.shape[0]
    K = vla.K
    assert len(modes) == n
    lam = torch.as_tensor(np.broadcast_to(np.asarray(lam, np.float32), (n,)).copy(),
                          device="cuda")[:, None, None]
    with torch.no_grad():
        ppm, kv = vla._prefix(batch, n)
    x = noise.to("cuda", torch.float32)
    dt = -1.0 / K
    sign = torch.tensor([{"down": -1.0, "up": 1.0, "rand": 0.0, "none": 0.0}[m] for m in modes],
                        device="cuda")[:, None, None]
    is_rand = torch.tensor([1.0 if m == "rand" else 0.0 for m in modes], device="cuda")[:, None, None]
    rng = rng or np.random.default_rng(0)
    flip = torch.as_tensor(rng.choice([-1.0, 1.0], size=tuple(x.shape)).astype(np.float32), device="cuda")
    use_grad = bool((lam > 0).any())

    for step in range(K):
        t = 1.0 + step * dt
        tt = torch.tensor(t, dtype=torch.float32, device="cuda").expand(n)
        with torch.no_grad():
            v = mm.denoise_step(prefix_pad_masks=ppm, past_key_values=kv, x_t=x, timestep=tt).float()
        x = x + dt * v
        if not use_grad:
            continue
        x1 = (x - max(t + dt, 1e-3) * v).detach()
        with torch.enable_grad():
            xh = x1.clone().requires_grad_(True)
            tap.reset()
            mm.denoise_step(prefix_pad_masks=ppm, past_key_values=kv, x_t=xh,
                            timestep=torch.full((n,), 1.0 / K, device="cuda"))
            r = head.torch_(tap.pop(1, n))
            (g,) = torch.autograd.grad(r.sum(), xh)
        unit = lambda z: z / z.flatten(1).norm(dim=1).clamp_min(1e-8)[:, None, None]
        d = sign * unit(g) + is_rand * unit(g.abs() * flip)   # same magnitudes, scrambled direction
        x = x + (lam / K) * d
    tap.reset()
    return vla.unnorm(x[:, :, :D_ACT]).cpu().numpy(), x.detach().cpu()
