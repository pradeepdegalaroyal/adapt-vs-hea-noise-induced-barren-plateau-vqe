"""Gradient-magnitude statistics of the hardware-efficient ansatz at random starts."""

from __future__ import annotations

import numpy as np
import torch

from .circuits import hea_cnots, hea_n_params, hea_state
from .dm import energy
from .hamiltonian import TFIM


def random_start_gradients(model: TFIM, depth: int, p: float, n_starts: int, seed: int) -> dict:
    """RMS energy-gradient component over ``n_starts`` uniform random angle sets."""
    gen = torch.Generator().manual_seed(104729 * seed + depth)
    sq, per_start = [], []
    for _ in range(n_starts):
        theta = (2 * np.pi * torch.rand(hea_n_params(model.n, depth), generator=gen,
                                        dtype=torch.float64)).requires_grad_(True)
        e = energy(hea_state(theta, model.n, depth, p), model.H)
        (grad,) = torch.autograd.grad(e, theta)
        sq.append(float((grad**2).mean()))
        per_start.append(float(e.detach()))
    rms = float(np.sqrt(np.mean(sq)))
    return {
        "n": model.n,
        "depth": depth,
        "p": p,
        "cnots": hea_cnots(model.n, depth),
        "grad_rms": rms,
        "shots_to_resolve": (model.sigma / rms) ** 2 if rms > 0 else float("inf"),
        "mean_random_energy": float(np.mean(per_start)),
    }
