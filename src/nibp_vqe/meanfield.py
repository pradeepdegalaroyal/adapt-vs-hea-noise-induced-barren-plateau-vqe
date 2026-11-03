"""Classical baseline: best real product state (mean-field) for the Ising chain."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .hamiltonian import TFIM


def meanfield_energy(model: TFIM, restarts: int = 8, seed: int = 0) -> float:
    """Minimise <H> over product states cos(a_i)|0> + sin(a_i)|1>."""
    rng = np.random.default_rng(seed)

    def energy(a: np.ndarray) -> float:
        z, x = np.cos(2 * a), np.sin(2 * a)
        return float(-np.sum(model.J * z[:-1] * z[1:]) - np.sum(model.g * x))

    best = np.inf
    for _ in range(restarts):
        res = minimize(energy, rng.uniform(0, np.pi, model.n), method="BFGS")
        best = min(best, float(res.fun))
    return best
