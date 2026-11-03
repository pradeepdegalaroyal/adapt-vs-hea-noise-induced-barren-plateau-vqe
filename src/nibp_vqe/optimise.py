"""Optimisation drivers: HEA energy minimisation and noisy ADAPT growth."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .circuits import (
    PoolOp,
    adaptive_state,
    apply_pool_op,
    hea_cnots,
    hea_n_params,
    hea_state,
    make_pool,
)
from .dm import energy
from .hamiltonian import TFIM


def lbfgs(fn, theta0: torch.Tensor, max_iter: int = 80) -> tuple[torch.Tensor, float]:
    """Minimise a scalar torch function with strong-Wolfe L-BFGS."""
    theta = theta0.clone().detach().requires_grad_(True)
    opt = torch.optim.LBFGS(
        [theta],
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=1e-8,
        tolerance_change=1e-11,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = fn(theta)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return theta.detach(), float(fn(theta))


def run_hea(model: TFIM, depth: int, p: float, seed: int, max_iter: int = 80) -> dict:
    """Noisy-energy minimisation of the RY + CNOT-ladder ansatz from a random start."""
    gen = torch.Generator().manual_seed(7919 * seed + 31 * depth + model.n)
    theta0 = (2 * torch.rand(hea_n_params(model.n, depth), generator=gen, dtype=torch.float64) - 1)
    theta, e = lbfgs(lambda t: energy(hea_state(t, model.n, depth, p), model.H), theta0, max_iter)
    return {
        "depth": depth,
        "n_params": theta.numel(),
        "cnots": hea_cnots(model.n, depth),
        "energy": e,
    }


@dataclass
class AdaptTrace:
    """Per-step record of a noisy ADAPT run (step 0 is the reference state)."""

    ops: list[str] = field(default_factory=list)
    cnots: list[int] = field(default_factory=lambda: [0])
    energy: list[float] = field(default_factory=list)
    max_grad: list[float] = field(default_factory=list)


def pool_gradients(
    pool: list[PoolOp], ops: list[int], theta: torch.Tensor, model: TFIM, p: float
) -> np.ndarray:
    """Exact noisy gradient of every pool operator appended at angle 0 (parameter shift)."""
    n = model.n
    with torch.no_grad():
        rho = adaptive_state(pool, ops, theta, n, p)
        half_pi = torch.tensor(np.pi / 2, dtype=torch.float64)
        grads = np.zeros(len(pool))
        for j, op in enumerate(pool):
            e_plus = energy(apply_pool_op(rho, op, half_pi, n, p), model.H)
            e_minus = energy(apply_pool_op(rho, op, -half_pi, n, p), model.H)
            grads[j] = 0.5 * float(e_plus - e_minus)
    return grads


def run_adapt(model: TFIM, p: float, max_ops: int, max_iter: int = 60) -> AdaptTrace:
    """Grow the ansatz one Pauli rotation at a time on the noisy device.

    The operator with the largest noisy gradient is appended and all angles are
    re-optimised on the noisy energy. The run is continued to ``max_ops`` so that
    different stopping rules can be applied to the same trajectory afterwards.
    """
    n = model.n
    pool = make_pool(n)
    trace = AdaptTrace()
    ops: list[int] = []
    theta = torch.zeros(0, dtype=torch.float64)
    with torch.no_grad():
        trace.energy.append(float(energy(adaptive_state(pool, ops, theta, n, p), model.H)))
    trace.max_grad.append(float("nan"))
    for _ in range(max_ops):
        grads = pool_gradients(pool, ops, theta, model, p)
        j = int(np.argmax(np.abs(grads)))
        ops.append(j)
        theta = torch.cat([theta, torch.zeros(1, dtype=torch.float64)])
        theta, e = lbfgs(
            lambda t: energy(adaptive_state(pool, ops, t, n, p), model.H), theta, max_iter
        )
        trace.ops.append(pool[j].label + "@" + "".join(str(q) for q in pool[j].qubits))
        trace.cnots.append(trace.cnots[-1] + pool[j].cnots)
        trace.energy.append(e)
        trace.max_grad.append(float(np.abs(grads).max()))
    return trace


def noise_aware_stop(energies: list[float]) -> int:
    """Index of the step with the lowest measured noisy energy (best prefix)."""
    return int(np.argmin(energies))
