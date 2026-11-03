"""Noisy circuits: hardware-efficient ansatz (HEA) and adaptive Pauli-rotation ansatz.

Noise model: every CNOT is followed by a single-qubit depolarising channel of Pauli
error probability ``p`` on each of its two qubits. A weight-2 Pauli rotation costs
two CNOTs, so it receives two such channels per qubit, attached after the exact
rotation. Single-qubit rotations are noiseless.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .dm import (
    cnot,
    conjugate,
    depolarize,
    embed,
    pauli_generator,
    pauli_rotation,
    ry_layer,
    zero_state,
)


def lam_of(p: float) -> float:
    """Depolarising shrink factor for Pauli error probability p."""
    return 4.0 * p / 3.0


def hea_n_params(n: int, depth: int) -> int:
    return (depth + 1) * n


def hea_cnots(n: int, depth: int) -> int:
    return depth * (n - 1)


def hea_state(theta: torch.Tensor, n: int, depth: int, p: float) -> torch.Tensor:
    """Kandala-style RY + CNOT-ladder ansatz, ``depth`` entangling layers."""
    lam = lam_of(p)
    theta = theta.reshape(depth + 1, n)
    rho = zero_state(n)
    for layer in range(depth + 1):
        rho = conjugate(rho, ry_layer(theta[layer]))
        if layer == depth:
            break
        for q in range(n - 1):
            rho = cnot(rho, n, q, q + 1)
            rho = depolarize(rho, n, q, lam)
            rho = depolarize(rho, n, q + 1, lam)
    return rho


@dataclass(frozen=True)
class PoolOp:
    """Pauli-rotation generator exp(-i theta P / 2), P with exactly one Y."""

    label: str
    qubits: tuple[int, ...]
    generator: torch.Tensor

    @property
    def cnots(self) -> int:
        return 2 if len(self.qubits) == 2 else 0


def make_pool(n: int) -> list[PoolOp]:
    """Qubit-ADAPT style pool: Y_i and {ZY, YZ, XY, YX} on nearest neighbours."""
    pool = [PoolOp("Y", (i,), pauli_generator("Y")) for i in range(n)]
    for i in range(n - 1):
        for label in ("ZY", "YZ", "XY", "YX"):
            pool.append(PoolOp(label, (i, i + 1), pauli_generator(label)))
    return pool


def apply_pool_op(
    rho: torch.Tensor, op: PoolOp, theta: torch.Tensor, n: int, p: float
) -> torch.Tensor:
    u = embed(pauli_rotation(theta, op.generator), op.qubits[0], n)
    rho = conjugate(rho, u)
    if op.cnots:
        lam_eff = 1.0 - (1.0 - lam_of(p)) ** op.cnots
        for q in op.qubits:
            rho = depolarize(rho, n, q, lam_eff)
    return rho


def adaptive_state(
    pool: list[PoolOp], ops: list[int], theta: torch.Tensor, n: int, p: float
) -> torch.Tensor:
    rho = zero_state(n)
    for k, j in enumerate(ops):
        rho = apply_pool_op(rho, pool[j], theta[k], n, p)
    return rho
