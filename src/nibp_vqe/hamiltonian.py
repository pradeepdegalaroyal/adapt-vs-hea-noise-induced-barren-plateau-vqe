"""Disordered transverse-field Ising chain used as the VQE target."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

I2 = np.eye(2)
X = np.array([[0.0, 1.0], [1.0, 0.0]])
Z = np.array([[1.0, 0.0], [0.0, -1.0]])
Y_IMAG = np.array([[0.0, -1.0], [1.0, 0.0]])  # equals -i * Pauli-Y (real)


@dataclass(frozen=True)
class TFIM:
    """H = -sum_i J_i Z_i Z_{i+1} - sum_i g_i X_i on an open chain of n qubits."""

    n: int
    J: np.ndarray
    g: np.ndarray
    H: torch.Tensor  # dense (2^n, 2^n) float64
    e0: float  # exact ground-state energy

    @property
    def sigma(self) -> float:
        """Root-sum-square of the Pauli coefficients (single-shot spread proxy)."""
        return float(np.sqrt(np.sum(self.J**2) + np.sum(self.g**2)))


def _kron_chain(ops: list[np.ndarray]) -> np.ndarray:
    out = np.ones((1, 1))
    for op in ops:
        out = np.kron(out, op)
    return out


def make_tfim(n: int, seed: int, disorder: float = 0.5) -> TFIM:
    """Draw couplings J_i, g_i ~ U(1 - disorder, 1 + disorder) and build H densely."""
    rng = np.random.default_rng(1000 * n + seed)
    J = rng.uniform(1 - disorder, 1 + disorder, n - 1)
    g = rng.uniform(1 - disorder, 1 + disorder, n)
    dim = 2**n
    H = np.zeros((dim, dim))
    for i in range(n - 1):
        ops = [I2] * n
        ops[i], ops[i + 1] = Z, Z
        H -= J[i] * _kron_chain(ops)
    for i in range(n):
        ops = [I2] * n
        ops[i] = X
        H -= g[i] * _kron_chain(ops)
    e0 = float(np.linalg.eigvalsh(H)[0])
    return TFIM(n=n, J=J, g=g, H=torch.tensor(H, dtype=torch.float64), e0=e0)
