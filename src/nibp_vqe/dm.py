"""Real-valued density-matrix simulator with gate-attached depolarising noise.

All circuits in this study use only real orthogonal gates (RY rotations, CNOT and
Pauli rotations with exactly one Y factor), so the state stays real and a float64
matrix of shape (2^n, 2^n) holds rho. Qubit 0 is the most significant bit.
Noise and CNOTs are applied as precomputed flat-index gathers; rotations are
embedded into full 2^n x 2^n orthogonal matrices.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from .hamiltonian import I2, X, Y_IMAG, Z


def zero_state(n: int) -> torch.Tensor:
    rho = torch.zeros((2**n, 2**n), dtype=torch.float64)
    rho[0, 0] = 1.0
    return rho


@lru_cache(maxsize=None)
def _cnot_table(n: int, control: int, target: int) -> torch.Tensor:
    """Flat gather index implementing rho -> P rho P^T for a CNOT permutation P."""
    dim = 2**n
    cb, tb = 1 << (n - 1 - control), 1 << (n - 1 - target)
    basis = np.arange(dim)
    perm = np.where(basis & cb, basis ^ tb, basis)  # involution
    return torch.tensor((perm[:, None] * dim + perm[None, :]).reshape(-1), dtype=torch.long)


def cnot(rho: torch.Tensor, n: int, control: int, target: int) -> torch.Tensor:
    return rho.reshape(-1)[_cnot_table(n, control, target)].reshape(rho.shape)


def depolarize(rho: torch.Tensor, n: int, q: int, lam: float) -> torch.Tensor:
    """Single-qubit depolarising channel rho -> (1-lam) rho + lam Tr_q(rho) (x) I/2.

    A Pauli error probability p per gate corresponds to lam = 4p/3.
    """
    if lam == 0.0:
        return rho
    dim = 2**n
    v = rho.reshape(2**q, 2, 2 ** (n - q - 1), 2**q, 2, 2 ** (n - q - 1))
    reduced = v[:, 0, :, :, 0, :] + v[:, 1, :, :, 1, :]
    add = torch.zeros_like(v)
    add[:, 0, :, :, 0, :] = reduced
    add[:, 1, :, :, 1, :] = reduced
    return ((1.0 - lam) * v + (0.5 * lam) * add).reshape(dim, dim)


def embed(u: torch.Tensor, first: int, n: int) -> torch.Tensor:
    """Full-register matrix for a gate ``u`` on consecutive qubits starting at ``first``."""
    k = int(np.log2(u.shape[0]))
    left, right = 2**first, 2 ** (n - first - k)
    eye = torch.eye
    return torch.kron(torch.kron(eye(left, dtype=torch.float64), u), eye(right, dtype=torch.float64))


def conjugate(rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """rho -> U rho U^T for a real orthogonal full-register U."""
    return u @ rho @ u.T


def energy(rho: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Tr(H rho) for symmetric real H."""
    return torch.sum(h * rho)


def ry(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def ry_layer(thetas: torch.Tensor) -> torch.Tensor:
    """Kronecker product of RY rotations, one per qubit."""
    out = ry(thetas[0])
    for t in thetas[1:]:
        out = torch.kron(out, ry(t))
    return out


def pauli_rotation(theta: torch.Tensor, generator: torch.Tensor) -> torch.Tensor:
    """exp(-i theta P / 2) = cos(theta/2) I + sin(theta/2) (-iP) with -iP real."""
    eye = torch.eye(generator.shape[0], dtype=torch.float64)
    return torch.cos(theta / 2) * eye + torch.sin(theta / 2) * generator


def pauli_generator(label: str) -> torch.Tensor:
    """Real matrix -iP for a Pauli string with exactly one Y factor (e.g. 'ZY')."""
    if label.count("Y") != 1:
        raise ValueError("generator must contain exactly one Y factor")
    mats = {"I": I2, "X": X, "Z": Z, "Y": Y_IMAG}
    out = np.ones((1, 1))
    for ch in label:
        out = np.kron(out, mats[ch])
    # Y_IMAG = -iY carries the single factor -i, so the product is -iP.
    return torch.tensor(out, dtype=torch.float64)
