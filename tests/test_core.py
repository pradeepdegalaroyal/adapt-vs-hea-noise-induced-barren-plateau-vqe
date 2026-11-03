"""Fast correctness tests for the density-matrix simulator and the VQE drivers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nibp_vqe.circuits import adaptive_state, hea_state, make_pool
from nibp_vqe.dm import cnot, depolarize, energy, zero_state
from nibp_vqe.hamiltonian import make_tfim
from nibp_vqe.meanfield import meanfield_energy
from nibp_vqe.optimise import pool_gradients, run_hea


def test_hamiltonian_ground_energy_is_lowest_eigenvalue():
    m = make_tfim(4, 0)
    assert m.e0 == pytest.approx(float(torch.linalg.eigvalsh(m.H)[0]))
    assert m.sigma > 0


def test_noiseless_hea_matches_statevector():
    m = make_tfim(3, 1)
    theta = torch.rand(6, dtype=torch.float64)
    rho = hea_state(theta, 3, 1, 0.0)
    assert torch.trace(rho).item() == pytest.approx(1.0)
    assert torch.allclose(rho @ rho, rho, atol=1e-10)  # pure state
    assert energy(rho, m.H).item() >= m.e0 - 1e-9  # variational bound


def test_depolarising_preserves_trace_and_full_noise_gives_identity_on_qubit():
    rho = hea_state(torch.rand(6, dtype=torch.float64), 3, 1, 0.0)
    noisy = depolarize(rho, 3, 1, 0.75 * 4 / 3)  # lam = 1: qubit fully depolarised
    assert torch.trace(noisy).item() == pytest.approx(1.0)
    r = noisy.reshape(2, 2, 2, 2, 2, 2)  # (q0 q1 q2 | q0' q1' q2')
    assert torch.allclose(r[:, 0, :, :, 1, :], torch.zeros_like(r[:, 0, :, :, 1, :]), atol=1e-12)
    assert torch.linalg.eigvalsh(noisy).min().item() > -1e-12


def test_cnot_is_involution():
    rho = zero_state(3)
    rho[0, 0] = 0.5
    rho[4, 4] = 0.5
    once = cnot(rho, 3, 0, 1)
    assert not torch.allclose(once, rho)
    assert torch.allclose(cnot(once, 3, 0, 1), rho)


def test_noise_raises_energy_of_optimised_hea():
    m = make_tfim(4, 0)
    clean = run_hea(m, 2, 0.0, 0, max_iter=40)["energy"]
    noisy = run_hea(m, 2, 0.02, 0, max_iter=40)["energy"]
    assert noisy > clean


def test_pool_gradient_matches_autograd():
    m = make_tfim(3, 2)
    pool = make_pool(3)
    ops = [0, 4]
    theta = torch.tensor([0.4, -0.3], dtype=torch.float64)
    g = pool_gradients(pool, ops, theta, m, 0.01)
    j = 5
    t = torch.cat([theta, torch.zeros(1, dtype=torch.float64)]).requires_grad_(True)
    e = energy(adaptive_state(pool, ops + [j], t, 3, 0.01), m.H)
    (grad,) = torch.autograd.grad(e, t)
    assert g[j] == pytest.approx(float(grad[-1]), abs=1e-9)


def test_meanfield_is_variational_upper_bound():
    m = make_tfim(4, 0)
    assert meanfield_energy(m) >= m.e0 - 1e-9
    assert np.isfinite(meanfield_energy(m))
