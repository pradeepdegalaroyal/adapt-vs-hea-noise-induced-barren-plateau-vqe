# Method

## Target problem

Open disordered transverse-field Ising chain of n qubits, H = -sum_i J_i Z_i Z_(i+1) - sum_i g_i X_i with J_i and g_i drawn uniformly from [0.5, 1.5]. The seed selects the disorder instance. The ground-state energy E0 comes from exact diagonalisation. The quality metric is the relative energy error eps = (E - E0) / |E0|, where E is the energy of the noisy density matrix, Tr(H rho).

## Noise model

Every controlled-NOT (CNOT) gate is followed by a single-qubit depolarising channel with Pauli error probability p on each of its two qubits. A weight-two Pauli rotation exp(-i theta P / 2) is charged two CNOTs and therefore two such channels per qubit, attached after the exact rotation. Single-qubit rotations are noiseless. This is a deliberately simple model, chosen because it is the one under which the noise-induced barren plateau (NIBP) theorem of Wang et al., 2021 applies and because it is exactly simulable as a density matrix (n <= 8).

## Methods compared

| Label | Description | Cost accounting |
|---|---|---|
| Mean field (classical baseline) | Best real product state, found by 8-restart Broyden-Fletcher-Goldfarb-Shanno (BFGS) minimisation; independent of p | 0 CNOT, no noise |
| Hardware-efficient ansatz (HEA; standard quantum baseline, Kandala et al., 2017) | Layers of single-qubit rotations about the Y axis (RY) followed by a CNOT ladder, depth L in {1, 2, 4}, random start in [-1, 1] rad, limited-memory BFGS (L-BFGS) on the noisy energy; the depth with the lowest noisy energy is kept ("best depth") | L (n - 1) CNOTs |
| Adaptive ansatz (adaptive derivative-assembled pseudo-Trotter, ADAPT; Grimsley et al., 2019) | Pool of Pauli rotations with one Y factor: Y_i and ZY, YZ, XY, YX on neighbours. At each step the operator with the largest noisy gradient (exact parameter-shift on the noisy state, including its own noise) is appended and all angles are re-optimised on the noisy energy. Run to 3n operators | 2 CNOTs per two-qubit rotation, 0 per single-qubit rotation |

**Stopping rules (ablation).** The trajectory of one ADAPT run is shared. "Full budget" takes the last step. "Noise-aware" takes the step with the lowest measured noisy energy, a quantity available on hardware. The difference isolates what a noise-aware stop is worth.

## Experiments

1. **Gradient scaling.** For n = 4 to 8, HEA with L = n (depth linear in n, the NIBP regime) and with L = 2, angles uniform in [0, 2 pi), 6 random starts per seed, p in {0, 0.002, 0.005, 0.01}. Reported: root-mean-square (RMS) gradient component, fitted exponential decay rate in n, and a single-shot-spread proxy for the shots needed to resolve a gradient, (sqrt(sum_i J_i^2 + sum_i g_i^2) / RMS gradient)^2.
2. **Noisy-energy comparison.** n = 4 to 7, p in {0, 0.001, 0.002, 0.005, 0.01}, three seeds. Reported: eps for every method, CNOT count of the chosen circuit, fraction of seeds in which a method wins.
3. **Break-even noise.** Largest grid p at which a method beats mean field (eps below 0.98 times the mean-field eps) in at least half of the seeds. The resolution is the p grid.
4. **Cost versus quality.** All evaluated circuits at the largest n for three noise levels, Pareto front of error against CNOT count.

## Controls and seeds

Seeds 0, 1, 2 select the disorder instance and the HEA starting angles; the ADAPT run is deterministic given the instance. All reported spreads are the population standard deviation over these seeds. Energies are exact expectation values of the noisy density matrix, so shot noise is not simulated in the energy comparison; shot cost enters only through the gradient-resolution proxy in experiment 1.

## Hypotheses

The three pre-registered hypotheses (H1 gradient decay, H2 adaptive growth, H3 break-even trend) and their falsification criteria are stated in `docs/RESEARCH_GAP.md`.

## Implementation notes

Real-valued density-matrix simulator in PyTorch (float64): RY rotations, CNOT permutations and Pauli rotations with exactly one Y factor keep the state real. Depolarising noise is applied through a closed-form partial-trace expression. Gradients of the HEA come from automatic differentiation; pool gradients come from the parameter-shift rule. Runs are distributed over four single-thread worker processes.
