# Research gap

## State of the art

Variational quantum eigensolvers (VQE) prepare a trial state with a parametrised circuit and minimise the measured energy. Kandala et al., 2017, doi:10.1038/nature23879 made the hardware-efficient ansatz (HEA), alternating single-qubit rotations and a fixed entangling pattern, the default for noisy devices. O'Malley et al., 2016, doi:10.1103/PhysRevX.6.031007 showed that VQE tolerates some coherent errors. The review of Tilly et al., 2022, doi:10.1016/j.physrep.2022.08.003 lists vanishing gradients and the effect of quantum noise as two of four open problems.

Trainability theory is now well developed. Cerezo et al., 2021, doi:10.1038/s41467-021-21728-w proved that global cost functions give exponentially vanishing gradients (barren plateaus) while local costs allow polynomial gradients for logarithmic depth. Wang et al., 2021, doi:10.1038/s41467-021-27045-6 proved that local Pauli noise produces a noise-induced barren plateau (NIBP) whenever the depth grows linearly with the number of qubits n. The review of Larocca et al., 2025, doi:10.1038/s42254-025-00813-9 states that ansatz, initial state, observable, loss and hardware noise can each cause a barren plateau. Sim et al., 2019, doi:10.1002/qute.201900070 supplies expressibility and entangling-capability descriptors for choosing compact circuits.

A structural remedy is to grow the circuit instead of fixing it. Grimsley et al., 2019, doi:10.1038/s41467-019-10988-2 (adaptive derivative-assembled pseudo-Trotter VQE, ADAPT-VQE) appends one operator at a time, chosen by its gradient, and reports shallow circuits with a small parameter count, but its evidence is noiseless.

## Related work

| Paper | Year | Role | What it does | What it does NOT cover |
|---|---|---|---|---|
| Kandala et al., doi:10.1038/nature23879 | 2017 | landmark | Hardware-efficient VQE on six superconducting qubits | Scaling of trainability with qubits and noise; no growth rule |
| O'Malley et al., doi:10.1103/PhysRevX.6.031007 | 2016 | landmark | Two-qubit hydrogen VQE, robustness to some errors | Larger registers, depolarising-noise sweeps |
| Schuld et al., doi:10.1103/PhysRevA.101.032308 | 2020 | landmark | Circuit-centric classifier; depth tied to data structure; noise resilience claim | Ground-state energy tasks; a break-even noise level |
| Grimsley et al., doi:10.1038/s41467-019-10988-2 | 2019 | landmark | ADAPT-VQE: gradient-ranked operator growth | Behaviour when the gradients and the energies are themselves noisy |
| Sim et al., doi:10.1002/qute.201900070 | 2019 | landmark | Expressibility and entangling-capability descriptors | Link between these descriptors and noisy-energy accuracy |
| Cerezo et al., doi:10.1038/s41467-021-21728-w | 2021 | landmark | Proof: cost locality controls barren plateaus | Hardware noise; concrete finite-size numbers |
| Wang et al., doi:10.1038/s41467-021-27045-6 | 2021 | landmark | Proof of noise-induced barren plateaus | Whether an adaptively grown circuit escapes them; finite-size onset |
| Tilly et al., doi:10.1016/j.physrep.2022.08.003 | 2022 | landmark (review of VQE) | Methods and best practices; four open problems | A measured comparison of ansatz families on a shared noise model |
| Larocca et al., doi:10.1038/s42254-025-00813-9 | 2025 | review | Barren-plateau theory, mitigation, links to other fields | Cost accounting against a free classical baseline |
| Huang and von Lilienfeld, doi:10.1021/acs.chemrev.0c01303 | 2021 | review | Machine learning for chemical compound space | Variational quantum circuits |
| Peng et al., doi:10.1016/j.optcom.2026.133879 | 2027 | adjacent | Semi-quantum protocol with explicit parameter-estimation bounds and loss cut-off distance | Variational algorithms; its habit of reporting a positive-rate cutoff is the template borrowed here |
| Akrom et al., doi:10.1016/j.optcom.2026.133925 | 2027 | adjacent | Migration-aware benchmark of standardised cryptography with statistical tests | Variational algorithms; it motivates reporting cost and significance together |

## Gap

The theory proves that noise flattens the landscape asymptotically, and ADAPT-VQE is argued to avoid deep circuits, yet three practical questions remain unanswered in the supplied literature:

1. At which concrete (n, p) does the gradient of a depth-linear HEA actually become unresolvable, and how large is the exponential rate at realistic two-qubit error probabilities p?
2. When the circuit is grown on the noisy device itself (gradients and energies both noisy, each rotation charged its controlled-NOT (CNOT) count), does it beat the best fixed-depth HEA?
3. Below which p does any entangling ansatz beat the free classical baseline, a product-state mean-field energy, so that the quantum circuit is worth running at all?

Question 3 is rarely asked: both reviews list noise as an open problem but do not give a break-even noise level against a zero-cost classical competitor.

**Novelty in one sentence.** This study measures, with the same noise model and the same CNOT cost accounting, the noise-induced gradient decay of the HEA, the noisy-device ADAPT growth trajectory with a measured-energy stopping rule, and the break-even noise level against mean field, on a disordered transverse-field Ising chain with n = 4 to 8 qubits (exact density-matrix simulation).

## Hypotheses (fixed before the full run)

- **H1 (noise-induced decay).** For the HEA with depth L = n at random angles, the fitted decay rate d ln(root-mean-square (RMS) gradient) / dn is more negative at p = 0.01 than at p = 0 by at least 0.1 ln-units per qubit. For fixed depth L = 2 the difference is smaller than for L = n. Falsified if the p = 0.01 and p = 0 rates differ by less than 0.1 or if L = 2 decays as fast as L = n.
- **H2 (adaptive growth).** For n >= 5 and p >= 0.002, noise-aware ADAPT (stop at the lowest measured noisy energy) has a lower relative energy error than the best-depth HEA in at least two of three seeds. Falsified otherwise.
- **H3 (break-even).** The noise level up to which the best quantum ansatz beats mean field does not increase with n. Falsified if the break-even level at n = 7 exceeds that at n = 4.

**Success metric.** Relative energy error (E - E0) / |E0| of the noisy-device energy, mean and standard deviation over three disorder seeds; gradient RMS over random starts; win fractions over seeds.

## Scale statement

All results are exact density-matrix simulations with n <= 8 qubits and a depolarising-only noise model. They indicate trends and break-even levels for this model; they are not hardware measurements and make no claim of quantum advantage.
