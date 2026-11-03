"""Figures (Okabe-Ito palette, 200 dpi)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OI = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
}
DPI = 200


def _p_label(p: float) -> str:
    return "0" if p == 0 else f"{p:g}"


def gradient_scaling(grad: pd.DataFrame, fits: pd.DataFrame, path: Path) -> None:
    """Log-scale RMS gradient versus qubits with fitted exponential decay per qubit."""
    noises = sorted(grad.p.unique())
    colours = [OI["black"], OI["sky"], OI["orange"], OI["vermillion"], OI["purple"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, rule in zip(axes, ("L=n", "L=2")):
        for c, p in zip(colours, noises):
            d = grad[(grad.rule == rule) & (grad.p == p)]
            m = d.groupby("n").grad_rms.apply(lambda x: float(np.sqrt(np.mean(np.square(x)))))
            f = fits[(fits.rule == rule) & (fits.p == p)]
            lab = f"p = {_p_label(p)}"
            if len(f):
                lab += f"  (slope {f.decay_per_qubit.iloc[0]:+.2f} ln-units/qubit)"
                xs = np.array([m.index.min(), m.index.max()], dtype=float)
                ax.plot(xs, np.exp(f.ln_intercept.iloc[0] + f.decay_per_qubit.iloc[0] * xs),
                        color=c, lw=1, ls="--")
            ax.plot(m.index, m.values, "o-", color=c, label=lab)
        ax.set_yscale("log")
        ax.set_xlabel("number of qubits n")
        ax.set_title("depth L = n (linear in n)" if rule == "L=n" else "fixed depth L = 2")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("RMS energy-gradient component (energy units)")
    fig.suptitle("Gradient magnitude of the hardware-efficient ansatz at random angles")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _grid(summary: pd.DataFrame, num: str, den: str) -> tuple[np.ndarray, list, list]:
    ns = sorted(summary.n.unique())
    ps = sorted(summary.p.unique())
    z = np.full((len(ns), len(ps)), np.nan)
    for i, n in enumerate(ns):
        for j, p in enumerate(ps):
            r = summary[(summary.n == n) & (summary.p == p)]
            if len(r):
                z[i, j] = np.log10(max(r[num].iloc[0], 1e-9) / max(r[den].iloc[0], 1e-9))
    return z, ns, ps


def phase_diagram(summary: pd.DataFrame, path: Path) -> None:
    """Filled contours of log10 error ratios (noisy cells only); zero contour = break-even."""
    summary = summary[summary.p > 0]
    panels = [
        ("eps_meanfield_mean", "eps_quantum_best_mean", "best noisy VQE vs classical mean field"),
        ("eps_hea_best_mean", "eps_adapt_aware_mean", "adaptive (noise-aware stop) vs hardware-efficient"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, (num, den, title) in zip(axes, panels):
        z, ns, ps = _grid(summary, num, den)
        lim = max(0.1, float(np.nanmax(np.abs(z))))
        levels = np.linspace(-lim, lim, 13)
        if z.shape[0] < 2 or z.shape[1] < 2:  # degenerate grid (smoke run): plain scatter
            cs = ax.scatter(np.tile(ps, len(ns)), np.repeat(ns, len(ps)), c=z.ravel(),
                            cmap="BrBG", vmin=-lim, vmax=lim, s=200)
        else:
            cs = ax.contourf(ps, ns, z, levels=levels, cmap="BrBG", extend="both")
            if np.nanmin(z) < 0 < np.nanmax(z):
                ax.contour(ps, ns, z, levels=[0.0], colors=OI["black"], linewidths=2)
        ax.plot(np.tile(ps, len(ns)), np.repeat(ns, len(ps)), "k.", ms=3)
        ax.set_xscale("log")
        ax.set_xticks(ps, [_p_label(p) for p in ps])
        ax.minorticks_off()
        ax.set_yticks(ns)
        ax.set_xlabel("depolarising error probability p per CNOT")
        ax.set_ylabel("number of qubits n")
        ax.set_title(title, fontsize=10)
        cb = fig.colorbar(cs, ax=ax)
        cb.set_label("log10 (error ratio, numerator / denominator)", fontsize=8)
    fig.suptitle("Phase diagram: value > 0 means the quantum / adaptive method wins "
                 "(black line = break-even)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _pareto_front(points: np.ndarray) -> np.ndarray:
    order = np.argsort(points[:, 0], kind="stable")
    front, best = [], np.inf
    for i in order:
        if points[i, 1] < best:
            front.append(points[i])
            best = points[i, 1]
    return np.array(front)


def pareto(tables: dict, per_instance: pd.DataFrame, path: Path) -> None:
    """Relative energy error versus CNOT count at the largest n, for three noise levels."""
    n = int(max(tables["hea_runs"].n))
    ps = sorted(tables["hea_runs"].p.unique())
    pick = [ps[0], ps[len(ps) // 2], ps[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, p in zip(axes, pick):
        h = tables["hea_runs"]
        h = h[(h.n == n) & (h.p == p)].copy()
        h["eps"] = (h.energy - h.e0) / h.e0.abs()
        a = tables["adapt_trajectories"]
        a = a[(a.n == n) & (a.p == p) & (a.step > 0)].copy()
        a["eps"] = (a.energy - a.e0) / a.e0.abs()
        mf = per_instance[(per_instance.n == n) & (per_instance.p == p)].eps_meanfield.mean()
        ax.scatter(h.cnots, h.eps, marker="s", color=OI["blue"], label="hardware-efficient (HEA)", s=28)
        ax.scatter(a.cnots + 0.0, a.eps, marker="o", color=OI["orange"], alpha=0.6,
                   label="adaptive (ADAPT) steps", s=22)
        pts = np.vstack([h[["cnots", "eps"]].to_numpy(), a[["cnots", "eps"]].to_numpy()])
        front = _pareto_front(pts)
        ax.step(front[:, 0], front[:, 1], where="post", color=OI["black"], lw=2, label="Pareto front")
        ax.axhline(mf, color=OI["vermillion"], ls="--", label="mean field (0 CNOT, classical)")
        ax.set_yscale("log")
        ax.set_xlabel("CNOT count of the circuit")
        ax.set_title(f"n = {n}, p = {_p_label(p)}")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("relative energy error (E - E0) / |E0|")
    axes[0].legend(fontsize=7)
    fig.suptitle("Cost versus quality: every circuit evaluated on the noisy device (all seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def ablation(summary: pd.DataFrame, path: Path) -> None:
    """Noise-aware stopping versus running the adaptive ansatz to the full operator budget."""
    n = int(summary.n.max())
    d = summary[summary.n == n].sort_values("p")
    xs = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for off, col, lab, c in (
        (-0.27, "eps_adapt_fixed", "ADAPT, full budget (no noise-aware stop)", OI["vermillion"]),
        (0.0, "eps_adapt_aware", "ADAPT, noise-aware stop", OI["orange"]),
        (0.27, "eps_hea_best", "HEA, best depth", OI["blue"]),
    ):
        ax.bar(xs + off, d[col + "_mean"], 0.26, yerr=d[col + "_std"], color=c, label=lab,
               capsize=2, edgecolor="black", linewidth=0.5)
    ax.plot(xs, d.eps_meanfield_mean, "k_", ms=28, mew=2, label="mean field (classical)")
    ax.set_yscale("log")
    ax.set_xticks(xs, [_p_label(p) for p in d.p])
    ax.set_xlabel("two-qubit depolarising error probability p per CNOT")
    ax.set_ylabel("relative energy error (mean +/- std over seeds)")
    ax.set_title(f"Ablation of the stopping rule, n = {n}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both", axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
