"""Experiment drivers: gradient scaling, noisy-energy comparison and aggregation."""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .gradients import random_start_gradients
from .hamiltonian import make_tfim
from .meanfield import meanfield_energy
from .optimise import run_adapt, run_hea


@dataclass(frozen=True)
class Config:
    sizes: tuple[int, ...]
    noise: tuple[float, ...]
    seeds: tuple[int, ...]
    hea_depths: tuple[int, ...]
    adapt_ops_per_qubit: float
    grad_sizes: tuple[int, ...]
    grad_noise: tuple[float, ...]
    grad_starts: int
    workers: int = 4


FULL = Config(
    sizes=(4, 5, 6, 7),
    noise=(0.0, 0.001, 0.002, 0.005, 0.01),
    seeds=(0, 1, 2),
    hea_depths=(1, 2, 4),
    adapt_ops_per_qubit=3.0,
    grad_sizes=(4, 5, 6, 7, 8),
    grad_noise=(0.0, 0.002, 0.005, 0.01),
    grad_starts=6,
)
QUICK = Config(
    sizes=(4,),
    noise=(0.0, 0.005),
    seeds=(0,),
    hea_depths=(1, 2),
    adapt_ops_per_qubit=2.0,
    grad_sizes=(4, 5),
    grad_noise=(0.0, 0.005),
    grad_starts=2,
    workers=2,
)


def _init_worker() -> None:
    torch.set_num_threads(1)


def _grad_task(args: tuple) -> list[dict]:
    n, seed, p, starts = args
    model = make_tfim(n, seed)
    rows = []
    for rule, depth in (("L=n", n), ("L=2", 2)):
        row = random_start_gradients(model, depth, p, starts, seed)
        row.update(seed=seed, rule=rule)
        rows.append(row)
    return rows


def _hea_task(args: tuple) -> dict:
    n, seed, p, depth = args
    model = make_tfim(n, seed)
    out = run_hea(model, depth, p, seed)
    out.update(n=n, seed=seed, p=p, e0=model.e0)
    return out


def _adapt_task(args: tuple) -> list[dict]:
    n, seed, p, max_ops = args
    model = make_tfim(n, seed)
    trace = run_adapt(model, p, max_ops)
    return [
        {
            "n": n,
            "seed": seed,
            "p": p,
            "step": k,
            "cnots": trace.cnots[k],
            "energy": trace.energy[k],
            "e0": model.e0,
            "max_grad": trace.max_grad[k],
            "op": "ref" if k == 0 else trace.ops[k - 1],
        }
        for k in range(len(trace.energy))
    ]


def _mf_task(args: tuple) -> dict:
    n, seed = args
    model = make_tfim(n, seed)
    return {"n": n, "seed": seed, "energy": meanfield_energy(model), "e0": model.e0}


def run_study(cfg: Config) -> dict[str, pd.DataFrame]:
    """Run all experiments for ``cfg`` and return raw result tables."""
    t0 = time.time()
    with ProcessPoolExecutor(cfg.workers, initializer=_init_worker) as pool:
        mf = list(pool.map(_mf_task, [(n, s) for n in cfg.sizes for s in cfg.seeds]))
        ad_tasks = [
            (n, s, p, int(round(cfg.adapt_ops_per_qubit * n)))
            for n in cfg.sizes
            for s in cfg.seeds
            for p in cfg.noise
        ]
        hea_tasks = [
            (n, s, p, d)
            for n in cfg.sizes
            for s in cfg.seeds
            for p in cfg.noise
            for d in cfg.hea_depths
        ]
        g_tasks = [
            (n, s, p, cfg.grad_starts)
            for n in cfg.grad_sizes
            for s in cfg.seeds
            for p in cfg.grad_noise
        ]
        # Submit the heaviest tasks (largest n) first so workers stay balanced.
        ad_futs = [pool.submit(_adapt_task, t) for t in sorted(ad_tasks, key=lambda t: -t[0])]
        hea_futs = [pool.submit(_hea_task, t) for t in sorted(hea_tasks, key=lambda t: -t[0])]
        g_futs = [pool.submit(_grad_task, t) for t in sorted(g_tasks, key=lambda t: -t[0])]
        grad = [r for f in g_futs for r in f.result()]
        print(f"  gradient scaling done ({time.time() - t0:.0f}s)", flush=True)
        adapt = [r for f in ad_futs for r in f.result()]
        hea = [f.result() for f in hea_futs]
    print(f"  energy comparison done ({time.time() - t0:.0f}s)", flush=True)
    return {
        "gradient_scaling": pd.DataFrame(grad),
        "hea_runs": pd.DataFrame(hea),
        "adapt_trajectories": pd.DataFrame(adapt),
        "meanfield": pd.DataFrame(mf),
    }


def _rel(e: pd.Series, e0: pd.Series) -> pd.Series:
    return (e - e0) / e0.abs()


def summarise(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-instance relative energy errors of every method."""
    hea = tables["hea_runs"].copy()
    ad = tables["adapt_trajectories"].copy()
    mf = tables["meanfield"].copy()
    keys = ["n", "seed", "p"]
    mf["eps_meanfield"] = _rel(mf.energy, mf.e0)
    hea["eps"] = _rel(hea.energy, hea.e0)
    best = hea.loc[hea.groupby(keys).eps.idxmin()].rename(
        columns={"eps": "eps_hea_best", "depth": "hea_depth", "cnots": "hea_cnots"}
    )
    deepest = hea[hea.depth == hea.depth.max()].rename(columns={"eps": "eps_hea_deepest"})
    ad["eps"] = _rel(ad.energy, ad.e0)
    last = ad.loc[ad.groupby(keys).step.idxmax()].rename(
        columns={"eps": "eps_adapt_fixed", "cnots": "adapt_fixed_cnots"}
    )
    stop = ad.loc[ad.groupby(keys).energy.idxmin()].rename(
        columns={"eps": "eps_adapt_aware", "cnots": "adapt_aware_cnots",
                 "step": "adapt_aware_step"}
    )
    df = best[keys + ["eps_hea_best", "hea_depth", "hea_cnots"]]
    df = df.merge(deepest[keys + ["eps_hea_deepest"]], on=keys)
    df = df.merge(last[keys + ["eps_adapt_fixed", "adapt_fixed_cnots"]], on=keys)
    df = df.merge(stop[keys + ["eps_adapt_aware", "adapt_aware_cnots", "adapt_aware_step"]], on=keys)
    df = df.merge(mf[["n", "seed", "eps_meanfield"]], on=["n", "seed"])
    df["eps_quantum_best"] = df[["eps_hea_best", "eps_adapt_aware"]].min(axis=1)
    return df


def aggregate(per_instance: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over seeds for every (n, p) cell."""
    cols = [
        "eps_meanfield", "eps_hea_best", "eps_hea_deepest", "eps_adapt_fixed",
        "eps_adapt_aware", "eps_quantum_best", "hea_cnots", "adapt_aware_cnots",
    ]
    g = per_instance.groupby(["n", "p"])[cols]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=0).add_suffix("_std"))
    flags = per_instance.assign(
        adapt_beats_hea=(per_instance.eps_adapt_aware < per_instance.eps_hea_best).astype(float),
        quantum_beats_meanfield=(
            per_instance.eps_quantum_best < 0.98 * per_instance.eps_meanfield
        ).astype(float),
        adapt_beats_meanfield=(
            per_instance.eps_adapt_aware < 0.98 * per_instance.eps_meanfield
        ).astype(float),
        hea_beats_meanfield=(
            per_instance.eps_hea_best < 0.98 * per_instance.eps_meanfield
        ).astype(float),
    )
    frac = flags.groupby(["n", "p"])[
        ["adapt_beats_hea", "quantum_beats_meanfield", "adapt_beats_meanfield",
         "hea_beats_meanfield"]
    ].mean()
    return out.join(frac).reset_index()


def grad_fits(grad: pd.DataFrame) -> pd.DataFrame:
    """Least-squares exponential decay rate d ln(grad_rms) / dn per (rule, p)."""
    rows = []
    for (rule, p), d in grad.groupby(["rule", "p"]):
        m = d.groupby("n").grad_rms.apply(lambda x: float(np.sqrt(np.mean(np.square(x)))))
        if len(m) < 2:
            continue
        slope, intercept = np.polyfit(m.index.values.astype(float), np.log(m.values), 1)
        rows.append({"rule": rule, "p": p, "decay_per_qubit": float(slope),
                     "ln_intercept": float(intercept)})
    return pd.DataFrame(rows)


def crossover_noise(summary: pd.DataFrame) -> pd.DataFrame:
    """Largest grid noise level at which each method still beats mean-field (frac >= 0.5)."""
    rows = []
    for n, d in summary.groupby("n"):
        row = {"n": n}
        for name, col in (("hea", "hea_beats_meanfield"), ("adapt", "adapt_beats_meanfield"),
                          ("quantum_best", "quantum_beats_meanfield")):
            ok = d[d[col] >= 0.5].p
            row[f"p_star_{name}"] = float(ok.max()) if len(ok) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
