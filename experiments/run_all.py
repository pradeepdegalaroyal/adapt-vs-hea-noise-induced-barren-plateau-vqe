"""Reproduce every result and figure: ``python experiments/run_all.py [--quick]``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    import torch

    from nibp_vqe import plots, study

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="under-one-minute smoke run")
    parser.add_argument("--plots-only", action="store_true",
                        help="rebuild figures from the saved CSVs in results/")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = study.QUICK if args.quick else study.FULL
    out = ROOT / ("results_quick" if args.quick else "results")
    fig = ROOT / ("figures_quick" if args.quick else "figures")
    out.mkdir(exist_ok=True)
    fig.mkdir(exist_ok=True)

    t0 = time.time()
    if args.plots_only:
        import pandas as pd

        names = ["gradient_scaling", "hea_runs", "adapt_trajectories", "meanfield"]
        tables = {k: pd.read_csv(out / f"{k}.csv") for k in names}
    else:
        tables = study.run_study(cfg)
    per_instance = study.summarise(tables)
    summary = study.aggregate(per_instance)
    fits = study.grad_fits(tables["gradient_scaling"])
    thresholds = study.crossover_noise(summary)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    per_instance.to_csv(out / "per_instance_errors.csv", index=False)
    summary.to_csv(out / "summary_by_size_and_noise.csv", index=False)
    fits.to_csv(out / "gradient_decay_fits.csv", index=False)
    thresholds.to_csv(out / "noise_crossover.csv", index=False)

    plots.gradient_scaling(tables["gradient_scaling"], fits, fig / "gradient_scaling.png")
    plots.phase_diagram(summary, fig / "phase_diagram.png")
    plots.pareto(tables, per_instance, fig / "pareto_cnots_vs_error.png")
    plots.ablation(summary, fig / "ablation_stopping_rule.png")

    if args.plots_only:
        print("figures rebuilt")
        return
    meta = {
        "quick": args.quick,
        "runtime_s": round(time.time() - t0, 1),
        "torch": torch.__version__,
        "config": str(cfg),
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done in {meta['runtime_s']} s -> {out}, {fig}")


if __name__ == "__main__":
    main()
