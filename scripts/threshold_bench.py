#!/usr/bin/env python
"""Crossover benchmark: at what dihedral count do the dimensionality-scaled lengthscale
prior and low-mode(PCA) search start beating the value-only baseline?

The result sets ``config.HIGH_D_DIHEDRAL_THRESHOLD`` -- the dihedral count at/above which
bouquet's "auto" defaults turn those two features on. Run on smiles/stopbench-hard.smi
(165 molecules, d=6-12, ~24/bin), paired by molecule and stratified by dihedral count.
All arms share the acquisition loop, init, and --auto budget (95 BO steps at d<=7, 195 at
d>=8); NO gradients here (a separate, established axis). Arms (each sets the two flags
explicitly so the CLI "auto" default can't confound them):

  base     : value-only GP, no prior, no low-mode (the historical default).
  prior    : + dim-scaled lengthscale prior (Hvarfner et al., ICML 2024).
  prior_lm : + low-mode (PCA) search (prob 0.5, warmup 20 so it fires within the budget).

Read: base->prior gives the prior's crossover d; prior->prior_lm gives low-mode's. We gate
both together, so the threshold is where the combined (prior_lm) arm starts winning.

  python scripts/threshold_bench.py run --input smiles/stopbench-hard.smi \
      --output thr.csv --seeds 1,2,3 --workers 8 --timeout 3600
  # crossover: paired gain vs base, stratified by dihedral count
  python scripts/threshold_bench.py analyze thr_s*.csv --dihedral-bins 7,9,11
  python scripts/threshold_bench.py traj    thr_s*_traj.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_common as sc  # noqa: E402

CONFIG_NAMES = ["base", "prior", "prior_lm"]
BASELINE_LABEL = "base"


def build_configurations() -> dict:
    """Arm -> extra CLI args. Each arm sets --lengthscale-prior and --lowmode-prob
    explicitly: the CLI "auto" default would otherwise switch them on at d >= the
    (provisional) threshold and confound exactly the crossover we're measuring."""
    return {
        "base": ["--lengthscale-prior", "none", "--lowmode-prob", "0"],
        "prior": ["--lengthscale-prior", "dim_scaled", "--lowmode-prob", "0"],
        "prior_lm": ["--lengthscale-prior", "dim_scaled",
                     "--lowmode-prob", "0.5", "--lowmode-warmup", "20"],
    }


if __name__ == "__main__":
    # No gradients here, so no single-surface requirement.
    sc.run_sweep_cli(CONFIG_NAMES, build_configurations, BASELINE_LABEL,
                     description=__doc__)
