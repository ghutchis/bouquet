#!/usr/bin/env python
"""
High-dimensional Phase-A benchmark: value-only vs bounded-full-gradient acquisition
across a dihedral-count ladder (Benchmark 1 of the HDBO plan).

All arms hold the acquisition loop, init, priors, budget, and energy/optimizer fixed;
they share the dimensionality-scaled lengthscale prior (`--lengthscale-prior dim_scaled`,
the established best value-only config). The only difference is the gradient phase:

  value     : value-only GP (dim_scaled prior).
  grad_s50  : --use-gradients --gradient-steps 50   (gradient GP for the first 50 BO
              steps, then value-only -- the "bounded-early gradient" Phase A).
  grad_s100 : --gradient-steps 100.
  grad_s150 : --gradient-steps 150.

The question (the d<=11 gradient win is established; this targets >10 dihedrals):
does the gradient-acquisition benefit persist at d=15..34, and at what wall-clock cost?
Paired by (molecule, seed) against `value`. The diversity pairs in the ladder
(fold4 vs polyala_n4 at d=19; fold1 vs napth at d=25) test whether any benefit is
chemistry-dependent rather than purely a function of d.

Single-surface requirement: --use-gradients with --relax needs --energy == --optimizer
(the torsion gradient is dE*/dtheta only at a constrained minimum of the energy
calculator); both default to gfnff and a mismatch is refused.

Phases (shared machinery in sweep_common.py):

  # 1. Run (writes summary CSV + per-evaluation trajectory CSV):
  python scripts/high_d.py run --input smiles/high-d-ladder.csv \
      --output high_d.csv --seeds 1,2,3,4,5 --workers 8 --timeout 14400
  # 2. Analyze (per-arm summary + paired-by-(molecule,seed) vs value):
  python scripts/high_d.py analyze high_d.csv
  # 3. Trajectory (anytime best-energy-vs-budget curves):
  python scripts/high_d.py traj high_d_traj.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_common as sc  # noqa: E402

S_VALUES = [50, 100, 150]
CONFIG_NAMES = ["value"] + [f"grad_s{s}" for s in S_VALUES]
BASELINE_LABEL = "value"
# Every arm uses the dimensionality-scaled lengthscale prior (the best value-only
# config); the gradient arms add a bounded gradient phase on top.
_PRIOR = ["--lengthscale-prior", "dim_scaled"]


def build_configurations() -> dict:
    """Arm -> extra CLI args appended to the bouquet command. Only the gradient phase
    differs across arms; the dim_scaled prior, init, budget, and surface are shared."""
    cfg = {"value": list(_PRIOR)}
    for s in S_VALUES:
        cfg[f"grad_s{s}"] = _PRIOR + ["--use-gradients", "--gradient-steps", str(s)]
    return cfg


if __name__ == "__main__":
    # Gradient arms need energy == optimizer under --relax (single_surface).
    sc.run_sweep_cli(CONFIG_NAMES, build_configurations, BASELINE_LABEL,
                     description=__doc__, single_surface=True)
