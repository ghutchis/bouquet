#!/usr/bin/env python
"""Phase C benchmark: does relaxation-enabled low-mode search (Phase 2.5) help on top of
the best gradient acquisition, and do ENM kick directions beat PCA?

All arms share **Phase A = bounded gradient acquisition** (grad_s50, the Benchmark-1
winner) + the dim_scaled prior. They differ only in the low-mode phase that follows:

  base       : no low-mode (gradient Phase A only -- the reference).
  lm_pca     : low-mode moves (prob 0.5) with PCA kick directions (data-derived).
  lm_enm     : low-mode moves (prob 0.5) with ENM kick directions (data-independent --
               the fold diagnostics found PCA misses the fold direction once stuck).
  lmonly_enm : low-mode-ONLY after Phase A (prob 1.0), ENM kicks (tests the schedule).

A low-mode move kicks the incumbent along a soft mode then relaxes UNCONSTRAINED, so the
geometry can slide along the curved fold valley the line-restricted BO step cannot cross
(Phase 2.4 diagnostics: a straight dihedral path to the fold has a ~58 kcal/mol clash
barrier). Run on the foldamer subset (d>=19, where folding matters and --auto's 195-step
budget leaves ~145 steps after the 50-step Phase A; --lowmode-warmup 55 = init 5 + 50).

  python scripts/high_d_phaseC.py run --input smiles/high-d-phaseC.csv \
      --output phaseC.csv --seeds 1,2,3,4,5 --workers 6 --timeout 14400
  python scripts/high_d_phaseC.py analyze phaseC_s*.csv      # vs 'base'
  python scripts/high_d_phaseC.py traj    phaseC_s*_traj.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_common as sc  # noqa: E402

# Phase A (shared by every arm): dim_scaled prior + 50-step bounded gradient acquisition.
PHASE_A = ["--lengthscale-prior", "dim_scaled", "--use-gradients", "--gradient-steps", "50"]
_WARMUP = ["--lowmode-warmup", "55"]  # init(5) + grad Phase A(50)
CONFIG_NAMES = ["base", "lm_pca", "lm_enm", "lmonly_enm"]
BASELINE_LABEL = "base"


def build_configurations() -> dict:
    """Arm -> extra CLI args. Phase A is shared; arms differ only in the low-mode phase
    (probability and kick-direction source)."""
    return {
        "base": list(PHASE_A),
        "lm_pca": PHASE_A + ["--lowmode-prob", "0.5"] + _WARMUP + ["--lowmode-kick-dir", "pca"],
        "lm_enm": PHASE_A + ["--lowmode-prob", "0.5"] + _WARMUP + ["--lowmode-kick-dir", "enm"],
        "lmonly_enm": PHASE_A + ["--lowmode-prob", "1.0"] + _WARMUP + ["--lowmode-kick-dir", "enm"],
    }


if __name__ == "__main__":
    # Gradient Phase A needs energy == optimizer under --relax (single_surface).
    sc.run_sweep_cli(CONFIG_NAMES, build_configurations, BASELINE_LABEL,
                     description=__doc__, single_surface=True)
