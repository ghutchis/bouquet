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
  cat        : category-tied moves (prob 0.5) instead of a low-mode kick -- every
               dihedral sharing a SMARTS prior category is set to one shared value (a
               chemistry-defined embedding, from --priors), chosen by a periodic GP over
               the reduced per-category space, then relaxed UNCONSTRAINED. The prior is
               loaded ONLY to define the categories + seed the reduced-space warmup;
               PiBO acquisition steering is OFF (--prior-exponent 0) so this arm differs
               from lm_pca in exactly one thing: the collective move (category tie vs
               PCA kick). Both share the identical grad Phase A.

A low-mode move kicks the incumbent along a soft mode then relaxes UNCONSTRAINED, so the
geometry can slide along the curved fold valley the line-restricted BO step cannot cross
(Phase 2.4 diagnostics: a straight dihedral path to the fold has a ~58 kcal/mol clash
barrier). The category move reaches the same fold as one reduced-space point, available
from the first move (no elite set to accumulate). Run on the foldamer subset (d>=19, where
folding matters and --auto's 195-step budget leaves ~145 steps after the 50-step Phase A;
--lowmode-warmup / --category-warmup 55 = init 5 + 50).

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
# Category arm: reuse the same post-Phase-A start (55 evals); the prior file supplies the
# SMARTS categories + warmup seeds, with PiBO steering disabled so only the move differs.
_CAT_WARMUP = ["--category-warmup", "55"]
PRIORS = "bouquet/data/gfn2_priors.json"  # bundled fitted 1D prior library; run cd's to $BOUQUET
CONFIG_NAMES = ["base", "lm_pca", "lm_enm", "lmonly_enm", "cat", "cat_pca"]
BASELINE_LABEL = "base"


def build_configurations() -> dict:
    """Arm -> extra CLI args. Phase A is shared; arms differ only in the collective phase
    (low-mode kick source, or the category-tied move)."""
    return {
        "base": list(PHASE_A),
        "lm_pca": PHASE_A + ["--lowmode-prob", "0.5"] + _WARMUP + ["--lowmode-kick-dir", "pca"],
        "lm_enm": PHASE_A + ["--lowmode-prob", "0.5"] + _WARMUP + ["--lowmode-kick-dir", "enm"],
        "lmonly_enm": PHASE_A + ["--lowmode-prob", "1.0"] + _WARMUP + ["--lowmode-kick-dir", "enm"],
        # Category-tied move in place of the PCA kick; prior loaded for grouping only
        # (--prior-exponent 0 turns off PiBO acquisition steering).
        "cat": PHASE_A + ["--category-prob", "0.5", "--lowmode-prob", "0"] + _CAT_WARMUP
        + ["--priors", PRIORS, "--prior-exponent", "0"],
        # Both collective moves enabled: the dispatch tries the category move first, then
        # the PCA low-mode on steps it didn't fire -- so P(cat)=0.5, P(pca)=0.25,
        # P(standard)=0.25. Tests whether cat + PCA together beats either alone (the
        # category move exploits repeats; PCA covers the rest).
        "cat_pca": PHASE_A + ["--category-prob", "0.5", "--lowmode-prob", "0.5"]
        + _CAT_WARMUP + _WARMUP + ["--lowmode-kick-dir", "pca"]
        + ["--priors", PRIORS, "--prior-exponent", "0"],
    }


if __name__ == "__main__":
    # Gradient Phase A needs energy == optimizer under --relax (single_surface).
    sc.run_sweep_cli(CONFIG_NAMES, build_configurations, BASELINE_LABEL,
                     description=__doc__, single_surface=True)
