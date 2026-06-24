#!/usr/bin/env python
"""
Stopping-rule calibration benchmark for bouquet's ``--auto`` step cap.

Goal: replace the hand-set ``--auto`` budget (the tiered AUTO_STEPS table) with an
*empirically calibrated* stopping rule for reliably finding the global-minimum
conformer, and characterize how the required budget scales with the dihedral
count ``d``. This script is the Phase B run loop: it drives ``bouquet.cli`` over a
molecule set x seeds at a generous per-molecule **ceiling** C(d) -- large enough
to keep right-censoring rare -- with per-BO-step certificate logging on, so every
candidate stopping rule can be evaluated offline by replaying the logged
trajectory (no re-running). See the design notes in the project for the rule
catalogue (GP certificate, log-EI, stall) and the survival/Pareto analysis.

The single arm is the production-candidate surrogate: the gradient-enhanced GP
with the default freeze schedule (``--use-gradients``). The certificate logs
mu_min, alpha_max, and a beta-grid of lower bounds lb_b<beta> per step, plus
e_best/n_calls/wall_s -- everything the offline rules need.

Ceiling (upper envelope on hitting time, NOT a budget): anchored to the auto
table's 25 at d=3, grown with a conservative global-best exponent and a safety
margin so the high-d tail gets headroom:

    C(d) = ceil( margin * anchor_d3 * (d / 3) ** p ),  capped at --ceiling-cap
    #  defaults margin=3, anchor=25, p=1.5:
    #  d=3 -> 75   d=5 -> 161   d=7 -> 267   d=10 -> 456   d=12 -> 600   d=20 -> 1290

Usage (mirrors the other sweep drivers; shared machinery in sweep_common.py):

  # Run (writes summary + trajectory + certificate CSVs; resumable):
  python scripts/stop_benchmark.py run --input smiles/platinum-50.smi \
      --output stopbench.csv --seeds 1234,12345,3141 --workers 8

  # Anytime best-energy-vs-budget curves (reuses the shared trajectory view):
  python scripts/stop_benchmark.py traj stopbench_traj.csv

The offline stopping-rule replay + survival/Pareto/calibration analysis over the
certificate CSV (rules.py / analyze.py) is the next milestone.
"""

import argparse
import math
import sys
from pathlib import Path

# Shared sweep machinery (same directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_common as sc  # noqa: E402

CONFIG_NAMES = ["grad"]
BASELINE_LABEL = "grad"


def ceiling(d: int, anchor_d3: float, p: float, margin: float, cap: int) -> int:
    """Per-molecule BO-step ceiling C(d) (see module docstring)."""
    return min(cap, math.ceil(margin * anchor_d3 * (d / 3.0) ** p))


def dihedral_count(smiles: str):
    """Bouquet's own rotatable-dihedral count for ``smiles`` (matches the CLI), or
    None if it can't be computed -- callers then fall back to the auto budget."""
    try:
        from bouquet.setup import detect_dihedrals, get_initial_structure

        _, mol = get_initial_structure(smiles)
        return len(detect_dihedrals(mol))
    except Exception:
        return None


def run(args: argparse.Namespace) -> None:
    """Drive the certificate-logging sweep at the per-molecule ceiling C(d)."""
    # Gradients need a single surface under --relax (sweep_common always relaxes);
    # both default to gfnff, so refuse a mismatched pair rather than crash later.
    sc.require_single_surface(args.energy, args.optimizer)
    # Certificate logging is the whole point of this driver -- force it on, along
    # with the geometry trail (needed for the RMSD-identity success criterion).
    args.certificate = True
    args.geometry = True

    # Dihedral count is reused by both the budget and the skip; cache it so the
    # (embedding) detection runs once per molecule.
    d_cache: dict = {}

    def d_of(smiles: str):
        if smiles not in d_cache:
            d_cache[smiles] = dihedral_count(smiles)
        return d_cache[smiles]

    def num_steps_fn(smiles: str):
        d = d_of(smiles)
        if d is None:  # unknown d -> let bouquet --auto pick the budget
            return None
        return ceiling(d, args.ceiling_anchor, args.ceiling_p,
                       args.ceiling_margin, args.ceiling_cap)

    def mol_skip_fn(smiles: str, name: str):
        if args.max_dihedrals is None:
            return False
        d = d_of(smiles)
        return d is not None and d > args.max_dihedrals

    sc.run_sweep(args, {"grad": ["--use-gradients"]},
                 num_steps_fn=num_steps_fn, mol_skip_fn=mol_skip_fn)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run the certificate sweep at the ceiling C(d)")
    sc.add_run_args(r, CONFIG_NAMES)
    r.add_argument("--ceiling-anchor", type=float, default=25.0,
                   help="C(d) anchor: BO steps at d=3 before the margin (default 25, "
                   "the auto-table value).")
    r.add_argument("--ceiling-p", type=float, default=1.5,
                   help="C(d) growth exponent in (d/3)**p (default 1.5, a conservative "
                   "global-best envelope).")
    r.add_argument("--ceiling-margin", type=float, default=3.0,
                   help="C(d) safety margin multiplier to keep censoring rare "
                   "(default 3).")
    r.add_argument("--ceiling-cap", type=int, default=1500,
                   help="Hard upper cap on C(d) regardless of d (default 1500).")
    r.add_argument("--max-dihedrals", type=int, default=None,
                   help="Skip molecules with more than this many rotatable dihedrals "
                   "(e.g. 12). Off by default.")
    r.set_defaults(func=run)

    a = sub.add_parser("analyze", help="Summarize a sweep CSV (shared view)")
    sc.add_analyze_args(a)
    a.set_defaults(func=lambda a: sc.analyze(a, BASELINE_LABEL))

    t = sub.add_parser("traj", help="Anytime best-energy-vs-budget curves + plots")
    sc.add_traj_args(t)
    t.set_defaults(func=lambda a: sc.trajectory(a, BASELINE_LABEL))

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
