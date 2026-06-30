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

import argparse
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


def run(args: argparse.Namespace) -> None:
    sc.require_single_surface(args.energy, args.optimizer)
    sc.run_sweep(args, build_configurations())


def _concat_to_tmp(paths, drop_traj: bool) -> Path:
    """Concatenate the per-(seed,arm) CSVs from the SLURM array into one tidy CSV
    (sweep_common's analyze/traj each read a single file). ``drop_traj`` filters out
    ``*_traj.csv`` so an ``analyze high_d_s*.csv`` glob that also matched the
    trajectory files still works."""
    import tempfile
    import pandas as pd
    files = [p for p in paths if not (drop_traj and "_traj" in str(p))]
    if not files:
        sys.exit("no input CSVs (after dropping _traj files)")
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    tmp = Path(tempfile.mkstemp(suffix=".csv")[1])
    df.to_csv(tmp, index=False)
    return tmp


def _accept_many(parser) -> None:
    """Make the single-file ``input`` positional accept many files, so an
    ``analyze high_d_s*.csv`` glob over the distributed array's per-(seed,arm) output
    works (sweep_common adds it as a single Path)."""
    for action in parser._actions:
        if action.dest == "input":
            action.nargs = "+"
            action.help = "per-(seed,arm) sweep CSV(s) from the array; concatenated"


def _multi(handler, drop_traj: bool):
    """Wrap an analyze/traj handler so its (now multi-valued) ``input`` positional is
    concatenated to one tidy CSV before the single-file handler runs. ``analyze`` drops
    any ``*_traj.csv`` a broad glob caught; ``traj`` keeps them."""
    def wrapped(args):
        args.input = _concat_to_tmp(args.input, drop_traj=drop_traj)
        handler(args)
    return wrapped


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run the sweep")
    sc.add_run_args(r, CONFIG_NAMES)
    r.set_defaults(func=run)

    a = sub.add_parser("analyze", help="Summarize sweep CSV(s)")
    sc.add_analyze_args(a)
    _accept_many(a)  # the SLURM array writes one CSV per (seed, arm)
    a.set_defaults(func=_multi(lambda a: sc.analyze(a, BASELINE_LABEL), drop_traj=True))

    t = sub.add_parser("traj", help="Anytime best-energy-vs-budget curves + plots")
    sc.add_traj_args(t)
    _accept_many(t)
    t.set_defaults(func=_multi(lambda a: sc.trajectory(a, BASELINE_LABEL), drop_traj=False))

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
