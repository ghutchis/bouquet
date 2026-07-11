#!/usr/bin/env python
"""Timing benchmark: wall-clock cost vs molecular size, broken down by phase.

Runs a fixed BO budget (default 300 steps) on the polyalanine oligomer size ladder
(polyalanine_n2..n8, d=11..35) with per-step certificate logging ON, so the cost of
each phase is recorded for every step:

  t_eval    -- GFN2 energy + relaxation      (the "GFN2" bucket)
  t_gp_fit  -- GP construction + fit/condition (the "GP" bucket)
  t_acq     -- acquisition build + optimize_acqf (the "acquisition" bucket)
  t_select  -- t_gp_fit + t_acq + certificate overhead (the residual = "other")
  wall_s    -- cumulative loop wall-time (gives the 50/100/.../300-step checkpoints)

Two arms are profiled (a subset of high_d_phaseC.py's, to keep the comparison to the
reference vs the production collective-move arm):

  base    -- Phase A only: dim_scaled prior + grad_s50 (the reference).
  cat_pca -- base + category-tied + PCA low-mode collective moves (production arm).

Unlike the phase-C sweep this forces a FIXED --num-steps (via num_steps_fn) instead of
the tiered --auto budget, so every molecule runs the same number of BO steps and the
wall-time-vs-size curves are directly comparable. Certificate output is always on.

  python scripts/timing_bench.py run --input smiles/timing-ala.csv \
      --output timing/timing.csv --seeds 1,2,3 --configs base,cat_pca \
      --energy gfn2 --optimizer gfn2 --num-steps 300 --workers 6 --timeout 0
  python scripts/timing_bench.py analyze timing/timing_s*.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_common as sc  # noqa: E402
from high_d_phaseC import build_configurations  # noqa: E402

CONFIG_NAMES = ["base", "cat_pca"]
BASELINE_LABEL = "base"
DEFAULT_STEPS = 300


def main() -> None:
    def _run(args: argparse.Namespace) -> None:
        sc.require_single_surface(args.energy, args.optimizer)
        configs = {k: v for k, v in build_configurations().items() if k in CONFIG_NAMES}
        # base is a PURE gradient-BO reference: pin the collective moves OFF so every
        # step is a standard logged step (bouquet otherwise auto-enables low-mode moves
        # at prob 0.5 for d >= 12 -- see solver.HIGH_D_DIHEDRAL_THRESHOLD -- whose time
        # would then land only in the separate t_collective bucket). cat_pca keeps its
        # own explicit --lowmode-prob/--category-prob, so it still exercises (and now
        # logs) the collective moves.
        configs["base"] = configs["base"] + ["--lowmode-prob", "0", "--category-prob", "0"]
        # Certificate logging is the whole point of this benchmark -- force it on.
        args.certificate = True
        n = args.num_steps
        sc.run_sweep(args, configs, num_steps_fn=lambda _smiles: n)

    def _analyze(args: argparse.Namespace) -> None:
        args.input = sc.concat_sweep_csvs(args.input, drop_traj=True)
        sc.analyze(args, BASELINE_LABEL)

    def _traj(args: argparse.Namespace) -> None:
        args.input = sc.concat_sweep_csvs(args.input, drop_traj=False)
        sc.trajectory(args, BASELINE_LABEL)

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run the timing benchmark")
    sc.add_run_args(r, CONFIG_NAMES)
    r.add_argument("--num-steps", type=int, default=DEFAULT_STEPS,
                   help=f"Fixed BO steps per molecule (default {DEFAULT_STEPS}); "
                   "overrides --auto so every size runs the same budget.")
    r.set_defaults(func=_run)

    a = sub.add_parser("analyze", help="Summarize the timing sweep CSV(s)")
    sc.add_analyze_args(a)
    sc.accept_multi_input(a)
    a.set_defaults(func=_analyze)

    t = sub.add_parser("traj", help="Anytime curves + plots")
    sc.add_traj_args(t)
    sc.accept_multi_input(t)
    t.set_defaults(func=_traj)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
