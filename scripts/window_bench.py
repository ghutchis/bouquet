#!/usr/bin/env python
"""Does capping the gradient window make high-d searches feasible without losing quality?

THE PROBLEM. The gradient-enhanced GP conditions on an augmented covariance of size
``n*(d+1)`` -- values plus one derivative per torsion for every observed point. At
d=8 and n=2138 (the stopping-rule ceiling C(8)) that is a 19,242 x 19,242 matrix:
2.76 GB, and a Cholesky of 2.4e12 flops on EVERY BO step. Measured on the previous
stopbench that is ~232 s per step against ~1.35 s for the xTB evaluation, so the
search is GP-bound, not chemistry-bound. It is why every d>=7 trial there died on
the 35 h per-trial wall rather than reaching its budget: at d=7 only 5% of trials
reached C(d), at d=8 none did, and those right-censored trials are exactly what
weakens the n*(d) fit where it matters most.

THE LEVER. ``--gradient-window W`` keeps gradients for only W high-leverage points,
so the matrix becomes ``n + W*d``. At d=8, n=2138 that is 3,738 for W=200 -- a 136x
smaller factorization and 0.10 GB instead of 2.76 GB. Predicted ~1.7 s/step, i.e.
the search returns to being xTB-bound.

WHAT IS UNTESTED is quality. Discarding gradients cannot help the surrogate; the
question is how much it costs and whether the extra steps bought are worth more.
This cannot be answered offline: the structure logs record coordinates and energies
but NOT gradients, so there is no way to cross-validate the gradient GP from
existing data. Hence fresh paired runs.

DESIGN. All arms run at the full C(8)=2138 budget with a generous wall. The
windowed arms are expected to finish it in a few hours; the ``all`` baseline is
expected to censor, exactly as it does today -- which is itself half the result.
Because the certificate logs e_best at every step, the arms can still be compared
at MATCHED STEPS over whatever range the baseline reaches, so one run answers both
questions:

  1. matched-step quality  -- does windowing cost accuracy per step?
  2. reachability          -- can a windowed run finish C(d) at all?

A windowed arm that is slightly worse per step but reaches 2138 while the baseline
stalls at 1200 is the outcome that matters, and it is not visible from either
question alone.

Usage (mirrors kernel_bench.py; shared machinery in sweep_common.py):

  python scripts/window_bench.py run --input smiles/stopbench-500.smi \\
      --output wb.csv --seeds 1 --num-steps 2138 --workers 25
  python scripts/window_bench.py analyze wb.csv
  python scripts/window_bench.py traj wb_traj.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_common as sc  # noqa: E402

CONFIG_NAMES = ["all", "w500", "w250", "w200", "w150", "w50"]
BASELINE_LABEL = "all"

# The arms differ by exactly one flag. --gradient-keep stays at its default ("both":
# the most recent and the best-so-far points), so this measures the WINDOW SIZE and
# not the selection rule -- those are separable knobs and mixing them would make a
# regression impossible to attribute.
_BASE = ["--use-gradients", "--priors"]
ARMS = {
    "all": _BASE,                                    # keep every gradient (shipped)
    "w500": _BASE + ["--gradient-window", "500"],
    "w250": _BASE + ["--gradient-window", "250"],
    "w200": _BASE + ["--gradient-window", "200"],
    "w150": _BASE + ["--gradient-window", "150"],
    "w50": _BASE + ["--gradient-window", "50"],
}

# Which arms are worth running depends on d, because `all` stops being runnable.
# At d=12, n=2000 the full augmented matrix is 26,000 square: ~69 GB and ~64 h of
# Cholesky for a 2000-step run. So d=8 asks "does windowing cost anything versus
# keeping everything", while d=12 can only ask "how small can W get before quality
# drops" -- which is the question that actually decides the N_max floor anyway.
SUGGESTED_ARMS = {
    8: "all,w250,w200,w150",
    12: "w500,w250,w150,w50",
}


def run(args: argparse.Namespace) -> None:
    sc.require_single_surface(args.energy, args.optimizer)
    # The certificate is what makes the matched-step comparison possible once the
    # baseline censors; the geometry trail feeds the RMSD-identity criterion.
    args.certificate = True
    args.geometry = True

    steps = args.num_steps
    sc.run_sweep(args, ARMS, num_steps_fn=lambda smiles: steps)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run the gradient-window arms, paired")
    sc.add_run_args(r, CONFIG_NAMES)
    # Same surface as the CREST reference, matching the stopping-rule work.
    r.set_defaults(energy="gfn2", optimizer="gfn2")
    r.add_argument("--num-steps", type=int, default=2138,
                   help="BO-step budget for every arm (default 2138 = C(8), the "
                        "stopping-rule ceiling this is meant to make reachable).")
    r.set_defaults(func=run)

    a = sub.add_parser("analyze", help="Paired summary (shared view)")
    sc.add_analyze_args(a)
    a.set_defaults(func=lambda a: sc.analyze(a, BASELINE_LABEL))

    t = sub.add_parser("traj", help="Anytime best-energy-vs-budget curves + plots")
    sc.add_traj_args(t)
    t.set_defaults(func=lambda a: sc.trajectory(a, BASELINE_LABEL))

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
