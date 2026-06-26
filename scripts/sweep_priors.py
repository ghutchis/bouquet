#!/usr/bin/env python
"""
Sweep PiBO prior settings over a molecule set and compare to a no-prior baseline.

For each (configuration x molecule x seed) it runs ``bouquet.cli`` (mirroring
batch.py: ``--auto --relax``) and records, per trial, a tidy summary row plus a
per-evaluation trajectory (running best-so-far). Runs are resumable at
(config, name, seed) granularity and can be parallelized.

Three phases (shared machinery lives in sweep_common.py):

  # 1. Run the sweep (writes a summary CSV + a per-evaluation trajectory CSV):
  python scripts/sweep_priors.py run --input mols.csv --output sweep.csv \
      --seeds 1234,12345,3141,314159,42 --workers 8

  # 2. Analyze (per-config summary + paired-by-(molecule, seed) vs no-prior):
  python scripts/sweep_priors.py analyze sweep.csv

  # 3. Trajectory (anytime curves: who is ahead at 25%/50%/... of the budget):
  python scripts/sweep_priors.py traj sweep_traj.csv

The configurations tested are defined in CONFIGURATIONS below; edit that list (or
select a subset with --configs) to control the sweep. Each configuration is a full
pass over every molecule and seed, so N configs ~= N x (one batch.py run).

Now that bouquet seeds every RNG (numpy + torch) from --seed, arms sharing a seed
share their randomness, so the paired-by-(molecule, seed) comparison cancels the
shared Bayesian-optimization noise and isolates each prior setting's effect.
"""

import argparse
import sys
from pathlib import Path

# Shared sweep machinery (same directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_common as sc  # noqa: E402

# ---------------------------------------------------------------------------
# Configurations to sweep. Each maps a label to the extra CLI args (beyond
# --priors, which is added automatically for every non-baseline arm). "noprior"
# passes no --priors flag at all and is the paired-comparison baseline -- keep it.
#
# Corrected-prior settings: concentration cap = 50 and global background_weight = 0
# (the fitted priors carry their own per-pattern uniform background, so the blunt
# global knob stays 0). Earlier sweeps drove the exponent toward ~0 only because
# the priors were wrong (off-by-one histogram->SMARTS mapping + smeared/spiky fits).
#
# DECAY sweep at exp=0.5 (decay is geometric: prior_exponent *= prior_decay every
# BO step, so smaller = faster fade). The exponent sweep on the EASY stopbench-50
# set (d=1..10) was null on final energy and showed no anytime/speed lead; the only
# positive signal was exp0.5 on d>6. So fix exponent at 0.5 (the best arm) and test
# whether a faster-fading prior -- strong early guidance that gets out of the way
# before it can trap the search at its mode -- helps, on the harder stopbench-hard
# set (d=6..12, where the headroom is). All previous runs used the 0.9 default.
# ---------------------------------------------------------------------------
_BASE = ["--prior-max-concentration", "50", "--prior-background-weight", "0",
         "--prior-exponent", "0.5"]
CONFIGURATIONS = {
    "noprior": [],
    "exp0.5_decay0.9": _BASE + ["--prior-decay", "0.9"],   # default decay, reference
    "exp0.5_decay0.7": _BASE + ["--prior-decay", "0.7"],   # faster fade
    "exp0.5_decay0.5": _BASE + ["--prior-decay", "0.5"],   # much faster fade
}

BASELINE_LABEL = "noprior"


def build_configurations(priors_file: str) -> dict:
    """Label -> full CLI args. Every non-baseline arm gets ``--priors <file>``
    prepended; the baseline runs with no priors at all."""
    out = {}
    for label, extra in CONFIGURATIONS.items():
        prefix = [] if label == BASELINE_LABEL else ["--priors", priors_file]
        out[label] = prefix + extra
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run the sweep")
    sc.add_run_args(r, list(CONFIGURATIONS), gradients_default=True)
    r.set_defaults(func=lambda a: sc.run_sweep(a, build_configurations(a.priors_file)))

    a = sub.add_parser("analyze", help="Summarize a sweep CSV")
    sc.add_analyze_args(a)
    a.set_defaults(func=lambda a: sc.analyze(a, BASELINE_LABEL))

    t = sub.add_parser("traj", help="Anytime best-energy-vs-budget curves + plots")
    sc.add_traj_args(t)
    t.set_defaults(func=lambda a: sc.trajectory(a, BASELINE_LABEL))

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
