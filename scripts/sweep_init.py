#!/usr/bin/env python
"""
Compare initial-point strategies (random vs prior-peak seeding) over a molecule
set, holding the acquisition loop fixed.

Two arms isolate the peak-seeded initialization. PiBO acquisition steering is OFF
in both (the peaks arm passes --prior-exponent 0), so gfn2_priors.json is used
only to place the initial points -- any difference is attributable to the init
method alone, not to a prior-guided search:

  random : --init-method random   (current default; Gaussian around the start)
  peaks  : --init-method peaks --priors <file> --prior-exponent 0
           (systematic grid / weighted sampling from the prior peaks)

Three phases (shared machinery lives in sweep_common.py):

  # 1. Run the sweep (writes a summary CSV + a per-evaluation trajectory CSV):
  python scripts/sweep_init.py run --input mols.csv --output sweep_init.csv \
      --seeds 1234,12345,3141,314159,42 --workers 8

  # 2. Analyze (per-config summary + paired-by-(molecule, seed) vs the baseline):
  python scripts/sweep_init.py analyze sweep_init.csv

  # 3. Trajectory (anytime curves: who is ahead at 25%/50%/... of the budget):
  python scripts/sweep_init.py traj sweep_init_traj.csv

Peak seeding is expected to win *early* even when the final energy is a wash at
full budget, so the trajectory view is the primary comparison.
"""

import argparse
import sys
from pathlib import Path

# Shared sweep machinery (same directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_common as sc  # noqa: E402

CONFIG_NAMES = ["random", "peaks"]
BASELINE_LABEL = "random"


def build_configurations(priors_file: str) -> dict:
    """Arm -> full CLI args. The peaks arm uses ``priors_file`` only for peak
    locations: ``--prior-exponent 0`` keeps the acquisition loop prior-free,
    matching the random baseline."""
    return {
        "random": ["--init-method", "random"],
        "peaks": [
            "--init-method", "peaks",
            "--priors", priors_file,
            "--prior-exponent", "0",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run the sweep")
    sc.add_run_args(r, CONFIG_NAMES)
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
