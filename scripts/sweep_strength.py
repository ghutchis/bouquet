#!/usr/bin/env python
"""
PiBO prior-strength sweep (exponent x decay) on the smoke molecule set, AFTER the
mean-over-dihedrals normalization (so the prior's magnitude is O(1) in dihedral
count and the exponent means the same thing across molecule sizes).

PiBO idea: the prior is a *hint* that should guide the BO early and then decay in
favor of real data. Effective strength at BO step t is exponent * decay**t, so:
  - exponent sets the initial hint strength vs logEI,
  - decay sets how fast it hands off to the data.

All non-baseline arms use the hand-crafted ideal prior (smoke_priors/ideal_prior.json).
Paired vs noprior by (molecule, seed).

  python scripts/sweep_strength.py run   --input smoke_priors/mols.csv \
      --output smoke_priors/strength.csv --seeds 1234,42,7 --workers 8
  python scripts/sweep_strength.py analyze smoke_priors/strength.csv
  python scripts/sweep_strength.py traj    smoke_priors/strength_traj.csv --plot-dir smoke_priors
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_common as sc  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
IDEAL = str(REPO / "smoke_priors" / "ideal_prior.json")


def _arm(exp: str, decay: str) -> list:
    return ["--priors", IDEAL, "--prior-exponent", exp, "--prior-decay", decay]


# Vary exponent at the default decay (0.9), plus one slower-decay variant to probe
# whether a weaker-but-more-persistent hint helps.
CONFIGURATIONS = {
    "noprior": [],
    "e2.0_d0.9": _arm("2.0", "0.9"),   # current default (prior-dominated)
    "e1.0_d0.9": _arm("1.0", "0.9"),
    "e0.5_d0.9": _arm("0.5", "0.9"),
    "e0.25_d0.9": _arm("0.25", "0.9"),
    # The gfnff exponent-floor probe found the sweet spot near ~0.1 (e0.10 beat
    # noprior on hexane); 0.25-2.0 over-steers. These bracket the low end.
    "e0.1_d0.9": _arm("0.1", "0.9"),
    "e0.05_d0.9": _arm("0.05", "0.9"),
    "e1.0_d0.95": _arm("1.0", "0.95"),  # weaker but persists longer
}
BASELINE_LABEL = "noprior"


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run the strength sweep")
    sc.add_run_args(r, list(CONFIGURATIONS), gradients_default=False)
    r.set_defaults(func=lambda a: sc.run_sweep(a, CONFIGURATIONS))

    a = sub.add_parser("analyze", help="Summarize a strength CSV")
    sc.add_analyze_args(a)
    a.set_defaults(func=lambda a: sc.analyze(a, BASELINE_LABEL))

    t = sub.add_parser("traj", help="Anytime curves")
    sc.add_traj_args(t)
    t.set_defaults(func=lambda a: sc.trajectory(a, BASELINE_LABEL))

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
