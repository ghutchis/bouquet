#!/usr/bin/env python
"""
Smoke test: does an *obvious, correct* dihedral prior beat no-prior on molecules
where the prior is almost entirely right?

Molecule class: straight-chain alkane diols/diamines/amino-alcohols. Their global
minimum folds the H-bonding ends together, so it is NOT fully extended -- but every
interior C-C-C-C backbone torsion is a predictable staggered/anti minimum. A correct
backbone prior should therefore give an obvious head start.

Three arms (paired by molecule+seed against noprior):
  - noprior : baseline, no PiBO steering.
  - gfn2    : the fitted gfn2_priors.json (whose generic sp3-sp3 entry actually puts
              ~83% of its weight near 0 deg / eclipsed -- backwards for an alkane).
  - ideal   : smoke_priors/ideal_prior.json, a hand-crafted anti-dominant staggered
              backbone prior (the "obvious" prior).

Reuses all sweep_common machinery (run/analyze/traj), so:
  python scripts/sweep_smoke.py run --input smoke_priors/mols.csv \
      --output smoke_priors/smoke.csv --seeds 1234,42,7 --workers 8
  python scripts/sweep_smoke.py analyze smoke_priors/smoke.csv
  python scripts/sweep_smoke.py traj    smoke_priors/smoke_traj.csv --plot-dir smoke_priors
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_common as sc  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
GFN2 = str(REPO / "gfn2_priors.json")
IDEAL = str(REPO / "smoke_priors" / "ideal_prior.json")

CONFIGURATIONS = {
    "noprior": [],
    "gfn2": ["--priors", GFN2],
    "ideal": ["--priors", IDEAL],
}
BASELINE_LABEL = "noprior"


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run the smoke sweep")
    sc.add_run_args(r, list(CONFIGURATIONS), gradients_default=False)
    r.set_defaults(func=lambda a: sc.run_sweep(a, CONFIGURATIONS))

    a = sub.add_parser("analyze", help="Summarize a smoke CSV")
    sc.add_analyze_args(a)
    a.set_defaults(func=lambda a: sc.analyze(a, BASELINE_LABEL))

    t = sub.add_parser("traj", help="Anytime curves")
    sc.add_traj_args(t)
    t.set_defaults(func=lambda a: sc.trajectory(a, BASELINE_LABEL))

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
