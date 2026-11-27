#!/usr/bin/env python
"""
Sweep correlated (bivariate) dihedral priors against univariate and no-prior.

The claim being tested is that a *correlated* prior finds a good conformer
SOONER, not that it finds a lower one: with enough steps every arm reaches the
same basin, so the asymptote is shared by construction and best-final mostly
measures ties. The interesting output is therefore ``traj``, not ``analyze``.

Three arms, and the middle one is the one that matters:

  noprior     no prior module at all
  univariate  the fitted 1D priors
  bivariate   the same 1D priors PLUS the correlated 2D priors

``bivariate`` differs from ``univariate`` by exactly one flag, so the comparison
isolates the correlated term. Comparing bivariate against *noprior* instead
confounds two things: a bivariate run still gets the built-in sp3_sp3/sp3_sp2
fallbacks on every uncovered dihedral, so that contrast measures "some prior vs
none" as much as "correlated vs independent". Keep noprior as the baseline for
reference, but read bivariate-vs-univariate.

  # 1. Run (writes a summary CSV + a per-evaluation trajectory CSV):
  python scripts/sweep_bivariate.py run --input mols.csv --output biv.csv \
      --bivariate-file bivariate_priors.json --seeds 1,2,3 --workers 8

  # 2. Anytime curves -- who is ahead at 25%/50%/75% of the budget:
  python scripts/sweep_bivariate.py traj biv_traj.csv --plot-dir plots

  # 3. Final-energy summary (expected to be a wash; see above):
  python scripts/sweep_bivariate.py analyze biv.csv

TWO THINGS THAT WILL SILENTLY RUIN THIS BENCHMARK
-------------------------------------------------
* **The kappa cap.** These bivariate priors are fitted under a bound of 500, and
  bouquet's default cap of 50 would clamp them at load time -- which is not the
  same model and is measurably worse (on one pattern, a capped fit scored below
  assuming independence). ``--prior-max-concentration`` must match the cap the
  priors were fitted at. It is applied to both prior arms here so they differ
  only in the correlated term.
* **Dilution.** ``DihedralPriorModule.forward`` AVERAGES the log-prior over
  active dimensions, so on a molecule with 8 dihedrals and one correlated pair
  the bivariate term is a quarter of the signal. Select the molecule set by the
  *fraction of dihedrals a correlated pair covers*, not by whether a pattern
  matches at all -- otherwise the arms are nearly identical by construction and
  the sweep measures noise.

Budget matters too: if the median trial captures only a fraction of the best
conformer at full budget, every arm is still in the noise-dominated early regime
and no anytime comparison can separate them. Use ``--auto`` (the default here) so
the step count scales with the molecule, and check the captured fraction that
``traj`` prints before drawing any conclusion.
"""

import argparse
import sys
from pathlib import Path

# Shared sweep machinery (same directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_common as sc  # noqa: E402

# Concentration cap. MUST match the cap the bivariate priors were fitted under
# (build_bivariate_priors.py --kappa-cap), and is applied to both prior arms so
# the only difference between them is the correlated term.
KAPPA_CAP = "500"

_BASE = ["--prior-max-concentration", KAPPA_CAP,
         "--prior-background-weight", "0",
         "--prior-exponent", "0.5"]

# Label -> extra args. "univariate" and "bivariate" both get --priors added by
# build_configurations; only "bivariate" additionally gets --bivariate-priors.
CONFIGURATIONS = {
    "noprior": [],
    "univariate": list(_BASE),
    "bivariate": list(_BASE),
}

BASELINE_LABEL = "noprior"
BIVARIATE_LABEL = "bivariate"


def build_configurations(priors_file: str, bivariate_file: str) -> dict:
    """Label -> full CLI args."""
    out = {}
    for label, extra in CONFIGURATIONS.items():
        if label == BASELINE_LABEL:
            out[label] = list(extra)
            continue
        args = ["--priors", priors_file] + extra
        if label == BIVARIATE_LABEL:
            args += ["--bivariate-priors", bivariate_file]
        out[label] = args
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run the sweep")
    sc.add_run_args(r, list(CONFIGURATIONS), gradients_default=True)
    r.add_argument("--bivariate-file", required=True,
                   help="JSON of correlated priors (build_bivariate_priors.py). "
                        "Fit it at the same cap passed to "
                        "--prior-max-concentration.")
    r.set_defaults(func=lambda a: sc.run_sweep(
        a, build_configurations(a.priors_file, a.bivariate_file)))

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
