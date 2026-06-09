#!/usr/bin/env python
"""
Compare the gradient-enhanced GP surrogate against the value-only GP over a
molecule set, holding the acquisition loop fixed.

Three arms isolate the surrogate. The acquisition loop, init method, priors, and
budget are identical; the only difference is whether each energy evaluation also
contributes a projected torsion gradient (dE/dtheta) to the GP:

  nograd      : value-only GP                      (current default)
  grad        : --use-gradients                    (gradient GP, whole run, full
                hyperparameter fit every step) -- per-step cost grows as
                O((n*(1+d))^3), so this is intractable late on large molecules
                (a reference; the predictive skip protects it).
  gradhybrid  : --use-gradients --gradient-steps N (gradient GP for N steps, then
                value-only) -- keeps the early gradient benefit at bounded cost.
                Set N with --grad-switch-step (default 50).
  gradfreeze  : --use-gradients --grad-refit-dense-until N -- gradient GP for the
                whole run, but hyperparameters are cold-fit only for the first N
                steps then FROZEN (later steps only re-condition). Keeps ALL
                gradient data, stays tractable, validated quality-neutral vs grad.
  gradcadence : gradfreeze + --grad-refit-every K -- additionally cold-refreshes the
                frozen hyperparameters every K post-dense steps (tests whether
                periodic refits beat a pure freeze, at extra cost). Both cadence/
                freeze arms are left in by the predictive skip.

The hypothesis behind gradient-enhanced BO is data efficiency: every evaluation
carries dE/dtheta as well, so the surrogate should locate the minimum in fewer
calls. The payoff is therefore expected EARLY -- the trajectory (anytime) view is
the primary comparison; parity at full budget is unremarkable. This mirrors
sweep_init.py; see scripts/gradient_benchmark.py for the standalone
calls-to-minimum variant that drives the solver directly.

Single-surface requirement: the sweep runs with ``--relax``, and --use-gradients
with --relax requires --energy and --optimizer to be the SAME method (the torsion
gradient is only dE*/dtheta at a constrained minimum of the energy calculator).
This script therefore defaults both to ``gfnff`` and refuses a mismatched pair.

Cost warning: the full ``grad`` arm runs the gradient GP for the whole budget, and
its per-step cost grows ~O((n*(1+d))^3) -- on molecules above ~70 BO steps it runs
for hours and hits the per-trial --timeout, wasting compute. Two defenses:
  * compare ``nograd`` vs ``gradhybrid`` only (drop the doomed arm entirely):
        ... run --configs nograd,gradhybrid ...
  * keep ``grad`` where it's feasible but predictively skip it where it isn't:
        ... --skip-grad-above-steps 70
    (estimates each molecule's budget from its dihedral count and skips only the
    arms whose gradient-GP phase would exceed the threshold; gradhybrid's capped
    phase stays in). Both avoid grinding large molecules to a timeout.

Three phases (shared machinery lives in sweep_common.py):

  # 1. Run the sweep (writes a summary CSV + a per-evaluation trajectory CSV).
  #    --skip-grad-above-steps keeps the full-grad arm from timing out on big mols:
  python scripts/sweep_gradient.py run --input mols.csv --output sweep_grad.csv \
      --seeds 1234,12345,3141,314159,42 --workers 8 --skip-grad-above-steps 70

  # 2. Analyze (per-config summary + paired-by-(molecule, seed) vs the baseline):
  python scripts/sweep_gradient.py analyze sweep_grad.csv

  # 3. Trajectory (anytime curves: who is ahead at 25%/50%/... of the budget):
  python scripts/sweep_gradient.py traj sweep_grad_traj.csv
"""

import argparse
import sys
from pathlib import Path

# Shared sweep machinery (same directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_common as sc  # noqa: E402

CONFIG_NAMES = ["nograd", "grad", "gradhybrid", "gradfreeze", "gradcadence"]
BASELINE_LABEL = "nograd"


def build_configurations(switch_step: int, dense_until: int, cadence_stride: int) -> dict:
    """Arm -> extra CLI args. Only the surrogate differs; init, priors, budget, and
    energy/optimizer are shared:

      nograd      : value-only GP.
      grad        : gradient-enhanced GP for the whole run, full hyperparameter fit
                    every step. Its per-step cost grows as O((n*(1+d))^3), so this
                    is intractable late on large molecules (the reference, not a
                    practical setting; the predictive skip protects it).
      gradhybrid  : gradient GP for the first ``switch_step`` BO steps, then
                    value-only -- keeps the early gradient benefit at bounded cost.
      gradfreeze  : gradient GP for the whole run; hyperparameters are cold-fit for
                    the first ``dense_until`` steps then FROZEN (later steps only
                    re-condition -- one Cholesky instead of ~200). Keeps all gradient
                    data, stays tractable, validated quality-neutral vs ``grad``.
      gradcadence : like gradfreeze but cold-REFRESHES the frozen hyperparameters
                    every ``cadence_stride`` post-dense steps -- tests whether
                    periodic refits buy anything over a pure freeze (at extra cost,
                    since late cold refits are expensive).
    """
    dense = ["--use-gradients", "--grad-refit-dense-until", str(dense_until)]
    return {
        "nograd": [],
        # grad = the slow full-refit reference: dense=0 opts out of the default freeze.
        "grad": ["--use-gradients", "--grad-refit-dense-until", "0"],
        "gradhybrid": ["--use-gradients", "--gradient-steps", str(switch_step)],
        "gradfreeze": dense,
        "gradcadence": dense + ["--grad-refit-every", str(cadence_stride)],
    }


def run(args: argparse.Namespace) -> None:
    """Default and validate the single-surface requirement, then sweep.

    The gradient arms need energy == optimizer under ``--relax`` (sweep_common always
    relaxes); both default to ``gfnff`` (see add_run_args), so refuse a mismatched
    pair rather than letting every gradient trial crash.
    """
    sc.require_single_surface(args.energy, args.optimizer)
    sc.run_sweep(args, build_configurations(
        args.grad_switch_step, args.grad_refit_dense_until, args.grad_cadence_stride,
    ))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run the sweep")
    sc.add_run_args(r, CONFIG_NAMES)
    r.add_argument("--grad-switch-step", type=int, default=50,
                   help="For the 'gradhybrid' arm: BO step at which to switch from "
                   "the gradient-enhanced GP to the value-only GP (default 50). "
                   "Molecules whose budget is below this never switch.")
    r.add_argument("--grad-refit-dense-until", type=int, default=20,
                   help="For the 'gradfreeze'/'gradcadence' arms: number of leading "
                   "BO steps with a cold full hyperparameter fit before freezing "
                   "(default 20, matching the bouquet default).")
    r.add_argument("--grad-cadence-stride", type=int, default=10,
                   help="For the 'gradcadence' arm only: cold-refresh the frozen "
                   "hyperparameters every N post-dense steps (default 10). "
                   "'gradfreeze' never refreshes.")
    r.set_defaults(func=run)

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
