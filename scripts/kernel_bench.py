#!/usr/bin/env python
"""End-to-end A/B of GP covariance arms: does a kernel change find better conformers?

Compares the shipped product kernel (one shared lengthscale) against its ARD
variant (a lengthscale per torsion), paired on the same code, molecules and seeds:

    product   the shipped kernel
    ard       per-torsion lengthscales

WHY ONLY THESE TWO. A first-order additive component and two pairwise ("trellis")
variants were also built and run through this harness. The additive term won
held-out prediction decisively -- +0.25 to +0.30 nats on 93% of 509 trials,
uniform across d=5-12 and n=50-200 -- and then LOST end-to-end, finishing +0.41
kcal/mol above product on 63% of paired trials, with the deficit growing from
+0.51 at step 25 to +0.88 at step 400. That growth is the over-exploitation
signature: a sharper posterior has smaller sigma, so expected improvement stops
exploring. Predictive accuracy and search utility are anti-correlated for this
surrogate, which is exactly why this end-to-end harness exists and why a
cross-validation win is not sufficient grounds to change the default.
ARD was the only arm never worse than product (54-59% of cells better at every
budget) but was not significant at the endpoint (p=0.94), so it stays a flag.

Running one arm and diffing against the stored stop_bench_gfn2 results would be
wrong: those predate the clash constraint, the bond-formation species guard, the
collective-move guards and the calculator reset, all of which hit the d>=8 regime
under test.

Budget is a FIXED matched number of BO steps rather than the production --auto
table. --auto reaches 2000 steps by d=11, which measures out at ~8400 CPU-hours
for 2 arms x 3 seeds. A matched budget also isolates the kernel at equal cost,
and because the certificate logs e_best every step, one run yields the whole
anytime curve, so the arms can be compared at every smaller budget too.

Usage (mirrors stop_benchmark.py; shared machinery in sweep_common.py):

  python scripts/kernel_bench.py run --input smiles/stopbench-500.smi \\
      --output ab.csv --seeds 1 --num-steps 400 --workers 16
  python scripts/kernel_bench.py analyze ab.csv
  python scripts/kernel_bench.py traj ab_traj.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_common as sc  # noqa: E402

CONFIG_NAMES = ["product", "ard"]
BASELINE_LABEL = "product"

# The two arms differ by exactly one flag. Everything else -- surface, gradients,
# priors, collective-move autodefaults, clash constraint, species guard -- is held
# identical, which is the entire point of the paired design.
ARMS = {
    # the shipped kernel: product of per-torsion periodic kernels, ONE lengthscale
    "product": ["--use-gradients", "--priors", "--kernel", "product"],
    # per-torsion lengthscales; identical to product at initialization, so the
    # differentiation has to be learned -- the arm most exposed to overfitting at
    # small budgets.
    "ard": ["--use-gradients", "--priors", "--kernel", "ard"],
}


def run(args: argparse.Namespace) -> None:
    sc.require_single_surface(args.energy, args.optimizer)
    # The certificate is what makes the anytime comparison possible (e_best per
    # step); the geometry trail is what lets the RMSD-identity criterion score
    # against the CREST reference. Both are the point of the run, so force them on.
    args.certificate = True
    args.geometry = True

    steps = args.num_steps
    sc.run_sweep(args, ARMS, num_steps_fn=lambda smiles: steps)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run both kernel arms, paired")
    sc.add_run_args(r, CONFIG_NAMES)
    # Score on the same surface as the CREST reference, as the stopping-rule work does.
    r.set_defaults(energy="gfn2", optimizer="gfn2")
    r.add_argument("--num-steps", type=int, default=400,
                   help="Fixed BO-step budget for BOTH arms (default 400). Matched "
                        "budget is the comparison; the certificate still yields the "
                        "anytime curve below it.")
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
