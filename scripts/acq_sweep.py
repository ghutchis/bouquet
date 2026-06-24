#!/usr/bin/env python
"""
Speed-vs-quality sweep of the two acquisition-side levers for high-d BO cost.

The timing study showed the per-step BO cost at d>=8 is dominated by the
acquisition optimization (optimize_acqf, ~60% and growing), NOT the GP fit (~6-8%,
already cheap via the freeze schedule) nor xTB (~20%). Two levers reduce it:

  * acquisition effort -- fewer optimize_acqf restarts/raw-samples (a ~constant
    factor; arms acq32/acq16/acq8 vs the 64/64 baseline);
  * gradient-step cap -- switch to the value-only GP after N steps, so each
    acquisition posterior eval is O(n) instead of O(n*(1+d)) (attacks the growth
    at high d; arms hybrid75/hybrid100).

Each is a speed gain only if search QUALITY holds, so this measures both on the
same molecules: per-step t_acq (speed) AND whether the arm still reaches the
best-across-arms minimum (quality), keyed on the certificate's new t_acq/t_eval
timers and e_best trajectory. Focus where acquisition dominates (d>=6).

  # Run (after the main benchmark finishes -- shares CPU):
  python scripts/acq_sweep.py run --input smiles/stopbench-500.smi \
      --output runs/acq/acq.csv --d-min 6 --d-max 12 --per-bin 4 \
      --seeds 1 --workers 8 --timeout 28800

  # Analyze (per-arm speed + quality vs the 64/64 baseline):
  python scripts/acq_sweep.py analyze runs/acq/acq.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_common as sc  # noqa: E402
from stop_benchmark import ceiling, dihedral_count  # noqa: E402

KCAL = 23.060541945
BASELINE = "base"

# Arms: the 64/64 full-gradient baseline, the acquisition-effort diagonal, and two
# value-only-late (hybrid) switch points. All keep gradients on otherwise.
CONFIGS = {
    "base":      ["--use-gradients"],  # acq 64/64 (bouquet default)
    "acq32":     ["--use-gradients", "--acq-num-restarts", "32", "--acq-raw-samples", "32"],
    "acq16":     ["--use-gradients", "--acq-num-restarts", "16", "--acq-raw-samples", "16"],
    "acq8":      ["--use-gradients", "--acq-num-restarts", "8", "--acq-raw-samples", "8"],
    "hybrid75":  ["--use-gradients", "--gradient-steps", "75"],
    "hybrid100": ["--use-gradients", "--gradient-steps", "100"],
}


def run(args):
    sc.require_single_surface(args.energy, args.optimizer)
    args.certificate = True   # need the per-step t_acq/t_eval timers
    args.geometry = True
    d_cache: dict = {}

    def d_of(smiles):
        if smiles not in d_cache:
            d_cache[smiles] = dihedral_count(smiles)
        return d_cache[smiles]

    def num_steps_fn(smiles):
        d = d_of(smiles)
        return None if d is None else ceiling(d, args.ceiling_anchor, args.ceiling_p,
                                              args.ceiling_margin, args.ceiling_cap)

    # Keep only the first --per-bin molecules in each d-bin within [d_min, d_max].
    bin_count: dict = {}

    def mol_skip_fn(smiles, name):
        d = d_of(smiles)
        if d is None or not (args.d_min <= d <= args.d_max):
            return True
        if bin_count.get(d, 0) >= args.per_bin:
            return True
        bin_count[d] = bin_count.get(d, 0) + 1
        return False

    requested = args.arms.split(",") if args.arms else list(CONFIGS)
    unknown = [a for a in requested if a not in CONFIGS]
    if unknown:
        raise SystemExit(
            f"Unknown arm(s) {unknown}; choose from {list(CONFIGS)}."
        )
    configs = {k: CONFIGS[k] for k in requested}
    print(f"acq_sweep arms: {list(configs)}")
    sc.run_sweep(args, configs, num_steps_fn=num_steps_fn, mol_skip_fn=mol_skip_fn)


def analyze(args):
    import numpy as np
    import pandas as pd
    cert = pd.read_csv(args.cert)
    cert["e_best_k"] = cert.e_best * KCAL
    cert["d"] = pd.to_numeric(cert.num_dihedrals, errors="coerce")
    # per trial: final e_best (kcal), total wall, per-step t_acq, energy-convergence
    per = []
    for (cfg, name, seed), g in cert.groupby(["config", "name", "seed"]):
        g = g.sort_values("n_calls")
        fin = g.e_best_k.min()
        # energy-convergence: first call within --eps of the trial's own final best
        hit = g.loc[g.e_best_k <= fin + args.eps, "n_calls"]
        d0 = g.d.iloc[0]  # num_dihedrals can be NaN if a trial's count didn't parse
        per.append({
            "config": cfg, "name": name, "seed": seed,
            "d": int(d0) if pd.notna(d0) else -1,
            "final_eb": fin, "wall_s": g.wall_s.max(),
            "t_acq_step": g.t_acq.mean() if "t_acq" in g else np.nan,
            "n_converge": int(hit.iloc[0]) if len(hit) else int(g.n_calls.max()),
        })
    df = pd.DataFrame(per)
    # cross-arm reference: best final_eb per molecule over all arms
    best = df.groupby("name").final_eb.min().rename("best_eb")
    df = df.merge(best, on="name")
    df["hit_best"] = df.final_eb <= df.best_eb + args.eps   # reached the best-of-arms

    print(f"acq_sweep: {df.name.nunique()} molecules, arms {sorted(df.config.unique())}, "
          f"d {int(df.d.min())}-{int(df.d.max())} (eps={args.eps} kcal)\n")
    base = df[df.config == BASELINE]
    base_wall = base.groupby("name").wall_s.first()
    print(f"{'arm':10s} {'reliab':>7s} {'med wall(s)':>11s} {'t_acq/step':>11s} "
          f"{'med n_conv':>10s} {'speedup':>8s}")
    for cfg, g in df.groupby("config"):
        # speedup vs baseline on the SAME molecules (paired by name)
        m = g.merge(base_wall.rename("base_wall"), on="name")
        speed = (m.base_wall / m.wall_s).median()
        print(f"{cfg:10s} {g.hit_best.mean():7.2f} {g.wall_s.median():11.0f} "
              f"{g.t_acq_step.mean():11.3f} {g.n_converge.median():10.0f} {speed:7.2f}x")
    print("\nreliability by d (fraction reaching best-of-arms minimum):")
    piv = df.pivot_table(index="d", columns="config", values="hit_best", aggfunc="mean")
    print(piv.round(2).to_string())
    df.to_csv(Path(args.cert).with_name("acq_sweep_summary.csv"), index=False)
    print(f"\nwrote {Path(args.cert).with_name('acq_sweep_summary.csv')}")
    print("Read: an arm wins if reliability ~= base AND speedup > 1. acq* = constant "
          "speedup; hybrid* should help more at high d (cheaper posterior).")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run the acquisition/hybrid arm sweep")
    sc.add_run_args(r, list(CONFIGS))  # provides --input (the manifest .smi), --output, etc.
    r.add_argument("--arms", default=None,
                   help=f"comma-subset of arms (default all: {','.join(CONFIGS)})")
    r.add_argument("--d-min", type=int, default=6)
    r.add_argument("--d-max", type=int, default=12)
    r.add_argument("--per-bin", type=int, default=4, help="molecules per d-bin")
    r.add_argument("--ceiling-anchor", type=float, default=25.0)
    r.add_argument("--ceiling-p", type=float, default=1.5)
    r.add_argument("--ceiling-margin", type=float, default=3.0)
    r.add_argument("--ceiling-cap", type=int, default=1500)
    r.set_defaults(func=run)

    a = sub.add_parser("analyze", help="Per-arm speed vs quality")
    a.add_argument("cert", type=Path, nargs="?", default=None,
                   help="acq_sweep certificate CSV (<output stem>_cert.csv)")
    a.add_argument("--eps", type=float, default=0.5,
                   help="kcal/mol tolerance for convergence / reaching best-of-arms")
    a.set_defaults(func=analyze)

    args = p.parse_args()
    if args.command == "analyze" and args.cert is None:
        sys.exit("analyze needs the *_cert.csv path")
    args.func(args)


if __name__ == "__main__":
    main()
