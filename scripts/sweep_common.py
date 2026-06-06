#!/usr/bin/env python
"""
Shared machinery for the bouquet sweep scripts (sweep_init.py, sweep_priors.py).

A "sweep" runs ``bouquet.cli`` (with ``--auto --relax``) over a grid of
(configuration x molecule x seed) and records, per trial:

  - a tidy summary row (best step, final E-E0, ...) -> the summary CSV, and
  - a per-evaluation trajectory (running best-so-far) -> the trajectory CSV.

The individual scripts only differ in (a) the configurations being compared and
(b) which one is the paired-comparison baseline; everything else -- running,
trajectory parsing, the paired-by-seed analysis, and the anytime plots -- lives
here so the two stay in lock-step.

Configurations are passed as ``{label: full_extra_cli_args}`` (each arm fully
specifies its own flags, including ``--priors`` where relevant), so ``run_one``
is configuration-agnostic.

Paired comparisons are by (molecule, seed): bouquet now seeds every RNG
(numpy + torch) from ``--seed``, so the arms share their randomness at a fixed
seed and differencing per (name, seed) cancels the shared Bayesian-optimization
noise (common random numbers), isolating the configuration effect.
"""

import argparse
import csv
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reuse the log parser from batch.py (same directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch import parse_log_output  # noqa: E402

ENERGY_CHOICES = ["ani", "b3lyp", "b97", "gfn0", "gfn2", "gfnff"]
PRIORS_FILE_DEFAULT = "gfn2_priors.json"

# Tidy summary-CSV columns.
FIELDNAMES = [
    "config",
    "name",
    "smiles",
    "num_dihedrals",
    "seed",
    "success",
    "best_step",
    "low_step",
    "good_step",
    "e_e0_constrained",
    "e_e0_unconstrained",
]

# Per-evaluation trajectory CSV columns (one row per search evaluation).
TRAJ_FIELDNAMES = [
    "config",
    "name",
    "seed",
    "num_dihedrals",
    "eval_index",      # 0 = start point, then 1..K in evaluation order
    "phase",           # start | init | bo
    "e_e0",            # this evaluation's relative energy (eV)
    "best_so_far",     # running min of e_e0 up to and including this evaluation
    "n_search_evals",  # K: total init+BO evaluations in this trial
]

# Per-evaluation log lines emitted by the solver, in evaluation order. "energy in
# step" is a Bayesian-optimization step; the others are initial guesses. The
# post-search "Performed final relaxation ..." lines are deliberately NOT matched,
# so broken final-relaxation geometries never enter the trajectory.
_EVAL_RE = re.compile(
    r"Evaluated (initial guess|peak guess|conformer|energy in step)"
    r"\s+\d+\s*/\s*\d+\.\s+Energy-E0:\s+(-?\d+\.\d+)"
)


# ---------------------------------------------------------------------------
# Running a sweep
# ---------------------------------------------------------------------------


def subprocess_run(cmd: List[str]) -> Tuple[str, int]:
    """Run a command, returning (combined stdout+stderr, returncode).

    Logging may go to either stream, so both are combined for parsing.
    """
    import subprocess

    result = subprocess.run(
        cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL
    )
    return result.stdout + "\n" + result.stderr, result.returncode


def parse_trajectory(log_text: str) -> List[Tuple[str, float]]:
    """Return per-evaluation ``(phase, e_e0)`` in evaluation order (search only).

    ``phase`` is ``"bo"`` for Bayesian-optimization steps and ``"init"`` for
    initial guesses (random/peaks/conformer).
    """
    traj: List[Tuple[str, float]] = []
    for m in _EVAL_RE.finditer(log_text):
        phase = "bo" if m.group(1) == "energy in step" else "init"
        traj.append((phase, float(m.group(2))))
    return traj


def build_traj_rows(
    config: str, name: str, seed: int, num_dihedrals, log_text: str
) -> List[Dict]:
    """Build trajectory rows (running best-so-far) for one trial's log."""
    traj = parse_trajectory(log_text)
    k = len(traj)
    nd = num_dihedrals if num_dihedrals is not None else ""
    # eval 0: the start point, which is evaluated with E-E0 = 0 by definition.
    rows = [{
        "config": config, "name": name, "seed": seed, "num_dihedrals": nd,
        "eval_index": 0, "phase": "start", "e_e0": 0.0,
        "best_so_far": 0.0, "n_search_evals": k,
    }]
    best = 0.0
    for i, (phase, e) in enumerate(traj, start=1):
        best = min(best, e)
        rows.append({
            "config": config, "name": name, "seed": seed, "num_dihedrals": nd,
            "eval_index": i, "phase": phase, "e_e0": e,
            "best_so_far": best, "n_search_evals": k,
        })
    return rows


def run_one(
    config: str,
    extra_args: List[str],
    smiles: str,
    name: str,
    seed: int,
    energy_method: Optional[str],
    optimizer_method: Optional[str],
) -> Tuple[Dict, List[Dict]]:
    """Run a single (config, molecule, seed) trial.

    Returns ``(summary_row, traj_rows)``: the tidy summary dict and the
    per-evaluation trajectory rows (running best-so-far) for this trial.
    ``extra_args`` fully specifies the arm (including ``--priors`` if needed).
    """
    cmd = [
        sys.executable, "-m", "bouquet.cli",
        "--smiles", smiles,
        "--name", name,
        "--seed", str(seed),
        "--auto",
        "--relax",
    ]
    cmd += extra_args
    if energy_method:
        cmd += ["--energy", energy_method]
    if optimizer_method:
        cmd += ["--optimizer", optimizer_method]

    proc, returncode = subprocess_run(cmd)
    parsed = parse_log_output(proc)
    # Mirror batch.py: a trial only counts as successful when the process exited
    # cleanly AND a best step was parsed from the log.
    ok = returncode == 0 and parsed["best_step"] is not None
    row = {
        "config": config,
        "name": name,
        "smiles": smiles,
        "num_dihedrals": parsed["num_dihedrals"] if parsed["num_dihedrals"] is not None else "",
        "seed": seed,
        "success": int(ok),
        "best_step": parsed["best_step"] if ok else "",
        "low_step": parsed["low_step"] if ok else "",
        "good_step": parsed["good_step"] if ok else "",
        "e_e0_constrained": parsed["e_e0_constrained"] if ok else "",
        "e_e0_unconstrained": parsed["e_e0_unconstrained"] if ok else "",
    }
    # Only emit trajectory rows for cleanly-exited runs; a crashed process may
    # have produced a partial/misleading log.
    traj_rows = (
        build_traj_rows(config, name, seed, parsed["num_dihedrals"], proc)
        if returncode == 0
        else []
    )
    return row, traj_rows


def load_molecules(
    input_path: Path, smiles_col: str, name_col: str
) -> List[Tuple[str, str]]:
    """Load (smiles, name) pairs from a CSV/TSV with or without a header."""
    with open(input_path, newline="") as f:
        sample = f.read()
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters="\t, ")
        rows = list(csv.reader(f, dialect))
    if not rows:
        sys.exit(f"Error: empty input file {input_path}")

    first = rows[0]
    # Require BOTH columns before treating row 0 as a header; otherwise the
    # header.index() calls below would ValueError on the missing one.
    has_header = smiles_col in first and name_col in first
    if has_header:
        header, data = first, rows[1:]
        si, ni = header.index(smiles_col), header.index(name_col)
    else:
        si, ni, data = 0, 1, rows
        print("No header detected; assuming column 0 = SMILES, column 1 = name.")

    return [(r[si], r[ni]) for r in data if len(r) > max(si, ni)]


def load_done_keys(output_path: Path) -> set:
    """Return the set of (config, name, seed) already present in the output CSV."""
    done = set()
    if not output_path.exists():
        return done
    with open(output_path, newline="") as f:
        for row in csv.DictReader(f):
            done.add((row["config"], row["name"], str(row["seed"])))
    return done


def run_sweep(args: argparse.Namespace, configurations: Dict[str, List[str]]) -> None:
    """Run every (config, molecule, seed) trial; append to summary + trajectory CSVs.

    ``configurations`` maps each arm label to its full extra CLI args.
    """
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    configs = args.configs.split(",") if args.configs else list(configurations)
    for c in configs:
        if c not in configurations:
            sys.exit(f"Unknown config '{c}'. Known: {', '.join(configurations)}")

    mols = load_molecules(args.input, args.smiles_column, args.name_column)
    print(f"Loaded {len(mols)} molecules; seeds={seeds}; configs={configs}")

    # Trajectory CSV path: explicit, or "<output stem>_traj<suffix>".
    traj_path = args.traj_output or args.output.with_name(
        f"{args.output.stem}_traj{args.output.suffix or '.csv'}"
    )

    # Build the full task list, then drop any already-done (resume).
    done = load_done_keys(args.output) if args.resume else set()
    if not args.output.exists():
        with open(args.output, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
    if not traj_path.exists():
        with open(traj_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=TRAJ_FIELDNAMES).writeheader()

    tasks = []
    for config in configs:
        for smiles, name in mols:
            for seed in seeds:
                if (config, name, str(seed)) in done:
                    continue
                tasks.append((config, smiles, name, seed))

    total = len(tasks)
    print(f"{total} trials to run ({len(done)} already done)"
          f"{' [DRY RUN]' if args.dry_run else ''}")
    if args.dry_run or total == 0:
        return

    write_lock = threading.Lock()
    counter = {"n": 0}

    def submit(task):
        config, smiles, name, seed = task
        return run_one(
            config, configurations[config], smiles, name, seed,
            args.energy, args.optimizer,
        )

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(submit, t): t for t in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                row, traj_rows = fut.result()
            except Exception as e:  # keep the sweep alive on a single failure
                config, smiles, name, seed = task
                row = {f: "" for f in FIELDNAMES}
                row.update({"config": config, "name": name, "smiles": smiles,
                            "seed": seed, "success": 0})
                traj_rows = []
                print(f"  ERROR {config}/{name}/{seed}: {e}")
            with write_lock:
                with open(args.output, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)
                if traj_rows:
                    with open(traj_path, "a", newline="") as f:
                        csv.DictWriter(f, fieldnames=TRAJ_FIELDNAMES).writerows(traj_rows)
                counter["n"] += 1
                n = counter["n"]
            e = row.get("e_e0_unconstrained", "")
            bs = row.get("best_step", "")
            print(f"[{n}/{total}] {row['config']:<16} {row['name']:<16} "
                  f"seed={row['seed']:<7} best_step={bs} E-E0={e}")

    prog = Path(sys.argv[0]).name
    print(f"\nDone in {time.time() - t0:.0f}s. Wrote {args.output} and {traj_path}")
    print(f"Next: python {prog} analyze {args.output}")
    print(f"      python {prog} traj {traj_path}")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _wilcoxon_p(delta) -> float:
    """Wilcoxon signed-rank p-value for paired deltas (NaN if undefined)."""
    import numpy as np
    from scipy.stats import wilcoxon

    nz = delta[np.abs(delta) > 0]
    if len(nz) < 6:  # too few non-zero pairs for a meaningful test
        return float("nan")
    try:
        return float(wilcoxon(nz).pvalue)
    except ValueError:
        return float("nan")


def analyze(args: argparse.Namespace, baseline: str) -> None:
    """Per-config summaries + a paired-by-(molecule, seed) comparison vs baseline."""
    import pandas as pd

    df = pd.read_csv(args.input)
    df = df[df["success"] == 1].copy()
    for c in ["best_step", "e_e0_unconstrained", "e_e0_constrained", "num_dihedrals"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    tol = args.tol

    # A conformational E-E0 can't be hugely negative; a |E-E0| above max_abs_e means
    # the final (unconstrained) relaxation left the starting species (bond broke,
    # proton transfer, collapse). Such trials are physically meaningless and would
    # dominate any mean, so they are excluded from the *energy* aggregates (step/
    # trapping metrics are independent of energy magnitude and use all trials).
    df["physical"] = df["e_e0_unconstrained"].abs() <= args.max_abs_e
    n_out = int((~df["physical"]).sum())
    if n_out:
        print(f"\n=== Outliers (|E-E0| > {args.max_abs_e} eV; excluded from energy "
              f"stats): {n_out} of {len(df)} trials ===")
        print(df.loc[~df["physical"], "config"].value_counts().to_string())
        worst = df.nsmallest(min(10, n_out), "e_e0_unconstrained")
        print("worst trials (likely broken geometries -- check solutions/):")
        print(worst[["config", "name", "seed", "best_step",
                     "e_e0_unconstrained"]].to_string(index=False))

    df["improved_init"] = df["e_e0_unconstrained"] < -tol   # better than start geom
    df["bo_improved"] = df["best_step"] > 1                  # BO beat the first guess
    df["stuck_step1"] = df["best_step"] == 1                 # best came from step 1

    # Step/trapping summary over ALL successful trials (energy magnitude irrelevant).
    print("\n=== Per-config step/trapping summary (all successful trials) ===")
    steps = df.groupby("config").agg(
        n_trials=("best_step", "size"),
        n_outlier=("physical", lambda s: int((~s).sum())),
        pct_bo_improved=("bo_improved", lambda s: 100 * s.mean()),
        pct_stuck_step1=("stuck_step1", lambda s: 100 * s.mean()),
        median_best_step=("best_step", "median"),
    )
    print(steps.round(4).to_string())

    # Energy summary over physical trials only (robust to the outliers above).
    dfp = df[df["physical"]]
    print("\n=== Per-config energy summary (physical trials only) ===")
    per_trial = dfp.groupby("config").agg(
        n_physical=("e_e0_unconstrained", "size"),
        e_e0_mean=("e_e0_unconstrained", "mean"),
        e_e0_median=("e_e0_unconstrained", "median"),
        pct_improved_init=("improved_init", lambda s: 100 * s.mean()),
    )
    print(per_trial.round(4).to_string())

    # Best-of-seeds per molecule (what you actually keep from a multi-seed run),
    # over physical trials only. Restrict the across-molecule summaries to molecules
    # present (physically) in EVERY config -- otherwise a molecule that breaks under
    # one config but not another makes the per-config medians compare different sets.
    best = dfp.groupby(["config", "name"])["e_e0_unconstrained"].min().reset_index()
    n_cfg = best["config"].nunique()
    counts = best.groupby("name")["config"].nunique()
    common = set(counts[counts == n_cfg].index)
    dropped = sorted(set(best["name"]) - common)
    best = best[best["name"].isin(common)]
    print(f"\n=== Best-of-seeds per molecule (common set: {len(common)} molecules "
          f"present in all {n_cfg} configs) ===")
    if dropped:
        print(f"dropped (missing/all-broken in some config): {', '.join(dropped)}")
    per_mol = best.groupby("config").agg(
        n_mols=("e_e0_unconstrained", "size"),
        bestofseed_mean=("e_e0_unconstrained", "mean"),
        bestofseed_median=("e_e0_unconstrained", "median"),
    )
    print(per_mol.round(4).to_string())

    # ---- Paired comparison vs the baseline, paired by (molecule, seed) ----
    # Every RNG is seeded from --seed (torch included), so arms sharing a seed share
    # their randomness. Differencing per (name, seed) cancels the shared BO noise
    # (common random numbers) and isolates the config effect -- far lower variance
    # than pairing on best-of-seeds.
    if baseline not in dfp["config"].unique():
        print(f"\n(No '{baseline}' rows; skipping paired comparison.)")
        return

    # One column per arm; rows are (name, seed). dropna keeps only pairs where both
    # arms ran and were physical for that exact seed.
    wide = dfp.pivot_table(
        index=["name", "seed"], columns="config", values="e_e0_unconstrained"
    )
    ndih = (
        dfp.dropna(subset=["num_dihedrals"]).groupby("name")["num_dihedrals"].median()
    )

    def paired_deltas(config: str) -> "pd.DataFrame":
        """Per-(name, seed) delta = baseline - config (>0 => config lower/better)."""
        if config not in wide:
            return pd.DataFrame(columns=["name", "seed", "delta", "num_dihedrals"])
        pair = wide[[baseline, config]].dropna()
        d = (pair[baseline] - pair[config]).rename("delta").reset_index()
        d["num_dihedrals"] = d["name"].map(ndih)
        return d

    print(f"\n=== Paired vs '{baseline}', paired by (molecule, seed) "
          f"(win = lower by > {tol} eV) ===")
    rows = []
    deltas_by_config = {}
    for config in sorted(c for c in wide.columns if c != baseline):
        d = paired_deltas(config)
        deltas_by_config[config] = d
        delta = d["delta"]
        # Significance test is on the per-MOLECULE mean delta (n = molecules): the
        # several seeds of one molecule are correlated, so testing every pair would
        # pseudo-replicate. CRN keeps each molecule's mean delta low-variance.
        per_mol = d.groupby("name")["delta"].mean()
        rows.append({
            "config": config,
            "n_pairs": len(delta),
            "n_mols": int(per_mol.size),
            "wins": int((delta > tol).sum()),
            "ties": int((delta.abs() <= tol).sum()),
            "losses": int((delta < -tol).sum()),
            "median_gain_eV": float(delta.median()) if len(delta) else float("nan"),
            "mean_gain_eV": float(delta.mean()) if len(delta) else float("nan"),
            "wilcoxon_p_permol": _wilcoxon_p(per_mol.to_numpy()),
        })
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("(wins/ties/losses count all (molecule, seed) pairs; wilcoxon_p_permol "
          "tests the per-molecule mean paired Δ vs 0 with n=molecules. "
          "p > ~0.05 => not distinguishable from noise.)")

    # Stratify the per-(molecule, seed) paired gain by molecule size. A config should
    # help most where the search is hard (many dihedrals). Spearman
    # rho(gain, num_dihedrals): positive => it helps more as molecules get larger.
    edges = [0] + [float(x) for x in args.dihedral_bins.split(",")] + [float("inf")]
    strat = pd.concat(
        [d.assign(config=c) for c, d in deltas_by_config.items() if not d.empty],
        ignore_index=True,
    ).dropna(subset=["num_dihedrals"]) if deltas_by_config else pd.DataFrame()
    if not strat.empty:
        strat["bin"] = pd.cut(strat["num_dihedrals"], bins=edges, right=True)
        print(f"\n=== Paired gain vs '{baseline}', stratified by num_dihedrals ===")
        by_bin = strat.groupby(["config", "bin"], observed=True).agg(
            n=("delta", "size"),
            wins=("delta", lambda s: int((s > tol).sum())),
            losses=("delta", lambda s: int((s < -tol).sum())),
            median_gain_eV=("delta", "median"),
        )
        print(by_bin.round(4).to_string())

        print("\nSpearman rho(gain, num_dihedrals) per config "
              "(positive => the config helps more on larger molecules):")
        from scipy.stats import spearmanr
        for config, g in strat.groupby("config"):
            if g["num_dihedrals"].nunique() < 3 or len(g) < 6:
                print(f"  {config:<24} rho=  n/a (too few points)")
                continue
            rho, p = spearmanr(g["num_dihedrals"], g["delta"])
            print(f"  {config:<24} rho={rho:+.3f}  p={p:.3f}  (n={len(g)})")

    # Noise floor: how much does a single molecule's best E-E0 move between seeds?
    # If this dwarfs the config gains above, the comparison is seed-noise-limited and
    # needs more molecules/seeds rather than more tuning.
    seed_spread = (
        dfp.groupby(["config", "name"])["e_e0_unconstrained"].std().dropna()
    )
    if len(seed_spread):
        print(f"\nNoise floor: median seed-to-seed std of best E-E0 = "
              f"{seed_spread.median():.4f} eV "
              f"(compare to |median_gain_eV| above).")

    if args.summary_out:
        per_trial.to_csv(args.summary_out)
        print(f"\nWrote per-config summary to {args.summary_out}")


def _config_order(present, baseline: str) -> List[str]:
    """Baseline first, then the remaining configs sorted alphabetically."""
    others = sorted(c for c in present if c != baseline)
    return ([baseline] if baseline in present else []) + others


def _safe_filename(name: str) -> str:
    """Make a molecule name safe to use as a filename component."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "mol"


def trajectory(args: argparse.Namespace, baseline: str) -> None:
    """Anytime-performance analysis: best-energy-seen vs fraction of search budget.

    A good configuration should win EARLY -- at 25-50% of the budget -- even when
    the final energy is a wash. This builds the running best-so-far curve per trial,
    normalizes the x-axis to the fraction of that trial's init+BO budget (so
    molecules of different sizes are comparable), prints a paired-by-(molecule, seed)
    checkpoint table, and plots the configs (median over seeds).
    """
    import numpy as np
    import pandas as pd

    trial_keys = ["config", "name", "seed"]
    df = pd.read_csv(args.input)
    for c in ["eval_index", "e_e0", "best_so_far", "n_search_evals",
              "num_dihedrals", "seed"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["n_search_evals"] > 0].copy()
    if df.empty:
        sys.exit("No usable trajectory rows (n_search_evals == 0 everywhere).")

    # Fraction of this trial's budget consumed at each evaluation (0 = start point).
    df["frac"] = df["eval_index"] / df["n_search_evals"]

    # Drop broken trials (final best_so_far below -max_abs_e == dissociation etc.),
    # mirroring analyze's outlier gate so one blown-up geometry can't dominate.
    final_best = df.sort_values("eval_index").groupby(trial_keys)["best_so_far"].last()
    bad = set(final_best[final_best < -args.max_abs_e].index)
    if bad:
        print(f"Dropping {len(bad)} broken trial(s) (final best_so_far < "
              f"-{args.max_abs_e} eV): {', '.join(sorted({n for _, n, _ in bad}))}")
        df = df[~df.set_index(trial_keys).index.isin(bad)].copy()

    checkpoints = [float(x) for x in args.checkpoints.split(",")]
    configs = _config_order(df["config"].unique(), baseline)

    def interp_trial(g: pd.DataFrame, fracs) -> "np.ndarray":
        """best_so_far at the given budget fractions for one trial (monotone)."""
        g = g.sort_values("frac")
        return np.interp(fracs, g["frac"].to_numpy(), g["best_so_far"].to_numpy())

    # Per-trial best_so_far at each checkpoint.
    rows = []
    for (cfg, name, seed), g in df.groupby(trial_keys):
        for f, v in zip(checkpoints, interp_trial(g, checkpoints)):
            rows.append({"config": cfg, "name": name, "seed": seed,
                         "frac": f, "best_so_far": v})
    chk = pd.DataFrame(rows)

    # Per-molecule reference = lowest energy seen by ANY arm/seed (common target).
    best_known = df.groupby("name")["best_so_far"].min()  # most negative
    chk["captured"] = chk.apply(
        lambda r: r["best_so_far"] / best_known[r["name"]]
        if best_known[r["name"]] < 0 else np.nan, axis=1
    )

    tol = args.tol
    print(f"\n=== Anytime performance vs budget fraction "
          f"({len(best_known)} molecules, paired by (molecule, seed)) ===")
    print("E = median best_so_far over trials; cap = median captured = "
          "best_so_far / best_known (1.00 = matches the best conformer found by "
          "any arm/seed); 'vs base' = paired wins/ties/losses against the baseline.")
    for f in checkpoints:
        sub = chk[chk["frac"] == f]
        base_pairs = (
            sub[sub["config"] == baseline].set_index(["name", "seed"])["best_so_far"]
        )
        print(f"  budget {f * 100:4.0f}%:")
        for cfg in configs:
            g = sub[sub["config"] == cfg]
            if g.empty:
                continue
            med_e = g["best_so_far"].median()
            med_cap = g["captured"].median()
            tag = "  (baseline)"
            if cfg != baseline and not base_pairs.empty:
                cur = g.set_index(["name", "seed"])["best_so_far"]
                d = (base_pairs - cur).dropna()  # >0 => cfg lower (better)
                w = int((d > tol).sum())
                t = int((d.abs() <= tol).sum())
                ls = int((d < -tol).sum())
                tag = f"  vs base: +{w}/={t}/-{ls} (n={len(d)})"
            print(f"      {cfg:<24} E={med_e:+.4f} cap={med_cap:.2f}{tag}")
    print("(a config should lead the baseline at low budget if it helps; parity at "
          "100% is expected -- full budget tends to converge.)")

    if not args.no_plot:
        _plot_trajectories(df, best_known, configs, baseline, args.plot_dir)


def _plot_trajectories(df, best_known, configs, baseline, plot_dir) -> None:
    """Write two PNGs: an aggregate normalized anytime curve and a per-molecule grid."""
    import numpy as np

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots (pass --no-plot to silence).")
        return

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Baseline drawn black; the rest cycle through the default color list.
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors, ci = {}, 0
    for cfg in configs:
        if cfg == baseline:
            colors[cfg] = "black"
        else:
            colors[cfg] = cycle[ci % len(cycle)]
            ci += 1
    trial_keys = ["config", "name", "seed"]

    # ---- Aggregate: median fraction-of-best captured vs budget fraction ----
    grid = np.linspace(0.0, 1.0, 51)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for cfg in configs:
        sub = df[df["config"] == cfg]
        curves = []
        for (_, name, _), g in sub.groupby(trial_keys):
            bk = best_known[name]
            if not (bk < 0):
                continue
            g = g.sort_values("frac")
            curves.append(np.interp(grid, g["frac"], g["best_so_far"]) / bk)
        if not curves:
            continue
        C = np.vstack(curves)
        med = np.median(C, axis=0)
        lo, hi = np.percentile(C, 25, axis=0), np.percentile(C, 75, axis=0)
        ls = "--" if cfg == baseline else "-"
        ax.plot(grid, med, color=colors[cfg], ls=ls, label=f"{cfg} (n={len(curves)})")
        ax.fill_between(grid, lo, hi, color=colors[cfg], alpha=0.12)
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("fraction of search budget (init + BO evaluations)")
    ax.set_ylabel("fraction of best conformer captured\n(best_so_far / best_known)")
    ax.set_title("Anytime performance (median over trials, IQR band)")
    ax.legend(fontsize=8)
    p1 = plot_dir / "traj_aggregate.pdf"
    fig.tight_layout()
    fig.savefig(p1)
    plt.close(fig)
    print(f"Wrote {p1}")

    # ---- Per-molecule: one PDF each, median best_so_far over seeds + IQR band ----
    # Individual files are easier to scan than one large grid. Within a
    # (molecule, config) every seed shares the same eval_index range (the init+BO
    # budget is deterministic per molecule), so aggregating across seeds at each
    # eval_index gives an aligned median and 25-75% band.
    mol_dir = plot_dir / "per_molecule"
    mol_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(df["name"].unique())
    for name in names:
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        drew = False
        for cfg in configs:
            g = df[(df["name"] == name) & (df["config"] == cfg)]
            if g.empty:
                continue
            stats = g.groupby("eval_index")["best_so_far"].agg(
                med="median",
                lo=lambda s: s.quantile(0.25),
                hi=lambda s: s.quantile(0.75),
            )
            n_seeds = g["seed"].nunique()
            ls = "--" if cfg == baseline else "-"
            ax.plot(stats.index, stats["med"], color=colors[cfg], ls=ls,
                    label=f"{cfg} (n={n_seeds})")
            ax.fill_between(stats.index, stats["lo"], stats["hi"],
                            color=colors[cfg], alpha=0.15)
            drew = True
        if not drew:
            plt.close(fig)
            continue
        ax.set_title(name)
        ax.set_xlabel("evaluation index (init + BO)")
        ax.set_ylabel("best E-E0 seen (eV)  [median over seeds, IQR band]")
        ax.legend(fontsize=8)
        out = mol_dir / f"traj_{_safe_filename(name)}.pdf"
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
    print(f"Wrote {len(names)} per-molecule plots to {mol_dir}/")


# ---------------------------------------------------------------------------
# Shared argparse builders
# ---------------------------------------------------------------------------


def add_run_args(parser, config_names, priors_default=PRIORS_FILE_DEFAULT) -> None:
    parser.add_argument("--input", required=True, type=Path, help="CSV with smiles,name")
    parser.add_argument("--output", required=True, type=Path, help="Tidy output CSV")
    parser.add_argument("--traj-output", type=Path, default=None,
                        help="Per-evaluation trajectory CSV (default: <output>_traj.csv)")
    parser.add_argument("--seeds", default="1234,12345,3141,314159,42")
    parser.add_argument("--configs", default=None,
                        help=f"Comma-separated subset of: {', '.join(config_names)}")
    parser.add_argument("--priors-file", default=priors_default)
    parser.add_argument("--energy", default=None, choices=ENERGY_CHOICES)
    parser.add_argument("--optimizer", default=None, choices=ENERGY_CHOICES)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--name-column", default="name")
    parser.add_argument("--workers", "-w", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and trial count, then exit")


def add_analyze_args(parser) -> None:
    parser.add_argument("input", type=Path, help="Tidy sweep CSV from 'run'")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="Energy tolerance (eV) for win/improvement (default 1e-3)")
    parser.add_argument("--max-abs-e", type=float, default=5.0,
                        help="Exclude trials with |E-E0| above this many eV from energy "
                        "stats as broken geometries (default 5.0)")
    parser.add_argument("--dihedral-bins", default="3,6",
                        help="Comma-separated upper edges for num_dihedrals strata; "
                        "'3,6' => bins (0,3], (3,6], (6,inf) (default '3,6')")
    parser.add_argument("--summary-out", type=Path, default=None,
                        help="Optional path to write the per-config summary CSV")


def add_traj_args(parser) -> None:
    parser.add_argument("input", type=Path, help="Trajectory CSV from 'run' (..._traj.csv)")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="Energy tolerance (eV) for a paired win (default 1e-3)")
    parser.add_argument("--max-abs-e", type=float, default=5.0,
                        help="Drop trials whose final best_so_far < -this (eV) as broken "
                        "geometries (default 5.0)")
    parser.add_argument("--checkpoints", default="0.25,0.5,0.75,1.0",
                        help="Comma-separated budget fractions for the table "
                        "(default '0.25,0.5,0.75,1.0')")
    parser.add_argument("--plot-dir", type=Path, default=Path("."),
                        help="Directory for output PDFs: traj_aggregate.pdf plus "
                        "per_molecule/traj_<name>.pdf (default '.')")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip the PDFs; print the checkpoint table only")
