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

Paired comparisons are by (molecule, seed): bouquet seeds every RNG (numpy +
torch) from ``--seed``, so the arms share their randomness at a fixed seed and
differencing per (name, seed) cancels much of the shared Bayesian-optimization
noise (common random numbers), isolating the configuration effect.

Caveat: common random numbers only hold *exactly* when runs are deterministic.
Multi-threaded BLAS (the default) has non-deterministic reduction order, and the
chaotic BO search amplifies that ~1e-6 noise into different basins -- two
identical runs can finish ~0.02 eV apart. That ~0.02 eV is the paired-comparison
noise floor unless you pass ``--single-thread`` (bit-reproducible, slower). For a
clean isolation of a single configuration knob, run single-threaded; for a broad
sweep, rely on the per-molecule-mean statistics in ``analyze`` to average it out.
"""

import argparse
import csv
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Reuse the log parser from batch.py (same directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch import parse_log_output  # noqa: E402
from bouquet.calculator import CalculatorFactory  # noqa: E402

# Energy/optimizer choices come from the calculator registry (installed subset),
# so they track new methods automatically and never offer an unavailable one.
ENERGY_CHOICES = list(CalculatorFactory.available_methods())
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


def subprocess_run(
    cmd: List[str], timeout: Optional[float] = None, env: Optional[dict] = None
) -> Tuple[str, int]:
    """Run a command, returning (combined stdout+stderr, returncode).

    Logging may go to either stream, so both are combined for parsing. If
    ``timeout`` (seconds) is given and exceeded, the child is killed and a
    non-zero return code (124, matching coreutils ``timeout``) is returned along
    with whatever output was captured before the kill -- some molecules (large,
    very flexible) can run effectively forever, so the sweep must not block on
    one trial. ``env`` overrides the child environment (e.g. pinning BLAS threads
    for run-to-run determinism); None inherits the parent's.
    """
    import subprocess

    def _text(x) -> str:  # TimeoutExpired output is sometimes bytes even w/ text=True
        if x is None:
            return ""
        return x.decode("utf-8", "replace") if isinstance(x, bytes) else x

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL,
            timeout=timeout, env=env,
        )
        return result.stdout + "\n" + result.stderr, result.returncode
    except subprocess.TimeoutExpired as e:
        out = _text(e.stdout) + "\n" + _text(e.stderr)
        return out + f"\n[sweep] trial killed after {timeout:g}s timeout\n", 124


def single_thread_env() -> dict:
    """Parent environment with BLAS/OpenMP thread counts pinned to 1.

    The gradient-GP fit and the acquisition optimizer use multi-threaded BLAS,
    whose reduction order is not deterministic run-to-run; the chaotic BO search
    amplifies the resulting ~1e-6 noise into different basins (two identical runs
    can finish ~0.02 eV apart). Pinning to one thread makes a run bit-reproducible
    from its seed, which is what the paired (common-random-number) comparison needs
    to actually cancel noise between arms. The trade-off is slower per-trial fits.
    """
    import os

    return dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )


def require_single_surface(energy: Optional[str], optimizer: Optional[str]) -> None:
    """Exit unless ``energy == optimizer``. ``--use-gradients`` with ``--relax``
    needs a single surface so the projected torsion gradient equals dE*/dtheta at the
    constrained minimum of the energy calculator (see bouquet.cli)."""
    if energy != optimizer:
        sys.exit(f"--use-gradients with --relax requires --energy == --optimizer "
                 f"(got {energy} / {optimizer}); the torsion gradient is only "
                 "dE*/dtheta at a constrained minimum of the energy calculator.")


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
    timeout: Optional[float] = None,
    fail_log_dir: Optional[Path] = None,
    env: Optional[dict] = None,
    num_steps: Optional[int] = None,
    cert_dir: Optional[Path] = None,
    cert_betas: Optional[str] = None,
    geom_dir: Optional[Path] = None,
) -> Tuple[Dict, List[Dict], List[Dict]]:
    """Run a single (config, molecule, seed) trial.

    Returns ``(summary_row, traj_rows, cert_rows)``: the tidy summary dict, the
    per-evaluation trajectory rows (running best-so-far), and the per-BO-step
    stopping-rule certificate rows (empty unless ``cert_dir`` is given).
    ``extra_args`` fully specifies the arm (including ``--priors`` if needed).
    A trial exceeding ``timeout`` seconds is killed and recorded as a failure.
    On any failure, the captured stdout+stderr is written under ``fail_log_dir``
    (if given) so crashes can be told apart from timeouts after the fact.

    ``num_steps`` overrides the budget: None uses ``--auto`` (the tiered table),
    an int passes ``--num-steps`` instead (e.g. the stopping-rule benchmark's
    per-molecule ceiling C(d), which must exceed the auto cap). ``cert_dir``
    enables per-step certificate logging: bouquet writes a per-trial CSV there
    (``--certificate-log``), which is read back and returned with id columns;
    ``cert_betas`` is the comma-separated beta grid passed to ``--certificate-betas``.
    """
    cmd = [
        sys.executable, "-m", "bouquet.cli",
        "--smiles", smiles,
        "--name", name,
        "--seed", str(seed),
        "--relax",
    ]
    # Budget: --auto (tiered table) by default, or an explicit per-molecule cap.
    cmd += ["--num-steps", str(num_steps)] if num_steps is not None else ["--auto"]
    # Per-step certificate logging (stopping-rule benchmark): bouquet writes the
    # CSV to cert_file; one file per trial key so concurrent workers never collide.
    cert_file = None
    if cert_dir is not None:
        cert_dir.mkdir(parents=True, exist_ok=True)
        cert_file = cert_dir / f"{config}_{_safe_filename(name)}_seed{seed}.csv"
        cmd += ["--certificate-log", str(cert_file)]
        if cert_betas:
            cmd += ["--certificate-betas", cert_betas]
    # Per-trial geometry trail (best-improvement + final relaxed XYZ). Written
    # incrementally by the solver, so a timed-out trial keeps its improvements.
    if geom_dir is not None:
        geom_dir.mkdir(parents=True, exist_ok=True)
        geom_file = geom_dir / f"{config}_{_safe_filename(name)}_seed{seed}.xyz"
        cmd += ["--geometry-log", str(geom_file)]
    cmd += extra_args
    if energy_method:
        cmd += ["--energy", energy_method]
    if optimizer_method:
        cmd += ["--optimizer", optimizer_method]

    proc, returncode = subprocess_run(cmd, timeout=timeout, env=env)
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
    # Persist the log of a failed trial for diagnosis (timeout vs. crash vs. a
    # ValueError like the gradient single-surface check). Unique filename per
    # trial key, so concurrent workers never collide -- no lock needed.
    if not ok and fail_log_dir is not None:
        fail_log_dir.mkdir(parents=True, exist_ok=True)
        log_path = fail_log_dir / f"{config}_{_safe_filename(name)}_seed{seed}.log"
        with open(log_path, "w") as f:
            f.write(f"# cmd: {' '.join(cmd)}\n# returncode: {returncode}"
                    f"{'  (124 = timeout)' if returncode == 124 else ''}\n\n")
            f.write(proc)

    # Only emit trajectory rows for cleanly-exited runs; a crashed process may
    # have produced a partial/misleading log.
    traj_rows = (
        build_traj_rows(config, name, seed, parsed["num_dihedrals"], proc)
        if returncode == 0
        else []
    )

    # Certificate rows: read bouquet's per-trial CSV and prepend trial-identifying
    # columns. Kept for clean exits AND timeouts (returncode 124): a timed-out
    # high-d trial is a *right-censored* observation, and its partial trajectory is
    # exactly what the survival analysis needs -- discarding it would bias the
    # budget. Each row is stamped ``censored`` so the replay treats it correctly.
    # A crash (other returncodes) may leave a truncated line, so guard the read and
    # drop a trailing partial row defensively.
    cert_rows: List[Dict] = []
    if cert_file is not None and returncode in (0, 124) and cert_file.exists():
        nd = parsed["num_dihedrals"] if parsed["num_dihedrals"] is not None else ""
        censored = int(returncode == 124)
        with open(cert_file, newline="") as f:
            raw = list(csv.DictReader(f))
        if censored and raw and any(v in (None, "") for v in raw[-1].values()):
            raw = raw[:-1]  # drop a half-written final row from the kill
        for crow in raw:
            cert_rows.append(
                {"config": config, "name": name, "seed": seed,
                 "num_dihedrals": nd, "censored": censored, **crow}
            )
    return row, traj_rows, cert_rows


def predict_bo_budget(smiles: str) -> Optional[int]:
    """Predict how many BO steps ``bouquet --auto`` will run for ``smiles``.

    Uses bouquet's own dihedral detection and tiered ``--auto`` budget so the
    estimate tracks the CLI exactly. Returns ``None`` if it can't be computed
    (bad SMILES, import failure) -- callers then decline to skip and just run it.
    Imported lazily so non-gradient sweeps never pay the rdkit/torch import.
    """
    try:
        from bouquet.config import Configuration
        from bouquet.setup import detect_dihedrals, get_initial_structure

        _, mol = get_initial_structure(smiles)
        d = len(detect_dihedrals(mol))
        cfg = Configuration(smiles=smiles, auto_steps=True)
        return cfg.compute_auto_steps(d, cfg.init_steps)
    except Exception:
        return None


def dihedral_count(smiles: str) -> Optional[int]:
    """Number of rotatable dihedrals bouquet detects for ``smiles`` (None if it
    can't be computed -- bad SMILES, import failure -- so callers can decline to
    skip and just run it).

    Uses bouquet's own ``detect_dihedrals`` so the count matches what the CLI
    sees under ``--auto``. Imported lazily so the analyze/traj paths never pay the
    rdkit import.
    """
    try:
        from rdkit import Chem
        from bouquet.setup import detect_dihedrals
    except ImportError:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return len(detect_dihedrals(Chem.AddHs(mol)))


def max_dihedral_skip(max_dihedrals: Optional[int]):
    """Build a ``mol_skip_fn`` (for ``run_sweep``) dropping molecules with more
    than ``max_dihedrals`` rotatable dihedrals; returns None when the cap is None
    (skip nothing). A molecule whose count can't be determined is kept and run.

    ``run_sweep`` memoizes the predicate by molecule name, so the dihedral
    detection runs once per molecule regardless of seeds/configs.
    """
    if max_dihedrals is None:
        return None
    if max_dihedrals < 0:
        raise ValueError("max_dihedrals must be >= 0")

    def skip(smiles: str, name: str) -> bool:
        d = dihedral_count(smiles)
        return d is not None and d > max_dihedrals

    return skip


def _flag_int(extra_args: List[str], flag: str) -> Optional[int]:
    """The int value following ``flag`` in a CLI-arg list, or None if the flag is
    absent or its value can't be parsed."""
    if flag in extra_args:
        try:
            return int(extra_args[extra_args.index(flag) + 1])
        except (IndexError, ValueError):
            return None
    return None


def arm_gradient_steps(extra_args: List[str], budget: int) -> int:
    """Number of BO steps this arm would run the *gradient-enhanced* GP for, given
    a molecule's BO ``budget``.

    0 if the arm is value-only. A ``--gradient-steps N`` cap (e.g. the hybrid arm)
    limits it to ``min(N, budget)``; an uncapped ``--use-gradients`` arm runs the
    gradient GP for the whole ``budget``. This is what the predictive skip tests
    against, so capped arms are never skipped for being long.
    """
    if "--use-gradients" not in extra_args:
        return 0
    cap = _flag_int(extra_args, "--gradient-steps")
    return min(cap, budget) if cap and cap > 0 else budget


def arm_freezes_hypers(extra_args: List[str]) -> bool:
    """True if the gradient arm uses the freeze schedule, so its whole-run gradient
    GP is tractable and the predictive skip should leave it in.

    Freezing is bouquet's default (``Configuration.grad_refit_dense_until`` defaults
    to 20), so a bare ``--use-gradients`` arm freezes and is exempt; only an explicit
    ``--grad-refit-dense-until 0`` -- the slow full-refit reference -- does NOT freeze
    and remains subject to the skip. (Mirrors that bouquet default; keep in sync.)"""
    return _flag_int(extra_args, "--grad-refit-dense-until") != 0


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


def drop_failed(
    output_path: Path, traj_path: Path, cert_path: Optional[Path] = None
) -> set:
    """Remove every failed (``success != 1``) trial from the summary CSV and purge
    its rows from the trajectory CSV (and certificate CSV, if given). Returns the
    set of removed (config, name, seed) keys so a ``--resume`` afterwards re-runs
    exactly those trials.

    A clean-exit-but-unparsed trial can leave trajectory/certificate rows while
    still being recorded ``success=0``; purging them prevents duplicate rows when
    the retry succeeds.
    """
    if not output_path.exists():
        return set()
    with open(output_path, newline="") as f:
        rows = list(csv.DictReader(f))
    failed = {
        (r["config"], r["name"], str(r["seed"]))
        for r in rows if str(r.get("success")) != "1"
    }
    if not failed:
        return failed

    kept = [r for r in rows
            if (r["config"], r["name"], str(r["seed"])) not in failed]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(kept)

    if traj_path.exists():
        with open(traj_path, newline="") as f:
            trows = list(csv.DictReader(f))
        tkept = [r for r in trows
                 if (r["config"], r["name"], str(r["seed"])) not in failed]
        if len(tkept) != len(trows):
            with open(traj_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=TRAJ_FIELDNAMES)
                w.writeheader()
                w.writerows(tkept)

    # Purge failed trials from the certificate CSV too, preserving its (dynamic,
    # beta-dependent) column set from the existing header.
    if cert_path is not None and cert_path.exists():
        with open(cert_path, newline="") as f:
            reader = csv.DictReader(f)
            cfields = reader.fieldnames
            crows = list(reader)
        ckept = [r for r in crows
                 if (r["config"], r["name"], str(r["seed"])) not in failed]
        if len(ckept) != len(crows):
            with open(cert_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cfields)
                w.writeheader()
                w.writerows(ckept)
    return failed


def run_sweep(
    args: argparse.Namespace,
    configurations: Dict[str, List[str]],
    num_steps_fn=None,
    mol_skip_fn=None,
) -> None:
    """Run every (config, molecule, seed) trial; append to summary + trajectory CSVs.

    ``configurations`` maps each arm label to its full extra CLI args.
    ``mol_skip_fn``: optional ``(smiles, name) -> bool``; when it returns True the
    molecule is dropped from the run entirely (e.g. skip d > 12 in the benchmark).
    ``num_steps_fn``: optional ``smiles -> Optional[int]`` giving a per-molecule
    budget that overrides ``--auto`` (e.g. the stopping-rule benchmark's ceiling
    C(d)); None (default) keeps every trial on the tiered ``--auto`` budget.

    With ``args.certificate`` set, each trial also logs the per-BO-step
    certificate; those rows are aggregated into a master certificate CSV
    (``args.certificate_output`` or ``<output stem>_cert.csv``).
    """
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    configs = args.configs.split(",") if args.configs else list(configurations)
    for c in configs:
        if c not in configurations:
            sys.exit(f"Unknown config '{c}'. Known: {', '.join(configurations)}")

    # --use-gradients passthrough: turn on the gradient-enhanced GP for every arm
    # (requires a single surface; see require_single_surface / bouquet.cli).
    if getattr(args, "use_gradients", False):
        require_single_surface(args.energy, args.optimizer)
        configurations = {
            label: a if "--use-gradients" in a else a + ["--use-gradients"]
            for label, a in configurations.items()
        }
        print("Gradients ON for all arms (--use-gradients; freeze schedule by default).")

    mols = load_molecules(args.input, args.smiles_column, args.name_column)
    print(f"Loaded {len(mols)} molecules; seeds={seeds}; configs={configs}")

    # Ensure the output directory exists so the CSV writers below don't fail.
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Trajectory CSV path: explicit, or "<output stem>_traj<suffix>".
    traj_path = args.traj_output or args.output.with_name(
        f"{args.output.stem}_traj{args.output.suffix or '.csv'}"
    )

    # Certificate aggregation (stopping-rule benchmark). Master CSV columns are the
    # id columns + bouquet's per-step certificate columns (one lb_b<beta> per beta);
    # per-trial CSVs are written under cert_dir then read back in run_one.
    cert_enabled = getattr(args, "certificate", False)
    cert_path = cert_dir = cert_fieldnames = cert_betas = None
    if cert_enabled:
        from bouquet.config import (
            DEFAULT_CERTIFICATE_BETAS,
            format_certificate_betas,
            parse_certificate_betas,
        )
        from bouquet.io import CERTIFICATE_BASE_FIELDNAMES, certificate_lb_column

        cert_betas = getattr(args, "certificate_betas", None) or format_certificate_betas(
            DEFAULT_CERTIFICATE_BETAS
        )
        betas = parse_certificate_betas(cert_betas)
        cert_path = getattr(args, "certificate_output", None) or args.output.with_name(
            f"{args.output.stem}_cert{args.output.suffix or '.csv'}"
        )
        cert_dir = getattr(args, "certificate_dir", None) or args.output.with_name(
            f"{args.output.stem}_certfiles"
        )
        cert_fieldnames = (
            ["config", "name", "seed", "num_dihedrals", "censored"]
            + CERTIFICATE_BASE_FIELDNAMES
            + [certificate_lb_column(b) for b in betas]
        )

    # Geometry trail: per-trial best-improvement XYZs under cert_dir's sibling dir.
    geom_enabled = getattr(args, "geometry", False)
    geom_dir = None
    if geom_enabled:
        geom_dir = getattr(args, "geometry_dir", None) or args.output.with_name(
            f"{args.output.stem}_geom"
        )

    # --retry-failed: strip prior failures from both CSVs so they re-run; everything
    # successful is still skipped (it implies resume). Without it, --resume skips ALL
    # recorded trials, failures included.
    if getattr(args, "retry_failed", False):
        removed = drop_failed(args.output, traj_path, cert_path)
        print(f"--retry-failed: removed {len(removed)} failed trial(s); they will be "
              f"re-run, successful trials skipped.")

    # Build the full task list, then drop any already-done (resume).
    done = load_done_keys(args.output) if (args.resume or
                                           getattr(args, "retry_failed", False)) else set()
    if not args.output.exists():
        with open(args.output, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
    if not traj_path.exists():
        with open(traj_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=TRAJ_FIELDNAMES).writeheader()
    if cert_enabled:
        cert_path.parent.mkdir(parents=True, exist_ok=True)
        if not cert_path.exists():
            with open(cert_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=cert_fieldnames).writeheader()
        else:
            # Resume/append: a header that doesn't match the current schema (e.g. the
            # --certificate-betas grid changed) would silently misalign appended rows.
            with open(cert_path, newline="") as f:
                existing_header = next(csv.reader(f), [])
            if existing_header != cert_fieldnames:
                raise SystemExit(
                    f"{cert_path} header does not match the current certificate schema "
                    "(did --certificate-betas change?). Move it aside or pass a fresh "
                    "--certificate-output."
                )

    # Predictive skip: drop any gradient arm whose gradient-GP phase would exceed
    # this many BO steps for a molecule (its per-step cost grows steeply, so big
    # molecules run for hours and time out). Estimated from dihedral count up front
    # so we never pay the wall-clock to discover it. 0 disables.
    skip_thresh = getattr(args, "skip_grad_above_steps", 0) or 0
    budget_cache: Dict[str, Optional[int]] = {}
    skipped = []

    def predicted_skip(config: str, smiles: str, name: str) -> bool:
        if skip_thresh <= 0:
            return False
        extra = configurations[config]
        if "--use-gradients" not in extra:
            return False
        if arm_freezes_hypers(extra):  # the freeze schedule keeps it tractable
            return False
        if smiles not in budget_cache:
            budget_cache[smiles] = predict_bo_budget(smiles)
        budget = budget_cache[smiles]
        if budget is None:  # couldn't predict -> don't skip, just run it
            return False
        gsteps = arm_gradient_steps(extra, budget)
        if gsteps > skip_thresh:
            skipped.append((config, name, gsteps, budget))
            return True
        return False

    # Per-molecule budget override (e.g. the ceiling C(d)); cached per smiles.
    num_steps_cache: Dict[str, Optional[int]] = {}

    def steps_for(smiles: str) -> Optional[int]:
        if num_steps_fn is None:
            return None
        if smiles not in num_steps_cache:
            num_steps_cache[smiles] = num_steps_fn(smiles)
        return num_steps_cache[smiles]

    # Molecule-level skip (e.g. d above a cap): drop the molecule for all configs
    # and seeds before the run. Cached by name so the predicate runs once each.
    mol_skipped = []
    skip_cache: Dict[str, bool] = {}

    def mol_skip(smiles: str, name: str) -> bool:
        if mol_skip_fn is None:
            return False
        if name not in skip_cache:
            skip_cache[name] = bool(mol_skip_fn(smiles, name))
            if skip_cache[name]:
                mol_skipped.append(name)
        return skip_cache[name]

    tasks = []
    for config in configs:
        for smiles, name in mols:
            if mol_skip(smiles, name) or predicted_skip(config, smiles, name):
                continue
            for seed in seeds:
                if (config, name, str(seed)) in done:
                    continue
                tasks.append((config, smiles, name, seed, steps_for(smiles)))

    if mol_skipped:
        print(f"Skipped {len(mol_skipped)} molecule(s) via mol_skip_fn "
              f"(e.g. above the dihedral cap).")

    if skipped:
        print(f"\nPredictively skipped {len(skipped)} (config, molecule) cell(s) "
              f"with > {skip_thresh} gradient-GP steps (all seeds each):")
        for config, name, gsteps, budget in skipped:
            print(f"  SKIP {config:<16} {name:<20} "
                  f"({gsteps} grad steps of {budget}-step budget)")
        print()

    total = len(tasks)
    print(f"{total} trials to run ({len(done)} already done)"
          f"{' [DRY RUN]' if args.dry_run else ''}")
    if args.dry_run or total == 0:
        return

    write_lock = threading.Lock()
    counter = {"n": 0}

    # A non-positive --timeout disables the per-trial kill.
    timeout = args.timeout if getattr(args, "timeout", 0) and args.timeout > 0 else None

    # Failed-trial logs land in <output stem>_failed/ by default; --no-fail-logs off.
    fail_log_dir = None
    if not getattr(args, "no_fail_logs", False):
        fail_log_dir = args.fail_log_dir or args.output.with_name(
            f"{args.output.stem}_failed"
        )

    # Pin BLAS threads to 1 for bit-reproducible runs (real common-random-number
    # pairing) at the cost of speed. On by default; --no-single-thread opts out.
    env = single_thread_env() if getattr(args, "single_thread", True) else None
    if env is not None:
        print("Single-threaded (OMP/MKL/OPENBLAS=1): runs are seed-reproducible for "
              "clean paired comparison. Pass --no-single-thread for raw speed.")
    else:
        print("WARNING: --no-single-thread: multi-threaded BLAS is non-deterministic; "
              "the paired comparison carries a ~0.02 eV run-to-run noise floor.")

    def submit(task):
        config, smiles, name, seed, num_steps = task
        return run_one(
            config, configurations[config], smiles, name, seed,
            args.energy, args.optimizer, timeout=timeout, fail_log_dir=fail_log_dir,
            env=env, num_steps=num_steps,
            cert_dir=cert_dir if cert_enabled else None, cert_betas=cert_betas,
            geom_dir=geom_dir,
        )

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(submit, t): t for t in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                row, traj_rows, cert_rows = fut.result()
            except Exception as e:  # keep the sweep alive on a single failure
                config, smiles, name, seed, _ = task
                row = {f: "" for f in FIELDNAMES}
                row.update({"config": config, "name": name, "smiles": smiles,
                            "seed": seed, "success": 0})
                traj_rows = []
                cert_rows = []
                print(f"  ERROR {config}/{name}/{seed}: {e}")
            with write_lock:
                with open(args.output, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)
                if traj_rows:
                    with open(traj_path, "a", newline="") as f:
                        csv.DictWriter(f, fieldnames=TRAJ_FIELDNAMES).writerows(traj_rows)
                if cert_enabled and cert_rows:
                    with open(cert_path, "a", newline="") as f:
                        csv.DictWriter(f, fieldnames=cert_fieldnames).writerows(cert_rows)
                counter["n"] += 1
                n = counter["n"]
            e = row.get("e_e0_unconstrained", "")
            bs = row.get("best_step", "")
            tag = "" if row.get("success") else "  FAILED (crash/timeout)"
            print(f"[{n}/{total}] {row['config']:<16} {row['name']:<16} "
                  f"seed={row['seed']:<7} best_step={bs} E-E0={e}{tag}")

    prog = Path(sys.argv[0]).name
    print(f"\nDone in {time.time() - t0:.0f}s. Wrote {args.output} and {traj_path}")
    if cert_enabled:
        print(f"      certificate rows -> {cert_path}")
    if geom_enabled:
        print(f"      geometry trails  -> {geom_dir}/")
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
    # than pairing on best-of-seeds. NOTE: CRN is only exact when runs are
    # deterministic; under default multi-threaded BLAS there is a ~0.02 eV
    # run-to-run noise floor (see the module docstring), so treat per-(name, seed)
    # deltas below that as noise unless the sweep was run with --single-thread.
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

    # Stratify the per-(molecule, seed) paired gain by a molecule-level covariate and
    # report a per-config Spearman rho(gain, covariate): positive => the config helps more
    # as that covariate grows. Done for num_dihedrals (search difficulty) and, when a
    # --suitability manifest is given, for max_spec (repeat structure) -- the axis that
    # predicts the category move's win, where raw num_dihedrals is confounded (large
    # molecules are usually repeat-rich).
    from scipy.stats import spearmanr

    def _strat_rho(strat: "pd.DataFrame", col: str, mid_edges: list, positive_msg: str) -> None:
        strat = strat.dropna(subset=[col])
        if strat.empty:
            return
        edges = [0.0] + mid_edges + [float("inf")]
        strat = strat.assign(bin=pd.cut(strat[col], bins=edges, right=True))
        print(f"\n=== Paired gain vs '{baseline}', stratified by {col} ===")
        by_bin = strat.groupby(["config", "bin"], observed=True).agg(
            n=("delta", "size"),
            wins=("delta", lambda s: int((s > tol).sum())),
            losses=("delta", lambda s: int((s < -tol).sum())),
            median_gain_eV=("delta", "median"),
        )
        print(by_bin.round(4).to_string())
        print(f"\nSpearman rho(gain, {col}) per config ({positive_msg}):")
        for config, g in strat.groupby("config"):
            if g[col].nunique() < 3 or len(g) < 6:
                print(f"  {config:<24} rho=  n/a (too few points)")
                continue
            rho, p = spearmanr(g[col], g["delta"])
            print(f"  {config:<24} rho={rho:+.3f}  p={p:.3f}  (n={len(g)})")

    strat_dfs = [d.assign(config=c) for c, d in deltas_by_config.items() if not d.empty]
    strat = (
        pd.concat(strat_dfs, ignore_index=True)
        if strat_dfs else pd.DataFrame()
    )
    if not strat.empty:
        _strat_rho(strat, "num_dihedrals",
                   [float(x) for x in args.dihedral_bins.split(",")],
                   "positive => the config helps more on larger molecules")
        # max_spec (repeat structure) from the optional suitability manifest.
        if getattr(args, "suitability", None) is not None:
            man = (
                pd.read_csv(args.suitability)
                .drop_duplicates("name").set_index("name")["max_spec"]
            )
            _strat_rho(strat.assign(max_spec=strat["name"].map(man)), "max_spec",
                       [float(x) for x in args.maxspec_bins.split(",")],
                       "positive => the config helps more where there is real repeat structure")

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


def add_run_args(parser, config_names, priors_default=PRIORS_FILE_DEFAULT,
                 gradients_default=None) -> None:
    """Shared 'run' arguments. ``gradients_default``: if not None, add a
    --use-gradients/--no-use-gradients flag with that default; when set, every arm
    gets --use-gradients appended (single-surface check enforced). Sweeps that vary
    gradients per-arm (sweep_gradient) leave it None and manage their own arms."""
    parser.add_argument("--input", required=True, type=Path, help="CSV with smiles,name")
    parser.add_argument("--output", required=True, type=Path, help="Tidy output CSV")
    parser.add_argument("--traj-output", type=Path, default=None,
                        help="Per-evaluation trajectory CSV (default: <output>_traj.csv)")
    parser.add_argument("--seeds", default="1234,12345,3141,314159,42")
    parser.add_argument("--configs", default=None,
                        help=f"Comma-separated subset of: {', '.join(config_names)}")
    parser.add_argument("--priors-file", default=priors_default)
    # Default both methods to gfnff: it is fast and, being a single surface, satisfies
    # the gradient single-surface requirement (energy == optimizer under --relax), so
    # gradients can be enabled across every arm.
    parser.add_argument("--energy", default="gfnff", choices=ENERGY_CHOICES)
    parser.add_argument("--optimizer", default="gfnff", choices=ENERGY_CHOICES)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--name-column", default="name")
    parser.add_argument("--workers", "-w", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800.0,
                        help="Per-trial wall-clock limit in seconds; a trial that "
                        "exceeds it is killed and recorded as a failure (some large, "
                        "flexible molecules never converge). 0 disables (default 1800).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip (config, molecule, seed) trials already in the "
                        "output CSV, successes and failures alike")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-run only the trials previously recorded as failures: "
                        "drop their rows from the summary + trajectory CSVs, then "
                        "resume (successful trials are still skipped). Use after a "
                        "sweep where some trials timed out or crashed.")
    parser.add_argument("--fail-log-dir", type=Path, default=None,
                        help="Directory for captured logs of FAILED trials "
                        "(default: <output stem>_failed/). One file per trial; "
                        "returncode 124 means it hit --timeout.")
    parser.add_argument("--no-fail-logs", action="store_true",
                        help="Do not save logs for failed trials")
    parser.add_argument("--single-thread", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Pin BLAS/OpenMP to 1 thread per trial so each run is "
                        "bit-reproducible from its seed (DEFAULT: on). Multi-threaded "
                        "BLAS is non-deterministic and the chaotic BO search amplifies "
                        "it (~0.02 eV run-to-run), which breaks the paired (CRN) "
                        "comparison and inflates the noise floor. Pass "
                        "--no-single-thread for raw speed when reproducibility isn't "
                        "needed; recover throughput with more --workers.")
    parser.add_argument("--skip-grad-above-steps", type=int, default=0,
                        help="Predictively skip any --use-gradients arm whose "
                        "gradient-GP phase would run more than this many BO steps "
                        "for a molecule (estimated from its dihedral count and the "
                        "--auto budget), instead of grinding to a timeout. The "
                        "gradient GP's per-step cost grows steeply, so large "
                        "molecules can run for hours. A --gradient-steps N cap "
                        "counts only N, so capped arms (e.g. gradhybrid) stay in. "
                        "0 (default) disables.")
    if gradients_default is not None:
        parser.add_argument("--use-gradients", action=argparse.BooleanOptionalAction,
                            default=gradients_default,
                            help="Append --use-gradients to every arm (gradient-enhanced "
                            f"GP; freeze schedule by bouquet default). Default: "
                            f"{'on' if gradients_default else 'off'}. Requires "
                            "--energy == --optimizer (single surface); --no-use-gradients "
                            "runs the value-only GP everywhere.")
    parser.add_argument("--certificate", action="store_true",
                        help="Log the per-BO-step stopping-rule certificate "
                        "(mu_min/alpha_max/lb-grid + e_eval/e_best/n_calls/wall_s) "
                        "for every trial and aggregate it into a master CSV "
                        "(--certificate-output). Used by the stopping-rule "
                        "calibration benchmark.")
    parser.add_argument("--certificate-betas", default=None,
                        help="Comma-separated beta grid for the certificate lower "
                        "bound (default: bouquet's DEFAULT_CERTIFICATE_BETAS). One "
                        "lb_b<beta> column is logged per value.")
    parser.add_argument("--certificate-output", type=Path, default=None,
                        help="Master certificate CSV (default <output stem>_cert.csv).")
    parser.add_argument("--certificate-dir", type=Path, default=None,
                        help="Directory for per-trial certificate CSVs "
                        "(default <output stem>_certfiles/).")
    parser.add_argument("--geometry", action="store_true",
                        help="Also save, per trial, the geometry at each best-so-far "
                        "improvement plus the final relaxed best (multi-frame XYZ "
                        "under --geometry-dir), for the RMSD-identity / "
                        "distinct-conformer analysis.")
    parser.add_argument("--geometry-dir", type=Path, default=None,
                        help="Directory for per-trial geometry XYZs "
                        "(default <output stem>_geom/).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and trial count, then exit")


def accept_multi_input(parser) -> None:
    """Let the single-file ``input`` positional added by ``add_analyze_args`` /
    ``add_traj_args`` accept many files, so an ``analyze <stem>_s*.csv`` glob over a
    distributed SLURM array's per-(seed, arm) output works. Pair with
    ``concat_sweep_csvs`` in the subcommand handler."""
    for action in parser._actions:
        if action.dest == "input":
            action.nargs = "+"
            action.help = "per-(seed, arm) sweep CSV(s); concatenated"


def concat_sweep_csvs(paths, drop_traj: bool = True) -> Path:
    """Concatenate the per-(seed, arm) CSVs a distributed array writes into one tidy CSV
    (``analyze``/``trajectory`` each read a single file). ``drop_traj`` filters out
    ``*_traj.csv`` so an ``analyze <stem>_s*.csv`` glob that also matched the trajectory
    files still works."""
    import tempfile

    import pandas as pd

    files = [p for p in paths if not (drop_traj and str(p).endswith("_traj.csv"))]
    if not files:
        sys.exit("no input CSVs (after dropping _traj files)")
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    tmp = Path(tmp_name)
    df.to_csv(tmp, index=False)
    return tmp


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
    parser.add_argument("--suitability", type=Path, default=None,
                        help="Optional manifest CSV (name,max_spec,...) from "
                        "scripts/cat_suitability.py. When given, the paired gain is also "
                        "stratified by max_spec (repeat structure) with a parallel "
                        "Spearman rho(gain, max_spec) per config -- the axis that "
                        "predicts the category move's win (vs raw num_dihedrals).")
    parser.add_argument("--maxspec-bins", default="4,7",
                        help="Comma-separated upper edges for max_spec strata (with "
                        "--suitability); '4,7' => (0,4], (4,7], (7,inf) (default '4,7')")
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


def run_sweep_cli(
    config_names: List[str],
    build_configurations: Callable[[], Dict[str, List[str]]],
    baseline_label: str,
    description: Optional[str] = None,
    single_surface: bool = False,
) -> None:
    """Full run/analyze/traj command-line front-end for a sweep script.

    The run/analyze/traj subcommand wiring is identical across the sweep scripts, so it
    lives here: a script collapses to just its ``CONFIG_NAMES``, ``build_configurations``
    (arm -> extra CLI args), and baseline label. ``single_surface=True`` runs the
    energy==optimizer check before the sweep (needed by gradient arms). The distributed
    SLURM array writes one CSV per (seed, arm); ``analyze``/``traj`` accept the glob and
    concatenate (dropping ``*_traj.csv`` for ``analyze``). Usage::

        if __name__ == "__main__":
            run_sweep_cli(CONFIG_NAMES, build_configurations, BASELINE_LABEL,
                          description=__doc__, single_surface=True)
    """
    def _run(args: argparse.Namespace) -> None:
        if single_surface:
            require_single_surface(args.energy, args.optimizer)
        run_sweep(args, build_configurations())

    def _analyze(args: argparse.Namespace) -> None:
        args.input = concat_sweep_csvs(args.input, drop_traj=True)
        analyze(args, baseline_label)

    def _traj(args: argparse.Namespace) -> None:
        args.input = concat_sweep_csvs(args.input, drop_traj=False)
        trajectory(args, baseline_label)

    p = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run the sweep")
    add_run_args(r, config_names)
    r.set_defaults(func=_run)

    a = sub.add_parser("analyze", help="Summarize sweep CSV(s) from the array")
    add_analyze_args(a)
    accept_multi_input(a)
    a.set_defaults(func=_analyze)

    t = sub.add_parser("traj", help="Anytime best-energy-vs-budget curves + plots")
    add_traj_args(t)
    accept_multi_input(t)
    t.set_defaults(func=_traj)

    args = p.parse_args()
    args.func(args)
