"""Input/Output and logging utilities for Bouquet"""

from __future__ import annotations

# ase is only needed when structures are actually written; defer it (and the xyz
# writer) so importing this module for logging setup alone stays cheap. 3.15+.
__lazy_modules__ = [
    "ase",
    "ase.io.xyz",
]

import hashlib
import json
import logging
import re
import sys
from csv import DictWriter
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Callable

from ase import Atoms
from ase.io.xyz import simple_write_xyz

from bouquet.config import KCAL_TO_EV


def setup_logging(out_dir: Path, logger_name: str = "bouquet") -> logging.Logger:
    """Configure logging to both file and stdout.

    Args:
        out_dir: Directory where runtime.log will be created
        logger_name: Name for the logger (default: "bouquet")

    Returns:
        Configured logger instance
    """
    # Configure the named "bouquet" logger directly rather than via
    # logging.basicConfig, which is a no-op after the first call: multiple
    # programmatic runs in one process would otherwise keep logging to the first
    # run's handlers/files. Replace (and close) any handlers left from a previous
    # run so this run's records go to this run's out_dir.
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(out_dir / "runtime.log")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    # Handlers live on this logger; don't also bubble to the root logger (avoids
    # duplicate lines if the root is configured elsewhere).
    logger.propagate = False

    return logger


def _slugify_name(name: str, max_length: int = 60) -> str:
    """Sanitize a display name/SMILES into a single safe path segment.

    Replaces any character outside ``[A-Za-z0-9._-]`` (including path separators)
    with ``_`` and caps the length so a long SMILES cannot blow out the path. Falls
    back to ``"mol"`` if nothing usable remains.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._")
    slug = slug[:max_length].strip("._")
    return slug or "mol"


def create_output_directory(name: str, seed: int, energy_method: str, args_dict: dict) -> Path:
    """Create the output directory for a run.

    Args:
        name: Name of the molecule/run
        seed: Random seed used
        energy_method: Energy method being used
        args_dict: Dictionary of all arguments (for hashing)

    Returns:
        Path to the created output directory
    """
    params_hash = hashlib.sha256(str(args_dict).encode()).hexdigest()
    # Sanitize the display name before putting it in a path: raw names/SMILES can
    # contain path separators and other filesystem-hostile characters (e.g.
    # r"C/C=C\C" would otherwise create nested "solutions/C/C=C\C-..." dirs). The
    # params hash keeps the directory unique, and the original name is preserved in
    # run_params.json.
    slug = _slugify_name(name)
    out_dir = Path.cwd() / f"solutions/{slug}-{seed}-{energy_method}-{params_hash[-6:]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_run_parameters(out_dir: Path, args_dict: dict) -> None:
    """Save run parameters to a JSON file.

    Args:
        out_dir: Output directory
        args_dict: Dictionary of arguments to save
    """
    with (out_dir / "run_params.json").open("w") as fp:
        json.dump(args_dict, fp)


def save_structure(out_dir: Path, atoms: Atoms, filename: str, comment: str = "") -> None:
    """Save an atomic structure to an XYZ file.

    Args:
        out_dir: Output directory
        atoms: Atoms object to save
        filename: Name of the file (e.g., "initial.xyz")
        comment: Optional comment for the XYZ file
    """
    with (out_dir / filename).open("w") as fp:
        simple_write_xyz(fp, [atoms], comment=comment)


def append_xyz_frame(path: Path, atoms: Atoms, comment: str = "") -> None:
    """Append one structure as a frame to a multi-frame XYZ file (created if absent).

    Used for the stopping-rule benchmark's geometry trail (one frame per best-so-far
    improvement, plus the final relaxed best); writing incrementally means a
    timed-out/censored run still keeps the improvements it found before the kill.
    """
    with path.open("a") as fp:
        simple_write_xyz(fp, [atoms], comment=comment)


def atoms_to_xyz_string(atoms: Atoms) -> str:
    """Convert an Atoms object to an XYZ format string.

    Args:
        atoms: Atoms object to convert

    Returns:
        XYZ format string
    """
    xyz = StringIO()
    simple_write_xyz(xyz, [atoms])
    return xyz.getvalue()


def initialize_structure_log(out_dir: Path) -> tuple[Path, Path]:
    """Initialize the structure logging files.

    Creates the CSV log file with headers and returns paths for logging.

    Args:
        out_dir: Output directory

    Returns:
        Tuple of (log_path, ensemble_path)
    """
    log_path = out_dir / "structures.csv"
    ens_path = out_dir / "ensemble.xyz"

    with log_path.open("w", newline="") as fp:
        writer = DictWriter(fp, ["time", "coords", "xyz", "energy", "ediff"])
        writer.writeheader()

    return log_path, ens_path


def create_structure_logger(
    log_path: Path, ens_path: Path, start_energy: float
) -> Callable[[any, Atoms, float], None]:
    """Create a function to log structure entries.

    Args:
        log_path: Path to the CSV log file
        ens_path: Path to the ensemble XYZ file
        start_energy: Reference energy for computing energy differences

    Returns:
        Function that takes (coords, atoms, energy) and logs the entry
    """

    def add_entry(coords, atoms: Atoms, energy: float) -> None:
        with log_path.open("a", newline="") as fp:
            writer = DictWriter(fp, ["time", "coords", "xyz", "energy", "ediff"])
            writer.writerow(
                {
                    "time": datetime.now().timestamp(),
                    "coords": coords.tolist(),
                    "xyz": atoms_to_xyz_string(atoms),
                    "energy": energy,
                    "ediff": energy - start_energy,
                }
            )
        with ens_path.open("a") as fp:
            simple_write_xyz(fp, [atoms], comment=f"\t{energy}")

    return add_entry


# Base columns for the per-step stopping-rule certificate log. Energies are in
# relative eV (the e_e0 convention); n_calls is cumulative energy evaluations. One
# ``lb_b<beta>`` column per calibration beta is appended at logger-creation time
# (see certificate_lb_column / create_certificate_logger).
CERTIFICATE_BASE_FIELDNAMES = [
    "step",       # 0-based BO step index
    "e_eval",     # relative energy evaluated this step (eV)
    "e_best",     # running best (min) relative energy after this step (eV)
    "mu_min",     # GP posterior global-min estimate min_x mu_t(x) (eV)
    "alpha_max",  # max expected improvement over the pool (eV, plain not log)
    "n_calls",    # cumulative energy evaluations (start + init + BO so far)
    "wall_s",     # seconds elapsed in the BO loop up to this step
    "t_select",   # seconds this step in GP fit/condition + acquisition (+certificate)
    "t_eval",     # seconds this step in the xTB energy evaluation + relaxation
    "t_gp_fit",   # subset of t_select: GP construction + fit/condition
    "t_acq",      # subset of t_select: acquisition build + optimize_acqf
    # ... then one lb_b<beta> column per beta (the certificate lower bound).
]


def certificate_lb_column(beta: float) -> str:
    """Column name for the certificate lower bound at confidence multiplier ``beta``."""
    return f"lb_b{beta:g}"


def create_certificate_logger(
    log_path: Path, betas: tuple[float, ...]
) -> Callable[..., None]:
    """Create a per-BO-step logger for the stopping-rule certificate.

    Writes the header to ``log_path`` (base columns + one ``lb_b<beta>`` column
    per entry in ``betas``), one row per BO step. Returned callable signature:
    ``(step, e_eval, e_best, n_calls, wall_s, cert)`` where ``cert`` is the dict
    produced by ``solver._compute_certificate`` (keys ``mu_min``/``alpha_max`` and
    ``lb``, a list of bounds aligned with ``betas``).

    Args:
        log_path: Path to the certificate CSV (overwritten with a fresh header).
        betas: Confidence multipliers whose lower bounds were logged, in the same
            order ``cert["lb"]`` is produced.

    Returns:
        Callable that appends one certificate row per step.
    """
    lb_cols = [certificate_lb_column(b) for b in betas]
    fieldnames = CERTIFICATE_BASE_FIELDNAMES + lb_cols
    with log_path.open("w", newline="") as fp:
        DictWriter(fp, fieldnames).writeheader()

    def add_cert(
        step: int,
        e_eval: float,
        e_best: float,
        n_calls: int,
        wall_s: float,
        cert: dict,
    ) -> None:
        row = {
            "step": step,
            "e_eval": e_eval,
            "e_best": e_best,
            "mu_min": cert.get("mu_min"),
            "alpha_max": cert.get("alpha_max"),
            "n_calls": n_calls,
            "wall_s": wall_s,
            "t_select": cert.get("t_select"),
            "t_eval": cert.get("t_eval"),
            "t_gp_fit": cert.get("t_gp_fit"),
            "t_acq": cert.get("t_acq"),
        }
        for col, val in zip(lb_cols, cert.get("lb", [])):
            row[col] = val
        with log_path.open("a", newline="") as fp:
            DictWriter(fp, fieldnames).writerow(row)

    return add_cert


def save_ensemble(
    out_dir: Path, ensemble: list[tuple[Atoms, float, float]]
) -> None:
    """Write the final Boltzmann ensemble to disk.

    Produces two files in ``out_dir``:
      * ``ensemble_final.xyz`` -- one frame per conformer (sorted by energy),
        each comment line carrying the relative energy (kcal/mol, measured from
        the lowest ensemble member) and the Boltzmann population; and
      * ``ensemble.csv`` -- a summary table of index, relative energy, and weight.

    Args:
        out_dir: Output directory.
        ensemble: List of ``(atoms, relative_energy_eV, weight)`` sorted by
            energy ascending (as returned by the solver). The energies are
            relative to the run start; this routine re-references them to the
            lowest member for reporting.
    """
    if not ensemble:
        return

    # Re-reference energies to the lowest member and convert eV -> kcal/mol once.
    e_min = min(e for _, e, _ in ensemble)
    rows = [
        (atoms, (energy - e_min) / KCAL_TO_EV, weight)
        for atoms, energy, weight in ensemble
    ]

    with (out_dir / "ensemble_final.xyz").open("w") as fp:
        for atoms, rel_kcal, weight in rows:
            simple_write_xyz(
                fp, [atoms], comment=f"E_rel={rel_kcal:.3f} kcal/mol  pop={weight:.3f}"
            )

    with (out_dir / "ensemble.csv").open("w", newline="") as fp:
        writer = DictWriter(fp, ["index", "rel_energy_kcal", "weight"])
        writer.writeheader()
        for i, (atoms, rel_kcal, weight) in enumerate(rows):
            writer.writerow(
                {"index": i, "rel_energy_kcal": rel_kcal, "weight": weight}
            )
