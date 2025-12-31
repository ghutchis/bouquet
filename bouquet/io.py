"""Input/Output and logging utilities for Bouquet"""

import hashlib
import json
import logging
import sys
from csv import DictWriter
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from ase import Atoms
from ase.io.xyz import simple_write_xyz


def setup_logging(out_dir: Path, logger_name: str = "bouquet") -> logging.Logger:
    """Configure logging to both file and stdout.

    Args:
        out_dir: Directory where runtime.log will be created
        logger_name: Name for the logger (default: "bouquet")

    Returns:
        Configured logger instance
    """
    handlers = [
        logging.FileHandler(out_dir / "runtime.log"),
        logging.StreamHandler(sys.stdout),
    ]

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=handlers,
    )

    return logging.getLogger(logger_name)


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
    out_dir = Path.cwd() / f"solutions/{name}-{seed}-{energy_method}-{params_hash[-6:]}"
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


def initialize_structure_log(out_dir: Path) -> Tuple[Path, Path]:
    """Initialize the structure logging files.

    Creates the CSV log file with headers and returns paths for logging.

    Args:
        out_dir: Output directory

    Returns:
        Tuple of (log_path, ensemble_path)
    """
    log_path = out_dir / "structures.csv"
    ens_path = out_dir / "ensemble.xyz"

    with log_path.open("w") as fp:
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
        with log_path.open("a") as fp:
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
