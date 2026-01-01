"""Pytest configuration and shared fixtures for Bouquet tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
from ase import Atoms


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run slow tests (functional tests with actual calculations)",
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --slow is passed."""
    if config.getoption("--slow"):
        # --slow given: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def simple_atoms() -> Atoms:
    """Create a simple water molecule for testing."""
    # Water molecule (H2O)
    positions = [
        [0.0, 0.0, 0.0],      # O
        [0.96, 0.0, 0.0],     # H
        [-0.24, 0.93, 0.0],   # H
    ]
    return Atoms('OH2', positions=positions)


@pytest.fixture
def ethane_atoms() -> Atoms:
    """Create an ethane molecule (C2H6) for testing dihedrals."""
    # Ethane molecule - staggered conformation
    positions = [
        [0.0, 0.0, 0.0],         # C1
        [1.54, 0.0, 0.0],        # C2
        [-0.36, 1.03, 0.0],      # H1
        [-0.36, -0.51, 0.89],    # H2
        [-0.36, -0.51, -0.89],   # H3
        [1.90, 1.03, 0.0],       # H4
        [1.90, -0.51, 0.89],     # H5
        [1.90, -0.51, -0.89],    # H6
    ]
    return Atoms('C2H6', positions=positions)


@pytest.fixture
def butane_smiles() -> str:
    """Return SMILES string for butane (has rotatable bonds)."""
    return "CCCC"


@pytest.fixture
def sample_xyz_file(temp_dir: Path) -> Path:
    """Create a sample XYZ file for testing."""
    xyz_content = """3
Water molecule
O     0.000000     0.000000     0.000000
H     0.960000     0.000000     0.000000
H    -0.240000     0.930000     0.000000
"""
    xyz_path = temp_dir / "water.xyz"
    xyz_path.write_text(xyz_content)
    return xyz_path


@pytest.fixture
def sample_multi_xyz_file(temp_dir: Path) -> Path:
    """Create a multi-frame XYZ file for testing conformer reading."""
    xyz_content = """3
Water conformer 1
O     0.000000     0.000000     0.000000
H     0.960000     0.000000     0.000000
H    -0.240000     0.930000     0.000000
3
Water conformer 2
O     0.000000     0.000000     0.100000
H     0.970000     0.000000     0.100000
H    -0.240000     0.940000     0.100000
"""
    xyz_path = temp_dir / "water_ensemble.xyz"
    xyz_path.write_text(xyz_content)
    return xyz_path


@pytest.fixture
def pentane_xyz_file(temp_dir: Path) -> Path:
    """Create an XYZ file for pentane (multiple rotatable bonds)."""
    # n-pentane coordinates
    xyz_content = """17
n-pentane
C     0.000000     0.000000     0.000000
C     1.540000     0.000000     0.000000
C     2.080000     1.440000     0.000000
C     3.620000     1.440000     0.000000
C     4.160000     2.880000     0.000000
H    -0.360000     0.510000     0.890000
H    -0.360000     0.510000    -0.890000
H    -0.360000    -1.020000     0.000000
H     1.900000    -0.510000     0.890000
H     1.900000    -0.510000    -0.890000
H     1.720000     1.950000    -0.890000
H     1.720000     1.950000     0.890000
H     3.980000     0.930000     0.890000
H     3.980000     0.930000    -0.890000
H     3.800000     3.390000    -0.890000
H     3.800000     3.390000     0.890000
H     5.240000     2.880000     0.000000
"""
    xyz_path = temp_dir / "pentane.xyz"
    xyz_path.write_text(xyz_content)
    return xyz_path
