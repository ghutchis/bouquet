"""Tests for Phase 4 dimension-scaled ETKDGv3 + MMFF94 cluster seeding."""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from bouquet.clash import build_feasibility
from bouquet.config import DEFAULT_INIT_METHOD, etkdg_seed_pool_size
from bouquet.setup import (
    _periodic_torsion_rms_deg,
    detect_dihedrals,
    get_initial_structure,
    plan_etkdg_initial_points,
)


def test_etkdg_seed_pool_size_scales_and_clamps():
    assert etkdg_seed_pool_size(1) == 12
    assert etkdg_seed_pool_size(2) == 24
    assert etkdg_seed_pool_size(6) == 72
    assert etkdg_seed_pool_size(9) == 108
    assert etkdg_seed_pool_size(15) == 128


def test_plan_etkdg_initial_points_distinct_clash_free_and_deterministic():
    smiles = "CCCC(=O)NCCc1ccccc1"
    atoms, mol = get_initial_structure(smiles, seed=3)
    dihs = detect_dihedrals(mol, drop_h_c3=True)
    assert len(dihs) >= 5

    feas = build_feasibility(atoms, mol, dihs)
    pts1 = plan_etkdg_initial_points(
        mol, atoms, dihs, max_points=5, seed=7, feasible=feas
    )
    pts2 = plan_etkdg_initial_points(
        mol, atoms, dihs, max_points=5, seed=7, feasible=feas
    )

    assert pts1.shape == (5, len(dihs))
    assert np.allclose(pts1, pts2)

    periods = np.array([d.period_deg for d in dihs], dtype=float)
    assert np.all(pts1 >= 0.0)
    assert np.all(pts1 < periods[None, :] + 1e-9)

    # Every returned seed is clash-free on start_atoms
    if feas is not None:
        assert bool(np.all(feas(pts1)))

    # Distinct from start_coords and from one another in symmetry-reduced torsion RMSD
    start = np.array([d.get_angle(atoms) for d in dihs], dtype=float) % periods
    for i in range(len(pts1)):
        assert _periodic_torsion_rms_deg(pts1[i], start, periods) >= 20.0 - 1e-6
        for j in range(i + 1, len(pts1)):
            assert _periodic_torsion_rms_deg(pts1[i], pts1[j], periods) >= 20.0 - 1e-6


def test_plan_etkdg_initial_points_small_d_and_zero_d():
    atoms, mol = get_initial_structure("CCCC", seed=1)
    dihs = detect_dihedrals(mol, drop_h_c3=True)
    assert len(dihs) == 1

    # Butane has only 3 staggered rotamers (1 start + at most 2 distinct others)
    pts = plan_etkdg_initial_points(mol, atoms, dihs, max_points=5, seed=1)
    assert pts.ndim == 2
    assert pts.shape[1] == 1
    assert 1 <= len(pts) <= 2

    # Zero-dihedral input returns empty (0, 0) array cleanly
    empty = plan_etkdg_initial_points(mol, atoms, [], max_points=5, seed=1)
    assert empty.shape == (0, 0)


def test_cli_init_method_default_and_choices():
    from bouquet.cli import build_parser

    assert DEFAULT_INIT_METHOD == "etkdg"
    parser = build_parser()
    assert parser.parse_args(["--smiles", "CCCC"]).init_method == "etkdg"
    assert (
        parser.parse_args(["--smiles", "CCCC", "--init-method", "random"]).init_method
        == "random"
    )
    assert (
        parser.parse_args(["--smiles", "CCCC", "--init-method", "peaks"]).init_method
        == "peaks"
    )
