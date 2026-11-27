"""Tests for Plan 2 Phase 2 post-BO ring-exocyclic genetic grafting."""

from __future__ import annotations

import pytest

from bouquet.calc_rdkit import RDKitMMFFCalculator
from bouquet.config import DEFAULT_RING_GRAFTS, RunOptions
from bouquet.setup import (
    _kabsch_rmsd,
    detect_dihedrals,
    get_initial_candidates,
    get_initial_structure,
)
from bouquet.solver import (
    _evaluate_initial_guesses,
    _setup_initial_state,
    run_optimization,
)
from bouquet.solver_moves import _flex_ring_heavy_indices, build_ring_graft_candidates


def test_ring_grafts_option_and_cli_flag():
    from bouquet.cli import build_parser

    assert RunOptions().ring_grafts == DEFAULT_RING_GRAFTS
    assert RunOptions(ring_grafts=0).ring_grafts == 0
    with pytest.raises(ValueError, match="ring_grafts"):
        RunOptions(ring_grafts=-1)

    parser = build_parser()
    assert parser.parse_args(["--smiles", "CCCC"]).ring_grafts == DEFAULT_RING_GRAFTS
    assert parser.parse_args(["--smiles", "CCCC", "--ring-grafts", "0"]).ring_grafts == 0
    assert parser.parse_args(["--smiles", "CCCC", "--ring-grafts", "4"]).ring_grafts == 4


def test_build_ring_graft_candidates_skips_acyclic_and_single_conformer():
    # Acyclic molecule -> no flexible rings -> empty list
    cands_ac, mol_ac = get_initial_candidates("CCCC(=O)Nc1ccccc1", seed=1, max_confs=1, pool_size=24)
    dihs_ac = detect_dihedrals(mol_ac, drop_h_c3=True)
    calc_ac = RDKitMMFFCalculator(mol_ac)
    st_ac = _setup_initial_state(cands_ac[0], dihs_ac, calc_ac, calc_ac, False, None)
    assert build_ring_graft_candidates(st_ac, mol_ac, dihs_ac, max_grafts=6) == []

    # Flexible ring with only 1 conformer on mol -> no alternate ring puckers -> empty
    atoms_1, mol_1 = get_initial_structure("O[C@@H]1CCCC[C@H]1CCC(=O)Nc1ccccc1", seed=1)
    dihs_1 = detect_dihedrals(mol_1, drop_h_c3=True)
    calc_1 = RDKitMMFFCalculator(mol_1)
    st_1 = _setup_initial_state(atoms_1, dihs_1, calc_1, calc_1, False, None)
    assert build_ring_graft_candidates(st_1, mol_1, dihs_1, max_grafts=6) == []


def test_build_ring_graft_candidates_produces_distinct_alternate_rings():
    # Flexible fused/substituted ring with multiple ring puckers + exocyclic rotors
    smiles = "O[C@@H]1CCCC[C@H]1CCC(=O)Nc1ccccc1"
    cands, mol = get_initial_candidates(smiles, seed=3, max_confs=16, pool_size=64)
    dihs = detect_dihedrals(mol, drop_h_c3=True)
    calc = RDKitMMFFCalculator(mol)
    state = _setup_initial_state(cands[0], dihs, calc, calc, False, None)
    _evaluate_initial_guesses(state, dihs, calc, calc, False, 6, seed=3, initial_conformers=None)

    grafts = build_ring_graft_candidates(state, mol, dihs, max_grafts=4)
    assert 1 <= len(grafts) <= 4
    ring_idx = _flex_ring_heavy_indices(mol)
    start_ring = state.start_atoms.get_positions()[ring_idx]
    for g in grafts:
        assert len(g) == len(state.start_atoms)
        # Graft geometry was built on a genuinely alternate ring scaffold
        assert _kabsch_rmsd(g.get_positions()[ring_idx], start_ring) >= 0.05


def test_end_to_end_run_optimization_with_ring_grafts_is_monotonic():
    smiles = "O[C@@H]1CCCC[C@H]1CCC(=O)Nc1ccccc1"
    cands, mol = get_initial_candidates(smiles, seed=2, max_confs=16, pool_size=64)
    dihs = detect_dihedrals(mol, drop_h_c3=True)
    calc = RDKitMMFFCalculator(mol)

    final_off = run_optimization(
        cands[0].copy(), dihs, n_steps=4, calc=calc, relaxCalc=calc,
        init_steps=3, out_dir=None, relax=True, seed=2, mol=mol,
        opts=RunOptions(ring_grafts=0, gc_every=0),
    )
    final_on = run_optimization(
        cands[0].copy(), dihs, n_steps=4, calc=calc, relaxCalc=calc,
        init_steps=3, out_dir=None, relax=True, seed=2, mol=mol,
        opts=RunOptions(ring_grafts=4, gc_every=0),
    )
    e_off = calc.get_potential_energy(final_off)
    e_on = calc.get_potential_energy(final_on)
    assert e_on <= e_off + 1e-5
