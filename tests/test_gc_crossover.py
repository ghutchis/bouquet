"""Tests for Phase 3 MMFF-screened graph-aware genetic subtree crossover."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from bouquet.calc_rdkit import RDKitMMFFCalculator
from bouquet.clash import build_feasibility
from bouquet.config import DEFAULT_GC_EVERY, GC_D_THRESHOLD, RunOptions
from bouquet.setup import (
    _periodic_torsion_rms_deg,
    detect_dihedrals,
    get_initial_structure,
)
from bouquet.solver import (
    _evaluate_initial_guesses,
    _setup_initial_state,
    run_optimization,
)
from bouquet.solver_moves import _propose_gc_crossover, build_graph_cuts


def test_build_graph_cuts_linear_and_branched():
    # Linear chain of d dihedrals has exactly d - 1 unique connected bipartitions
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC(=O)NCCc1ccccc1"))
    dihs = detect_dihedrals(mol, drop_h_c3=True)
    cuts = build_graph_cuts(mol, dihs)
    assert len(cuts) == len(dihs) - 1
    for mask in cuts:
        assert mask.shape == (len(dihs),)
        assert mask.dtype == bool
        assert 0 < int(mask.sum()) < len(dihs)

    # Star-like trisubstituted benzene (every arm branches off a central ring)
    mol_star = Chem.AddHs(Chem.MolFromSmiles("CCOc1cc(OCC)cc(OCC)c1"))
    dihs_star = detect_dihedrals(mol_star, drop_h_c3=True)
    cuts_star = build_graph_cuts(mol_star, dihs_star)
    assert len(cuts_star) >= 3

    # Single dihedral has no valid bipartition
    mol_1d = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    dihs_1d = detect_dihedrals(mol_1d, drop_h_c3=True)
    assert len(dihs_1d) == 1
    assert build_graph_cuts(mol_1d, dihs_1d) == []


def test_run_options_resolve_gc_every():
    opts = RunOptions()
    assert opts.gc_every is None
    assert opts.resolve_gc_every(GC_D_THRESHOLD - 1, has_mol=True) == 0
    assert opts.resolve_gc_every(GC_D_THRESHOLD, has_mol=True) == DEFAULT_GC_EVERY
    assert opts.resolve_gc_every(GC_D_THRESHOLD, has_mol=False) == 0
    assert opts.resolve_gc_every(1, has_mol=True) == 0

    assert RunOptions(gc_every=0).resolve_gc_every(10, has_mol=True) == 0
    assert RunOptions(gc_every=3).resolve_gc_every(3, has_mol=True) == 3
    assert RunOptions(gc_every=3).resolve_gc_every(1, has_mol=True) == 0

    with pytest.raises(ValueError, match="gc_every"):
        RunOptions(gc_every=-1)
    with pytest.raises(ValueError, match="gc_warmup"):
        RunOptions(gc_warmup=-1)


def test_propose_gc_crossover_novel_clash_free_and_deterministic():
    smiles = "CCCC(=O)NCCc1ccccc1"
    atoms, mol = get_initial_structure(smiles, seed=2)
    dihs = detect_dihedrals(mol, drop_h_c3=True)
    calc = RDKitMMFFCalculator(mol)
    state = _setup_initial_state(atoms, dihs, calc, calc, relax=False, out_dir=None)
    state.clash_feasibility = build_feasibility(atoms, mol, dihs)
    state.gc_cuts = build_graph_cuts(mol, dihs)
    state.gc_mol = mol

    # Too few distinct basins (< 3) returns None cleanly
    state.gc_rng = np.random.default_rng(123)
    assert _propose_gc_crossover(state, dihs) is None

    # Populate with diverse random observations
    _evaluate_initial_guesses(state, dihs, calc, calc, False, 10, seed=5, initial_conformers=None)

    state.gc_rng = np.random.default_rng(42)
    cand1 = _propose_gc_crossover(state, dihs)
    state.gc_rng = np.random.default_rng(42)
    cand2 = _propose_gc_crossover(state, dihs)

    assert cand1 is not None
    assert np.allclose(cand1, cand2)
    periods = np.array([d.period_deg for d in dihs], dtype=float)
    assert np.all(cand1 >= 0.0) and np.all(cand1 < periods + 1e-9)
    if state.clash_feasibility is not None:
        assert bool(state.clash_feasibility(cand1[None, :])[0])
    obs = state.observed_coords.cpu().numpy() % periods
    for row in obs:
        assert _periodic_torsion_rms_deg(cand1, row, periods) >= 15.0 - 1e-6


def test_end_to_end_run_optimization_with_gc_and_cli_flag():
    from bouquet.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["--smiles", "CCCC"]).gc_every is None
    assert parser.parse_args(["--smiles", "CCCC", "--gc-every", "0"]).gc_every == 0
    assert parser.parse_args(["--smiles", "CCCC", "--gc-every", "4"]).gc_every == 4

    atoms, mol = get_initial_structure("CCCC(=O)NCCc1ccccc1", seed=1)
    dihs = detect_dihedrals(mol, drop_h_c3=True)
    calc = RDKitMMFFCalculator(mol)
    final_atoms = run_optimization(
        atoms,
        dihs,
        n_steps=6,
        calc=calc,
        relaxCalc=calc,
        init_steps=6,
        out_dir=None,
        relax=True,
        seed=1,
        mol=mol,
        opts=RunOptions(gc_every=2, gc_warmup=1),
    )
    assert len(final_atoms) == len(atoms)
