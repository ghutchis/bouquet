# Copyright (c) 2026 Geoffrey Hutchison
# SPDX-License-Identifier: MIT
"""Fallbacks in _perform_final_relaxation when the final relaxation fails.

The final relaxation re-relaxes the best observed structure, first with the
dihedrals pinned and then released. If the calculator fails on that geometry
(e.g. xtb SCF non-convergence -> sentinel failure energy), the released
structure is garbage; the function must fall back rather than emit it. These
tests force the failure deterministically by patching the relaxation calls.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")

from bouquet import solver  # noqa: E402
from bouquet.assess import RELAX_FAILURE_ENERGY_EV  # noqa: E402
from bouquet.calc_rdkit import RDKitMMFFCalculator  # noqa: E402
from bouquet.setup import detect_dihedrals, get_initial_structure  # noqa: E402


@pytest.fixture
def populated_state():
    """A butane state with a handful of real (valid) observed evaluations."""
    atoms, mol = get_initial_structure("CCCC")
    dihedrals = detect_dihedrals(mol)
    calc = RDKitMMFFCalculator(mol)
    state = solver._setup_initial_state(atoms, dihedrals, calc, calc, False, None)
    solver._evaluate_initial_guesses(state, dihedrals, calc, calc, False, 4, 0, None, None)
    return state, dihedrals, calc


def test_final_relaxation_reverts_to_constrained(populated_state, monkeypatch):
    """If only the UNCONSTRAINED relax fails, keep the valid constrained result."""
    state, dihedrals, calc = populated_state

    # The unconstrained relax goes through solver.relax_structure; the constrained
    # one goes through solver.evaluate_energy -> assess.relax_structure, which we
    # leave real. So patching solver.relax_structure fails only the released step.
    def failing_relax(atoms, energyCalc, relaxCalc, steps):
        return RELAX_FAILURE_ENERGY_EV, atoms

    monkeypatch.setattr(solver, "relax_structure", failing_relax)

    _, rel_energy = solver._perform_final_relaxation(state, dihedrals, calc, calc)

    # Must not be the sentinel: it reverted to the (valid) constrained relaxation.
    assert rel_energy + state.start_energy < RELAX_FAILURE_ENERGY_EV


def test_final_relaxation_reverts_on_formed_bond(populated_state, monkeypatch):
    """If the UNCONSTRAINED relax changes connectivity (a formed/broken bond) and
    lands on a bogusly low energy -- a different species, e.g. ring closure -- the
    function must revert to the constrained best rather than emit the transmuted
    structure as a spurious global minimum."""
    import numpy as np
    from bouquet.setup import geometry_bond_set

    state, dihedrals, calc = populated_state

    # Patch only the unconstrained relax (solver.relax_structure): return a geometry
    # with a NEW bond (collapse the two terminal carbons together) at an absurdly low
    # energy. The constrained relax (assess.evaluate_energy) is left real and valid.
    def transmuting_relax(atoms, energyCalc, relaxCalc, steps):
        bad = atoms.copy()
        pos = bad.get_positions()
        pos[3] = pos[0] + np.array([0.5, 0.0, 0.0])  # C3 onto C0 -> forms a bond
        bad.set_positions(pos)
        return -500.0, bad  # bogusly low "energy" of the wrong species

    monkeypatch.setattr(solver, "relax_structure", transmuting_relax)

    best_atoms, rel_energy = solver._perform_final_relaxation(state, dihedrals, calc, calc)

    # Reverted: energy is the (valid, physical) constrained result, not the -500 species.
    assert rel_energy + state.start_energy > -100.0
    # And the returned geometry keeps the original connectivity (no formed bond).
    assert geometry_bond_set(best_atoms) == geometry_bond_set(state.start_atoms)


def test_final_relaxation_reverts_to_best_observed(populated_state, monkeypatch):
    """If BOTH constrained and unconstrained relaxes fail, keep the best observed."""
    state, dihedrals, calc = populated_state
    best_idx = state.observed_energies.argmin().item()
    best_observed_eV = state.observed_energies[best_idx].item()

    def failing_relax(atoms, energyCalc, relaxCalc, steps):
        return RELAX_FAILURE_ENERGY_EV, atoms

    def failing_eval(*args, **kwargs):
        # (energy, atoms) with the constrained relaxation failing too.
        atoms = args[1]
        return RELAX_FAILURE_ENERGY_EV, atoms.copy()

    monkeypatch.setattr(solver, "relax_structure", failing_relax)
    monkeypatch.setattr(solver, "evaluate_energy", failing_eval)

    _, rel_energy = solver._perform_final_relaxation(state, dihedrals, calc, calc)

    # Reverted to the lowest structure actually observed during the search.
    assert rel_energy == pytest.approx(best_observed_eV)
    assert best_observed_eV < RELAX_FAILURE_ENERGY_EV
