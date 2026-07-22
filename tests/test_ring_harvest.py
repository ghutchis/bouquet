"""Tests for bouquet.rings.harvest (§7 of RING_MTD_v2), focused on the stereo filter.

The filter must retain BOTH atom (tetrahedral R/S) and bond (E/Z) configuration, referenced
to the START geometry's full 3D stereo -- including stereocentres the input SMILES leaves
unspecified (spiro carbons), which the embedding concretizes and the bias could invert.
"""

from __future__ import annotations

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from bouquet.rings.harvest import (  # noqa: E402
    cluster_ring_states, rank_and_window, stereo_retained, stereo_signature,
    topology_retained,
)
from bouquet.rings.systems import perceive_ring_systems  # noqa: E402
from bouquet.setup import mol_to_ase_atoms  # noqa: E402


def _embed(smiles, seed=1):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3(); params.randomSeed = seed
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def _mirror(atoms):
    a = atoms.copy()
    p = a.get_positions(); p[:, 0] *= -1
    a.set_positions(p)
    return a


# --- atom stereo -----------------------------------------------------------------

def test_stereo_retained_for_same_geometry():
    mol = _embed("N[C@@H]1CCC2(CCNC2)C1")
    atoms = mol_to_ase_atoms(mol, 0)
    ref = stereo_signature(mol, atoms)
    assert stereo_retained(mol, atoms, ref)


def test_signature_includes_unspecified_spiro_center():
    """The SMILES specifies one centre; the 3D signature must also catch the spiro carbon."""
    mol = _embed("N[C@@H]1CCC2(CCNC2)C1")
    atom_cip, _ = stereo_signature(mol, mol_to_ase_atoms(mol, 0))
    assert len(atom_cip) >= 2  # the specified CH plus the unspecified spiro C


def test_atom_inversion_rejected():
    """Mirroring inverts every tetrahedral centre (specified AND spiro) -> reject."""
    mol = _embed("N[C@@H]1CCC2(CCNC2)C1")
    atoms = mol_to_ase_atoms(mol, 0)
    ref = stereo_signature(mol, atoms)
    assert not stereo_retained(mol, _mirror(atoms), ref)


def test_second_conformer_same_stereo_is_retained():
    """A different conformer (same molecule/stereo) passes the filter."""
    mol = Chem.AddHs(Chem.MolFromSmiles("N[C@@H]1CCC2(CCNC2)C1"))
    p = AllChem.ETKDGv3(); p.randomSeed = 7
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=5, params=p)
    AllChem.MMFFOptimizeMoleculeConfs(mol)
    ref = stereo_signature(mol, mol_to_ase_atoms(mol, cids[0]))
    # at least one other embedding shares the exact stereo assignment
    assert any(stereo_retained(mol, mol_to_ase_atoms(mol, c), ref) for c in cids[1:])


# --- bond stereo -----------------------------------------------------------------

def test_bond_ez_retained_and_flip_rejected():
    """E reference: an E geometry passes, a Z geometry (same atom order) is rejected."""
    molE = _embed("C/C=C/C")
    molZ = _embed(r"C/C=C\C")
    ref = stereo_signature(molE, mol_to_ase_atoms(molE, 0))
    _, ref_bonds = ref
    assert ref_bonds and set(ref_bonds.values()) == {"E"}
    assert stereo_retained(molE, mol_to_ase_atoms(molE, 0), ref)
    # molZ shares molE's connectivity/atom order, so its coords sit on molE's topology
    assert not stereo_retained(molE, mol_to_ase_atoms(molZ, 0), ref)


# --- topology --------------------------------------------------------------------

def test_topology_retained_and_broken():
    mol = _embed("C1CCCCC1")
    atoms = mol_to_ase_atoms(mol, 0)
    assert topology_retained(mol, atoms)
    broken = atoms.copy()
    p = broken.get_positions(); p[0] += np.array([6.0, 0.0, 0.0])  # yank an atom out
    broken.set_positions(p)
    assert not topology_retained(mol, broken)


# --- ring-state clustering + ranking ---------------------------------------------

def test_cluster_ring_states_labels_and_ranks():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC1CCCCC1"))  # methylcyclohexane -> several states
    p = AllChem.ETKDGv3(); p.randomSeed = 7
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=40, params=p)
    energies = [e for _, e in AllChem.MMFFOptimizeMoleculeConfs(mol)]
    systems = perceive_ring_systems(mol)
    entries = [(mol_to_ase_atoms(mol, c), float(energies[i])) for i, c in enumerate(cids)]

    clusters = cluster_ring_states(entries, systems, tol=0.30)
    assert len(clusters) >= 2
    sid = systems[0].id
    for atoms, energy, ring_state, size in clusters:
        assert sid in ring_state                 # per-system state label present
        assert isinstance(ring_state[sid], int)

    seeds = rank_and_window(clusters, temperature=298.15, energy_window_kcal=12.0)
    assert seeds
    assert seeds[0].rel_energy_kcal == pytest.approx(0.0, abs=1e-9)  # sorted, min first
    assert abs(sum(s.weight for s in seeds) - 1.0) < 1e-9            # Boltzmann normalised
