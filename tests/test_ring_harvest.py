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
    cluster_ring_states, coarse_prefilter, harvest, rank_and_window, stereo_retained,
    stereo_signature, topology_retained,
)
from bouquet.rings.systems import _kabsch_rmsd, perceive_ring_systems  # noqa: E402
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



# --- coarse pre-dedup: tolerance is correctness, budget is cost -------------------

def test_coarse_tol_stays_below_fine_tol():
    """The pre-filter may only discard what the classifier would certainly merge.

    Its metric (ring heavy atoms, proper-Kabsch) is strictly weaker than the classifier's
    (ring + first shell incl. H, iRMSD), so an inverted pair silently destroys ring states
    before they are classified. Regression for RING_MTD_BENCH §9.7: at the old 0.5 vs
    0.20 A the 5-ring molecules harvested ONE state from 404 frames.
    """
    from inspect import signature
    from bouquet.rings.mtd import RingMTDConfig

    cfg = RingMTDConfig()
    assert cfg.coarse_dedup_A < cfg.fine_dedup_A
    p = signature(harvest).parameters
    assert p["coarse_tol"].default < p["fine_tol"].default


def test_coarse_prefilter_tolerance_separates_close_frames():
    """A tolerance below the separation must keep both frames; above it, merge them."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 3))
    b = a.copy()
    b[0] += 0.30                       # one atom displaced -> ~0.12 A ring RMSD
    ring_idx = np.arange(6)
    sep = _kabsch_rmsd(a[ring_idx], b[ring_idx])
    assert len(coarse_prefilter([a, b], ring_idx, coarse_tol=sep * 0.5)) == 2
    assert len(coarse_prefilter([a, b], ring_idx, coarse_tol=sep * 2.0)) == 1


def test_coarse_prefilter_budget_keeps_the_spread_not_a_prefix():
    """max_coarse must farthest-point sample, so the extremes survive the budget."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(6, 3))
    ring_idx = np.arange(6)
    # 20 near-duplicates followed by two far-apart outliers: a prefix cut keeps only
    # near-duplicates, farthest-point sampling has to keep the outliers.
    frames = [base + 1e-3 * rng.normal(size=(6, 3)) for _ in range(20)]
    frames += [base * 4.0, base * -4.0]
    kept = coarse_prefilter(frames, ring_idx, coarse_tol=1e-9, max_coarse=4)
    assert len(kept) == 4
    spread = max(_kabsch_rmsd(x[ring_idx], y[ring_idx]) for x in kept for y in kept)
    allspread = max(_kabsch_rmsd(x[ring_idx], y[ring_idx]) for x in frames for y in frames)
    prefix = coarse_prefilter(frames, ring_idx, coarse_tol=1e-9)[:4]
    prefix_spread = max(_kabsch_rmsd(x[ring_idx], y[ring_idx]) for x in prefix for y in prefix)
    # Farthest-point keeps essentially the whole spread; a prefix cut keeps ~none of it.
    assert spread > 0.9 * allspread, "budget dropped the extremes"
    assert prefix_spread < 0.01 * allspread, "test setup no longer discriminates"
