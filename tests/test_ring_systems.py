"""Tests for bouquet.rings.systems (§11.2 of RING_MTD_v2).

Covers ring-system perception (fused/bridged/spiro/aromatic handling) and the
iRMSD-based ``ring_state_distance`` -- in particular the first-shell regression guard:
methylcyclohexane's axial and equatorial chairs must NOT collapse to the same state.
"""

from __future__ import annotations

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from bouquet.rings.systems import perceive_ring_systems, ring_state_distance  # noqa: E402


def _molh(smiles, seed=1):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def _coords(mol):
    return mol.GetConformer().GetPositions()


def _distinct_ring_states(smiles, n_conf=50, tol=0.30, seed=7):
    """Number of distinct ring states over an ETKDG ensemble (greedy dedup)."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, params=params)
    AllChem.MMFFOptimizeMoleculeConfs(mol)
    sysm = perceive_ring_systems(mol)[0]
    confs = [mol.GetConformer(c).GetPositions() for c in cids]
    reps = []
    for cf in confs:
        if all(ring_state_distance(sysm, cf, r) >= tol for r in reps):
            reps.append(cf)
    return len(reps), sysm, reps


# --- perception -----------------------------------------------------------------

def test_cyclohexane_one_system_six_ring_atoms():
    mol = Chem.AddHs(Chem.MolFromSmiles("C1CCCCC1"))
    systems = perceive_ring_systems(mol)
    assert len(systems) == 1
    assert len(systems[0].ring_idx) == 6
    # dedup includes the ring hydrogens (mandatory first shell).
    assert len(systems[0].dedup_idx) > 6
    # dedup carries its atomic numbers so ring_state_distance can build the iRMSD subset.
    assert len(systems[0].dedup_numbers) == len(systems[0].dedup_idx)


def test_decalin_fused_is_one_system():
    # two six-rings sharing an edge (2 atoms) -> fused -> single system, 10 ring atoms.
    mol = Chem.AddHs(Chem.MolFromSmiles("C1CCC2CCCCC2C1"))
    systems = perceive_ring_systems(mol)
    assert len(systems) == 1
    assert len(systems[0].ring_idx) == 10


def test_spiro_split_toggle():
    smi = "C1CCC2(CC1)CCCCC2"  # spiro[5.5]undecane: two rings sharing one atom
    split = perceive_ring_systems(Chem.AddHs(Chem.MolFromSmiles(smi)), spiro_split=True)
    lumped = perceive_ring_systems(Chem.AddHs(Chem.MolFromSmiles(smi)), spiro_split=False)
    assert len(split) == 2
    assert len(lumped) == 1
    # split systems know they are spiro partners of each other.
    assert split[0].is_spiro_member and split[1].is_spiro_member
    assert split[0].spiro_partners == (1,) and split[1].spiro_partners == (0,)


def test_skip_aromatic_biphenyl_has_no_bias_groups():
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1-c1ccccc1"))
    assert perceive_ring_systems(mol, skip_aromatic=True) == []
    # with skip_aromatic off, the two aromatic rings are seen as two systems.
    assert len(perceive_ring_systems(mol, skip_aromatic=False)) == 2


def test_acyclic_returns_empty():
    assert perceive_ring_systems(Chem.AddHs(Chem.MolFromSmiles("CCCCO"))) == []


# --- ring_state_distance --------------------------------------------------------

def test_distance_self_and_rigid_motion_are_zero():
    mol = _molh("C1CCCCC1")
    sysm = perceive_ring_systems(mol)[0]
    coords = _coords(mol)
    # iRMSD has a ~1e-6 A numerical alignment floor; well below any real ring difference.
    assert ring_state_distance(sysm, coords, coords) < 1e-3
    moved = coords @ np.linalg.qr(np.random.default_rng(0).standard_normal((3, 3)))[0].T
    moved += np.array([2.0, -1.0, 0.5])
    assert ring_state_distance(sysm, coords, moved) < 1e-3


def test_mirror_image_embedding_merges():
    """A conformer and its mirror image are the same ring state (reflection invariance).

    Distance geometry emits achiral conformers as either mirror form; a proper-rotation
    RMSD cannot align them and over-splits locked systems. iRMSD merges them -- the fix for
    the trans-decalin over-counting the physics gate caught."""
    mol = _molh("C1CCCCC1")
    sysm = perceive_ring_systems(mol)[0]
    coords = _coords(mol)
    mirror = coords.copy(); mirror[:, 0] *= -1
    assert ring_state_distance(sysm, coords, mirror) < 1e-3


def test_locked_ring_gives_single_state():
    """trans-decalin is conformationally locked: its ground-state embeddings must NOT
    over-split. (With proper-rotation RMSD they split into ~7; iRMSD merges them to 1.)"""
    mol = Chem.AddHs(Chem.MolFromSmiles("C1CC[C@H]2CCCC[C@H]2C1"))
    p = AllChem.ETKDGv3(); p.randomSeed = 7
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=30, params=p)
    energies = [e for _, e in AllChem.MMFFOptimizeMoleculeConfs(mol)]
    emin = min(energies)
    ground = [mol.GetConformer(c).GetPositions()
              for i, c in enumerate(cids) if energies[i] - emin < 0.5]
    sysm = perceive_ring_systems(mol)[0]
    reps = []
    for cf in ground:
        if all(ring_state_distance(sysm, cf, r) >= 0.20 for r in reps):
            reps.append(cf)
    assert len(reps) == 1


def test_first_shell_detects_substituent_placement():
    """First-shell regression guard: moving only the methyl (ring fixed) is detected.

    The methyl is a first-shell atom, not a ring atom, so a distance restricted to
    ring atoms would score this as identical -- the exact axial/equatorial merge bug."""
    mol = _molh("CC1CCCCC1")
    sysm = perceive_ring_systems(mol)[0]
    coords = _coords(mol)
    ringset = {int(i) for i in sysm.ring_idx}
    methyl_c = next(a.GetIdx() for a in mol.GetAtoms()
                    if a.GetAtomicNum() == 6 and a.GetIdx() not in ringset)
    group = [methyl_c] + [nb.GetIdx() for nb in mol.GetAtomWithIdx(methyl_c).GetNeighbors()
                          if nb.GetAtomicNum() == 1]
    ring = coords[sysm.ring_idx]
    _, _, vt = np.linalg.svd(ring - ring.mean(axis=0))
    moved = coords.copy()
    moved[group] += 2.0 * vt[-1]   # shove the methyl off the ring plane
    assert ring_state_distance(sysm, coords, moved) > 0.30


def test_methylcyclohexane_resolves_distinct_ring_states():
    """Real ETKDG conformers of methylcyclohexane resolve into >=2 well-separated ring
    states (axial/equatorial and puckers) -- they must not all collapse to one."""
    n_states, sysm, reps = _distinct_ring_states("CC1CCCCC1")
    assert n_states >= 2
    assert min(ring_state_distance(sysm, reps[0], r) for r in reps[1:]) > 0.30
