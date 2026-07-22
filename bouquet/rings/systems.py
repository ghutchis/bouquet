"""Ring-system perception and the atom sets the ring-MTD needs.

Partitions a molecule into independent ring systems (fused/bridged rings merged; spiro
junctions optionally kept separate so the MTD explores the product of ring states), and
for each system exposes:

* ``bias_idx``  -- ring-system atoms, the coordinate set the RMSD bias acts on;
* ``dedup_idx`` -- ``bias_idx`` plus every atom bonded to a ring atom, INCLUDING hydrogens.
  The first shell is mandatory: methylcyclohexane's axial and equatorial chairs have
  identical ring-atom geometries and differ only in where the methyl points; deduplicating
  on ring atoms alone silently merges them.

``ring_state_distance`` is the iRMSD (rotation-, permutation-, AND reflection-invariant)
over ``dedup_idx``. Reflection invariance matters: distance geometry emits achiral
conformers as either mirror form, and a proper-rotation RMSD cannot align a conformer to
its own mirror image -- it over-splits locked systems (e.g. trans-decalin, which has an
inversion centre). It is safe here because the harvest stereo filter runs BEFORE any
dedup, so every compared structure already shares the start's stereochemistry -- no two of
them are mirror images -- while the first-shell atoms keep axial vs equatorial distinct.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from bouquet.ensemble import _HAVE_IRMSD, _rmsd
from bouquet.rings.bias import kabsch_rotation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RingSystem:
    """One independent ring system and the atom sets the MTD needs for it."""

    id: int
    ring_idx: np.ndarray          # ring-system atom indices (into the full molecule)
    bias_idx: np.ndarray          # == ring_idx; the RMSD-bias coordinate set
    dedup_idx: np.ndarray         # ring_idx + first-shell neighbours (incl. H)
    dedup_numbers: np.ndarray     # atomic numbers of dedup_idx atoms (for iRMSD)
    is_spiro_member: bool
    spiro_partners: tuple[int, ...]


def _fuse(rings: list[frozenset[int]], spiro_split: bool) -> list[set[int]]:
    """Union-find over rings: merge two rings sharing >=2 atoms (fused/bridged). With
    ``spiro_split`` False also merge rings sharing exactly 1 atom (spiro)."""
    parent = list(range(len(rings)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    thresh = 1 if spiro_split else 0
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if len(rings[i] & rings[j]) > thresh:
                union(i, j)

    comps: dict[int, set[int]] = {}
    for i in range(len(rings)):
        comps.setdefault(find(i), set()).update(rings[i])
    return list(comps.values())


def perceive_ring_systems(mol, spiro_split: bool = True,
                          skip_aromatic: bool = True) -> list[RingSystem]:
    """Partition ``mol`` (RDKit Mol, hydrogens expected) into :class:`RingSystem` objects.

    Args:
        mol: RDKit molecule. Add explicit Hs first if you want them in ``dedup_idx``.
        spiro_split: keep spiro-linked rings (sharing exactly one atom) as separate
            systems (default). The shared spiro atom then belongs to both systems.
        skip_aromatic: drop ring systems that contain no non-aromatic (puckerable) ring.

    Returns an empty list for an acyclic molecule (caller short-circuits to one ETKDG seed).
    """
    ri = mol.GetRingInfo()
    atom_rings = [frozenset(r) for r in ri.AtomRings()]
    if not atom_rings:
        return []

    aromatic_ring = [all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)
                     for r in atom_rings]
    components = _fuse(atom_rings, spiro_split)

    # Keep a component only if it has a non-aromatic ring (unless skip_aromatic is off).
    def has_puckerable_ring(atomset: set[int]) -> bool:
        return any((not aromatic_ring[k]) and atom_rings[k] <= atomset
                   for k in range(len(atom_rings)))

    kept = [c for c in components if (not skip_aromatic) or has_puckerable_ring(c)]
    kept.sort(key=lambda c: min(c))  # stable, index-ordered ids

    # Spiro partners: two kept systems sharing exactly one atom.
    spiro_pairs: dict[int, set[int]] = {i: set() for i in range(len(kept))}
    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            if len(kept[i] & kept[j]) == 1:
                spiro_pairs[i].add(j)
                spiro_pairs[j].add(i)

    systems = []
    for sid, atomset in enumerate(kept):
        ring_idx = np.array(sorted(atomset), dtype=int)
        dedup = set(atomset)
        for a in atomset:
            for nb in mol.GetAtomWithIdx(int(a)).GetNeighbors():
                dedup.add(nb.GetIdx())
        dedup_idx = np.array(sorted(dedup), dtype=int)
        dedup_numbers = np.array(
            [mol.GetAtomWithIdx(int(i)).GetAtomicNum() for i in dedup_idx], dtype=int)
        systems.append(RingSystem(
            id=sid, ring_idx=ring_idx, bias_idx=ring_idx.copy(), dedup_idx=dedup_idx,
            dedup_numbers=dedup_numbers, is_spiro_member=bool(spiro_pairs[sid]),
            spiro_partners=tuple(sorted(spiro_pairs[sid])),
        ))
    return systems


def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Proper-rotation Kabsch RMSD between two (n,3) sets (centered internally).

    Used for the cheap coarse pre-dedup in harvest (where over-splitting is harmless) and
    as the iRMSD fallback when the wheel is absent.
    """
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    R, _ = kabsch_rotation(Pc, Qc)
    diff = Pc @ R.T - Qc
    return float(np.sqrt((diff * diff).sum() / len(P)))


def ring_state_distance(system: RingSystem, coords_a: np.ndarray,
                        coords_b: np.ndarray) -> float:
    """iRMSD over the ring + first-shell atoms of two full-molecule geometries.

    Small => the same ring conformer (including the same axial/equatorial substituent
    placement); large => a genuinely different ring state. Rotation/permutation/reflection
    invariant, so two embeddings of one conformer merge even when distance geometry emitted
    them as mirror images -- see the module docstring for why reflection invariance is safe
    given the upstream stereo filter.
    """
    idx = system.dedup_idx
    if _HAVE_IRMSD:
        from ase import Atoms
        a = Atoms(numbers=system.dedup_numbers, positions=coords_a[idx])
        b = Atoms(numbers=system.dedup_numbers, positions=coords_b[idx])
        return _rmsd(a, b)  # reuse ensemble's iRMSD wrapper (same backend used elsewhere)
    logger.warning("irmsd unavailable; ring_state_distance falls back to proper-rotation "
                   "Kabsch, which over-splits mirror-image embeddings")
    return _kabsch_rmsd(coords_a[idx], coords_b[idx])
