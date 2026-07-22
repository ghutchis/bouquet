"""Ring-system perception and the atom sets the ring-MTD needs.

Partitions a molecule into independent ring systems (fused/bridged rings merged; spiro
junctions optionally kept separate so the MTD explores the product of ring states), and
for each system exposes:

* ``bias_idx``  -- ring-system atoms, the coordinate set the RMSD bias acts on;
* ``dedup_idx`` -- ``bias_idx`` plus every atom bonded to a ring atom, INCLUDING hydrogens.
  The first shell is mandatory: methylcyclohexane's axial and equatorial chairs have
  identical ring-atom geometries and differ only in where the methyl points; deduplicating
  on ring atoms alone silently merges them.
* ``automorphisms`` -- graph symmetries that map ``dedup_idx`` onto itself, as index maps
  into ``dedup_idx``. These let the graph, not hand-coded chemistry, decide identity:
  cyclohexane's ring hydrogens are graph-equivalent so an automorphism merges its two
  chairs, while methylcyclohexane has none swapping methyl<->H so axial/equatorial stay
  distinct.

``ring_state_distance`` uses those automorphisms with PROPER-rotation Kabsch, so it keeps
enantiomeric ring states apart (unlike the reflection-invariant whole-molecule iRMSD used
for ensemble dedup elsewhere in bouquet).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bouquet.rings.bias import kabsch_rotation

_MAX_AUTOMORPHISMS = 1000


@dataclass(frozen=True)
class RingSystem:
    """One independent ring system and the atom sets the MTD needs for it."""

    id: int
    ring_idx: np.ndarray          # ring-system atom indices (into the full molecule)
    bias_idx: np.ndarray          # == ring_idx; the RMSD-bias coordinate set
    dedup_idx: np.ndarray         # ring_idx + first-shell neighbours (incl. H)
    automorphisms: np.ndarray     # (n_auto, len(dedup_idx)) index maps INTO dedup_idx
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

    automorphisms = _molecule_automorphisms(mol)

    systems = []
    for sid, atomset in enumerate(kept):
        ring_idx = np.array(sorted(atomset), dtype=int)
        dedup = set(atomset)
        for a in atomset:
            for nb in mol.GetAtomWithIdx(int(a)).GetNeighbors():
                dedup.add(nb.GetIdx())
        dedup_idx = np.array(sorted(dedup), dtype=int)
        auto_maps = _restrict_automorphisms(automorphisms, dedup_idx)
        systems.append(RingSystem(
            id=sid, ring_idx=ring_idx, bias_idx=ring_idx.copy(), dedup_idx=dedup_idx,
            automorphisms=auto_maps, is_spiro_member=bool(spiro_pairs[sid]),
            spiro_partners=tuple(sorted(spiro_pairs[sid])),
        ))
    return systems


def _molecule_automorphisms(mol) -> list[tuple[int, ...]]:
    """Graph automorphisms as full-molecule index permutations (ignoring 3D/chirality)."""
    matches = mol.GetSubstructMatches(mol, uniquify=False, useChirality=False,
                                      maxMatches=_MAX_AUTOMORPHISMS)
    return list(matches)


def _restrict_automorphisms(automorphisms, dedup_idx: np.ndarray) -> np.ndarray:
    """Keep permutations that map ``dedup_idx`` onto itself; return index maps into it.

    Result row ``a`` gives, for each position ``p`` in (sorted) ``dedup_idx``, the position
    the automorphism sends it to. The identity map is always present.
    """
    dedup_set = {int(i) for i in dedup_idx}
    pos = {int(a): p for p, a in enumerate(dedup_idx)}
    maps = []
    for perm in automorphisms:
        if all(perm[a] in dedup_set for a in dedup_idx):
            maps.append([pos[perm[a]] for a in dedup_idx])
    if not maps:  # perm may not cover dedup if graph asymmetric; identity always valid
        maps = [list(range(len(dedup_idx)))]
    # De-duplicate identical maps (GetSubstructMatches can repeat).
    uniq = {tuple(m) for m in maps}
    return np.array(sorted(uniq), dtype=int)


def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Proper-rotation Kabsch RMSD between two (n,3) sets (centered internally)."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    R, _ = kabsch_rotation(Pc, Qc)
    diff = Pc @ R.T - Qc
    return float(np.sqrt((diff * diff).sum() / len(P)))


def ring_state_distance(system: RingSystem, coords_a: np.ndarray,
                        coords_b: np.ndarray) -> float:
    """Automorphism-minimized ring-state RMSD between two full-molecule geometries.

    ``D = min over Aut  RMSD(A_dedup, pi(B_dedup))`` with proper-rotation Kabsch. Small
    => same ring conformer (including the same axial/equatorial substituent placement);
    large => a genuinely different ring state. Proper rotations + graph automorphisms keep
    mirror-image (enantiomeric) ring states distinct.
    """
    A = coords_a[system.dedup_idx]
    B = coords_b[system.dedup_idx]
    return min(_kabsch_rmsd(A, B[amap]) for amap in system.automorphisms)
