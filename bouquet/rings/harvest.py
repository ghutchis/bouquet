"""Turn ring-MTD snapshots into a ranked, deduplicated set of ring-state seeds.

Pipeline: quench each snapshot -> reject anything that changed stereochemistry or bonding
-> cluster into distinct ring states (per ring system) -> rank by energy within a window.

The stereo filter is the load-bearing correctness check. An aggressive RMSD bias can
invert a stereocenter (spiro carbons are frequent offenders) or flip a double bond, and a
seed with inverted stereo is a DIFFERENT molecule that silently poisons whatever consumes
it. Both ATOM (tetrahedral R/S) and BOND (E/Z) configuration must be retained.

The reference is the START geometry's full 3D stereo signature, NOT the input SMILES: the
SMILES routinely leaves real stereocenters unspecified (e.g. spiro carbons), which the
embedding then concretizes -- those must be preserved too, so we perceive the reference
from 3D and require every harvested state to match it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ase import Atoms

from bouquet.ensemble import _boltzmann_weights
from bouquet.rings.systems import RingSystem, _kabsch_rmsd, ring_state_distance
from bouquet.setup import connectivity_changed

KCAL_TO_EV = 0.0433641153
StereoSignature = tuple  # (atom_idx -> "R"/"S", (i, j) -> "E"/"Z"), both dicts


# --------------------------------------------------------------------------- #
# Stereochemistry filter (atom R/S + bond E/Z)
# --------------------------------------------------------------------------- #
def stereo_signature(ref_mol, atoms: Atoms) -> StereoSignature:
    """Full 3D stereo signature perceived from ``atoms`` on ``ref_mol``'s topology.

    Returns ``(atom_cip, bond_cip)`` where ``atom_cip`` maps a stereocenter atom index to
    its CIP label ("R"/"S") and ``bond_cip`` maps a stereo double bond (sorted index pair)
    to "E"/"Z". Perceived from coordinates (``AssignStereochemistryFrom3D``) then labelled
    with the modern CIP algorithm, so it captures every real centre -- including ones the
    input SMILES left unspecified. ``ref_mol`` must carry explicit Hs (atom count matches).
    """
    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler
    from rdkit.Geometry import Point3D

    m = Chem.Mol(ref_mol)
    pos = atoms.get_positions()
    if m.GetNumAtoms() != len(pos):
        raise ValueError("ref_mol atom count does not match the geometry")
    conf = Chem.Conformer(m.GetNumAtoms())
    for i, p in enumerate(pos):
        conf.SetAtomPosition(i, Point3D(float(p[0]), float(p[1]), float(p[2])))
    m.RemoveAllConformers()
    m.AddConformer(conf, assignId=True)
    Chem.AssignStereochemistryFrom3D(m)
    rdCIPLabeler.AssignCIPLabels(m)
    atom_cip = {a.GetIdx(): a.GetProp("_CIPCode")
                for a in m.GetAtoms() if a.HasProp("_CIPCode")}
    bond_cip = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): b.GetProp("_CIPCode")
                for b in m.GetBonds() if b.HasProp("_CIPCode")}
    return atom_cip, bond_cip


def stereo_retained(ref_mol, atoms: Atoms, reference: StereoSignature) -> bool:
    """True iff every atom and bond stereocentre matches the ``reference`` signature.

    ``reference`` is the START geometry's :func:`stereo_signature`. A differing, missing, or
    newly-flattened centre all count as a change and reject the structure.
    """
    return stereo_signature(ref_mol, atoms) == reference


def topology_retained(ref_mol, atoms: Atoms, tol: float = 1.3) -> bool:
    """True iff the covalent bond graph is unchanged (no bond formed/broken).

    A no-op for a fixed-topology engine like GFN-FF, but real for GFN2 / AIMNet2 and a
    cheap backstop against an over-biased structure that dissociated or cyclised.
    """
    return not connectivity_changed(atoms, ref_mol, tol)


# --------------------------------------------------------------------------- #
# Quench
# --------------------------------------------------------------------------- #
def quench(atoms: Atoms, calc, fmax: float = 0.02, steps: int = 300):
    """Local (unconstrained) optimization; returns ``(atoms, energy_eV)`` or None on failure."""
    from ase.optimize import LBFGS
    a = atoms.copy()
    a.calc = calc
    try:
        LBFGS(a, logfile=None).run(fmax=fmax, steps=steps)
        energy = float(a.get_potential_energy())
    except Exception:
        return None
    return a, energy


# --------------------------------------------------------------------------- #
# Ring-state clustering + ranking
# --------------------------------------------------------------------------- #
@dataclass
class RingStateSeed:
    """One ranked ring-state seed for bouquet."""

    atoms: Atoms
    energy_eV: float
    ring_state: dict[int, int]      # ring-system id -> per-system state id
    cluster_size: int = 1
    rel_energy_kcal: float = 0.0
    weight: float = 0.0             # Boltzmann population (filled at ranking)


@dataclass
class HarvestResult:
    seeds: list[RingStateSeed]
    diagnostics: dict = field(default_factory=dict)


def _ring_union(systems: list[RingSystem]) -> np.ndarray:
    return np.array(sorted({int(i) for s in systems for i in s.ring_idx}), dtype=int)


def cluster_ring_states(entries, systems: list[RingSystem], tol: float):
    """Group ``(atoms, energy)`` entries into distinct ring states, labelled per system.

    Two entries share a ring state iff, for EVERY ring system, their geometries are within
    ``tol`` by :func:`ring_state_distance` (automorphism-minimized, stereo-correct). Each
    system gets its own greedy state-id list, so a seed's ``ring_state`` is
    ``{system_id: state_id}`` and the overall state is the tuple of those -- which is what
    lets a later categorical GP transfer torsion info across states.

    Returns ``[(atoms, energy, ring_state, size)]``, lowest energy per state kept.
    """
    entries = sorted(entries, key=lambda t: t[1])  # low energy first -> best rep per state
    per_system_reps: list[list[np.ndarray]] = [[] for _ in systems]

    def state_ids(pos):
        ids = {}
        for si, sysm in enumerate(systems):
            reps = per_system_reps[si]
            match = next((k for k, rc in enumerate(reps)
                          if ring_state_distance(sysm, pos, rc) < tol), None)
            if match is None:
                match = len(reps)
                reps.append(pos)
            ids[sysm.id] = match
        return ids

    clusters: dict[tuple, list] = {}
    for atoms, energy in entries:
        pos = atoms.get_positions()
        ids = state_ids(pos)
        key = tuple(sorted(ids.items()))
        if key not in clusters:
            clusters[key] = [atoms, energy, ids, 1]
        else:
            clusters[key][3] += 1  # keep the first (lowest-energy) rep, bump the count
    return [tuple(v) for v in clusters.values()]


def rank_and_window(clusters, temperature: float, energy_window_kcal: float
                    ) -> list[RingStateSeed]:
    """Rank ring-state clusters by energy, keep within a window of the min, weight them.

    The window is deliberately generous: each snapshot carries a random exocyclic torsion
    state (the MD was not optimizing those), so a good ring state paired with a bad rotamer
    can look several kcal/mol too high. bouquet relaxes the exocyclic DOF and reorders --
    tightening this discards ring states you needed.
    """
    if not clusters:
        return []
    e_min = min(c[1] for c in clusters)
    window_eV = energy_window_kcal * KCAL_TO_EV
    kept = [c for c in clusters if (c[1] - e_min) <= window_eV]
    kept.sort(key=lambda c: c[1])
    weights = _boltzmann_weights(np.array([c[1] for c in kept]), temperature)
    seeds = []
    for (atoms, energy, ids, size), w in zip(kept, weights):
        seeds.append(RingStateSeed(
            atoms=atoms, energy_eV=energy, ring_state=ids, cluster_size=size,
            rel_energy_kcal=(energy - e_min) / KCAL_TO_EV, weight=float(w)))
    return seeds


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def harvest(frames, ref_mol, systems: list[RingSystem], calc, *,
            start_signature: StereoSignature,
            coarse_tol: float = 0.5, fine_tol: float = 0.20,
            temperature: float = 298.15, energy_window_kcal: float = 12.0,
            fmax: float = 0.02, steps: int = 300, conn_tol: float = 1.3,
            topology_check: bool = False) -> HarvestResult:
    """Frames -> ranked ring-state seeds.

    Steps: coarse ring-RMSD dedup of raw frames (cheap, cuts the quench count roughly in
    half) -> quench -> stereo filter (always) + topology filter (``topology_check``, needed
    only for bond-breaking engines) -> per-system ring-state clustering at ``fine_tol`` ->
    rank within ``energy_window_kcal``. ``start_signature`` is the walker start's
    :func:`stereo_signature`; a reject rate >5% means the bias (``k_hill``) is too strong.
    """
    ring_idx = _ring_union(systems)

    # Coarse pre-dedup on ring atoms (fast proper-Kabsch) so we don't quench near-duplicates.
    coarse: list[np.ndarray] = []
    for pos in frames:
        if all(_kabsch_rmsd(pos[ring_idx], c[ring_idx]) >= coarse_tol for c in coarse):
            coarse.append(pos)

    n_stereo_rej = n_topo_rej = n_quench_fail = 0
    survivors = []
    for pos in coarse:
        cand = Atoms(numbers=ref_mol_numbers(ref_mol), positions=pos)
        q = quench(cand, calc, fmax=fmax, steps=steps)
        if q is None:
            n_quench_fail += 1
            continue
        qa = q[0]
        if not stereo_retained(ref_mol, qa, start_signature):
            n_stereo_rej += 1
            continue
        if topology_check and not topology_retained(ref_mol, qa, conn_tol):
            n_topo_rej += 1
            continue
        survivors.append((qa, q[1]))

    clusters = cluster_ring_states(survivors, systems, fine_tol)
    seeds = rank_and_window(clusters, temperature, energy_window_kcal)

    n_frames = len(frames)
    reject_rate = (n_stereo_rej / len(coarse)) if coarse else 0.0
    diagnostics = {
        "n_frames_raw": n_frames,
        "n_after_coarse_dedup": len(coarse),
        "n_quench_failed": n_quench_fail,
        "n_stereo_rejects": n_stereo_rej,
        "n_topology_rejects": n_topo_rej,
        "n_clusters": len(clusters),
        "n_within_window": len(seeds),
        "stereo_reject_rate": reject_rate,
        "stereo_reject_alarm": reject_rate > 0.05,  # k_hill likely too large
    }
    return HarvestResult(seeds=seeds, diagnostics=diagnostics)


def ref_mol_numbers(ref_mol):
    """Atomic numbers in ref_mol atom order (for rebuilding ASE Atoms from raw coords)."""
    return [a.GetAtomicNum() for a in ref_mol.GetAtoms()]
