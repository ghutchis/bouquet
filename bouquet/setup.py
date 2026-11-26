"""Tools for assessing the bond structure of a molecule and finding the dihedrals to move"""

from __future__ import annotations

# With annotations as strings, networkx/ase/rdkit are only used inside function
# bodies, so defer them until a structure is actually parsed (Python 3.15+).
__lazy_modules__ = [
    "numpy",
    "networkx",
    "ase",
    "ase.io",
    "rdkit",
    "rdkit.Chem",
]

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from ase import Atoms
from ase.io import read
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds, rdMolTransforms

if TYPE_CHECKING:
    from bouquet.calculator import Calculator

logger = logging.getLogger(__name__)


def mol_to_ase_atoms(mol: Chem.Mol, conf_id: int = -1) -> Atoms:
    """Convert an RDKit Mol to an ASE Atoms object.

    Args:
        mol: RDKit molecule with at least one conformer
        conf_id: Conformer ID to use (-1 for first/default conformer)

    Returns:
        ASE Atoms object with positions and charges set
    """
    conf = mol.GetConformer(conf_id)
    positions = conf.GetPositions()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    formal_charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]

    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.set_initial_charges(formal_charges)

    # Store total molecular charge
    total_charge = Chem.GetFormalCharge(mol)
    atoms.info["charge"] = total_charge
    # Also set as attribute for backward compatibility
    atoms.charge = total_charge

    return atoms


def apply_charge_spin(atoms: Atoms, charge: int, multiplicity: int) -> None:
    """Stamp total charge and spin on an Atoms object, in place.

    xtb's ASE calculator reads charge and the number of unpaired electrons from the
    atoms (``get_initial_charges().sum()`` and ``get_initial_magnetic_moments().sum()``),
    so charge/spin must live on the Atoms (psi4 instead takes them via its
    constructor). These arrays are preserved across ``Atoms.copy()`` and relaxation,
    so stamping the starting structure carries them into every evaluation.

    ``charge`` is applied as the sum of per-atom initial charges (lumped onto the
    first atom; only the sum matters to xtb), preserving any existing per-atom
    formal charges only when they already match. ``multiplicity`` (2S+1) maps to
    ``uhf = multiplicity - 1`` unpaired electrons, lumped likewise.
    """
    n = len(atoms)
    if n == 0:
        return
    # Charge: keep the per-atom formal charges from mol_to_ase_atoms when they sum
    # to the requested total; otherwise set a sum-correct array (lumped on atom 0).
    if abs(float(atoms.get_initial_charges().sum()) - float(charge)) > 1e-6:
        q = [0.0] * n
        q[0] = float(charge)
        atoms.set_initial_charges(q)
    atoms.info["charge"] = int(charge)
    # Spin: uhf = multiplicity - 1 unpaired electrons (0 for a singlet -> no-op).
    mag = [0.0] * n
    mag[0] = float(multiplicity - 1)
    atoms.set_initial_magnetic_moments(mag)


def default_multiplicity(atoms: Atoms, charge: int = 0) -> int:
    """Lowest-spin multiplicity consistent with the electron count.

    The total electron count is ``sum(atomic numbers) - charge``. An odd electron
    count cannot pair into a closed shell, so it defaults to a doublet (2);
    an even count defaults to a singlet (1). This is only a sensible default for
    the common case -- genuinely high-spin even-electron species (e.g. O2 triplet)
    must be requested explicitly via ``--spin``.
    """
    electrons = int(atoms.get_atomic_numbers().sum()) - int(charge)
    return 2 if electrons % 2 else 1


def get_initial_structure(smiles: str, seed: int = 42) -> tuple[Atoms, Chem.Mol]:
    """Generate an initial guess for a molecular structure from SMILES.

    Args:
        smiles: SMILES string
        seed: ETKDG random seed for the embedding. Pass the run seed so different
            runs start from different 3D embeddings -- this is the ONLY source of
            ring-pucker / non-rotatable-DOF diversity across seeds, since the BO
            loop and its initial guesses perturb rotatable dihedrals only. Default
            42 (reproducible single embedding).

    Returns:
        Tuple of (ASE Atoms object, RDKit Mol object)
    """
    # A single embedding is just the multi-embedding path with the cap pinned to
    # one (see get_initial_candidates), so delegate to keep one embed/optimize
    # implementation.
    candidates, mol = get_initial_candidates(smiles, seed=seed, max_confs=1)
    return candidates[0], mol


def count_flexible_ring_atoms(mol: Chem.Mol) -> int:
    """Count sp3 atoms that sit in a non-aromatic ring.

    These atoms carry the ring-puckering degrees of freedom that the BO loop
    never explores (it perturbs rotatable dihedrals only, never ring bonds), so
    they are the right proxy for how many distinct ETKDG embeddings are worth
    generating: a rigid or fully aromatic molecule has none, while saturated
    mono-/poly-cyclic systems have several. Aromatic / sp2 ring atoms (planar,
    no pucker) and acyclic atoms are excluded.

    Args:
        mol: RDKit molecule (hydrogens may or may not be present; only heavy
            ring atoms can match).

    Returns:
        Number of sp3 non-aromatic ring atoms.
    """
    return sum(
        1
        for atom in mol.GetAtoms()
        if atom.IsInRing()
        and not atom.GetIsAromatic()
        and atom.GetHybridization() == Chem.HybridizationType.SP3
    )


def num_initial_embeddings(mol: Chem.Mol, cap: int) -> int:
    """Number of ETKDG embeddings to try for the initial structure.

    Scales with ring flexibility -- one embedding per flexible ring atom plus a
    baseline embedding -- and is clamped to ``[1, cap]``. A ``cap`` <= 1 forces
    the single embedding used historically.

    Args:
        mol: RDKit molecule to inspect for ring flexibility.
        cap: Upper bound on the embedding count.

    Returns:
        Embedding count in ``[1, cap]``.
    """
    if cap <= 1:
        return 1
    return max(1, min(cap, 1 + count_flexible_ring_atoms(mol)))


def get_initial_candidates(
    smiles: str, seed: int = 42, max_confs: int = 32
) -> tuple[list[Atoms], Chem.Mol]:
    """Generate several ETKDG conformers as initial-structure candidates.

    The number of embeddings scales with ring flexibility (see
    :func:`num_initial_embeddings`), capped at ``max_confs``; every conformer is
    MMFF94-optimized (UFF fallback). Because the BO loop perturbs rotatable
    dihedrals only, these embeddings are the main way to sample distinct ring
    puckers, and scoring them with the run's energy calculator lets the caller
    start from the lowest-energy basin rather than a seed-dependent one.

    Args:
        smiles: SMILES string.
        seed: ETKDG random seed (the run seed, so different seeds explore
            different embeddings).
        max_confs: Upper bound on the number of conformers to generate.

    Returns:
        Tuple of (list of ASE Atoms, one per generated conformer, in conformer
        order) and the RDKit Mol carrying every embedded conformer.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    n_confs = num_initial_embeddings(mol, max_confs)

    # ETKDG embedding (seed-dependent). EmbedMultipleConfs returns the IDs of the
    # conformers it managed to embed (may be fewer than requested).
    params = AllChem.ETKDGv3()
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.randomSeed = seed
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params))

    if not conf_ids:
        logger.warning("ETKDG embedding failed, trying with random coordinates")
        fallback = AllChem.ETKDGv3()
        fallback.randomSeed = seed
        fallback.useRandomCoords = True
        conf_ids = list(
            AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=fallback)
        )

    if mol.GetNumConformers() == 0:
        raise ValueError(f"Could not generate 3D coordinates for: {smiles}")

    # Optimize each conformer with MMFF94 (UFF fallback for unparameterized atoms).
    try:
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200)
        if any(rc == -1 for rc, _ in results):
            logger.warning("MMFF optimization failed for a conformer, trying UFF")
            AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=200)
    except Exception as e:
        logger.warning(f"Force field optimization failed: {e}")

    candidates = [mol_to_ase_atoms(mol, conf_id=cid) for cid in conf_ids]
    return candidates, mol


def select_initial_structure(
    candidates: list[Atoms], calc: Calculator
) -> tuple[Atoms, float]:
    """Pick the lowest single-point-energy structure from a candidate list.

    Used to choose the starting geometry among the ETKDG embeddings from
    :func:`get_initial_candidates`. A single candidate is returned unscored.
    Scoring is a single-point energy (no relaxation), so it stays cheap relative
    to the BO loop. Candidates whose energy evaluation raises are skipped.

    Args:
        candidates: Candidate structures (at least one), e.g. from
            get_initial_candidates; charge/spin should already be stamped.
        calc: Energy calculator used for the single-point scoring.

    Returns:
        Tuple of (chosen Atoms, its energy in eV). When only one candidate is
        supplied its energy is ``nan`` (it is not evaluated).
    """
    if len(candidates) == 1:
        return candidates[0], float("nan")

    best_atoms, best_energy = None, float("inf")
    last_error: Exception | None = None
    for i, cand in enumerate(candidates):
        try:
            energy = calc.get_potential_energy(cand)
        except Exception as e:
            logger.warning(f"Energy evaluation failed for candidate {i}: {e}")
            last_error = e
            continue
        logger.info(f"Initial candidate {i}: {energy:.6f} eV")
        if energy < best_energy:
            best_atoms, best_energy = cand, energy

    if best_atoms is None:
        raise RuntimeError(
            f"Energy evaluation failed for all {len(candidates)} initial candidates"
        ) from last_error

    logger.info(
        f"Selected lowest-energy initial structure from {len(candidates)} "
        f"ETKDG embeddings ({best_energy:.6f} eV)"
    )
    return best_atoms, best_energy


def get_initial_structure_from_file(filename: str) -> tuple[Atoms, Chem.Mol]:
    """Generate an initial molecular structure from a file.

    Supported formats: .xyz, .mol, .sdf, .mol2, .pdb

    Args:
        filename: Path to a file containing the structure

    Returns:
        Tuple of (ASE Atoms object, RDKit Mol object)
    """
    filepath = Path(filename)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = filepath.suffix.lower()

    if ext == ".xyz":
        # XYZ files don't have bond info - need to determine bonds
        mol = Chem.MolFromXYZFile(str(filepath))
        if mol is None:
            raise ValueError(f"Could not read XYZ file: {filepath}")
        # Determine connectivity from geometry
        try:
            rdDetermineBonds.DetermineBonds(mol)
        except Exception as e:
            logger.warning(f"Bond determination failed: {e}. Trying with charge=0")
            rdDetermineBonds.DetermineBonds(mol, charge=0)

    elif ext in (".mol", ".sdf", ".mdl"):
        mol = Chem.MolFromMolFile(str(filepath), removeHs=False, sanitize=True)
        if mol is None:
            # Try without sanitization
            mol = Chem.MolFromMolFile(str(filepath), removeHs=False, sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except Exception as e:
                    logger.warning(f"Sanitization failed: {e}")

    elif ext == ".mol2":
        mol = Chem.MolFromMol2File(str(filepath), removeHs=False, sanitize=True)
        if mol is None:
            mol = Chem.MolFromMol2File(str(filepath), removeHs=False, sanitize=False)

    elif ext == ".pdb":
        mol = Chem.MolFromPDBFile(str(filepath), removeHs=False, sanitize=True)
        if mol is None:
            mol = Chem.MolFromPDBFile(str(filepath), removeHs=False, sanitize=False)

    else:
        # todo optionally try using pybel
        raise ValueError(
            f"Unsupported file format: {ext}. "
            f"Supported formats: .xyz, .mol, .sdf, .mol2, .pdb"
        )

    if mol is None:
        raise ValueError(f"Could not read molecule from file: {filepath}")

    # Add hydrogens if not present
    if not any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        logger.info("No hydrogens found, adding hydrogens")
        mol = Chem.AddHs(mol, addCoords=True)

    # Ensure we have 3D coordinates
    if mol.GetNumConformers() == 0:
        logger.info("No conformer found in file, generating 3D coordinates")
        embed_result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if embed_result == -1:
            # Fallback to random coordinates
            AllChem.EmbedMolecule(mol, AllChem.EmbedParameters())
        if mol.GetNumConformers() == 0:
            raise ValueError(f"Could not generate 3D coordinates for molecule from: {filepath}")
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            pass  # Optimization failure is non-fatal

    atoms = mol_to_ase_atoms(mol)
    return atoms, mol


def get_conformers_from_file(
    filename: str, max_conformers: int = 0, select_rule: str = "energy"
) -> list[Atoms]:
    """Read multiple conformers from a file.

    Args:
        filename: Path to a file containing multiple structures
                  (e.g., multi-frame XYZ or multi-molecule SDF)
        max_conformers: Cap on how many conformers to return (0 = all). Only the
                  ring-seed ``.json`` format selects intelligently under a cap -- the
                  other formats have no ranking to select on and are simply truncated.
        select_rule: Selection rule for ring-seed files under ``max_conformers``; see
                  :func:`bouquet.rings.select.select_seeds`.

    Returns:
        List of Atoms objects, one for each conformer
    """
    filepath = Path(filename)

    if not filepath.exists():
        raise FileNotFoundError(f"Conformer file not found: {filepath}")

    ext = filepath.suffix.lower()
    conformers = []

    if ext == ".xyz":
        # Use ASE's read function for multi-frame XYZ
        conformers = read(str(filepath), index=":", format="xyz")
        # XYZ files don't contain charge info, default to neutral
        for atoms in conformers:
            atoms.charge = 0
            atoms.info["charge"] = 0
            atoms.set_initial_charges([0] * len(atoms))

    elif ext == ".sdf":
        # SDF files can contain multiple molecules/conformers
        suppl = Chem.SDMolSupplier(str(filepath), removeHs=False, sanitize=True)
        for mol in suppl:
            if mol is not None:
                # Handle case where mol has no conformer
                if mol.GetNumConformers() == 0:
                    logger.warning("Molecule in SDF has no conformer, skipping")
                    continue
                conformers.append(mol_to_ase_atoms(mol))

    elif ext in (".mol2", ".pdb", ".mol", ".mdl"):
        # in principle, these could contain multiple conformers
        # For now, just read the first molecule
        atoms, _ = get_initial_structure_from_file(filename)
        conformers = [atoms]

    elif ext == ".json":
        # Ring-MTD seed file (ring_seeds schema). The seed geometries are in
        # AddHs(MolFromSmiles(smiles)) order -- the same order bouquet builds its molecule in
        # -- so they line up atom-for-atom. Charge/spin are (re)stamped by the caller,
        # matching the other branches.
        #
        # CAUTION: conformers loaded here do NOT carry their ring pucker into the search.
        # `solver._evaluate_point` reads each conformer's DIHEDRAL ANGLES and applies them to
        # the single `state.start_atoms` geometry, so only torsions transfer; the geometries
        # themselves reach `ensemble_seed_geometries`, which feeds the returned ensemble and
        # only under --ensemble. To actually start a search FROM a ring state, make it the
        # start structure (`--file`). See RING_MTD_BENCH §9.17.
        #
        # The file holds every ring state the metadynamics found (up to ~100 on a spiro
        # product) and bouquet runs one BO per conformer, so a cap is applied HERE rather
        # than when the file is written: the file stays the complete, energy-ranked record
        # that any selection rule can be re-scored against offline.
        from bouquet.rings.seeds import read_seeds
        seeds = read_seeds(str(filepath))
        conformers = (seeds.select(max_conformers, select_rule) if max_conformers
                      else seeds.conformers())
        for atoms in conformers:
            atoms.charge = 0
            atoms.info["charge"] = 0
            atoms.set_initial_charges([0] * len(atoms))

    else:
        raise ValueError(f"Unsupported file format for conformers: {ext}")

    # Formats without a ranking get a plain truncation; the ring-seed branch above has
    # already applied its rule and is left alone.
    if max_conformers and ext != ".json" and len(conformers) > max_conformers:
        logger.info(
            "Truncating %d conformers from %s to %d", len(conformers), filepath, max_conformers
        )
        conformers = conformers[:max_conformers]

    if len(conformers) == 0:
        raise ValueError(f"No conformers found in file: {filepath}")

    logger.info(f"Read {len(conformers)} conformers from {filepath}")
    return conformers


@dataclass(slots=True)
class DihedralInfo:
    """Describes a dihedral angle within a molecule."""

    chain: tuple[int, int, int, int] = None
    """Atoms that form the dihedral. ASE rotates the last atom when setting a dihedral angle"""
    group: set[int] = None
    """List of atoms that should rotate along with this dihedral"""
    type: str = None
    """Optional classification of the dihedral (e.g., 'backbone', 'sidechain')"""
    prior_type: str | int | None = None
    """Prior type ID for PiBO (assigned by SMARTS matching)"""
    correlated_with: int | None = None
    """Index of correlated dihedral (for bivariate priors)"""
    symmetry_order: int = 1
    """Exact local rotational symmetry order m (period = 360 / m degrees).

    1 for an asymmetric rotor (360 deg), 2 for a rigid planar C2 group (phenyl,
    para-Cl/F/CN-phenyl, -NO2, -CO2-; 180 deg), 3 for a tetrahedral C3 terminal
    top (-CH3, -CF3, -NH3+; 120 deg), or lcm(m_u, m_v) when both sides are
    symmetric (e.g. 6 for toluene / N-methylpyridinium; 60 deg).
    """

    @property
    def period_deg(self) -> float:
        """Irreducible angular period in degrees (360 / symmetry_order)."""
        return 360.0 / self.symmetry_order

    def get_angle(self, atoms: Atoms) -> float:
        """Get the value of the specified dihedral angle.

        Args:
            atoms: Structure to assess

        Returns:
            Dihedral angle in degrees
        """
        return atoms.get_dihedral(*self.chain)


def get_bonding_graph(mol: Chem.Mol) -> nx.Graph:
    """Generate a bonding graph from an RDKit molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        NetworkX Graph describing the connectivity
    """
    g = nx.Graph()

    # Add nodes with atomic number
    g.add_nodes_from(
        [(i, dict(z=atom.GetAtomicNum())) for i, atom in enumerate(mol.GetAtoms())]
    )

    # Add edges with bond properties
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        is_single = bond.GetBondType() == Chem.BondType.SINGLE
        is_in_ring = bond.IsInRing()
        g.add_edge(
            i,
            j,
            data={
                "rotor": is_single and not is_in_ring,
                "ring": is_in_ring,
            },
        )

    return g


def geometry_bond_set(atoms: Atoms, tol: float = 1.3) -> set[tuple[int, int]]:
    """Perceive bonds from a 3D geometry by covalent-radius distance cutoff.

    A pair (i, j) is bonded when ``dist < tol * (r_cov[i] + r_cov[j])``. Returns a
    set of sorted index pairs. ``tol`` 1.3 is a standard tolerance that flags both
    broken bonds (dissociation) and newly formed bonds (rearrangement).
    """
    import numpy as np
    from ase.data import covalent_radii

    z = atoms.get_atomic_numbers()
    pos = atoms.get_positions()
    rcov = covalent_radii[z]
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    cutoff = tol * (rcov[:, None] + rcov[None, :])
    iu = np.triu_indices(len(z), k=1)
    mask = d[iu] < cutoff[iu]
    return {(int(i), int(j)) for i, j, m in zip(iu[0], iu[1], mask, strict=True) if m}


def mol_bond_set(mol: Chem.Mol) -> set[tuple[int, int]]:
    """Expected bond set (sorted index pairs) from an RDKit mol's bond table. The
    mol must have explicit Hs and atom order matching the geometry being checked
    (e.g. both from ``AddHs(MolFromSmiles(...))``)."""
    return {
        tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
        for b in mol.GetBonds()
    }


def bonds_broken(atoms: Atoms, required_bonds: set[tuple[int, int]],
                 tol: float = 1.3) -> bool:
    """True if any of ``required_bonds`` is no longer present in the geometry (the
    structure dissociated or rearranged). Only *broken* bonds count -- newly *close*
    non-bonded contacts (folded or strained/caged conformers, e.g. bicyclo[1.1.1]
    bridgeheads) are legitimate conformations, not connectivity changes, so they are
    deliberately ignored. This is the "retain the molecule's bonds" criterion."""
    return bool(set(required_bonds) - geometry_bond_set(atoms, tol))


# Bond-*formation* tolerance for the species guard: a non-bonded contact counts as
# a newly formed bond only inside this multiple of the summed covalent radii. Much
# stricter than the 1.3 breaking tolerance -- see species_changed.
BOND_FORMATION_TOL = 1.0


def bonds_formed(atoms: Atoms, reference_bonds: set[tuple[int, int]],
                 tol: float = BOND_FORMATION_TOL) -> bool:
    """True if the geometry has a covalent contact absent from ``reference_bonds``
    -- atoms drove into each other and made a new bond.

    The counterpart to :func:`bonds_broken`, and deliberately much stricter: a
    legitimately strained or folded conformer can push non-bonded atoms close
    (bicyclo[1.1.1] bridgeheads sit near 1.2x the summed covalent radii), so only
    a contact inside genuine covalent-bond range (``tol`` 1.0) counts. That
    asymmetry is what makes this safe to apply to every evaluation.

    The failure it catches is not subtle. In the d=8-12 stopbench runs the
    collapsed geometries formed 4-42 bonds at once with *zero* broken, at
    contacts of 0.9-1.2 A between heavy atoms -- one relaxation had driven atoms
    straight through each other. :func:`bonds_broken` is blind to that by
    construction, which is why those runs were never rejected and instead became
    absorbing states that consumed the rest of the budget.

    ``reference_bonds`` should be perceived at the *looser* breaking tolerance
    (as :func:`geometry_bond_set` defaults to). A pair that is already close in
    the reference -- e.g. a 1-3 Cl...Cl contact perceived as bonded -- is then in
    the reference set and cannot be reported as newly formed.
    """
    return bool(geometry_bond_set(atoms, tol) - set(reference_bonds))


def species_changed(atoms: Atoms, required_bonds: set[tuple[int, int]],
                    tol_break: float = 1.3,
                    tol_form: float = BOND_FORMATION_TOL) -> bool:
    """True if the geometry is no longer the same species -- a bond broke OR formed.

    The union of :func:`bonds_broken` and :func:`bonds_formed`; see those for why
    the two directions use different tolerances.
    """
    return (bonds_broken(atoms, required_bonds, tol_break)
            or bonds_formed(atoms, required_bonds, tol_form))


def connectivity_changed(atoms: Atoms, mol: Chem.Mol, tol: float = 1.3) -> bool:
    """True if the geometry has lost any of the mol's bonds -- i.e. it dissociated
    or rearranged and is no longer the same molecule. Used to reject broken-bond
    conformers (the optimizer can rearrange odd species). Newly formed close
    contacts are ignored (see bonds_broken). ``atoms``/``mol`` share atom ordering."""
    return bonds_broken(atoms, mol_bond_set(mol), tol)


def get_dihedral_info(
    graph: nx.Graph, bond: tuple[int, int], backbone_atoms: set[int]
) -> DihedralInfo | None:
    """For a rotatable bond, get the atoms defining the dihedral and the rotating group.

    Args:
        graph: Bond graph of the molecule
        bond: Left and right indices of the bond, respectively
        backbone_atoms: List of atoms defined as part of the backbone (non-hydrogen)

    Returns:
        DihedralInfo object, or None if the bond doesn't form a valid dihedral
    """
    # Pick the atoms to use in the dihedral, starting with the left
    points = list(bond)

    # Find neighbors of the left atom (excluding the bond partner)
    choices = set(graph[bond[0]]).difference(bond)
    if not choices:
        return None

    # Prefer backbone atoms for defining the dihedral
    bb_choices = choices.intersection(backbone_atoms)
    if bb_choices:
        choices = bb_choices
    points.insert(0, min(choices))

    # Find neighbors of the right atom (excluding the bond partner)
    choices = set(graph[bond[1]]).difference(bond)
    if not choices:
        return None

    bb_choices = choices.intersection(backbone_atoms)
    if bb_choices:
        choices = bb_choices
    points.append(min(choices))

    # Get the atoms that will rotate with this bond
    # by finding connected components when the bond is removed
    h = graph.copy()
    h.remove_edge(*bond)

    components = list(nx.connected_components(h))
    if len(components) != 2:
        return None
    a, b = components

    # Determine which component rotates
    if bond[1] in a:
        return DihedralInfo(chain=tuple(points), group=a, type="backbone")
    else:
        return DihedralInfo(chain=tuple(points), group=b, type="backbone")


def _side_symmetry_order(
    mol: Chem.Mol,
    anchor_idx: int,
    other_idx: int,
    comp_atoms: set[int],
    ranks: list[int],
    rot_bonds: set[tuple[int, int]],
) -> int:
    """Exact rigid C_m rotational symmetry order of fragment ``comp_atoms``
    around the ``other_idx -> anchor_idx`` bond axis.

    Conservative by design (zero false positives): returns ``m > 1`` only when
    rotating ``comp_atoms`` by ``360 / m`` degrees is an exact symmetry of the
    3D geometry regardless of the rest of the molecule (no internal rotatable
    bonds, no non-aromatic puckerable rings, and no pyramidal lone-pair traps
    such as neutral -NH2).
    """
    anchor = mol.GetAtomWithIdx(anchor_idx)
    nbrs = [n for n in anchor.GetNeighbors() if n.GetIdx() != other_idx]
    k = len(nbrs)
    if k < 2:
        return 1

    # Internal rotatable bonds or puckerable non-aromatic rings break exact
    # invariance under a rigid rotation of this fragment.
    for i, j in rot_bonds:
        if i in comp_atoms and j in comp_atoms:
            return 1
    for idx in comp_atoms:
        at = mol.GetAtomWithIdx(idx)
        if at.IsInRing() and not at.GetIsAromatic():
            return 1

    hyb = anchor.GetHybridization()

    # Case 1: Tetrahedral sp3 terminal C3 top (-CH3, -CF3, -CCl3, -NH3+, -SiH3).
    if k == 3 and hyb == Chem.HybridizationType.SP3:
        if all(n.GetDegree() == 1 for n in nbrs):
            sigs = {(n.GetAtomicNum(), n.GetFormalCharge(), n.GetIsotope()) for n in nbrs}
            if len(sigs) == 1:
                return 3
        return 1

    # Case 2: Planar sp2 C2 rotor.
    if k == 2 and hyb == Chem.HybridizationType.SP2:
        n0, n1 = nbrs
        # (a) Terminal planar C2 group: resonance-symmetric -NO2, -CO2-, -CS2-
        # (exclude neutral N/P which pyramidalize, e.g. -NH2).
        if n0.GetDegree() == 1 and n1.GetDegree() == 1:
            if (
                n0.GetAtomicNum() == n1.GetAtomicNum()
                and n0.GetIsotope() == n1.GetIsotope()
            ):
                if ranks[n0.GetIdx()] == ranks[n1.GetIdx()]:
                    if anchor.GetAtomicNum() == 6 or (
                        anchor.GetAtomicNum() == 7 and anchor.GetFormalCharge() == 1
                    ):
                        return 2
                elif n0.GetAtomicNum() in (8, 16):
                    b0 = mol.GetBondBetweenAtoms(anchor_idx, n0.GetIdx()).GetBondType()
                    b1 = mol.GetBondBetweenAtoms(anchor_idx, n1.GetIdx()).GetBondType()
                    pair = {(b0, n0.GetFormalCharge()), (b1, n1.GetFormalCharge())}
                    if pair == {
                        (Chem.BondType.DOUBLE, 0),
                        (Chem.BondType.SINGLE, -1),
                    }:
                        return 2

        # (b) Rigid planar aromatic ring with graph-equivalent ortho neighbors
        # (phenyl, para-F/Cl/Br/CN-phenyl, 3,5-disubstituted phenyl, 4-pyridyl,
        # 9-anthryl, etc.): no chiral centers and no out-of-plane sp3 atoms.
        if anchor.GetIsAromatic() and ranks[n0.GetIdx()] == ranks[n1.GetIdx()]:
            if any(
                mol.GetAtomWithIdx(i).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
                for i in comp_atoms
            ):
                return 1
            if all(
                mol.GetAtomWithIdx(i).GetDegree() == 1
                or mol.GetAtomWithIdx(i).GetHybridization()
                in (Chem.HybridizationType.SP2, Chem.HybridizationType.SP)
                for i in comp_atoms
            ):
                return 2

    return 1


def assign_dihedral_symmetries(
    mol: Chem.Mol, dihedrals: list[DihedralInfo]
) -> None:
    """Compute and store exact ``symmetry_order`` on each :class:`DihedralInfo`."""
    if not dihedrals:
        return
    ranks = list(
        Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=True)
    )
    rot_bonds = {tuple(sorted((d.chain[1], d.chain[2]))) for d in dihedrals}
    all_atoms = set(range(mol.GetNumAtoms()))
    for d in dihedrals:
        u, v = d.chain[1], d.chain[2]
        comp_v = set(d.group)
        comp_u = all_atoms - comp_v
        mu = _side_symmetry_order(mol, u, v, comp_u, ranks, rot_bonds)
        mv = _side_symmetry_order(mol, v, u, comp_v, ranks, rot_bonds)
        d.symmetry_order = math.lcm(mu, mv)


def is_trivial_h_c3_rotor(mol: Chem.Mol, dihedral: DihedralInfo) -> bool:
    """Return True if either side of the rotatable bond is a terminal tetrahedral
    sp3 hydrogen-only C3 top (e.g. ``-CH3``, ``-NH3+``, ``-SiH3``).

    All three staggered orientations of such a top are symmetry-identical and
    separated by a low barrier, so local relaxation (``--relax``) places the
    hydrogens without needing a dedicated BO dimension. Mixed-isotope tops
    (e.g. ``-CH2D``) and heavy tops (e.g. ``-CF3``) return False.
    """
    u, v = dihedral.chain[1], dihedral.chain[2]
    for anchor_idx, other_idx in ((u, v), (v, u)):
        anchor = mol.GetAtomWithIdx(anchor_idx)
        if anchor.GetHybridization() != Chem.HybridizationType.SP3:
            continue
        nbrs = [n for n in anchor.GetNeighbors() if n.GetIdx() != other_idx]
        if (
            len(nbrs) == 3
            and all(n.GetAtomicNum() == 1 and n.GetDegree() == 1 for n in nbrs)
            and len({n.GetIsotope() for n in nbrs}) == 1
        ):
            return True
    return False


def detect_dihedrals(
    mol: Chem.Mol, *, drop_h_c3: bool = False
) -> list[DihedralInfo]:
    """Detect the bonds to be treated as rotors.

    Uses the RDKit definition of rotatable bonds:
    Single bonds between two atoms that each have at least one other
    non-triple bond neighbor, and the bond is not in a ring.

    Args:
        mol: RDKit Mol object
        drop_h_c3: If True, exclude trivial terminal H-only C3 tops
            (``-CH3``, ``-NH3+``, ``-SiH3``; see :func:`is_trivial_h_c3_rotor`)
            from the returned list, reducing BO dimensionality when local
            relaxation is enabled.

    Returns:
        List of DihedralInfo objects describing each rotatable dihedral
    """
    dihedrals = []

    # Build the bonding graph
    g = get_bonding_graph(mol)

    # Get indices of non-hydrogen atoms (backbone)
    backbone = {i for i, d in g.nodes(data=True) if d["z"] > 1}

    # SMARTS pattern for rotatable bonds
    # Matches: [not triple bond and not terminal]-[single, not in ring]-[not triple bond and not terminal]
    rot_bond_smarts = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")

    if rot_bond_smarts is None:
        logger.error("Failed to parse rotatable bond SMARTS")
        return dihedrals

    matches = mol.GetSubstructMatches(rot_bond_smarts)

    for match in matches:
        i, j = match
        info = get_dihedral_info(g, (i, j), backbone)
        if info is not None:
            dihedrals.append(info)

    assign_dihedral_symmetries(mol, dihedrals)
    n_total = len(dihedrals)
    n_dropped = 0
    if drop_h_c3:
        dihedrals = [d for d in dihedrals if not is_trivial_h_c3_rotor(mol, d)]
        n_dropped = n_total - len(dihedrals)

    n_sym = sum(1 for d in dihedrals if d.symmetry_order > 1)
    parts = []
    if n_dropped:
        parts.append(f"dropped {n_dropped} H-only C3 rotor(s) -> {len(dihedrals)} active")
    if n_sym:
        vol_cut = math.prod(d.symmetry_order for d in dihedrals)
        parts.append(f"{n_sym} symmetric; {vol_cut}x search-volume reduction")
    if parts:
        logger.info(
            f"Detected {n_total} rotatable dihedral angles ({'; '.join(parts)})"
        )
    else:
        logger.info(f"Detected {n_total} rotatable dihedral angles")
    return dihedrals


def _periodic_torsion_rms_deg(
    a: np.ndarray, b: np.ndarray, periods_deg: np.ndarray
) -> float:
    """Symmetry-reduced wrapped RMS torsion distance in degrees."""
    diff = np.abs(a - b) % periods_deg
    diff = np.minimum(diff, periods_deg - diff)
    return float(np.sqrt(np.mean(diff**2)))


def plan_etkdg_initial_points(
    mol: Chem.Mol,
    start_atoms: Atoms,
    dihedrals: list[DihedralInfo],
    max_points: int,
    seed: int = 42,
    pool_size: int | None = None,
    cluster_deg: float = 35.0,
    fallback_cluster_deg: float = 20.0,
    feasible=None,
    num_threads: int = 1,
) -> np.ndarray:
    """Generate distinct low-energy initial dihedral seeds via ETKDGv3 + MMFF94.

    Embeds a dimension-scaled pool of ETKDGv3 conformers, optimizes them with
    MMFF94 (UFF fallback), filters out any torsion vector that clashes when
    applied to ``start_atoms``, sorts feasible conformers by force-field energy,
    and greedily selects up to ``max_points`` distinct clusters separated in
    symmetry-reduced wrapped torsion RMSD from ``start_atoms`` and each other.

    If a small molecule genuinely has fewer than ``max_points`` distinct
    rotamers, returns only the unique clusters found (avoiding wasted
    evaluations on duplicate conformers); returns shape ``(0, d)`` if embedding
    fails or no distinct feasible clusters exist.
    """
    from bouquet.config import etkdg_seed_pool_size

    d_eff = len(dihedrals)
    if d_eff == 0 or max_points <= 0:
        return np.empty((0, d_eff), dtype=float)

    n_pool = pool_size if pool_size is not None else etkdg_seed_pool_size(d_eff)
    periods_deg = np.array([di.period_deg for di in dihedrals], dtype=float)
    start_coords = (
        np.array([di.get_angle(start_atoms) for di in dihedrals], dtype=float)
        % periods_deg
    )

    work = Chem.Mol(mol)
    work.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.randomSeed = int(seed) + 10007
    if hasattr(params, "numThreads"):
        params.numThreads = num_threads
    conf_ids = list(AllChem.EmbedMultipleConfs(work, numConfs=n_pool, params=params))
    if not conf_ids:
        fallback = AllChem.ETKDGv3()
        fallback.randomSeed = int(seed) + 10007
        fallback.useRandomCoords = True
        if hasattr(fallback, "numThreads"):
            fallback.numThreads = num_threads
        conf_ids = list(
            AllChem.EmbedMultipleConfs(work, numConfs=n_pool, params=fallback)
        )
    if not conf_ids:
        return np.empty((0, d_eff), dtype=float)

    try:
        res = AllChem.MMFFOptimizeMoleculeConfs(
            work, numThreads=num_threads, maxIters=200
        )
        if any(rc == -1 for rc, _ in res):
            res = AllChem.UFFOptimizeMoleculeConfs(
                work, numThreads=num_threads, maxIters=200
            )
    except Exception:
        res = AllChem.UFFOptimizeMoleculeConfs(
            work, numThreads=num_threads, maxIters=200
        )
    energies = [float(e) for _, e in res]

    raw_angles = np.empty((len(conf_ids), d_eff), dtype=float)
    chains = [di.chain for di in dihedrals]
    for idx, cid in enumerate(conf_ids):
        conf = work.GetConformer(cid)
        raw_angles[idx] = (
            np.array(
                [rdMolTransforms.GetDihedralDeg(conf, *c) for c in chains],
                dtype=float,
            )
            % periods_deg
        )

    if feasible is not None:
        ok_mask = np.asarray(feasible(raw_angles), dtype=bool)
    else:
        ok_mask = np.ones(len(conf_ids), dtype=bool)

    feas_idx = np.nonzero(ok_mask)[0]
    if feas_idx.size == 0:
        return np.empty((0, d_eff), dtype=float)
    order = feas_idx[np.argsort([energies[i] for i in feas_idx])]

    selected: list[np.ndarray] = []
    thresholds = (
        (cluster_deg, fallback_cluster_deg)
        if fallback_cluster_deg < cluster_deg
        else (cluster_deg,)
    )
    for thresh in thresholds:
        for i in order:
            if len(selected) >= max_points:
                break
            cand = raw_angles[i]
            if _periodic_torsion_rms_deg(cand, start_coords, periods_deg) < thresh:
                continue
            if all(
                _periodic_torsion_rms_deg(cand, prev, periods_deg) >= thresh
                for prev in selected
            ):
                selected.append(cand)
        if len(selected) >= max_points:
            break

    logger.info(
        f"ETKDG cluster seeding: selected {len(selected)}/{max_points} distinct "
        f"cluster(s) from {len(conf_ids)} embeddings ({int(ok_mask.sum())} clash-free)"
    )
    return np.asarray(selected, dtype=float) if selected else np.empty((0, d_eff), dtype=float)
