"""Tools for assessing the bond structure of a molecule and finding the dihedrals to move"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from ase import Atoms
from ase.io import read
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

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


def get_initial_structure(smiles: str) -> Tuple[Atoms, Chem.Mol]:
    """Generate an initial guess for a molecular structure from SMILES.

    Args:
        smiles: SMILES string

    Returns:
        Tuple of (ASE Atoms object, RDKit Mol object)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    # Add hydrogens
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates using ETKDG
    params = AllChem.ETKDGv3()
    params.randomSeed = 42  # For reproducibility
    embed_result = AllChem.EmbedMolecule(mol, params)

    if embed_result == -1:
        # Embedding failed, try with random coordinates
        logger.warning("ETKDG embedding failed, trying with random coordinates")
        AllChem.EmbedMolecule(mol, AllChem.EmbedParameters())

    if mol.GetNumConformers() == 0:
        raise ValueError(f"Could not generate 3D coordinates for: {smiles}")

    # Optimize with MMFF94
    try:
        mmff_result = AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        if mmff_result == -1:
            logger.warning("MMFF optimization failed, trying UFF")
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception as e:
        logger.warning(f"Force field optimization failed: {e}")

    atoms = mol_to_ase_atoms(mol)
    return atoms, mol


def get_initial_structure_from_file(filename: str) -> Tuple[Atoms, Chem.Mol]:
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

    elif ext in (".mol", ".sdf"):
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


def get_conformers_from_file(filename: str) -> List[Atoms]:
    """Read multiple conformers from a file.

    Args:
        filename: Path to a file containing multiple structures
                  (e.g., multi-frame XYZ or multi-molecule SDF)

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

    else:
        raise ValueError(f"Unsupported file format for conformers: {ext}")

    if len(conformers) == 0:
        raise ValueError(f"No conformers found in file: {filepath}")

    logger.info(f"Read {len(conformers)} conformers from {filepath}")
    return conformers


@dataclass()
class DihedralInfo:
    """Describes a dihedral angle within a molecule."""

    chain: Tuple[int, int, int, int] = None
    """Atoms that form the dihedral. ASE rotates the last atom when setting a dihedral angle"""
    group: Set[int] = None
    """List of atoms that should rotate along with this dihedral"""
    type: str = None

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


def get_dihedral_info(
    graph: nx.Graph, bond: Tuple[int, int], backbone_atoms: Set[int]
) -> Optional[DihedralInfo]:
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

    try:
        components = list(nx.connected_components(h))
        if len(components) != 2:
            return None
        a, b = components
    except ValueError:
        return None

    # Determine which component rotates
    if bond[1] in a:
        return DihedralInfo(chain=tuple(points), group=a, type="backbone")
    else:
        return DihedralInfo(chain=tuple(points), group=b, type="backbone")


def detect_dihedrals(mol: Chem.Mol) -> List[DihedralInfo]:
    """Detect the bonds to be treated as rotors.

    Uses the RDKit definition of rotatable bonds:
    Single bonds between two atoms that each have at least one other
    non-triple bond neighbor, and the bond is not in a ring.

    Args:
        mol: RDKit Mol object

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

    logger.info(f"Detected {len(dihedrals)} rotatable dihedral angles")
    return dihedrals
