"""Tools for assessing the bond structure of a molecule and finding the dihedrals to move"""

import logging
import os
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np
from ase import Atoms
from ase.io import read
from ase.io.xyz import read_xyz
from openbabel import openbabel as ob
from openbabel import pybel

logger = logging.getLogger(__name__)

def get_initial_structure_from_file(filename: str) -> Tuple[Atoms, pybel.Molecule]:
    """Generate an initial molecular structure from a file

    Args:
        filename: Path to a file containing the structure (including extension)
    Returns:
        Generate an Atoms object and a pybel Molecule
    """

    filepath = Path(filename)

    # Make the 3D structure
    extension = filepath.suffix.lstrip('.')
    mol = next(pybel.readfile(extension, str(filepath)))

    # Convert it to ASE
    atoms = next(read_xyz(StringIO(mol.write('xyz')), slice(None)))
    atoms.charge = mol.charge
    atoms.set_initial_charges([a.formalcharge for a in mol.atoms])

    return atoms, mol


def get_conformers_from_file(filename: str) -> List[Atoms]:
    """Read multiple conformers from a file

    Args:
        filename: Path to a file containing multiple structures (e.g., XYZ with multiple frames)
    Returns:
        List of Atoms objects, one for each conformer
    """
    conformers = []
    filepath = Path(filename)
    extension = filepath.suffix.lstrip('.').lower()

    # Make sure the file exists
    if not filepath.exists():
        raise FileNotFoundError(f"Conformer file not found: {filepath}")

    if extension.lower() == 'xyz':
        # Use ASE's read function which can handle multi-frame XYZ
        conformers = read(str(filepath), index=':', format='xyz')
        # XYZ files don't contain charge info, default to neutral
        for atoms in conformers:
            atoms.charge = 0
            atoms.set_initial_charges([0] * len(atoms))
    else:
        # For other formats, use pybel
        for mol in pybel.readfile(extension, str(filepath)):
            atoms = next(read_xyz(StringIO(mol.write('xyz')), slice(None)))
            atoms.charge = mol.charge
            atoms.set_initial_charges([a.formalcharge for a in mol.atoms])
            conformers.append(atoms)

    # Make sure we found something
    if len(conformers) == 0:
        raise ValueError(f"No conformers found in file: {filepath}")

    logger.info(f'Read {len(conformers)} conformers from {filepath}')
    return conformers


def get_initial_structure(smiles: str) -> Tuple[Atoms, pybel.Molecule]:
    """Generate an initial guess for a molecular structure

    Args:
        smiles: SMILES string
    Returns:
        Generate an Atoms object and a pybel Molecule
    """

    # Make the 3D structure
    mol = pybel.readstring("smi", smiles)
    mol.make3D()

    ff = pybel._forcefields["mmff94"]
    success = ff.Setup(mol.OBMol)
    if not success:
        ff = pybel._forcefields["uff"]
        success = ff.Setup(mol.OBMol)

    if success:
        # quick cleanup
        # TODO: make this an option to skip
        ff.ConjugateGradients(100, 1.0e-3)
        ff.FastRotorSearch(True) # permute central rotors
        ff.ConjugateGradients(100, 1.0e-4)
        # update the coordinates
        ff.GetCoordinates(mol.OBMol)

    # Convert it to ASE
    atoms = next(read_xyz(StringIO(mol.write('xyz')), slice(None)))
    atoms.charge = mol.charge
    atoms.set_initial_charges([a.formalcharge for a in mol.atoms])

    return atoms, mol


@dataclass()
class DihedralInfo:
    """Describes a dihedral angle within a molecule"""

    chain: Tuple[int, int, int, int] = None
    """Atoms that form the dihedral. ASE rotates the last atom when setting a dihedral angle"""
    group: Set[int] = None
    """List of atoms that should rotate along with this dihedral"""
    type: str = None

    def get_angle(self, atoms: Atoms) -> float:
        """Get the value of the specified dihedral angle

        Args:
            atoms: Structure to assess
        """
        return atoms.get_dihedral(*self.chain)


def detect_dihedrals(mol: pybel.Molecule) -> List[DihedralInfo]:
    """Detect the bonds to be treated as rotors.

    We use the more generous definition from RDKit:
    https://github.com/rdkit/rdkit/blob/1bf6ef3d65f5c7b06b56862b3fb9116a3839b229/rdkit/Chem/Lipinski.py#L47%3E

    It matches pairs of atoms that are connected by a single bond,
    both bonds have at least one other bond that is not a triple bond
    and they are not part of the same ring.

    Args:
        mol: Molecule to assess
    Returns:
        List of dihedral angles. Most are defined
    """
    dihedrals = []

    # Compute the bonding graph
    g = get_bonding_graph(mol)

    # Get the indices of backbond atoms
    backbone = {i for i, d in g.nodes(data=True) if d['z'] > 1}

    # Step 1: Get the bonds from a simple matching
    smarts = pybel.Smarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
    for i, j in smarts.findall(mol):
        dihedrals.append(get_dihedral_info(g, (i - 1, j - 1), backbone))
    return dihedrals


def get_bonding_graph(mol: pybel.Molecule) -> nx.Graph:
    """Generate a bonding graph from a molecule

    Args:
        mol: Molecule to be assessed
    Returns:
        Graph describing the connectivity
    """

    # Get the bonding graph
    g = nx.Graph()
    g.add_nodes_from([
        (i, dict(z=a.atomicnum))
        for i, a in enumerate(mol.atoms)
    ])
    for bond in ob.OBMolBondIter(mol.OBMol):
        g.add_edge(bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1,
                   data={"rotor": bond.IsRotor(), "ring": bond.IsInRing()})
    return g


def get_dihedral_info(graph: nx.Graph, bond: Tuple[int, int], backbone_atoms: Set[int]) -> DihedralInfo:
    """For a rotatable bond in a model, get the atoms which define the dihedral angle
    and the atoms that should rotate along with the right half of the molecule

    Args:
        graph: Bond graph of the molecule
        bond: Left and right indicies of the bond, respectively
        backbone_atoms: List of atoms defined as part of the backbone
    Returns:
        - Atom indices defining the dihedral. Last atom is the one that will be moved
          by ase's "set_dihedral" function
        - List of atoms being rotated along with set_dihedral
    """

    # Pick the atoms to use in the dihedral, starting with the left
    points = list(bond)
    choices = set(graph[bond[0]]).difference(bond)
    bb_choices = choices.intersection(backbone_atoms)
    if len(bb_choices) > 0:  # Pick a backbone if available
        choices = bb_choices
    points.insert(0, min(choices))

    # Then the right
    choices = set(graph[bond[1]]).difference(bond)
    bb_choices = choices.intersection(backbone_atoms)
    if len(bb_choices) > 0:  # Pick a backbone if available
        choices = bb_choices
    points.append(min(choices))

    # Get the points that will rotate along with the bond
    h = graph.copy()
    h.remove_edge(*bond)
    a, b = nx.connected_components(h)
    if bond[1] in a:
        return DihedralInfo(chain=points, group=a, type='backbone')
    else:
        return DihedralInfo(chain=points, group=b, type='backbone')
