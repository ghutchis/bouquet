"""Unit tests for bouquet.setup module (structure and dihedral detection)."""

import pytest
from pathlib import Path

import numpy as np
from ase import Atoms

from bouquet.setup import (
    get_initial_structure,
    get_initial_structure_from_file,
    get_conformers_from_file,
    detect_dihedrals,
    get_bonding_graph,
    get_dihedral_info,
    DihedralInfo,
)


class TestGetInitialStructure:
    """Tests for SMILES to 3D structure conversion."""

    def test_simple_smiles(self):
        """Test converting simple SMILES to structure."""
        atoms, mol = get_initial_structure("C")  # Methane
        assert len(atoms) == 5  # 1 C + 4 H
        assert "C" in atoms.get_chemical_symbols()
        assert "H" in atoms.get_chemical_symbols()

    def test_returns_atoms_and_mol(self):
        """Test that function returns both Atoms and pybel Molecule."""
        atoms, mol = get_initial_structure("CC")  # Ethane
        assert isinstance(atoms, Atoms)
        assert hasattr(mol, "atoms")  # pybel molecule

    def test_butane_structure(self):
        """Test that butane generates expected structure."""
        atoms, mol = get_initial_structure("CCCC")
        # Butane: 4 C + 10 H = 14 atoms
        assert len(atoms) == 14

    def test_charge_is_set(self):
        """Test that charge attribute is set on atoms."""
        atoms, mol = get_initial_structure("C")
        assert hasattr(atoms, "charge")

    def test_aromatic_smiles(self):
        """Test aromatic molecule (benzene)."""
        atoms, mol = get_initial_structure("c1ccccc1")
        # Benzene: 6 C + 6 H = 12 atoms
        assert len(atoms) == 12

    def test_3d_coordinates(self):
        """Test that 3D coordinates are generated."""
        atoms, mol = get_initial_structure("CC")
        positions = atoms.get_positions()

        # Should have 3D coordinates (not all zeros)
        assert positions.shape[1] == 3
        # At least some coordinates should be non-zero
        assert np.any(positions != 0)


class TestGetInitialStructureFromFile:
    """Tests for reading structures from files."""

    def test_read_xyz_file(self, sample_xyz_file):
        """Test reading structure from XYZ file."""
        atoms, mol = get_initial_structure_from_file(str(sample_xyz_file))
        assert len(atoms) == 3  # Water
        assert "O" in atoms.get_chemical_symbols()

    def test_returns_atoms_and_mol(self, sample_xyz_file):
        """Test that function returns both Atoms and molecule."""
        atoms, mol = get_initial_structure_from_file(str(sample_xyz_file))
        assert isinstance(atoms, Atoms)

    def test_charge_is_set_from_file(self, sample_xyz_file):
        """Test that charge is set when reading from file."""
        atoms, mol = get_initial_structure_from_file(str(sample_xyz_file))
        assert hasattr(atoms, "charge")


class TestGetConformersFromFile:
    """Tests for reading multiple conformers from files."""

    def test_read_multi_frame_xyz(self, sample_multi_xyz_file):
        """Test reading multiple conformers from XYZ file."""
        conformers = get_conformers_from_file(str(sample_multi_xyz_file))
        assert len(conformers) == 2
        for conf in conformers:
            assert isinstance(conf, Atoms)
            assert len(conf) == 3  # Water molecules

    def test_missing_file_raises_error(self, temp_dir):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            get_conformers_from_file(str(temp_dir / "nonexistent.xyz"))

    def test_single_frame_xyz(self, sample_xyz_file):
        """Test reading single-frame XYZ returns list with one conformer."""
        conformers = get_conformers_from_file(str(sample_xyz_file))
        assert len(conformers) == 1

    def test_conformers_have_charge(self, sample_multi_xyz_file):
        """Test that conformers have charge attribute set."""
        conformers = get_conformers_from_file(str(sample_multi_xyz_file))
        for conf in conformers:
            assert hasattr(conf, "charge")


class TestDihedralInfo:
    """Tests for the DihedralInfo dataclass."""

    def test_create_dihedral_info(self):
        """Test creating DihedralInfo instance."""
        info = DihedralInfo(
            chain=(0, 1, 2, 3),
            group={2, 3, 4, 5},
            type="backbone",
        )
        assert info.chain == (0, 1, 2, 3)
        assert info.group == {2, 3, 4, 5}
        assert info.type == "backbone"

    def test_get_angle(self, ethane_atoms):
        """Test getting dihedral angle from atoms."""
        # For ethane, we can define a H-C-C-H dihedral
        info = DihedralInfo(
            chain=(2, 0, 1, 5),  # H-C-C-H
            group={1, 5, 6, 7},
            type="backbone",
        )
        angle = info.get_angle(ethane_atoms)
        assert isinstance(angle, float)
        assert -180 <= angle <= 180 or 0 <= angle <= 360


class TestDetectDihedrals:
    """Tests for dihedral angle detection."""

    def test_butane_has_dihedrals(self):
        """Test that butane has detectable rotatable bonds."""
        atoms, mol = get_initial_structure("CCCC")
        dihedrals = detect_dihedrals(mol)

        # Butane should have rotatable bonds
        assert len(dihedrals) > 0

    def test_methane_no_dihedrals(self):
        """Test that methane has no rotatable bonds."""
        atoms, mol = get_initial_structure("C")
        dihedrals = detect_dihedrals(mol)
        assert len(dihedrals) == 0

    def test_ethane_one_dihedral(self):
        """Test that ethane has exactly one rotatable bond."""
        atoms, mol = get_initial_structure("CC")
        dihedrals = detect_dihedrals(mol)
        # Ethane has one C-C rotatable bond
        assert len(dihedrals) == 1

    def test_pentane_dihedrals(self):
        """Test that pentane has multiple rotatable bonds."""
        atoms, mol = get_initial_structure("CCCCC")
        dihedrals = detect_dihedrals(mol)
        # Pentane should have 2 backbone rotatable bonds
        assert len(dihedrals) >= 2

    def test_dihedral_chain_has_four_atoms(self):
        """Test that dihedral chains have exactly 4 atoms."""
        atoms, mol = get_initial_structure("CCCC")
        dihedrals = detect_dihedrals(mol)

        for d in dihedrals:
            assert len(d.chain) == 4

    def test_dihedral_group_is_set(self):
        """Test that dihedral groups are sets of integers."""
        atoms, mol = get_initial_structure("CCCC")
        dihedrals = detect_dihedrals(mol)

        for d in dihedrals:
            assert isinstance(d.group, set)
            assert all(isinstance(i, int) for i in d.group)

    def test_ring_bonds_not_rotatable(self):
        """Test that ring bonds are not detected as rotatable."""
        # Cyclohexane - all bonds are in ring
        atoms, mol = get_initial_structure("C1CCCCC1")
        dihedrals = detect_dihedrals(mol)
        # Ring bonds should not be rotatable
        assert len(dihedrals) == 0


class TestGetBondingGraph:
    """Tests for bonding graph generation."""

    def test_graph_has_correct_nodes(self):
        """Test that graph has correct number of nodes."""
        atoms, mol = get_initial_structure("CC")  # Ethane
        g = get_bonding_graph(mol)

        # Should have node for each atom
        assert g.number_of_nodes() == len(atoms)

    def test_nodes_have_atomic_number(self):
        """Test that nodes have atomic number attribute."""
        atoms, mol = get_initial_structure("CO")  # Methanol
        g = get_bonding_graph(mol)

        for node, data in g.nodes(data=True):
            assert "z" in data
            assert data["z"] > 0

    def test_graph_has_edges(self):
        """Test that graph has bonds as edges."""
        atoms, mol = get_initial_structure("CC")
        g = get_bonding_graph(mol)

        # Ethane has 7 bonds (1 C-C + 6 C-H)
        assert g.number_of_edges() == 7

    def test_edge_has_rotor_info(self):
        """Test that edges have rotor information."""
        atoms, mol = get_initial_structure("CCCC")
        g = get_bonding_graph(mol)

        for u, v, data in g.edges(data=True):
            assert "data" in data
            assert "rotor" in data["data"]
            assert "ring" in data["data"]


class TestGetDihedralInfo:
    """Tests for dihedral info extraction from bonds."""

    def test_returns_dihedral_info(self):
        """Test that function returns DihedralInfo object."""
        atoms, mol = get_initial_structure("CCCC")
        g = get_bonding_graph(mol)
        backbone = {i for i, d in g.nodes(data=True) if d["z"] > 1}

        # Get a rotatable bond (C-C bond between carbon atoms)
        # Find carbon atoms first
        carbon_nodes = [n for n, d in g.nodes(data=True) if d["z"] == 6]
        if len(carbon_nodes) >= 2:
            bond = (carbon_nodes[0], carbon_nodes[1])
            if g.has_edge(*bond):
                info = get_dihedral_info(g, bond, backbone)
                assert isinstance(info, DihedralInfo)

    def test_chain_is_valid(self):
        """Test that returned chain defines valid dihedral."""
        atoms, mol = get_initial_structure("CCCC")
        g = get_bonding_graph(mol)
        backbone = {i for i, d in g.nodes(data=True) if d["z"] > 1}

        carbon_nodes = [n for n, d in g.nodes(data=True) if d["z"] == 6]
        if len(carbon_nodes) >= 2:
            # Find a bond between two carbons
            for i, c1 in enumerate(carbon_nodes):
                for c2 in carbon_nodes[i + 1 :]:
                    if g.has_edge(c1, c2):
                        info = get_dihedral_info(g, (c1, c2), backbone)
                        # Chain should have 4 unique atoms
                        assert len(info.chain) == 4
                        assert len(set(info.chain)) == 4
                        break
