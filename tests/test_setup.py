#!/usr/bin/env python3
"""Test the refactored bouquet/setup.py with RDKit"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bouquet.setup import (
    get_initial_structure,
    get_initial_structure_from_file,
    get_conformers_from_file,
    detect_dihedrals,
    get_bonding_graph,
    mol_to_ase_atoms,
)


def test_smiles_to_structure():
    """Test generating structure from SMILES"""
    print("=" * 60)
    print("Test 1: SMILES to structure")
    print("=" * 60)

    test_cases = [
        ("CCCC", "butane"),
        ("c1ccccc1", "benzene"),
        ("CC(=O)O", "acetic acid"),
        ("CCO", "ethanol"),
        ("c1ccccc1CCCC", "butylbenzene"),
    ]

    for smiles, name in test_cases:
        atoms, mol = get_initial_structure(smiles)
        dihedrals = detect_dihedrals(mol)
        print(
            f"  {name:15} ({smiles:15}): {len(atoms):3} atoms, {len(dihedrals):2} dihedrals"
        )

        # Verify atoms object has expected attributes
        assert hasattr(atoms, "charge"), "atoms should have charge attribute"
        assert len(atoms.get_positions()) == len(
            atoms
        ), "positions should match atom count"

    print("  ✓ All SMILES tests passed\n")


def test_xyz_file_reading():
    """Test reading from XYZ file"""
    print("=" * 60)
    print("Test 2: XYZ file reading")
    print("=" * 60)

    # Create a test XYZ file (ethane)
    xyz_content = """8
ethane
C     0.000000     0.000000     0.762736
C     0.000000     0.000000    -0.762736
H     1.018470     0.000000     1.158568
H    -0.509235     0.881823     1.158568
H    -0.509235    -0.881823     1.158568
H    -1.018470     0.000000    -1.158568
H     0.509235    -0.881823    -1.158568
H     0.509235     0.881823    -1.158568
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
        f.write(xyz_content)
        xyz_path = f.name

    try:
        atoms, mol = get_initial_structure_from_file(xyz_path)
        print(f"  Loaded ethane from XYZ: {len(atoms)} atoms")

        # Check bond detection worked
        g = get_bonding_graph(mol)
        print(f"  Bond graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

        dihedrals = detect_dihedrals(mol)
        print(f"  Detected {len(dihedrals)} dihedrals")

        print("  ✓ XYZ reading test passed\n")

    finally:
        Path(xyz_path).unlink()


def test_sdf_file_reading():
    """Test reading from SDF file"""
    print("=" * 60)
    print("Test 3: SDF file reading")
    print("=" * 60)

    # Create a test SDF file (methane)
    sdf_content = """methane
     RDKit          3D

  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.6276    0.6276    0.6276 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6276   -0.6276    0.6276 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6276    0.6276   -0.6276 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.6276   -0.6276   -0.6276 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
  1  5  1  0
M  END
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sdf", delete=False) as f:
        f.write(sdf_content)
        sdf_path = f.name

    try:
        atoms, mol = get_initial_structure_from_file(sdf_path)
        print(f"  Loaded methane from SDF: {len(atoms)} atoms")
        print("  ✓ SDF reading test passed\n")

    finally:
        Path(sdf_path).unlink()


def test_multi_conformer_xyz():
    """Test reading multiple conformers from XYZ"""
    print("=" * 60)
    print("Test 4: Multi-conformer XYZ reading")
    print("=" * 60)

    # Create multi-frame XYZ
    xyz_content = """3
water conf 1
O     0.000000     0.000000     0.117300
H     0.756950     0.000000    -0.469200
H    -0.756950     0.000000    -0.469200
3
water conf 2
O     0.000000     0.000000     0.120000
H     0.760000     0.000000    -0.480000
H    -0.760000     0.000000    -0.480000
3
water conf 3
O     0.000000     0.000000     0.115000
H     0.750000     0.000000    -0.460000
H    -0.750000     0.000000    -0.460000
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
        f.write(xyz_content)
        xyz_path = f.name

    try:
        conformers = get_conformers_from_file(xyz_path)
        print(f"  Loaded {len(conformers)} water conformers")
        assert len(conformers) == 3, "Should have 3 conformers"
        print("  ✓ Multi-conformer test passed\n")

    finally:
        Path(xyz_path).unlink()


def test_dihedral_detection():
    """Test dihedral angle detection and manipulation"""
    print("=" * 60)
    print("Test 5: Dihedral detection and manipulation")
    print("=" * 60)

    # Test with n-pentane (should have multiple rotatable bonds)
    smiles = "CCCCC"
    atoms, mol = get_initial_structure(smiles)
    dihedrals = detect_dihedrals(mol)

    print(f"  n-pentane: {len(atoms)} atoms, {len(dihedrals)} dihedrals")

    for i, dih in enumerate(dihedrals):
        angle = dih.get_angle(atoms)
        print(
            f"    Dihedral {i+1}: atoms {dih.chain}, angle = {angle:.1f}°, group size = {len(dih.group)}"
        )

    # Verify dihedral info structure
    assert len(dihedrals) > 0, "Should detect at least one dihedral"
    for dih in dihedrals:
        assert len(dih.chain) == 4, "Dihedral chain should have 4 atoms"
        assert dih.group is not None, "Dihedral should have a rotation group"
        assert dih.type is not None, "Dihedral should have a type"

    print("  ✓ Dihedral detection test passed\n")


def test_bonding_graph():
    """Test bonding graph generation"""
    print("=" * 60)
    print("Test 6: Bonding graph generation")
    print("=" * 60)

    atoms, mol = get_initial_structure("c1ccccc1C")  # toluene

    g = get_bonding_graph(mol)

    print(f"  Toluene graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    # Count carbons and hydrogens
    carbons = sum(1 for _, d in g.nodes(data=True) if d["z"] == 6)
    hydrogens = sum(1 for _, d in g.nodes(data=True) if d["z"] == 1)
    print(f"  Atoms: {carbons} C, {hydrogens} H")

    # Count ring vs non-ring bonds
    ring_bonds = sum(1 for _, _, d in g.edges(data=True) if d["data"]["ring"])
    rotor_bonds = sum(1 for _, _, d in g.edges(data=True) if d["data"]["rotor"])
    print(f"  Bonds: {ring_bonds} in ring, {rotor_bonds} rotatable")

    assert carbons == 7, "Toluene should have 7 carbons"
    assert hydrogens == 8, "Toluene should have 8 hydrogens"
    print("  ✓ Bonding graph test passed\n")


def test_charge_handling():
    """Test handling of charged molecules"""
    print("=" * 60)
    print("Test 7: Charged molecule handling")
    print("=" * 60)

    test_cases = [
        ("[NH4+]", "ammonium", 1),
        ("CC(=O)[O-]", "acetate", -1),
        ("CCCC", "butane", 0),
    ]

    for smiles, name, expected_charge in test_cases:
        atoms, mol = get_initial_structure(smiles)
        from rdkit import Chem

        actual_charge = Chem.GetFormalCharge(mol)
        print(
            f"  {name:12}: expected charge = {expected_charge:+d}, actual = {actual_charge:+d}"
        )
        assert actual_charge == expected_charge, f"Charge mismatch for {name}"

    print("  ✓ Charge handling test passed\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("TESTING REFACTORED BOUQUET/SETUP.PY (RDKit-based)")
    print("=" * 60 + "\n")

    test_smiles_to_structure()
    test_xyz_file_reading()
    test_sdf_file_reading()
    test_multi_conformer_xyz()
    test_dihedral_detection()
    test_bonding_graph()
    test_charge_handling()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
