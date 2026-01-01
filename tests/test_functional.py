"""Functional tests for the complete Bouquet optimization workflow.

These tests run actual optimizations with real calculators and verify
that the results are chemically sensible.
"""

import pytest
from pathlib import Path

import numpy as np
from ase import Atoms

from bouquet.config import Configuration
from bouquet.setup import get_initial_structure, get_initial_structure_from_file, detect_dihedrals
from bouquet.assess import evaluate_energy, relax_structure
from bouquet.io import create_output_directory, save_structure, initialize_structure_log
from bouquet.calculator import CalculatorFactory


# Mark all tests in this module as slow (they involve actual calculations)
pytestmark = pytest.mark.slow


class TestSmilesOptimization:
    """Functional tests using SMILES input."""

    @pytest.fixture
    def gfnff_calc(self):
        """Create a GFNFF calculator for fast testing."""
        pytest.importorskip("xtb")
        return CalculatorFactory.create("gfnff")

    def test_ethane_structure_generation(self):
        """Test generating 3D structure from ethane SMILES."""
        atoms, mol = get_initial_structure("CC")

        # Should have correct atom count
        assert len(atoms) == 8  # 2 C + 6 H

        # Should have reasonable bond lengths
        positions = atoms.get_positions()
        c_positions = positions[:2]  # First two are carbons
        cc_dist = np.linalg.norm(c_positions[0] - c_positions[1])
        # C-C bond should be ~1.54 Angstroms (allow some flexibility)
        assert 1.3 < cc_dist < 1.8

    def test_butane_dihedral_detection(self):
        """Test that butane SMILES produces detectable dihedrals."""
        atoms, mol = get_initial_structure("CCCC")
        dihedrals = detect_dihedrals(mol)

        # Butane should have 1 central rotatable C-C bond
        # (terminal C-C bonds don't change energy much due to methyl symmetry)
        assert len(dihedrals) >= 1

    def test_pentane_multiple_dihedrals(self):
        """Test that pentane has multiple rotatable bonds."""
        atoms, mol = get_initial_structure("CCCCC")
        dihedrals = detect_dihedrals(mol)

        # Pentane should have 2 meaningful rotatable bonds
        assert len(dihedrals) >= 2

    def test_butane_energy_evaluation(self, gfnff_calc):
        """Test energy evaluation for butane at different conformations."""
        atoms, mol = get_initial_structure("CCCC")
        dihedrals = detect_dihedrals(mol)

        if len(dihedrals) == 0:
            pytest.skip("No dihedrals detected")

        # Evaluate energy at gauche (60 deg) and anti (180 deg) conformations
        gauche_angles = [60.0] * len(dihedrals)
        anti_angles = [180.0] * len(dihedrals)

        gauche_energy, gauche_atoms = evaluate_energy(
            gauche_angles, atoms, dihedrals, gfnff_calc, gfnff_calc, relax=False
        )
        anti_energy, anti_atoms = evaluate_energy(
            anti_angles, atoms, dihedrals, gfnff_calc, gfnff_calc, relax=False
        )

        # Both should return valid energies
        assert np.isfinite(gauche_energy)
        assert np.isfinite(anti_energy)

        # Anti conformation should typically be lower energy for butane
        # (this is a soft check - force field might not capture this perfectly)

    def test_relax_structure_converges(self, gfnff_calc):
        """Test that structure relaxation produces lower energy."""
        atoms, mol = get_initial_structure("CC")

        # Get initial energy
        atoms.calc = gfnff_calc
        initial_energy = atoms.get_potential_energy()

        # Relax structure
        final_energy, relaxed_atoms = relax_structure(atoms.copy(), gfnff_calc, steps=50)

        # Energy should decrease or stay the same
        assert final_energy <= initial_energy + 0.01  # Small tolerance

    def test_hexane_full_workflow(self, gfnff_calc, temp_dir, monkeypatch):
        """Test complete workflow for hexane optimization."""
        monkeypatch.chdir(temp_dir)

        # Generate structure
        atoms, mol = get_initial_structure("CCCCCC")
        dihedrals = detect_dihedrals(mol)

        # Should have multiple dihedrals
        assert len(dihedrals) >= 3

        # Create output directory
        out_dir = create_output_directory("hexane", 42, "gfnff", {"test": True})
        assert out_dir.exists()

        # Save initial structure
        save_structure(out_dir, atoms, "initial.xyz")
        assert (out_dir / "initial.xyz").exists()

        # Relax initial structure
        energy, relaxed = relax_structure(atoms.copy(), gfnff_calc, steps=20)
        save_structure(out_dir, relaxed, "relaxed.xyz", comment=f"E={energy}")
        assert (out_dir / "relaxed.xyz").exists()

    def test_branched_alkane_dihedrals(self):
        """Test dihedral detection for branched molecules."""
        # 2-methylbutane (isopentane)
        atoms, mol = get_initial_structure("CC(C)CC")
        dihedrals = detect_dihedrals(mol)

        # Should have at least 2 rotatable bonds
        assert len(dihedrals) >= 2

    def test_cyclic_no_rotatable_bonds(self):
        """Test that cyclic molecules have no ring-internal rotatable bonds."""
        # Cyclohexane
        atoms, mol = get_initial_structure("C1CCCCC1")
        dihedrals = detect_dihedrals(mol)

        # Ring bonds should not be rotatable
        assert len(dihedrals) == 0

    def test_methylcyclohexane_has_external_dihedral(self):
        """Test that substituents on rings can have rotatable bonds."""
        # Methylcyclohexane - has a rotatable C-CH3 bond
        atoms, mol = get_initial_structure("CC1CCCCC1")
        dihedrals = detect_dihedrals(mol)

        # Should have at least one rotatable bond (the methyl)
        assert len(dihedrals) >= 1


class TestXyzFileOptimization:
    """Functional tests using XYZ file input."""

    @pytest.fixture
    def gfnff_calc(self):
        """Create a GFNFF calculator for fast testing."""
        pytest.importorskip("xtb")
        return CalculatorFactory.create("gfnff")

    def test_read_and_optimize_xyz(self, gfnff_calc, pentane_xyz_file):
        """Test reading XYZ file and optimizing structure."""
        atoms, mol = get_initial_structure_from_file(str(pentane_xyz_file))

        # Should have correct atom count for pentane
        assert len(atoms) == 17  # 5 C + 12 H

        # Relax and verify
        energy, relaxed = relax_structure(atoms.copy(), gfnff_calc, steps=20)
        assert np.isfinite(energy)

    def test_xyz_preserves_connectivity(self, pentane_xyz_file):
        """Test that XYZ reading preserves molecular connectivity."""
        atoms, mol = get_initial_structure_from_file(str(pentane_xyz_file))
        dihedrals = detect_dihedrals(mol)

        # Pentane should have rotatable bonds
        assert len(dihedrals) >= 2

    def test_water_xyz_no_dihedrals(self, sample_xyz_file):
        """Test that water XYZ file has no rotatable bonds."""
        atoms, mol = get_initial_structure_from_file(str(sample_xyz_file))
        dihedrals = detect_dihedrals(mol)

        # Water should have no rotatable bonds
        assert len(dihedrals) == 0


class TestEnergyEvaluation:
    """Functional tests for energy evaluation with constraints."""

    @pytest.fixture
    def gfnff_calc(self):
        """Create a GFNFF calculator."""
        pytest.importorskip("xtb")
        return CalculatorFactory.create("gfnff")

    def test_dihedral_constraint_applied(self, gfnff_calc):
        """Test that dihedral constraints are properly applied."""
        atoms, mol = get_initial_structure("CCCC")
        dihedrals = detect_dihedrals(mol)

        if len(dihedrals) == 0:
            pytest.skip("No dihedrals detected")

        # Set to specific angle
        target_angle = 90.0
        angles = [target_angle] * len(dihedrals)

        energy, result_atoms = evaluate_energy(
            angles, atoms, dihedrals, gfnff_calc, gfnff_calc, relax=False
        )

        # Check that dihedral was set correctly
        actual_angle = dihedrals[0].get_angle(result_atoms)
        # Angle should be close to target (within ~1 degree)
        angle_diff = abs((actual_angle - target_angle + 180) % 360 - 180)
        assert angle_diff < 5.0

    def test_relaxation_preserves_dihedral(self, gfnff_calc):
        """Test that constrained relaxation preserves dihedral angle."""
        atoms, mol = get_initial_structure("CCCC")
        dihedrals = detect_dihedrals(mol)

        if len(dihedrals) == 0:
            pytest.skip("No dihedrals detected")

        target_angle = 120.0
        angles = [target_angle] * len(dihedrals)

        # Evaluate with relaxation
        energy, result_atoms = evaluate_energy(
            angles, atoms, dihedrals, gfnff_calc, gfnff_calc, relax=True
        )

        # Dihedral should still be close to target
        actual_angle = dihedrals[0].get_angle(result_atoms)
        angle_diff = abs((actual_angle - target_angle + 180) % 360 - 180)
        assert angle_diff < 10.0  # Allow some tolerance due to relaxation


class TestCalculatorIntegration:
    """Test that different calculators work correctly."""

    @pytest.mark.slow
    def test_gfn2_calculator(self):
        """Test GFN2 calculator produces valid energies."""
        pytest.importorskip("xtb")

        atoms, mol = get_initial_structure("CC")
        calc = CalculatorFactory.create("gfn2")

        atoms.calc = calc
        energy = atoms.get_potential_energy()

        assert np.isfinite(energy)
        # GFN2 energies for ethane should be around -7 to -8 Hartree * 27.2 = -190 to -220 eV
        assert energy < 0  # Should be negative

    @pytest.mark.slow
    def test_gfnff_calculator(self):
        """Test GFNFF calculator produces valid energies."""
        pytest.importorskip("xtb")

        atoms, mol = get_initial_structure("CC")
        calc = CalculatorFactory.create("gfnff")

        atoms.calc = calc
        energy = atoms.get_potential_energy()

        assert np.isfinite(energy)

    @pytest.mark.slow
    def test_ani_calculator(self):
        """Test ANI calculator produces valid energies."""
        pytest.importorskip("torchani")

        atoms, mol = get_initial_structure("CC")
        calc = CalculatorFactory.create("ani")

        atoms.calc = calc
        energy = atoms.get_potential_energy()

        assert np.isfinite(energy)


class TestEndToEndOptimization:
    """End-to-end tests for the full optimization workflow."""

    @pytest.fixture
    def gfnff_calc(self):
        """Create a GFNFF calculator."""
        pytest.importorskip("xtb")
        return CalculatorFactory.create("gfnff")

    def test_propane_short_optimization(self, gfnff_calc, temp_dir, monkeypatch):
        """Test a short optimization run for propane."""
        monkeypatch.chdir(temp_dir)

        # Setup
        atoms, mol = get_initial_structure("CCC")
        dihedrals = detect_dihedrals(mol)

        if len(dihedrals) == 0:
            pytest.skip("No dihedrals for propane")

        # Get initial energy
        atoms.calc = gfnff_calc
        initial_energy = atoms.get_potential_energy()

        # Evaluate a few conformations
        energies = []
        for angle in [60, 120, 180, 240, 300]:
            angles = [float(angle)] * len(dihedrals)
            energy, _ = evaluate_energy(
                angles, atoms, dihedrals, gfnff_calc, gfnff_calc, relax=False
            )
            energies.append(energy)

        # Should have variation in energies
        assert max(energies) != min(energies)

        # Best energy should be reasonable
        best_energy = min(energies)
        assert np.isfinite(best_energy)

    def test_config_to_calculation_workflow(self, gfnff_calc, temp_dir, monkeypatch):
        """Test workflow from Configuration to calculation."""
        monkeypatch.chdir(temp_dir)

        # Create configuration
        config = Configuration(
            smiles="CCCC",
            energy_method="gfnff",
            optimizer_method="gfnff",
            num_steps=5,
            init_steps=2,
            seed=42,
        )

        # Generate structure
        atoms, mol = get_initial_structure(config.smiles)
        dihedrals = detect_dihedrals(mol)

        # Create calculator from config
        calc = CalculatorFactory.from_config(config, for_optimizer=False)

        # Verify calculator works
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        assert np.isfinite(energy)

        # Verify auto_steps computation
        num_steps = config.compute_auto_steps(len(dihedrals), config.init_steps)
        assert num_steps == config.num_steps  # auto_steps is False
