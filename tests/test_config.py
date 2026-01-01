"""Unit tests for bouquet.config module."""

import pytest
from pathlib import Path

from bouquet.config import (
    Configuration,
    DEFAULT_NUM_STEPS,
    DEFAULT_INIT_STEPS,
    AUTO_STEPS_THRESHOLDS,
    AUTO_STEPS_DEFAULT,
    SUPPORTED_METHODS,
)


class TestConfiguration:
    """Tests for the Configuration dataclass."""

    def test_configuration_with_smiles(self):
        """Test creating configuration with SMILES input."""
        config = Configuration(smiles="CCCC")
        assert config.smiles == "CCCC"
        assert config.input_file is None
        assert config.name == "CCCC"

    def test_configuration_with_input_file(self, tmp_path):
        """Test creating configuration with input file."""
        test_file = tmp_path / "test.xyz"
        test_file.touch()
        config = Configuration(input_file=test_file)
        assert config.input_file == test_file
        assert config.smiles is None
        assert config.name == "test"

    def test_configuration_requires_input(self):
        """Test that configuration requires either smiles or input_file."""
        with pytest.raises(ValueError, match="Must specify either smiles or input_file"):
            Configuration()

    def test_configuration_default_values(self):
        """Test default values are set correctly."""
        config = Configuration(smiles="C")
        assert config.energy_method == "gfn2"
        assert config.optimizer_method == "gfnff"
        assert config.num_steps == DEFAULT_NUM_STEPS
        assert config.init_steps == DEFAULT_INIT_STEPS
        assert config.auto_steps is False
        assert config.relax is True
        assert config.psi4_num_threads == 4
        assert config.charge == 0
        assert config.multiplicity == 1

    def test_configuration_string_path_conversion(self, tmp_path):
        """Test that string paths are converted to Path objects."""
        test_file = tmp_path / "test.xyz"
        test_file.touch()

        config = Configuration(
            smiles="C",
            input_file=str(test_file),
            conformer_file=str(test_file),
            out_dir=str(tmp_path),
        )

        assert isinstance(config.input_file, Path)
        assert isinstance(config.conformer_file, Path)
        assert isinstance(config.out_dir, Path)

    def test_configuration_custom_values(self):
        """Test setting custom configuration values."""
        config = Configuration(
            smiles="CCCC",
            energy_method="ani",
            optimizer_method="gfn2",
            num_steps=100,
            init_steps=10,
            auto_steps=True,
            relax=False,
            seed=42,
            charge=1,
            multiplicity=2,
        )

        assert config.energy_method == "ani"
        assert config.optimizer_method == "gfn2"
        assert config.num_steps == 100
        assert config.init_steps == 10
        assert config.auto_steps is True
        assert config.relax is False
        assert config.seed == 42
        assert config.charge == 1
        assert config.multiplicity == 2


class TestComputeAutoSteps:
    """Tests for the compute_auto_steps method."""

    def test_auto_steps_disabled(self):
        """Test that auto_steps returns num_steps when disabled."""
        config = Configuration(smiles="C", num_steps=50, auto_steps=False)
        result = config.compute_auto_steps(num_dihedrals=10, num_initial=5)
        assert result == 50

    def test_auto_steps_few_dihedrals(self):
        """Test auto_steps with few dihedrals (<=3)."""
        config = Configuration(smiles="C", auto_steps=True)
        result = config.compute_auto_steps(num_dihedrals=2, num_initial=5)
        # Should use threshold of 25, minus initial guesses
        assert result == 25 - 5

    def test_auto_steps_medium_dihedrals(self):
        """Test auto_steps with medium dihedrals (<=5)."""
        config = Configuration(smiles="C", auto_steps=True)
        result = config.compute_auto_steps(num_dihedrals=4, num_initial=5)
        # Should use threshold of 50, minus initial guesses
        assert result == 50 - 5

    def test_auto_steps_many_dihedrals(self):
        """Test auto_steps with many dihedrals (<=7)."""
        config = Configuration(smiles="C", auto_steps=True)
        result = config.compute_auto_steps(num_dihedrals=6, num_initial=5)
        # Should use threshold of 100, minus initial guesses
        assert result == 100 - 5

    def test_auto_steps_very_many_dihedrals(self):
        """Test auto_steps with very many dihedrals (>7)."""
        config = Configuration(smiles="C", auto_steps=True)
        result = config.compute_auto_steps(num_dihedrals=10, num_initial=5)
        # Should use default of 200, minus initial guesses
        assert result == AUTO_STEPS_DEFAULT - 5

    def test_auto_steps_returns_zero_minimum(self):
        """Test that auto_steps doesn't return negative values."""
        config = Configuration(smiles="C", auto_steps=True)
        result = config.compute_auto_steps(num_dihedrals=2, num_initial=100)
        # Should return max(0, 25 - 100) = 0
        assert result == 0


class TestConstants:
    """Tests for module-level constants."""

    def test_supported_methods(self):
        """Test that supported methods are correctly defined."""
        expected = {"ani", "b3lyp", "b97", "gfn0", "gfn2", "gfnff"}
        assert SUPPORTED_METHODS == expected

    def test_auto_steps_thresholds_ordering(self):
        """Test that auto_steps thresholds are properly ordered."""
        thresholds = sorted(AUTO_STEPS_THRESHOLDS.keys())
        assert thresholds == [3, 5, 7]

        # Values should increase with thresholds
        values = [AUTO_STEPS_THRESHOLDS[k] for k in thresholds]
        assert values == sorted(values)

    def test_default_values(self):
        """Test default constants are reasonable."""
        assert DEFAULT_NUM_STEPS > 0
        assert DEFAULT_INIT_STEPS > 0
        assert AUTO_STEPS_DEFAULT > max(AUTO_STEPS_THRESHOLDS.values())
