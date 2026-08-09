"""Unit tests for bouquet.config module."""

import pytest
from pathlib import Path

from bouquet.calculator import CalculatorFactory
from bouquet.config import (
    Configuration,
    DEFAULT_NUM_STEPS,
    DEFAULT_INIT_STEPS,
    AUTO_STEPS_FLOOR,
    AUTO_STEPS_CAP,
    auto_total_budget,
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
        assert config.num_threads == 4
        assert config.charge == 0
        assert config.multiplicity == 1
        # Gradient GP defaults to the "gradfreeze" schedule (cold-fit 20 steps,
        # then freeze); full refitting every step is the opt-out (dense=0). The
        # tuning knobs live on the nested RunOptions (config.run).
        assert config.run.gradient_steps == 0
        assert config.run.grad_refit_dense_until == 20
        assert config.run.grad_refit_every == 0

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
        """Test auto_steps with few dihedrals: budget = round(1.95*d^3) minus initial."""
        config = Configuration(smiles="C", auto_steps=True)
        result = config.compute_auto_steps(num_dihedrals=4, num_initial=5)
        assert result == auto_total_budget(4) - 5  # 125 - 5

    def test_auto_steps_d1_floor(self):
        """d=1 gets the fixed floor budget, not the near-zero power law."""
        config = Configuration(smiles="C", auto_steps=True)
        result = config.compute_auto_steps(num_dihedrals=1, num_initial=5)
        assert result == AUTO_STEPS_FLOOR - 5

    def test_auto_steps_high_d_cap(self):
        """Very high d saturates at the cap."""
        config = Configuration(smiles="C", auto_steps=True)
        result = config.compute_auto_steps(num_dihedrals=15, num_initial=5)
        assert result == AUTO_STEPS_CAP - 5

    def test_auto_steps_returns_zero_minimum(self):
        """Test that auto_steps doesn't return negative values."""
        config = Configuration(smiles="C", auto_steps=True)
        result = config.compute_auto_steps(num_dihedrals=2, num_initial=100)
        # budget(2)=16, so max(0, 16 - 100) = 0
        assert result == 0


class TestConstants:
    """Tests for module-level constants."""

    def test_available_methods(self):
        """available_methods() is the installed subset of the registry."""
        from bouquet.calculator import _FULL_REGISTRY

        avail = CalculatorFactory.available_methods()
        assert isinstance(avail, tuple)
        assert list(avail) == sorted(avail)  # stable, sorted
        # every available method must be a registered one
        assert set(avail) <= set(_FULL_REGISTRY)
        # rdkit force fields have no optional deps, so they're always available
        assert {"mmff", "uff"} <= set(avail)

    def test_auto_steps_budget_monotonic_floor_cap(self):
        """The budget is non-decreasing in d, floored at d=1 and capped at high d."""
        vals = [auto_total_budget(d) for d in range(0, 16)]
        assert vals == sorted(vals)                      # non-decreasing
        assert auto_total_budget(0) == AUTO_STEPS_FLOOR   # floor
        assert auto_total_budget(1) == AUTO_STEPS_FLOOR
        assert auto_total_budget(4) == 125                # round(1.95*64)
        assert max(vals) <= AUTO_STEPS_CAP
        assert auto_total_budget(20) == AUTO_STEPS_CAP    # saturates

    def test_default_values(self):
        """Test default constants are reasonable."""
        assert DEFAULT_NUM_STEPS > 0
        assert DEFAULT_INIT_STEPS > 0
        assert AUTO_STEPS_CAP > AUTO_STEPS_FLOOR
