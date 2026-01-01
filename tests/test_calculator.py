"""Unit tests for bouquet.calculator module."""

import pytest
from unittest.mock import patch, MagicMock

from bouquet.calculator import CalculatorFactory
from bouquet.config import Configuration


class TestCalculatorFactory:
    """Tests for the CalculatorFactory class."""

    def test_create_gfn2_calculator(self):
        """Test creating a GFN2 calculator."""
        # The XTB import happens inside the create method, so we need to
        # patch it at the module level before calling create
        mock_xtb_class = MagicMock()
        mock_calc = MagicMock()
        mock_xtb_class.return_value = mock_calc

        mock_module = MagicMock()
        mock_module.XTB = mock_xtb_class

        with patch.dict("sys.modules", {"xtb.ase.calculator": mock_module, "xtb": MagicMock()}):
            calc = CalculatorFactory.create("gfn2")
            assert calc is not None
            mock_xtb_class.assert_called_once()

    def test_create_gfn0_calculator(self):
        """Test creating a GFN0 calculator."""
        mock_xtb_class = MagicMock()
        mock_calc = MagicMock()
        mock_xtb_class.return_value = mock_calc

        mock_module = MagicMock()
        mock_module.XTB = mock_xtb_class

        with patch.dict("sys.modules", {"xtb.ase.calculator": mock_module, "xtb": MagicMock()}):
            calc = CalculatorFactory.create("gfn0")
            assert calc is not None
            mock_xtb_class.assert_called_once_with(method="gfn0")

    def test_create_gfnff_calculator(self):
        """Test creating a GFNFF calculator."""
        mock_xtb_class = MagicMock()
        mock_calc = MagicMock()
        mock_xtb_class.return_value = mock_calc

        mock_module = MagicMock()
        mock_module.XTB = mock_xtb_class

        with patch.dict("sys.modules", {"xtb.ase.calculator": mock_module, "xtb": MagicMock()}):
            calc = CalculatorFactory.create("gfnff")
            assert calc is not None
            mock_xtb_class.assert_called_once_with(method="gfnff")

    def test_create_invalid_method(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized calculation method"):
            CalculatorFactory.create("invalid_method")

    def test_from_config_energy_calculator(self, tmp_path):
        """Test creating energy calculator from config."""
        config = Configuration(
            smiles="C",
            energy_method="gfn2",
            optimizer_method="gfnff",
        )

        with patch.object(CalculatorFactory, "create") as mock_create:
            mock_calc = MagicMock()
            mock_create.return_value = mock_calc

            calc = CalculatorFactory.from_config(config, for_optimizer=False)

            mock_create.assert_called_once_with(
                method="gfn2",
                num_threads=config.psi4_num_threads,
                charge=config.charge,
                multiplicity=config.multiplicity,
            )

    def test_from_config_optimizer_calculator(self, tmp_path):
        """Test creating optimizer calculator from config."""
        config = Configuration(
            smiles="C",
            energy_method="gfn2",
            optimizer_method="gfnff",
        )

        with patch.object(CalculatorFactory, "create") as mock_create:
            mock_calc = MagicMock()
            mock_create.return_value = mock_calc

            calc = CalculatorFactory.from_config(config, for_optimizer=True)

            mock_create.assert_called_once_with(
                method="gfnff",
                num_threads=config.psi4_num_threads,
                charge=config.charge,
                multiplicity=config.multiplicity,
            )

    def test_from_config_passes_charge_and_multiplicity(self):
        """Test that charge and multiplicity are passed correctly."""
        config = Configuration(
            smiles="C",
            energy_method="b3lyp",
            charge=1,
            multiplicity=2,
            psi4_num_threads=8,
        )

        with patch.object(CalculatorFactory, "create") as mock_create:
            CalculatorFactory.from_config(config)

            mock_create.assert_called_once_with(
                method="b3lyp",
                num_threads=8,
                charge=1,
                multiplicity=2,
            )


class TestCalculatorFactoryIntegration:
    """Integration tests that actually create calculators (require dependencies)."""

    @pytest.mark.slow
    def test_create_gfn2_real(self):
        """Test creating a real GFN2 calculator."""
        pytest.importorskip("xtb")
        calc = CalculatorFactory.create("gfn2")
        assert calc is not None
        # Check it has the expected interface
        assert hasattr(calc, "get_potential_energy")

    @pytest.mark.slow
    def test_create_gfnff_real(self):
        """Test creating a real GFNFF calculator."""
        pytest.importorskip("xtb")
        calc = CalculatorFactory.create("gfnff")
        assert calc is not None
        assert hasattr(calc, "get_potential_energy")

    @pytest.mark.slow
    def test_create_ani_real(self):
        """Test creating a real ANI calculator."""
        pytest.importorskip("torchani")
        calc = CalculatorFactory.create("ani")
        assert calc is not None
        assert hasattr(calc, "get_potential_energy")
