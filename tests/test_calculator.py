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
            mock_xtb_class.assert_called_once_with(method="GFN0xTB")

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

    def test_create_gfn2_calculator_with_solvent(self):
        """Test that --solvent reaches xtb's native solvent kwarg."""
        mock_xtb_class = MagicMock()
        mock_module = MagicMock()
        mock_module.XTB = mock_xtb_class

        with patch.dict("sys.modules", {"xtb.ase.calculator": mock_module, "xtb": MagicMock()}):
            calc = CalculatorFactory.create("gfn2", solvent="water")
            assert calc is not None
            mock_xtb_class.assert_called_once_with(method="GFN2xTB", solvent="water")

    def test_create_gfn0_calculator_with_solvent_raises(self):
        """GFN0-xTB has no fitted GBSA parameters; requesting a solvent should
        raise a clear error here rather than reach xtb's opaque
        'Cannot construct calculator for xtb' InputError."""
        mock_module = MagicMock()
        mock_module.XTB = MagicMock()

        with patch.dict("sys.modules", {"xtb.ase.calculator": mock_module, "xtb": MagicMock()}):
            with pytest.raises(ValueError, match="does not support implicit solvent"):
                CalculatorFactory.create("gfn0", solvent="water")

    def test_create_ani_with_solvent_raises(self):
        """ANI-2x is gas-phase only; requesting a solvent should raise, not be ignored."""
        with patch.dict("sys.modules", {"torchani": MagicMock()}):
            with pytest.raises(ValueError, match="does not support implicit solvent"):
                CalculatorFactory.create("ani", solvent="water")

    def test_psi4_solvent_sets_ddx_options_on_calculate(self):
        """--solvent for psi4 methods should engage DDX continuum solvation via
        psi4.set_options; the ASE Psi4 wrapper doesn't forward unknown kwargs
        (like a plain "solvent" kwarg) into any actual psi4 call, so this must
        go through the _SolvatedPsi4 subclass's calculate() override."""
        from ase import Atoms
        from bouquet.calculator import _psi4_calculator

        mock_psi4 = MagicMock()
        mock_psi4.energy.return_value = -1.0

        with patch.dict("sys.modules", {"psi4": mock_psi4}):
            calc = _psi4_calculator(
                "b3lyp-d4", basis="def2-svp", num_threads=1,
                charge=0, multiplicity=1, solvent="water",
            )
            atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
            calc.calculate(atoms=atoms, properties=["energy"])

        mock_psi4.set_options.assert_any_call({"ddx": True, "ddx_solvent": "water"})

    def test_psi4_no_solvent_skips_ddx_options(self):
        """Without --solvent, psi4 methods get the plain (unpatched) Psi4 class."""
        from bouquet.calculator import _psi4_calculator

        mock_psi4 = MagicMock()

        with patch.dict("sys.modules", {"psi4": mock_psi4}):
            calc = _psi4_calculator(
                "b3lyp-d4", basis="def2-svp", num_threads=1,
                charge=0, multiplicity=1,
            )

        assert type(calc).__name__ == "Psi4"

    @pytest.mark.parametrize(
        "method,expected",
        [
            ("b3lyp", True),
            ("wb97x", True),
            ("mp2", True),
            ("hf3c", True),
            ("b973c", True),
            ("r2scan3c", True),
            ("gfn2", False),
            ("gfn0", False),
            ("gfnff", False),
            ("ani", False),
            ("aimnet2", False),
            ("mmff", False),
            ("uff", False),
            ("not_a_real_method", False),
        ],
    )
    def test_uses_psi4(self, method, expected):
        """Used by the CLI to reject --solvent + --relax on a psi4 optimizer
        (DDX solvation gradients under relaxation aren't validated)."""
        assert CalculatorFactory.uses_psi4(method) is expected

    def test_create_aimnet2_passes_charge(self):
        """AIMNet2 is charge-aware: charge must reach the AIMNet2ASE constructor."""
        mock_aimnet2_class = MagicMock()
        mock_module = MagicMock()
        mock_module.AIMNet2ASE = mock_aimnet2_class

        with patch.dict(
            "sys.modules",
            {"aimnet.calculators": mock_module, "aimnet": MagicMock()},
        ):
            calc = CalculatorFactory.create("aimnet2", charge=-1)
            assert calc is not None
            mock_aimnet2_class.assert_called_once_with("aimnet2", charge=-1)

    def test_create_invalid_method(self):
        """Test that an unregistered/unavailable method raises ValueError."""
        with pytest.raises(ValueError, match="not available"):
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
                mol=None,
                num_threads=config.num_threads,
                charge=config.charge,
                multiplicity=config.multiplicity,
                solvent=config.solvent,
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
                mol=None,
                num_threads=config.num_threads,
                charge=config.charge,
                multiplicity=config.multiplicity,
                solvent=config.solvent,
            )

    def test_from_config_passes_charge_and_multiplicity(self):
        """Test that charge and multiplicity are passed correctly."""
        config = Configuration(
            smiles="C",
            energy_method="b3lyp",
            charge=1,
            multiplicity=2,
            num_threads=8,
        )

        with patch.object(CalculatorFactory, "create") as mock_create:
            CalculatorFactory.from_config(config)

            mock_create.assert_called_once_with(
                method="b3lyp",
                mol=None,
                num_threads=8,
                charge=1,
                multiplicity=2,
                solvent=None,
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

    @pytest.mark.slow
    def test_create_aimnet2_real(self):
        """Test creating a real AIMNet2 calculator."""
        pytest.importorskip("aimnet")
        calc = CalculatorFactory.create("aimnet2")
        assert calc is not None
        assert hasattr(calc, "get_potential_energy")
