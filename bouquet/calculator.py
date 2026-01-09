"""Factory for creating ASE calculators."""

from typing import TYPE_CHECKING, Literal, Optional, get_args

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from bouquet.config import Configuration
    from rdkit import Chem

# Type alias for supported methods - this is the single source of truth
MethodType = Literal["ani", "b3lyp", "b97", "gfn0", "gfn2", "gfnff", "mmff", "uff"]

# Derive SUPPORTED_METHODS from MethodType for runtime checks
SUPPORTED_METHODS: frozenset[str] = frozenset(get_args(MethodType))

# Default number of threads (matches config.NUM_THREADS)
_DEFAULT_NUM_THREADS = 4


class CalculatorFactory:
    """Factory for creating ASE calculators based on method name."""

    # Class-level access to supported methods
    SUPPORTED_METHODS = SUPPORTED_METHODS

    @staticmethod
    def create(
        method: MethodType,
        mol: Optional["Chem.Mol"] = None,
        num_threads: int = _DEFAULT_NUM_THREADS,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> "Calculator":
        """Create an ASE-compatible calculator for the specified method.

        Args:
            method: The calculation method to use
            mol: RDKit molecule (required for mmff/uff methods)
            num_threads: Number of threads for Psi4 calculations
            charge: Molecular charge
            multiplicity: Spin multiplicity

        Returns:
            An ASE Calculator instance

        Raises:
            ValueError: If the method is not recognized or mol is missing for RDKit methods
        """
        if method == "ani":
            import torchani

            return torchani.models.ANI2x().ase()

        elif method == "gfn2":
            from xtb.ase.calculator import XTB

            return XTB(method="GFN2xTB")

        elif method == "gfn0":
            from xtb.ase.calculator import XTB

            return XTB(method="GFN0xTB")

        elif method == "gfnff":
            from xtb.ase.calculator import XTB

            return XTB(method="gfnff")

        elif method == "b3lyp":
            from ase.calculators.psi4 import Psi4

            return Psi4(
                method="b3lyp-D3MBJ2B",
                basis="def2-svp",
                num_threads=num_threads,
                multiplicity=multiplicity,
                charge=charge,
            )

        elif method == "b97":
            from ase.calculators.psi4 import Psi4

            return Psi4(
                method="b97-d3bj",
                basis="def2-svp",
                num_threads=num_threads,
                multiplicity=multiplicity,
                charge=charge,
            )

        elif method == "mmff":
            from bouquet.calc_rdkit import RDKitMMFFCalculator

            if mol is None:
                raise ValueError("RDKit MMFF calculator: requires a molecule (mol)")
            return RDKitMMFFCalculator(mol)

        elif method == "uff":
            from bouquet.calc_rdkit import RDKitUFFCalculator

            if mol is None:
                raise ValueError("RDKit UFF calculator: requires a molecule (mol)")
            return RDKitUFFCalculator(mol)

        else:
            raise ValueError(f"Unrecognized calculation method: {method}")

    @classmethod
    def from_config(
        cls,
        config: "Configuration",
        for_optimizer: bool = False,
        mol: Optional["Chem.Mol"] = None,
    ) -> "Calculator":
        """Create a calculator from a Configuration object.

        Args:
            config: The configuration object
            for_optimizer: If True, create the optimizer calculator;
                          if False, create the energy calculator
            mol: RDKit molecule (required for mmff/uff methods)

        Returns:
            An ASE Calculator instance
        """
        method = config.optimizer_method if for_optimizer else config.energy_method
        return cls.create(
            method=method,
            mol=mol,
            num_threads=config.num_threads,
            charge=config.charge,
            multiplicity=config.multiplicity,
        )
