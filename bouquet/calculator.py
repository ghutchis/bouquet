"""Factory for creating ASE calculators."""

from typing import TYPE_CHECKING

from bouquet.config import PSI4_NUM_THREADS, MethodType

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from bouquet.config import Configuration


class CalculatorFactory:
    """Factory for creating ASE calculators based on method name."""

    @staticmethod
    def create(
        method: MethodType,
        num_threads: int = PSI4_NUM_THREADS,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> "Calculator":
        """Create an ASE calculator for the specified method.

        Args:
            method: The calculation method to use
            num_threads: Number of threads for Psi4 calculations
            charge: Molecular charge
            multiplicity: Spin multiplicity

        Returns:
            An ASE Calculator instance

        Raises:
            ValueError: If the method is not recognized
        """
        if method == "ani":
            import torchani

            return torchani.models.ANI2x().ase()

        elif method == "gfn2":
            from xtb.ase.calculator import XTB

            return XTB()

        elif method == "gfn0":
            from xtb.ase.calculator import XTB

            return XTB(method="gfn0")

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

        else:
            raise ValueError(f"Unrecognized calculation method: {method}")

    @classmethod
    def from_config(cls, config: "Configuration", for_optimizer: bool = False) -> "Calculator":
        """Create a calculator from a Configuration object.

        Args:
            config: The configuration object
            for_optimizer: If True, create the optimizer calculator;
                          if False, create the energy calculator

        Returns:
            An ASE Calculator instance
        """
        method = config.optimizer_method if for_optimizer else config.energy_method
        return cls.create(
            method=method,
            num_threads=config.psi4_num_threads,
            charge=config.charge,
            multiplicity=config.multiplicity,
        )
