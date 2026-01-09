"""RDKit force field calculators compatible with ASE interface."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from rdkit import Chem
from rdkit.Chem import AllChem

from bouquet.config import KCAL_TO_EV


class RDKitCalculator(Calculator, ABC):
    """Base ASE Calculator wrapper for RDKit force fields."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, mol: Chem.Mol, **kwargs):
        """Initialize the RDKit calculator.

        Args:
            mol: RDKit molecule (connectivity information used for force field)
        """
        super().__init__(**kwargs)
        self.mol = Chem.Mol(mol)  # Make a copy

        # Ensure we have a conformer to work with
        # (shouldn't ever happen, but just in case)
        if self.mol.GetNumConformers() == 0:
            result = AllChem.EmbedMolecule(self.mol)
            if result == -1:
                raise ValueError(
                    "Could not generate 3D conformer for molecule. "
                    "Ensure the molecule has valid connectivity."
                )
        # Subclasses should initialize their force field properties
        self._setup_force_field()

    @abstractmethod
    def _setup_force_field(self) -> None:
        """Set up force field-specific properties. Called during __init__."""
        pass

    @abstractmethod
    def _create_force_field(self, conf_id: int = 0):
        """Create a force field instance for the current geometry.

        Args:
            conf_id: Conformer ID to use

        Returns:
            RDKit force field object
        """
        pass

    def calculate(
        self,
        atoms=None,
        properties: List[str] = None,
        system_changes=all_changes,
    ):
        """Calculate energy (and optionally forces) using the force field.

        Args:
            atoms: ASE Atoms object (uses self.atoms if None)
            properties: List of properties to calculate
            system_changes: List of changes since last calculation
        """
        if properties is None:
            properties = ["energy"]
        super().calculate(atoms, properties, system_changes)

        # Update RDKit conformer with current ASE positions
        conf = self.mol.GetConformer(0)
        positions = self.atoms.get_positions()
        for i, pos in enumerate(positions):
            conf.SetAtomPosition(i, pos.tolist())

        # Create force field with current geometry
        ff = self._create_force_field(conf_id=0)
        if ff is None:
            raise ValueError(
                f"Could not create {self.__class__.__name__} force field for current geometry"
            )

        # Energy in kcal/mol, convert to eV for ASE consistency
        energy_kcal = ff.CalcEnergy()
        self.results["energy"] = energy_kcal * KCAL_TO_EV

        if "forces" in properties:
            # RDKit returns gradient (dE/dx), forces are negative gradient
            grad = ff.CalcGrad()
            grad = np.array(grad).reshape(-1, 3)
            # Convert kcal/(mol·Å) to eV/Å and negate for forces
            self.results["forces"] = -grad * KCAL_TO_EV


class RDKitMMFFCalculator(RDKitCalculator):
    """ASE Calculator wrapper for RDKit's MMFF94 force field."""

    def __init__(self, mol: Chem.Mol, mmff_variant: str = "MMFF94", **kwargs):
        """Initialize the MMFF calculator.

        Args:
            mol: RDKit molecule (connectivity information used for force field)
            mmff_variant: "MMFF94" or "MMFF94s" (the 's' variant uses
                         different parameters for planar nitrogens)
        """
        self.mmff_variant = mmff_variant
        self.mmff_props = None
        super().__init__(mol, **kwargs)

    def _setup_force_field(self) -> None:
        """Set up MMFF properties (atom types, etc.)."""
        self.mmff_props = AllChem.MMFFGetMoleculeProperties(
            self.mol, mmffVariant=self.mmff_variant
        )
        if self.mmff_props is None:
            raise ValueError(
                f"Could not get {self.mmff_variant} properties for molecule. "
                "The molecule may contain unsupported atom types."
            )

    def _create_force_field(self, conf_id: int = 0):
        """Create an MMFF force field instance."""
        return AllChem.MMFFGetMoleculeForceField(
            self.mol, self.mmff_props, confId=conf_id
        )


class RDKitUFFCalculator(RDKitCalculator):
    """ASE Calculator wrapper for RDKit's UFF (Universal Force Field)."""

    def __init__(self, mol: Chem.Mol, **kwargs):
        """Initialize the UFF calculator.

        Args:
            mol: RDKit molecule (connectivity information used for force field)
        """
        super().__init__(mol, **kwargs)

    def _setup_force_field(self) -> None:
        """Verify UFF can handle this molecule."""
        # UFF doesn't require pre-computed properties like MMFF,
        # but we verify it can be created for this molecule
        ff = AllChem.UFFGetMoleculeForceField(self.mol, confId=0)
        if ff is None:
            raise ValueError(
                "Could not create UFF force field for molecule. "
                "The molecule may contain unsupported atom types."
            )

    def _create_force_field(self, conf_id: int = 0):
        """Create a UFF force field instance."""
        return AllChem.UFFGetMoleculeForceField(self.mol, confId=conf_id)
