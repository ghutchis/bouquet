"""Tools for computing the energy of a molecule"""

import os
from typing import List, Tuple, Union

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixInternals
from ase.optimize import LBFGS

from bouquet.config import DEFAULT_FMAX, DEFAULT_RELAXATION_STEPS
from bouquet.setup import DihedralInfo


def evaluate_energy(
    angles: Union[List[float], np.ndarray],
    atoms: Atoms,
    dihedrals: List[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
    relax: bool = True,
) -> Tuple[float, Atoms]:
    """
    Compute the potential energy for a set of dihedral angles and optionally relax non-dihedral degrees of freedom.
    
    Parameters:
        angles (Union[List[float], np.ndarray]): Dihedral angles in degrees, in the same order as `dihedrals`.
        atoms (Atoms): Atomic structure to copy and modify; returned Atoms is the copy after any applied constraints/relaxation.
        dihedrals (List[DihedralInfo]): Descriptors of dihedral definitions (expected to provide `chain` and `group` for setting angles).
        calc (Calculator): Calculator used to evaluate the potential energy of the (constrained) structure.
        relaxCalc (Calculator): Calculator assigned to the Atoms for geometry optimization when relaxation is enabled.
        relax (bool): If True, relax non-dihedral degrees of freedom after fixing dihedrals; if False, return the energy of the constrained structure as-is.
    
    Returns:
        Tuple[float, Atoms]: energy (in eV) and the Atoms object after applying dihedral constraints and optional relaxation. An energy value of 1000.0 is used to indicate a failed energy evaluation.
    """
    # Make a copy of the input
    atoms = atoms.copy()

    # Set the dihedral angles to desired settings
    dihedral_constraints = []
    for a, di in zip(angles, dihedrals):
        atoms.set_dihedral(*di.chain, a, indices=di.group)

        # Define the constraints
        dihedral_constraints.append((a, di.chain))

    # First, try to compute the energy
    # if it doesn't converge, return a high energy
    # (e.g. 1000 eV)
    try:
        energy = calc.get_potential_energy(atoms)
    except:
        energy = 1000.0

    if not relax or energy >= 1000.0:
        # too high energy, just return
        return energy, atoms

    # set the dihedral constraints and relax
    atoms.set_constraint()
    atoms.set_constraint(FixInternals(dihedrals_deg=dihedral_constraints))

    # A quick relaxation to get the structure in the right ballpark
    return relax_structure(atoms, calc, relaxCalc, DEFAULT_RELAXATION_STEPS)


def relax_structure(atoms: Atoms, energyCalc: Calculator, calc: Calculator, steps: int) -> Tuple[float, Atoms]:
    """
    Relax the atomic geometry using the provided optimizer and evaluate its potential energy.
    
    Constraints on dihedral angles should be applied to `atoms` before calling. `calc` is used to perform the geometry optimization; `energyCalc` is used to evaluate the final potential energy.
    
    Parameters:
    	atoms (Atoms): The atomic configuration to relax.
    	energyCalc (Calculator): Calculator used to compute the potential energy after relaxation.
    	calc (Calculator): Calculator used for the geometry optimization.
    	steps (int or None): Maximum number of optimization steps to perform; if `None`, run until convergence.
    
    Returns:
    	tuple: A pair `(energy, atoms)` where `energy` is the potential energy of the (possibly relaxed) structure and `atoms` is the resulting Atoms object. If the energy evaluation fails, `energy` will be `1000.0`.
    """

    atoms.set_calculator(calc)

    try:
        dyn = LBFGS(atoms, logfile=os.devnull)
        if steps is not None:
            dyn.run(fmax=DEFAULT_FMAX, steps=steps)
        else:
            dyn.run(fmax=DEFAULT_FMAX)
    except ValueError:  # LBFGS failed to converge, probably high energy
        pass

    # if the energy calculation fails, return a high energy
    try:
        energy = energyCalc.get_potential_energy(atoms)
    except:
        energy = 1000.0

    return energy, atoms