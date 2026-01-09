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
    """Compute the energy of a molecule given dihedral angles

    Args:
        angles: List of dihedral angles in degrees
        atoms: Structure to optimize
        dihedrals: Description of the dihedral angles
        calc: Calculator used to compute energy
        relaxCalc: Calculator used to optimize geometry
        relax: Whether to relax the non-dihedral degrees of freedom
    Returns:
        - (float) energy of the structure
        - (Atoms) relaxed structure
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
    """Relax and return the energy and geometry of the ground state

    Any constraints on the dihedral angles should be applied (or removed) on atoms
    before calling.

    One calculator will be used to relax the geometry, and another to compute the energy
    of the final structure

    Args:
        atoms: Atoms object to be optimized
        energyCalc: Calculator used to compute the energy
        calc: Calculator used to optimize
        steps: Number of steps to perform (or None to run until convergence)
    Returns:
        Energy of the minimized structure
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
