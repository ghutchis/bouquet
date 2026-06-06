"""Tools for computing the energy of a molecule"""

import os
from typing import List, Optional, Tuple, Union

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixInternals
from ase.optimize import FIRE2, LBFGS

from bouquet.config import (
    DEFAULT_FMAX,
    DEFAULT_RELAXATION_STEPS,
    RELAX_FAILURE_ENERGY_EV,
    TIGHT_FMAX,
)
from bouquet.gradients import compute_torsion_gradient
from bouquet.setup import DihedralInfo


def evaluate_energy(
    angles: Union[List[float], np.ndarray],
    atoms: Atoms,
    dihedrals: List[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
    relax: bool = True,
    steps: Optional[int] = DEFAULT_RELAXATION_STEPS,
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
        steps (int or None): Maximum number of relaxation steps. A positive limit selects the quick FIRE2/ABC-FIRE relaxation (`DEFAULT_FMAX`); `None` selects the tight L-BFGS relaxation (`TIGHT_FMAX`, no step limit).

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
        energy = RELAX_FAILURE_ENERGY_EV

    if not relax or energy >= RELAX_FAILURE_ENERGY_EV:
        # too high energy, just return
        return energy, atoms

    # set the dihedral constraints and relax
    atoms.set_constraint()
    atoms.set_constraint(FixInternals(dihedrals_deg=dihedral_constraints))

    # Relax non-dihedral degrees of freedom (quick when `steps` is a limit,
    # tight when `steps` is None).
    return relax_structure(atoms, calc, relaxCalc, steps)


def relax_structure(atoms: Atoms, energyCalc: Calculator, calc: Calculator, steps: Optional[int]) -> Tuple[float, Atoms]:
    """
    Relax the atomic geometry using the provided optimizer and evaluate its potential energy.

    Two relaxation regimes are selected by `steps`:
      * a positive step limit runs a quick FIRE2 / ABC-FIRE relaxation to
        `DEFAULT_FMAX`, used during the Bayesian-optimization search; and
      * `None` runs a tight L-BFGS relaxation to `TIGHT_FMAX` with no step
        limit, used for the final geometry.

    Constraints on dihedral angles should be applied to `atoms` before calling. `calc` is used to perform the geometry optimization; `energyCalc` is used to evaluate the final potential energy.

    Parameters:
    	atoms (Atoms): The atomic configuration to relax.
    	energyCalc (Calculator): Calculator used to compute the potential energy after relaxation.
    	calc (Calculator): Calculator used for the geometry optimization.
    	steps (int or None): Maximum number of optimization steps to perform; if `None`, run a tight relaxation until convergence.

    Returns:
    	tuple: A pair `(energy, atoms)` where `energy` is the potential energy of the (possibly relaxed) structure and `atoms` is the resulting Atoms object. If the energy evaluation fails, `energy` will be `1000.0`.
    """

    atoms.calc = calc

    try:
        if steps is not None:
            # Quick relaxation during the search: ABC-FIRE for fast, robust
            # progress within a tight step budget.
            dyn = FIRE2(atoms, logfile=os.devnull, use_abc=True)
            dyn.run(fmax=DEFAULT_FMAX, steps=steps)
        else:
            # Tight final relaxation: L-BFGS to TIGHT_FMAX, no step limit.
            dyn = LBFGS(atoms, logfile=os.devnull)
            dyn.run(fmax=TIGHT_FMAX)
    except ValueError:  # optimizer failed to converge, probably high energy
        pass

    # if the energy calculation fails, return a high energy
    try:
        energy = energyCalc.get_potential_energy(atoms)
    except:
        energy = RELAX_FAILURE_ENERGY_EV

    return energy, atoms


def evaluate_energy_with_gradient(
    angles: Union[List[float], np.ndarray],
    atoms: Atoms,
    dihedrals: List[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
    relax: bool = True,
    steps: Optional[int] = DEFAULT_RELAXATION_STEPS,
    per_degree: bool = False,
) -> Tuple[float, Atoms, np.ndarray]:
    """Evaluate the energy and its gradient with respect to the torsion angles.

    Thin wrapper around :func:`evaluate_energy` that additionally returns
    ``dE/dtheta`` for each dihedral, obtained by projecting the calculator's
    Cartesian forces onto the torsion coordinates (see :mod:`bouquet.gradients`).

    When ``relax`` is True the gradient is that of the relaxed energy surface
    ``E*(theta)`` evaluated at the constrained minimum (valid by the envelope
    theorem); when ``relax`` is False it is the rigid-scan gradient at the given
    geometry. The gradient is always projected from the energy calculator
    ``calc``.

    Consistency requirement: the envelope-theorem identity only holds when the
    geometry is a constrained minimum of ``calc``. With ``relax=True`` the
    geometry is minimized on ``relaxCalc``, so ``dE*/dtheta`` from ``calc`` forces
    is only correct when ``calc`` and ``relaxCalc`` are the same surface. If they
    differ (e.g. gfn2 energy on a gfnff-optimized geometry), the projection drops
    the relaxation-response term and the gradient is biased -- callers that feed
    these to a model (``use_gradients``) must use matching calculators. The CLI
    enforces this; ``relax=False`` is always consistent (rigid scan of ``calc``).

    Parameters mirror :func:`evaluate_energy`, plus:
        per_degree (bool): If True, the gradient is in eV/degree; otherwise
            eV/radian (default).

    Returns:
        Tuple[float, Atoms, np.ndarray]: ``(energy, atoms, gradient)``. On a
        failed energy evaluation the energy is ``RELAX_FAILURE_ENERGY_EV`` and
        the gradient is filled with ``nan`` (it is meaningless there and should
        be dropped by the caller).
    """
    energy, relaxed = evaluate_energy(
        angles, atoms, dihedrals, calc, relaxCalc, relax=relax, steps=steps
    )

    if energy >= RELAX_FAILURE_ENERGY_EV:
        gradient = np.full(len(dihedrals), np.nan, dtype=float)
        return energy, relaxed, gradient

    gradient = compute_torsion_gradient(
        relaxed, dihedrals, calc, per_degree=per_degree
    )
    return energy, relaxed, gradient
