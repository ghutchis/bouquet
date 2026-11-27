"""Tools for computing the energy of a molecule

   Actual energy evaluation is done via the calculator.py module
   (or calc_rdkit.py if using UFF and MMFF94 via RDKit)
"""

from __future__ import annotations

# Annotations are strings (PEP 563), so numpy/ase types in signatures no longer
# pin those modules at import; defer them and the optimizer submodules until an
# energy is actually evaluated (Python 3.15+).
__lazy_modules__ = [
    "numpy",
    "ase",
    "ase.calculators.calculator",
    "ase.constraints",
    "ase.optimize",
    "bouquet.clash",
    "bouquet.gradients",
]

import logging
import os

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixInternals
from ase.optimize import FIRE2, LBFGS

from bouquet.clash import DegenerateDihedralError, TorsionGeometry
from bouquet.config import (
    DEFAULT_FMAX,
    DEFAULT_RELAXATION_STEPS,
    RELAX_FAILURE_ENERGY_EV,
    TIGHT_FMAX,
    TIGHT_MAX_STEPS,
)
from bouquet.gradients import compute_torsion_gradient
from bouquet.setup import DihedralInfo

logger = logging.getLogger(__name__)


def evaluate_energy(
    angles: list[float] | np.ndarray,
    atoms: Atoms,
    dihedrals: list[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
    relax: bool = True,
    steps: int | None = DEFAULT_RELAXATION_STEPS,
) -> tuple[float, Atoms]:
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
        Tuple[float, Atoms]: energy (in eV) and the Atoms object after applying dihedral constraints and optional relaxation. An energy value of 1000.0 (``RELAX_FAILURE_ENERGY_EV``) indicates a failed evaluation -- either the calculator/relaxation failed, or the requested dihedrals are undefined for this structure (a planar inner angle), in which case the returned Atoms is the unmodified copy.
    """
    # Make a copy of the input
    atoms = atoms.copy()

    # Set the dihedral angles to desired settings. TorsionGeometry applies exactly
    # the sequential set_dihedral loop this used to run inline (verified against
    # ASE to ~1e-13 A in tests/test_clash.py) but vectorized, and it is the same
    # code the acquisition optimizer screens candidates with -- so the geometry a
    # candidate is accepted on is the geometry that gets evaluated, by
    # construction rather than by coincidence.
    dihedral_constraints = [(a, di.chain) for a, di in zip(angles, dihedrals)]
    try:
        new_positions = TorsionGeometry(atoms, dihedrals).build(angles)
    except DegenerateDihedralError:
        # A planar inner angle leaves the torsion undefined. ASE raises here too,
        # but uncaught -- which killed the whole run. It is just a bad geometry:
        # report the failure energy and let the caller reject the point.
        return RELAX_FAILURE_ENERGY_EV, atoms

    # Write the coordinates directly rather than via set_positions(): the incoming
    # Atoms may still carry the FixInternals constraint from a previous evaluation,
    # and set_positions() would try to enforce it against the new torsions (and
    # fail to converge). ASE's own set_dihedral bypasses constraints the same way,
    # assigning into the positions array, so this preserves the original behaviour.
    atoms.positions[:] = new_positions

    # Cheap pre-relaxation guard: when relaxing, probe the unrelaxed geometry
    # with the (typically cheap) optimizer calculator rather than burning an
    # expensive energy-calculator single point (e.g. GFN2/DFT) that is immediately
    # discarded. When not relaxing, evaluate directly on `calc`.
    guard_calc = relaxCalc if relax else calc
    try:
        energy = guard_calc.get_potential_energy(atoms)
    except Exception:
        # A THROWN evaluation poisons the calculator exactly like a rejected one:
        # xtb leaves its restart wavefunction in whatever state failed to converge
        # and feeds it to every later call (see _reset_calculators). The geometry
        # guards reset on REJECTION, but nothing here was rejected -- the point was
        # simply unevaluable -- so without this reset the run continues on a broken
        # calculator and reports ~-2700 eV for perfectly intact geometries.
        _reset_calculators(guard_calc)
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


def relax_structure(atoms: Atoms, energyCalc: Calculator, calc: Calculator, steps: int | None) -> tuple[float, Atoms]:
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
            # Tight final relaxation: L-BFGS to TIGHT_FMAX with a safety cap so
            # noisy potentials (ML/SCF noise floors) cannot hang indefinitely.
            dyn = LBFGS(atoms, logfile=os.devnull)
            dyn.run(fmax=TIGHT_FMAX, steps=TIGHT_MAX_STEPS)
    except Exception:
        # The optimizer step failed -- either a non-convergent line search
        # (ValueError) or a calculator failure mid-relaxation (e.g. xtb SCF
        # non-convergence raised as CalculationFailed) when the geometry walks
        # into a clashing region. Both mean "this geometry is bad"; fall through
        # and let the energy evaluation below assign the failure energy. This is
        # essential for collective category / low-mode moves, whose broadcast
        # kick can land in a compact clashing coil the constrained pre-relax does
        # not fully clean up -- an uncaught CalculationFailed there crashes the run.
        #
        # Reset BEFORE falling through. If the failure came from the calculator
        # rather than the line search, xtb is now holding the wavefunction that
        # failed to converge, and the evaluation below would not raise -- it would
        # SUCCEED and return a poisoned energy (~-2700 eV on an intact geometry),
        # which is far worse than the failure sentinel because nothing downstream
        # can tell it from a real minimum. Resetting means that evaluation is a
        # clean SCF on the partially relaxed geometry.
        _reset_calculators(calc, energyCalc)

    # if the energy calculation fails, return a high energy
    try:
        energy = energyCalc.get_potential_energy(atoms)
    except Exception:
        _reset_calculators(energyCalc)  # same poisoning risk as above
        energy = RELAX_FAILURE_ENERGY_EV

    return energy, atoms


def evaluate_energy_with_gradient(
    angles: list[float] | np.ndarray,
    atoms: Atoms,
    dihedrals: list[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
    relax: bool = True,
    steps: int | None = DEFAULT_RELAXATION_STEPS,
    per_degree: bool = False,
) -> tuple[float, Atoms, np.ndarray]:
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


def _reset_calculators(*calcs: Calculator) -> None:
    """Clear cached calculator state after a bad geometry.

    xtb's ASE calculator keeps the converged wavefunction between calls and feeds
    it back as the SCF restart guess. That is a speed win on a smooth trajectory
    and a catastrophe after a collapsed geometry: the SCF settles into a spurious
    solution and every *subsequent* evaluation inherits it, returning energies
    ~100 eV below the start for geometries that are perfectly intact. Measured on
    m216_d09, one clashed initial guess made the next four guesses report -102.8,
    -102.2, -72.7 and -105.8 eV; after a reset the same four give -0.5, +0.2,
    +3.4 and +2.0 eV.

    This is the mechanism behind the absorbing failure seen in the d=8-12
    benchmark -- runs did not keep *choosing* the collapsed region, their
    calculator was simply broken from that point on. Resetting on rejection is
    what makes the geometry guards actually recover the run instead of merely
    labelling the damage.

    ``Calculator.reset()`` is standard ASE and clears the cached results; the xtb
    calculator additionally drops its API handle and restart wavefunction.
    Calculators without a usable ``reset`` are skipped.

    Contract for new backends: ASE's base ``reset()`` only clears ``self.atoms``
    and ``self.results`` -- the *result cache*, not backend state. Calling it is
    therefore sufficient only for a calculator that either overrides ``reset()``
    to drop its own persistent state (as xtb does) or never carries any. Before
    adding a backend, check whether it feeds a previous result back in as a
    restart guess; if it does and its ``reset()`` does not clear that, this
    function silently will not protect it. As of writing only xtb is known to
    restart from its own output; the psi4 wrapper rebuilds and re-solves from
    scratch each call, and the neural-network potentials have no SCF at all.
    """
    for calc in calcs:
        reset = getattr(calc, "reset", None)
        if callable(reset):
            try:
                reset()
            except Exception:  # a calculator that can't reset must not kill the run
                logger.debug("Calculator reset failed", exc_info=True)
