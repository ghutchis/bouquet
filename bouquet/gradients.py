"""Gradient of the energy with respect to torsion angles.

bouquet sets each dihedral by rigidly rotating a defined group of atoms about
the central bond (see :class:`bouquet.setup.DihedralInfo` and
:func:`bouquet.assess.evaluate_energy`). That makes the Jacobian of the atomic
positions with respect to a torsion angle trivial: every atom ``k`` in the
rotating group moves as

    dx_k / dtheta = u x (x_k - p)

where ``u`` is the unit vector along the rotation axis (the central bond of the
dihedral) and ``p`` is any point on that axis. The generalized force conjugate
to the torsion is therefore

    dE/dtheta = - sum_k F_k . (u x (x_k - p))        [theta in radians]

with ``F_k`` the Cartesian force on atom ``k`` (``F = -dE/dx``). This is just
the negative torque of the rotating group about the bond axis.

For the *relaxed* energy surface ``E*(theta) = min_{other DOF} E(x; theta)`` the
same projection, evaluated at the relaxed (constrained) geometry, yields
``dE*/dtheta`` by the envelope theorem: at a constrained minimum the residual
generalized force along every *unconstrained* degree of freedom is ~0, so the
Cartesian forces project cleanly onto the constrained torsion. The expensive
energy evaluation already computes these forces analytically (xTB, Psi4 and the
RDKit force fields all return them), so the torsion gradient is essentially a
free by-product of the energy call.
"""

from typing import List, Sequence

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from bouquet.setup import DihedralInfo

# 1 radian in degrees; dE/dtheta[deg] = dE/dtheta[rad] / DEG_PER_RAD.
DEG_PER_RAD = 180.0 / np.pi


def _torsion_axis(positions: np.ndarray, chain: Sequence[int]) -> tuple:
    """Return the unit rotation axis and a pivot point for a dihedral.

    The rotating group is turned about the central bond of the dihedral, i.e.
    the bond between the second and third atoms of ``chain`` (matching ASE's
    ``Atoms.set_dihedral(*chain, angle, indices=group)`` convention).

    Args:
        positions: Atomic positions, shape (n_atoms, 3).
        chain: The four atom indices defining the dihedral.

    Returns:
        Tuple ``(axis_unit, pivot)`` where ``axis_unit`` is the normalized bond
        vector and ``pivot`` is a point on the axis (the second chain atom).
    """
    a2, a3 = chain[1], chain[2]
    axis = positions[a3] - positions[a2]
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        raise ValueError(
            f"Degenerate torsion axis for chain {tuple(chain)}: "
            "the two central atoms coincide."
        )
    return axis / norm, positions[a2]


def project_torsion_gradient(
    positions: np.ndarray,
    dihedrals: List[DihedralInfo],
    forces: np.ndarray,
    per_degree: bool = False,
) -> np.ndarray:
    """Project Cartesian forces onto each torsion coordinate.

    Computes ``dE/dtheta`` for every dihedral as the negative torque of its
    rotating group about the bond axis (see the module docstring). This is a
    pure function of geometry and forces; it performs no energy evaluation.

    Args:
        positions: Atomic positions, shape (n_atoms, 3), in Angstrom.
        dihedrals: Dihedral definitions providing ``chain`` and ``group``.
        forces: Cartesian forces (``-dE/dx``), shape (n_atoms, 3), in eV/Angstrom.
            These must be the *unconstrained* forces; if the structure carries a
            ``FixInternals`` constraint, fetch the forces from the calculator
            directly rather than via ``Atoms.get_forces`` (which would zero out
            the torsion component). :func:`compute_torsion_gradient` handles this.
        per_degree: If True, return eV/degree; otherwise eV/radian (default).

    Returns:
        np.ndarray: Gradient ``dE/dtheta`` for each dihedral, length
        ``len(dihedrals)``, in eV/radian (or eV/degree if ``per_degree``).
    """
    positions = np.asarray(positions, dtype=float)
    forces = np.asarray(forces, dtype=float)

    grad = np.empty(len(dihedrals), dtype=float)
    for i, di in enumerate(dihedrals):
        axis, pivot = _torsion_axis(positions, di.chain)
        group = np.fromiter(di.group, dtype=int)
        # dx_k/dtheta = axis x (x_k - pivot); on-axis atoms contribute zero.
        velocity = np.cross(axis, positions[group] - pivot)
        generalized_force = float(np.einsum("ij,ij->", forces[group], velocity))
        grad[i] = -generalized_force

    if per_degree:
        grad = grad / DEG_PER_RAD
    return grad


def compute_torsion_gradient(
    atoms: Atoms,
    dihedrals: List[DihedralInfo],
    calc: Calculator,
    per_degree: bool = False,
) -> np.ndarray:
    """Evaluate ``dE/dtheta`` for each dihedral at the geometry of ``atoms``.

    Fetches the unconstrained Cartesian forces from ``calc`` and projects them
    onto the torsion coordinates. The input ``atoms`` is not modified, and any
    constraint it carries (e.g. ``FixInternals`` from a constrained relaxation)
    is dropped on the working copy so the torsion force component is preserved.

    Args:
        atoms: Structure at which to evaluate the gradient.
        dihedrals: Dihedral definitions providing ``chain`` and ``group``.
        calc: Calculator used to compute the Cartesian forces (the energy
            surface whose gradient is wanted).
        per_degree: If True, return eV/degree; otherwise eV/radian (default).

    Returns:
        np.ndarray: Gradient ``dE/dtheta`` for each dihedral, in eV/radian
        (or eV/degree if ``per_degree``).
    """
    work = atoms.copy()
    work.set_constraint()  # drop FixInternals so forces are unconstrained
    work.calc = calc
    forces = work.get_forces()
    return project_torsion_gradient(
        work.get_positions(), dihedrals, forces, per_degree=per_degree
    )
