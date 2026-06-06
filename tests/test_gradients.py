"""Unit tests for bouquet.gradients (torsion-angle energy gradients).

The core correctness check is a finite-difference comparison: projecting the
Cartesian forces onto a torsion coordinate must reproduce dE/dtheta obtained by
rigidly rotating the dihedral and differencing the energy. These use the fast
RDKit MMFF force field (a core dependency), so they are not marked slow.
"""

import numpy as np
import pytest

from bouquet.assess import evaluate_energy_with_gradient
from bouquet.config import RELAX_FAILURE_ENERGY_EV
from bouquet.gradients import (
    DEG_PER_RAD,
    compute_torsion_gradient,
    project_torsion_gradient,
)
from bouquet.setup import detect_dihedrals, get_initial_structure


@pytest.fixture
def butane():
    """Butane structure, RDKit Mol, and detected dihedrals."""
    atoms, mol = get_initial_structure("CCCC")
    dihedrals = detect_dihedrals(mol)
    return atoms, mol, dihedrals


@pytest.fixture
def mmff_calc(butane):
    """Fast MMFF94 calculator for the butane fixture."""
    from bouquet.calc_rdkit import RDKitMMFFCalculator

    _, mol, _ = butane
    return RDKitMMFFCalculator(mol)


def _finite_difference_gradient(atoms, di, calc, mol, delta_deg=0.5):
    """Central finite-difference dE/dtheta (eV/rad) for one rigid-rotation dihedral."""
    from bouquet.calc_rdkit import RDKitMMFFCalculator

    angle = atoms.get_dihedral(*di.chain)

    def energy_at(value):
        probe = atoms.copy()
        probe.calc = RDKitMMFFCalculator(mol)
        probe.set_dihedral(*di.chain, value, indices=di.group)
        return probe.get_potential_energy()

    per_deg = (energy_at(angle + delta_deg) - energy_at(angle - delta_deg)) / (
        2 * delta_deg
    )
    return per_deg * DEG_PER_RAD


class TestProjectionAgainstFiniteDifference:
    """The projected gradient must match a rigid-scan finite difference."""

    def test_matches_finite_difference_off_minimum(self, butane, mmff_calc):
        atoms, mol, dihedrals = butane
        di = dihedrals[0]

        # Move off any stationary point so the gradient is clearly nonzero.
        atoms = atoms.copy()
        atoms.set_dihedral(*di.chain, 75.0, indices=di.group)
        atoms.calc = mmff_calc

        grad = project_torsion_gradient(
            atoms.get_positions(), [di], atoms.get_forces()
        )[0]
        fd = _finite_difference_gradient(atoms, di, mmff_calc, mol)

        assert abs(grad) > 1e-3  # genuinely off a minimum
        assert grad == pytest.approx(fd, rel=2e-2, abs=1e-4)

    def test_all_dihedrals_match_finite_difference(self, butane):
        from bouquet.calc_rdkit import RDKitMMFFCalculator

        atoms, mol, dihedrals = butane
        atoms = atoms.copy()
        # Perturb every rotatable bond to a distinct, non-stationary angle.
        for offset, di in zip((40.0, 95.0, 155.0), dihedrals):
            atoms.set_dihedral(*di.chain, offset, indices=di.group)
        atoms.calc = RDKitMMFFCalculator(mol)

        grad = compute_torsion_gradient(atoms, dihedrals, atoms.calc)
        assert grad.shape == (len(dihedrals),)

        for i, di in enumerate(dihedrals):
            fd = _finite_difference_gradient(atoms, di, atoms.calc, mol)
            assert grad[i] == pytest.approx(fd, rel=3e-2, abs=1e-4)

    def test_per_degree_is_radian_scaled(self, butane, mmff_calc):
        atoms, _, dihedrals = butane
        atoms = atoms.copy()
        atoms.set_dihedral(*dihedrals[0].chain, 75.0, indices=dihedrals[0].group)
        atoms.calc = mmff_calc

        rad = compute_torsion_gradient(atoms, dihedrals, mmff_calc, per_degree=False)
        deg = compute_torsion_gradient(atoms, dihedrals, mmff_calc, per_degree=True)
        np.testing.assert_allclose(deg, rad / DEG_PER_RAD, rtol=1e-12)


class TestProjectionProperties:
    """Geometry/bookkeeping properties of the projection."""

    def test_on_axis_atoms_contribute_zero(self, butane, mmff_calc):
        # The third chain atom lies on the rotation axis; adding it to the group
        # must not change the projected gradient.
        atoms, _, dihedrals = butane
        di = dihedrals[0]
        atoms = atoms.copy()
        atoms.set_dihedral(*di.chain, 75.0, indices=di.group)
        atoms.calc = mmff_calc

        forces = atoms.get_forces()
        base = project_torsion_gradient(atoms.get_positions(), [di], forces)[0]

        import copy

        di_axis = copy.copy(di)
        di_axis.group = set(di.group) | {di.chain[1], di.chain[2]}
        with_axis = project_torsion_gradient(
            atoms.get_positions(), [di_axis], forces
        )[0]

        assert with_axis == pytest.approx(base, abs=1e-9)

    def test_degenerate_axis_raises(self, butane, mmff_calc):
        atoms, _, dihedrals = butane
        di = dihedrals[0]
        positions = atoms.get_positions().copy()
        # Collapse the two central atoms onto each other.
        positions[di.chain[2]] = positions[di.chain[1]]
        with pytest.raises(ValueError, match="Degenerate torsion axis"):
            project_torsion_gradient(positions, [di], np.zeros_like(positions))

    def test_does_not_mutate_input_atoms(self, butane, mmff_calc):
        atoms, _, dihedrals = butane
        atoms = atoms.copy()
        atoms.set_dihedral(*dihedrals[0].chain, 75.0, indices=dihedrals[0].group)
        before = atoms.get_positions().copy()

        compute_torsion_gradient(atoms, dihedrals, mmff_calc)

        np.testing.assert_array_equal(atoms.get_positions(), before)


class TestEvaluateEnergyWithGradient:
    """End-to-end behaviour of the assess-level wrapper."""

    def test_relaxed_gradient_matches_finite_difference(self, butane):
        """Envelope-theorem check: dE*/dtheta vs finite difference of the
        relaxed energy surface (both other DOF re-relaxed at each angle)."""
        from bouquet.assess import evaluate_energy
        from bouquet.calc_rdkit import RDKitMMFFCalculator

        atoms, mol, dihedrals = butane
        di = dihedrals[0]
        calc = RDKitMMFFCalculator(mol)

        angle = atoms.get_dihedral(*di.chain) + 60.0  # away from the minimum
        angles = [d.get_angle(atoms) for d in dihedrals]
        angles[0] = angle

        energy, relaxed, grad = evaluate_energy_with_gradient(
            angles, atoms, dihedrals, calc, calc, relax=True, steps=None
        )
        assert energy < RELAX_FAILURE_ENERGY_EV

        # Finite difference of the relaxed surface along dihedral 0.
        delta = 1.0
        plus = list(angles)
        plus[0] = angle + delta
        minus = list(angles)
        minus[0] = angle - delta
        e_plus, _ = evaluate_energy(
            plus, atoms, dihedrals, calc, calc, relax=True, steps=None
        )
        e_minus, _ = evaluate_energy(
            minus, atoms, dihedrals, calc, calc, relax=True, steps=None
        )
        fd_rad = (e_plus - e_minus) / (2 * delta) * DEG_PER_RAD

        assert grad[0] == pytest.approx(fd_rad, rel=0.1, abs=2e-3)

    def test_failed_evaluation_returns_nan_gradient(self, butane):
        atoms, _, dihedrals = butane

        class BoomCalc:
            def get_potential_energy(self, atoms=None):
                raise RuntimeError("calculation failed")

        angles = [d.get_angle(atoms) for d in dihedrals]
        energy, _, grad = evaluate_energy_with_gradient(
            angles, atoms, dihedrals, BoomCalc(), BoomCalc(), relax=True
        )

        assert energy >= RELAX_FAILURE_ENERGY_EV
        assert grad.shape == (len(dihedrals),)
        assert np.all(np.isnan(grad))
