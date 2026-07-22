"""Tests for bouquet.rings.bias.RMSDBias (§11.1 of RING_MTD_v2).

The finite-difference gradient check is the single most important test in the module: it
validates the envelope-theorem claim that the Kabsch rotation can be treated as constant
when differentiating RMSD^2. Everything downstream depends on it.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from bouquet.rings.bias import KCAL_TO_EV, RMSDBias, rmsd_sq_and_grad


def _atoms(positions):
    # Element identity is irrelevant to the bias (it acts on coordinates only).
    return Atoms("X" * len(positions), positions=positions)


def _biased(positions, groups, refs_per_group, alpha=6.0, k=0.5 * KCAL_TO_EV, max_refs=25):
    atoms = _atoms(positions)
    bias = RMSDBias(groups, k=k, alpha=alpha, max_refs=max_refs)
    for ref_positions in refs_per_group:
        bias.deposit(_atoms(ref_positions))
    atoms.calc = bias
    return atoms, bias


@pytest.fixture
def rng():
    return np.random.default_rng(20260722)


def test_fd_gradient_single_group(rng):
    """Analytic forces match central finite differences of the energy (the key test)."""
    n = 8
    X = rng.standard_normal((n, 3))
    refs = [X + 0.4 * rng.standard_normal((n, 3)) for _ in range(3)]  # RMSD>0, R smooth
    atoms, _ = _biased(X, [np.arange(n)], refs, alpha=5.0)

    F = atoms.get_forces()
    h = 1e-5
    F_fd = np.zeros_like(X)
    pos = atoms.get_positions()
    for i in range(n):
        for d in range(3):
            p = pos.copy(); p[i, d] += h
            atoms.set_positions(p); e_plus = atoms.get_potential_energy()
            p = pos.copy(); p[i, d] -= h
            atoms.set_positions(p); e_minus = atoms.get_potential_energy()
            F_fd[i, d] = -(e_plus - e_minus) / (2 * h)
        atoms.set_positions(pos)
    assert np.max(np.abs(F - F_fd)) < 1e-6


def test_fd_gradient_two_groups(rng):
    """Envelope-theorem gradient holds with multiple independent bias groups + per-group alpha."""
    pos = rng.standard_normal((10, 3))
    g0, g1 = np.array([0, 1, 2, 3, 4]), np.array([5, 6, 7, 8, 9])
    refs = [pos + 0.3 * rng.standard_normal((10, 3)) for _ in range(2)]
    atoms = _atoms(pos)
    bias = RMSDBias([g0, g1], alpha=[4.0, 9.0])
    for r in refs:
        bias.deposit(_atoms(r))
    atoms.calc = bias

    F = atoms.get_forces()
    h = 1e-5
    base = atoms.get_positions()
    for i in range(len(pos)):
        for d in range(3):
            p = base.copy(); p[i, d] += h
            atoms.set_positions(p); ep = atoms.get_potential_energy()
            p = base.copy(); p[i, d] -= h
            atoms.set_positions(p); em = atoms.get_potential_energy()
            assert abs(F[i, d] - (-(ep - em) / (2 * h))) < 1e-6
        atoms.set_positions(base)


def test_translational_invariance(rng):
    X = rng.standard_normal((7, 3))
    refs = [X + 0.3 * rng.standard_normal((7, 3)) for _ in range(2)]
    atoms, _ = _biased(X, [np.arange(7)], refs)
    e0 = atoms.get_potential_energy()
    atoms.set_positions(atoms.get_positions() + np.array([3.1, -2.7, 5.0]))
    assert abs(atoms.get_potential_energy() - e0) < 1e-10


def test_rotational_invariance(rng):
    from scipy.spatial.transform import Rotation
    X = rng.standard_normal((7, 3))
    refs = [X + 0.3 * rng.standard_normal((7, 3)) for _ in range(2)]
    atoms, _ = _biased(X, [np.arange(7)], refs)
    e0 = atoms.get_potential_energy()
    R = Rotation.random(random_state=1).as_matrix()
    atoms.set_positions(atoms.get_positions() @ R.T)
    assert abs(atoms.get_potential_energy() - e0) < 1e-10


def test_net_force_and_torque_zero(rng):
    """A pure internal bias exerts no net force or torque on its group."""
    X = rng.standard_normal((9, 3))
    refs = [X + 0.5 * rng.standard_normal((9, 3)) for _ in range(4)]
    idx = np.arange(9)
    atoms, _ = _biased(X, [idx], refs, alpha=7.0)
    F = atoms.get_forces()
    assert np.linalg.norm(F.sum(axis=0)) < 1e-9
    Xc = atoms.get_positions() - atoms.get_positions().mean(axis=0)
    torque = np.cross(Xc, F).sum(axis=0)
    assert np.linalg.norm(torque) < 1e-9


def test_self_reference_energy_is_k(rng):
    """A reference equal to the current geometry contributes exactly k, with zero force."""
    X = rng.standard_normal((6, 3))
    k = 0.5 * KCAL_TO_EV
    atoms, _ = _biased(X, [np.arange(6)], [X.copy()], alpha=6.0, k=k)
    assert abs(atoms.get_potential_energy() - k) < 1e-12
    assert np.max(np.abs(atoms.get_forces())) < 1e-9


def test_reflection_is_not_zero_rmsd(rng):
    """Mirror image gives RMSD>0 -- proper-rotation Kabsch must not superimpose enantiomers.

    Catches a missing det-sign (d) correction, which would let the SVD return an improper
    rotation and make a structure look identical to its own mirror image.
    """
    X = rng.standard_normal((8, 3))
    mirror = X.copy(); mirror[:, 0] *= -1.0  # reflect through the yz-plane
    rmsd2, _ = rmsd_sq_and_grad(X, mirror)
    assert np.sqrt(rmsd2) > 0.1


def test_multiple_hills_accumulate(rng):
    """Energy at a reference geometry grows with the number of deposited hills there."""
    X = rng.standard_normal((6, 3))
    k = 0.5 * KCAL_TO_EV
    atoms, bias = _biased(X, [np.arange(6)], [X.copy()], k=k)
    e1 = atoms.get_potential_energy()
    bias.deposit(_atoms(X)); atoms.calc.results.clear()
    e2 = atoms.get_potential_energy()
    assert abs(e1 - k) < 1e-12
    assert abs(e2 - 2 * k) < 1e-12


def test_ring_buffer_caps_references(rng):
    X = rng.standard_normal((5, 3))
    _, bias = _biased(X, [np.arange(5)], [X + 0.1 * rng.standard_normal((5, 3))
                                          for _ in range(40)], max_refs=25)
    assert bias.n_refs() == [25]
