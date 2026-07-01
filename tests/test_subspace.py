"""Tests for bouquet.subspace.LowEnergySubspace (Phase 2/3 collective-coordinate
extraction). Uses a synthetic quadratic valley whose soft direction is known."""

import numpy as np
import pytest

from bouquet.subspace import LowEnergySubspace, _wrap_to_pi


def _valley(seed=0, d=10, m=300, ksoft=1.0, kstiff=100.0):
    """A quadratic bowl with one soft (valley) direction v and stiff walls across it.
    Elite points spread far along v, slightly across; returns (coords_deg, E, grad, v,
    incumbent_deg)."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(W)
    v, across = Q[:, 0], Q[:, 1:]
    H = ksoft * np.outer(v, v) + kstiff * (np.eye(d) - np.outer(v, v))
    t = rng.normal(0, 1.0, size=m)             # large spread along the valley
    s = rng.normal(0, 0.1, size=(m, d - 1))    # small spread up the walls
    dth = t[:, None] * v + s @ across.T
    E = 0.5 * np.einsum("ij,jk,ik->i", dth, H, dth)
    g = dth @ H
    inc = np.linspace(20, 320, d)
    coords = np.rad2deg(np.deg2rad(inc) + dth) % 360
    return coords, E, g, v, inc


def _cos(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))


def test_pca_recovers_valley():
    coords, E, g, v, _ = _valley()
    ss = LowEnergySubspace(10, dedup_thresh_deg=0.0, temperature_kcal=None)
    ss.update(coords, E, g)
    _, Vp = ss.pca_basis(1)
    assert _cos(Vp, v) > 0.99


def test_gradient_soft_mode_is_valley_stiff_is_orthogonal():
    coords, E, g, v, _ = _valley()
    ss = LowEnergySubspace(10, dedup_thresh_deg=0.0, temperature_kcal=None)
    ss.update(coords, E, g)
    assert _cos(ss.gradient_modes(1, soft=True), v) > 0.99   # valley = soft
    assert _cos(ss.gradient_modes(1, soft=False), v) < 0.1   # stiff walls orthogonal


def test_overlap_high_when_consistent():
    coords, E, g, _, _ = _valley()
    ss = LowEnergySubspace(10, dedup_thresh_deg=0.0, temperature_kcal=None)
    ss.update(coords, E, g)
    assert ss.overlap(1) > 0.95


def test_unlift_roundtrip_exact():
    coords, E, g, _, _ = _valley(d=6, m=200)
    ss = LowEnergySubspace(6, dedup_thresh_deg=0.0, temperature_kcal=None)
    ss.update(coords, E, g)
    mean, V = ss.pca_basis(2)
    z = np.array([0.7, -0.4])
    tan = _wrap_to_pi(np.deg2rad(ss.unlift(z, mean, V)) - ss._incumbent)
    assert np.allclose(tan, mean + V @ z, atol=1e-10)


def test_dedup_collapses_duplicates():
    # Three clusters of near-duplicates -> dedup should keep ~3 points.
    base = np.array([[10.0, 200.0], [100.0, 30.0], [300.0, 150.0]])
    rng = np.random.default_rng(0)
    coords = np.repeat(base, 50, axis=0) + rng.normal(0, 1.0, size=(150, 2))
    E = rng.uniform(0, 0.01, size=150)
    ss = LowEnergySubspace(2, dedup_thresh_deg=15.0, temperature_kcal=None)
    ss.update(coords % 360, E)
    assert ss.n_elite == 3


def test_level_set_window():
    coords, E, g, _, _ = _valley()
    ss = LowEnergySubspace(10, dedup_thresh_deg=0.0, temperature_kcal=None)
    ss.update(coords, E, g)
    idx = ss.level_set(window_kcal=0.0)        # only the incumbent-energy point(s)
    assert ss._energy[idx].max() - ss._energy.min() <= 1e-9


def test_failed_evaluations_dropped():
    coords, E, g, v, _ = _valley(m=100)
    E = E.copy()
    E[::10] = 1000.0  # failure sentinels
    ss = LowEnergySubspace(10, dedup_thresh_deg=0.0, temperature_kcal=None)
    ss.update(coords, E, g)
    assert ss.n_elite == 90
    assert _cos(ss.pca_basis(1)[1], v) > 0.99


def test_gradient_modes_requires_gradients():
    coords, E, _, _, _ = _valley()
    ss = LowEnergySubspace(10, dedup_thresh_deg=0.0, temperature_kcal=None)
    ss.update(coords, E, None)
    with pytest.raises(ValueError):
        ss.gradient_modes(1)
