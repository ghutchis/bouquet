"""Tests for bouquet.enm (elastic-network soft modes projected into torsion space)."""

import numpy as np
import torch

from bouquet.enm import (
    _dihedral_rad,
    anm_hessian,
    dihedral_bmatrix,
    enm_dihedral_modes,
)


def _dih(pos, chain):
    return float(_dihedral_rad(torch.tensor(pos[list(chain)], dtype=torch.float64)))


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def test_bmatrix_matches_finite_difference():
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(6, 3))
    chains = [(0, 1, 2, 3), (1, 2, 3, 4)]
    B = dihedral_bmatrix(pos, chains)
    eps = 1e-6
    for i, ch in enumerate(chains):
        for a in range(6):
            for c in range(3):
                pp, pm = pos.copy(), pos.copy()
                pp[a, c] += eps
                pm[a, c] -= eps
                fd = _wrap(_dih(pp, ch) - _dih(pm, ch)) / (2 * eps)
                assert abs(B[i, 3 * a + c] - fd) < 1e-4


def test_bmatrix_degenerate_dihedral_is_finite():
    # Coincident central-bond atoms (b2 = p[2]-p[1] = 0) make b2/|b2| a 0/0 NaN that
    # propagates to the autograd gradient without the guard. dihedral_bmatrix must
    # return a finite row, not push NaN into the ANM projection.
    pos = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],   # atom 2 coincides with atom 1 -> central bond b2 = 0
        [1.0, 1.0, 0.0],
    ])
    B = dihedral_bmatrix(pos, [(0, 1, 2, 3)])
    assert np.isfinite(B).all()


def test_anm_hessian_symmetric_with_rigid_zero_modes():
    rng = np.random.default_rng(1)
    pos = rng.normal(scale=2.0, size=(12, 3))
    H = anm_hessian(pos, cutoff=6.0)
    assert np.allclose(H, H.T)
    # ANM springs are rigid-motion invariant -> >= 6 (near-)zero eigenvalues.
    vals = np.linalg.eigvalsh(H)
    assert int(np.sum(np.abs(vals) < 1e-6)) >= 6


def test_enm_dihedral_modes_shape_and_normalized():
    rng = np.random.default_rng(2)
    pos = rng.normal(scale=2.0, size=(8, 3))
    chains = [(0, 1, 2, 3), (2, 3, 4, 5), (4, 5, 6, 7)]
    V = enm_dihedral_modes(pos, chains, k=2, cutoff=6.0)
    assert V.shape[0] == len(chains)
    assert V.shape[1] <= 2
    for j in range(V.shape[1]):
        assert abs(np.linalg.norm(V[:, j]) - 1.0) < 1e-9
