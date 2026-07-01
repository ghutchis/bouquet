"""Elastic-network-model (ANM) soft modes projected into torsion space.

Data-*independent* kick directions for the low-mode conformer search (Phase 2.5). The
fold diagnostics (D1) found that data-derived modes (PCA of low-energy positions, the
gradient-covariance active subspace) miss the fold direction because the fold basin was
never sampled. The Anisotropic Network Model reads the global bend/compaction motions
off the molecular geometry + connectivity alone, so its softest modes should capture the
fold direction regardless of what the search has sampled.

Pipeline: build the ANM Hessian at a Cartesian geometry, take its softest *non-trivial*
modes (the global collective motions, after the 6 rigid translation/rotation zero
modes), and project each into torsion space with the Wilson B-matrix ``dtheta/dx`` so it
can drive a dihedral kick. The B-matrix is obtained by autograd of the analytic dihedral
(exact, and it handles the ``atan2`` branch correctly).
"""

from __future__ import annotations

import numpy as np
import torch


def _dihedral_rad(p: torch.Tensor) -> torch.Tensor:
    """Dihedral angle (radians) of 4 points ``p`` (4, 3), standard i-j-k-l convention."""
    b1, b2, b3 = p[1] - p[0], p[2] - p[1], p[3] - p[2]
    n1 = torch.linalg.cross(b1, b2)
    n2 = torch.linalg.cross(b2, b3)
    m1 = torch.linalg.cross(n1, b2 / b2.norm())
    return torch.atan2((m1 * n2).sum(), (n1 * n2).sum())


def dihedral_bmatrix(positions: np.ndarray, chains) -> np.ndarray:
    """Wilson B-matrix ``dtheta_i/dx`` of shape (d, 3N) (rad/Angstrom), via autograd.

    Sparse: only the 12 coordinates of each dihedral's 4 atoms are nonzero. ``chains``
    is a list of 4-tuples (atom indices into ``positions``)."""
    n_atoms = positions.shape[0]
    B = np.zeros((len(chains), 3 * n_atoms))
    for i, chain in enumerate(chains):
        idx = list(chain)
        p = torch.tensor(positions[idx], dtype=torch.float64, requires_grad=True)
        _dihedral_rad(p).backward()
        g = p.grad.numpy()  # (4, 3)
        for j, at in enumerate(idx):
            B[i, 3 * at:3 * at + 3] = g[j]
    return B


def anm_hessian(positions: np.ndarray, cutoff: float = 10.0, gamma: float = 1.0) -> np.ndarray:
    """Anisotropic Network Model Hessian (3N, 3N): a spring of constant ``gamma``
    between every atom pair within ``cutoff`` Angstrom.

    Vectorized over atom pairs. The off-diagonal super-block for pair (i, j) is
    ``-gamma (u u^T)`` with ``u`` the unit separation, and each diagonal super-block
    is minus the sum of its row's off-diagonal blocks (rigid-motion invariance)."""
    n = positions.shape[0]
    diff = positions[:, None, :] - positions[None, :, :]     # (n, n, 3)
    d2 = (diff * diff).sum(-1)                                # (n, n)
    within = (d2 > 1e-6) & (d2 < cutoff * cutoff)
    inv_d2 = np.where(within, gamma / np.where(within, d2, 1.0), 0.0)  # (n, n)
    # off-diagonal super-blocks -gamma u u^T = -(gamma/d2) diff diff^T   -> (n, n, 3, 3)
    blocks = -inv_d2[:, :, None, None] * (diff[:, :, :, None] * diff[:, :, None, :])
    idx = np.arange(n)
    blocks[idx, idx] = -blocks.sum(axis=1)                    # diagonal = -sum of row
    return blocks.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)


def enm_dihedral_modes(
    positions: np.ndarray, chains, k: int, cutoff: float = 10.0, n_skip: int = 6
) -> np.ndarray:
    """The ``k`` softest non-trivial ANM modes projected into torsion space.

    Returns a (d, k) array whose columns are dihedral-space directions (radians),
    each normalized. ``n_skip`` drops the 6 rigid translation/rotation zero modes.
    A column can be near-zero if a global Cartesian mode induces little dihedral
    motion (rare for the soft modes); such columns are dropped, so the result may
    have fewer than ``k`` columns."""
    H = anm_hessian(positions, cutoff)
    _vals, vecs = np.linalg.eigh(H)  # ascending eigenvalues
    B = dihedral_bmatrix(positions, chains)  # (d, 3N)
    cols = []
    for m in range(n_skip, min(n_skip + k, vecs.shape[1])):
        dtheta = B @ vecs[:, m]
        nrm = np.linalg.norm(dtheta)
        if nrm > 1e-9:
            cols.append(dtheta / nrm)
    return np.array(cols).T if cols else np.zeros((len(chains), 0))
