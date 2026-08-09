"""An ASE calculator implementing a multi-group RMSD bias potential.

This is the barrier-crossing engine for ring-conformer metadynamics: it deposits
reference geometries of one or more atom groups and adds a sum-of-Gaussians penalty on
the RMSD to each reference, pushing the dynamics out of already-visited basins (the
mechanism CREST's iMTD-GC uses, narrowed here to ring atoms). It is deliberately
standalone -- it knows nothing about rings -- so it is reusable for any RMSD-biased
sampling.

For a group with bias atoms ``X`` (n x 3) and stored references ``{Y_j}``::

    V(X) = sum_j  k * exp(-alpha * RMSD^2(X, Y_j))

using ``RMSD^2`` (not ``RMSD``) so the gradient is smooth at ``RMSD -> 0`` -- exactly
where the system sits right after a reference is deposited. ``RMSD`` is Kabsch-aligned
with the proper-rotation (det = +1) correction, so a conformer is never accidentally
superimposed on its own mirror image.

Gradient (no backprop through the SVD): the Kabsch rotation ``R`` is stationary at the
optimum, so by the envelope theorem ``d RMSD^2 / dx_i = (2/n)(x_i - R^T y_i)`` (centered
coordinates). Two invariants follow and are asserted in the tests: the net bias force
and net bias torque on each group are exactly zero.

Compose with a real potential via ``ase.calculators.mixing.SumCalculator([engine, bias])``.
Units are ASE-native (eV, Angstrom): ``k`` in eV, ``alpha`` in Angstrom^-2.
"""

from __future__ import annotations

import logging

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

logger = logging.getLogger(__name__)

# Below this ratio of smallest/largest singular value the optimal rotation is not unique
# (planar or degenerate bias set); fall back to the identity rotation rather than emit NaNs.
_DEGENERATE_SV_RATIO = 1e-6

KCAL_TO_EV = 0.0433641153  # 1 kcal/mol in eV


def kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Proper rotation ``R`` (det = +1) minimizing ``sum_i || P_i R^T - Q_i ||^2``.

    ``P`` and ``Q`` are (n, 3) and assumed already centered on their centroids. Returns
    ``(R, singular_values)``; the caller uses the singular values for the degeneracy
    guard. On a degenerate/planar set (rotation non-unique) returns the identity.
    """
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    if S[0] < 1e-12 or S[-1] / S[0] < _DEGENERATE_SV_RATIO:
        logger.debug("RMSDBias: degenerate/planar bias set; using identity rotation")
        return np.eye(3), S
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, S


def rmsd_sq_and_grad(X: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray]:
    """Kabsch ``RMSD^2`` between (n,3) sets and its gradient w.r.t. ``X``.

    Both are centered internally. The gradient uses the envelope theorem (``R`` treated
    as constant at the Kabsch optimum): ``d/dX = (2/n)(Xc - Yc @ R)``. The centering
    Jacobian contributes nothing (its row-sum is zero), so this is also the gradient
    w.r.t. the uncentered ``X``.
    """
    n = len(X)
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    R, _ = kabsch_rotation(Xc, Yc)
    diff = Xc @ R.T - Yc
    rmsd2 = float((diff * diff).sum() / n)
    grad = (2.0 / n) * (Xc - Yc @ R)
    return rmsd2, grad


class RMSDBias(Calculator):
    """Sum-of-Gaussians-on-RMSD bias over one or more atom groups.

    Args:
        groups: list of atom-index arrays, one per bias group (e.g. one ring system's
            ``bias_idx``). Indices are into the full ``Atoms`` object.
        k: Gaussian height per deposited hill, in eV (default 0.5 kcal/mol).
        alpha: Gaussian width in Angstrom^-2, scalar (shared) or one per group.
        max_refs: ring-buffer length per group; oldest references drop once full.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, groups, k: float = 0.5 * KCAL_TO_EV,
                 alpha=8.0, max_refs: int = 25, **kwargs):
        super().__init__(**kwargs)
        self.groups = [np.asarray(g, dtype=int) for g in groups]
        if np.isscalar(alpha):
            self.alpha = np.full(len(self.groups), float(alpha))
        else:
            self.alpha = np.asarray(alpha, dtype=float)
            if len(self.alpha) != len(self.groups):
                raise ValueError("alpha must be a scalar or one value per group")
        self.k = float(k)
        self.max_refs = int(max_refs)
        # Per-group reference lists; each entry is the (n_g, 3) bias-atom geometry.
        self.refs: list[list[np.ndarray]] = [[] for _ in self.groups]

    def deposit(self, atoms) -> None:
        """Append the current geometry of every group to its reference ring buffer."""
        pos = atoms.get_positions()
        for gi, idx in enumerate(self.groups):
            self.refs[gi].append(pos[idx].copy())
            if len(self.refs[gi]) > self.max_refs:
                self.refs[gi].pop(0)

    def n_refs(self) -> list[int]:
        """Number of stored references per group (for diagnostics/provenance)."""
        return [len(r) for r in self.refs]

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes) -> None:
        super().calculate(atoms, properties, system_changes)
        pos = self.atoms.get_positions()
        energy = 0.0
        forces = np.zeros_like(pos)
        for gi, idx in enumerate(self.groups):
            refs = self.refs[gi]
            if not refs:
                continue
            X = pos[idx]
            a = self.alpha[gi]
            dV_dX = np.zeros_like(X)  # dV/dX on the group's bias atoms
            for Y in refs:
                rmsd2, grad = rmsd_sq_and_grad(X, Y)
                w = self.k * np.exp(-a * rmsd2)
                energy += w
                dV_dX += (-a * w) * grad     # d/dX [ k exp(-a RMSD^2) ] = -a w * dRMSD2/dX
            forces[idx] += -dV_dX            # F = -dV/dX
        self.results["energy"] = energy
        self.results["forces"] = forces
