"""Steric-clash feasibility for the torsion domain.

The acquisition optimizer searches the full box [0, 360)^d, but most of that box
is sterically impossible: at d=8 roughly 90% of uniformly sampled dihedral
vectors put some non-bonded pair below 0.8x the sum of its van der Waals radii,
rising to ~99% by d=12. The clash-free subset is exactly the correlated-motion
manifold -- dihedrals cannot move independently without atoms colliding -- so
restricting the acquisition to it both saves wasted evaluations and encodes the
correlation the GP would otherwise have to learn from data.

Two pieces live here:

* :class:`TorsionGeometry` -- a vectorized re-implementation of the sequential
  ``Atoms.set_dihedral`` loop in :func:`bouquet.assess.evaluate_energy`. It exists
  purely for speed: screening a few hundred candidates per BO step through ASE
  costs more than the xTB evaluation it is meant to save. It reproduces ASE's
  geometry to floating-point tolerance (see ``tests/test_clash.py``).

* :class:`ClashDetector` -- the feasibility test itself, the worst non-bonded
  contact ratio ``min_ij dist_ij / (r_i + r_j)``.

Two exclusions matter and are not optional:

* **1-4 and closer pairs.** A dihedral rotation changes 1-4 distances by
  construction; an eclipsed 1-4 contact is ordinary torsional strain, not a
  clash. Only pairs separated by >= ``MIN_PATH_LENGTH`` bonds are considered.

* **Hydrogen bonds.** A real intramolecular O-H...O contact sits near 1.3-1.4 A
  against a 2.75 A vdW sum -- a ratio near 0.50, below any useful clash cut.
  Screening those pairs would reject precisely the folded, H-bonded conformers
  that are usually the global minimum. Polar-H...(N, O, F, S) pairs are dropped.
  Measured on the 492-molecule stopbench set, the worst contact of the *relaxed
  start* geometries has a naive minimum of 0.483 (p5 = 0.622) but an
  H-bond-aware minimum of 0.761 (p5 = 0.806).

Threshold: the test runs on the RIGID geometry, before the constrained
relaxation, so it must be looser than what a relaxed conformer would show -- the
relaxation exists to clean up exactly this. Calibrated against GFN2 on 1200
uniform points over 20 molecules at d=5-12, ``DEFAULT_CLASH_RATIO`` = 0.55
rejects 62% of the box while losing 6.6% of the points that would have relaxed
to within 10 kcal/mol of the start, and keeps the global best. Tightening to
0.80 rejects 94% of the box but loses 83% of the good points and degrades the
best reachable conformer from -9.3 to -2.0 kcal/mol.
"""

from __future__ import annotations

__lazy_modules__ = ["numpy"]

from typing import TYPE_CHECKING, Sequence

import numpy as np

from bouquet.config import DEFAULT_CLASH_RATIO

if TYPE_CHECKING:
    from ase import Atoms
    from rdkit import Chem

    from bouquet.setup import DihedralInfo

# Pairs closer than this many bonds are excluded from the clash test (1-2, 1-3
# and 1-4 contacts are bonded geometry / torsional strain, not clashes).
MIN_PATH_LENGTH = 4

# Elements treated as hydrogen-bond donors/acceptors for the polar-H exclusion.
POLAR_ELEMENTS = frozenset({7, 8, 9, 16})  # N, O, F, S

# DEFAULT_CLASH_RATIO (worst permitted non-bonded contact as a fraction of the
# summed vdW radii) is defined in bouquet.config and re-exported here.

# Length (A) below which a projected dihedral reference vector counts as collapsed.
# See _dihedral_angles_batch for why this is safe to set this tight.
DEGENERATE_TOL = 1e-9


class DegenerateDihedralError(ValueError):
    """A dihedral in the chain is undefined -- its inner angle is planar.

    The counterpart of the ``ZeroDivisionError('Undefined dihedral for planar
    inner angle')`` that ``ase.geometry.get_dihedrals`` raises for the same
    geometry. Raised (rather than silently returning a wrong geometry) because a
    finite-but-meaningless structure would go on to get a real energy and enter
    the surrogate as though it were a valid conformer.
    """


class TorsionGeometry:
    """Apply a dihedral vector to a reference geometry, vectorized over candidates.

    Reproduces the sequential loop in :func:`bouquet.assess.evaluate_energy`::

        for a, di in zip(angles, dihedrals):
            atoms.set_dihedral(*di.chain, a, indices=di.group)

    ASE's ``set_dihedral`` sets an *absolute* angle: it measures the current
    dihedral and rotates ``indices`` about the ``chain[1]-chain[2]`` axis by the
    difference. Applying the dihedrals in order matters -- for nested rotations a
    later one moves atoms an earlier one already placed -- so the order and the
    absolute-angle semantics are both preserved here.
    """

    def __init__(self, atoms: Atoms, dihedrals: Sequence[DihedralInfo]):
        self.ref_positions = np.asarray(atoms.get_positions(), dtype=float)
        self.n_atoms = len(self.ref_positions)
        self.chains = [tuple(int(k) for k in di.chain) for di in dihedrals]
        # Boolean masks are what the rotation step needs; build them once.
        self.groups = []
        for di in dihedrals:
            mask = np.zeros(self.n_atoms, dtype=bool)
            mask[np.fromiter(di.group, dtype=int, count=len(di.group))] = True
            self.groups.append(mask)

    def __len__(self) -> int:
        return len(self.chains)

    def build(self, angles: np.ndarray) -> np.ndarray:
        """Positions for one dihedral vector ``angles`` (degrees), shape (n_atoms, 3).

        Raises:
            DegenerateDihedralError: if any dihedral in the chain is undefined.
        """
        return self.build_batch(np.asarray(angles, dtype=float)[None, :])[0]

    def build_batch(self, angles: np.ndarray) -> np.ndarray:
        """Positions for a batch of dihedral vectors (degrees).

        Args:
            angles: shape (n_candidates, n_dihedrals), in degrees.

        Returns:
            shape (n_candidates, n_atoms, 3).

        Raises:
            DegenerateDihedralError: if any candidate has an undefined dihedral,
                mirroring the ``ZeroDivisionError`` ASE raises for the same input.
                Use :meth:`build_batch_flagged` to screen candidates instead.
        """
        positions, degenerate = self.build_batch_flagged(angles)
        if degenerate.any():
            raise DegenerateDihedralError(
                f"Undefined dihedral for planar inner angle in "
                f"{int(degenerate.sum())} of {len(degenerate)} geometries"
            )
        return positions

    def build_batch_flagged(self, angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Positions plus a per-candidate 'this dihedral was undefined' mask.

        The non-raising counterpart of :meth:`build_batch`, for screening a pool
        where one bad candidate must not abort the rest. Flagged candidates are
        returned with the offending rotation skipped, so their positions are
        finite but meaningless -- callers must treat them as rejected, not build
        on them.

        The dihedral loop stays sequential (order is part of the semantics), but
        every candidate is rotated at once, which is what makes screening a few
        hundred candidates per BO step cost far less than one xTB evaluation.
        """
        angles = np.atleast_2d(np.asarray(angles, dtype=float))
        pos = np.repeat(self.ref_positions[None, :, :], len(angles), axis=0)
        degenerate = np.zeros(len(angles), dtype=bool)
        for k, ((a1, a2, a3, a4), mask) in enumerate(zip(self.chains, self.groups)):
            current, bad = _dihedral_angles_batch(
                pos[:, a1], pos[:, a2], pos[:, a3], pos[:, a4]
            )
            degenerate |= bad
            # Skip the rotation for degenerate candidates: there is no meaningful
            # angle to rotate to, and leaving them finite keeps the batch usable
            # for the candidates that are fine.
            delta = np.where(bad, 0.0, np.radians(angles[:, k] - current))
            axis = pos[:, a3] - pos[:, a2]
            norm = np.linalg.norm(axis, axis=1)
            axis = np.where(bad[:, None], 0.0,
                            axis / np.where(norm > 0.0, norm, 1.0)[:, None])
            pos[:, mask, :] = _rotate_about_axis_batch(
                pos[:, mask, :], pos[:, a2], axis, delta
            )
        return pos, degenerate


def _dihedral_angles_batch(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                           p3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dihedrals p0-p1-p2-p3 in degrees, and a mask of undefined ones.

    Matches ``ase.geometry.get_dihedrals``: the in-plane reference is the
    projection of ``-v0`` (i.e. p0 - p1), not ``+v0``. Projecting ``+v0`` instead
    flips both atan2 arguments and silently yields a *different* angle, so the
    set/measure pair would no longer round-trip.

    A dihedral is undefined when either projection collapses -- the inner angle
    is planar, so there is no plane to measure the torsion in. ASE tests this
    with an exact ``== 0`` comparison; ``DEGENERATE_TOL`` is the same test with
    floating-point slack. It is deliberately far below real chemistry: a 179.9
    degree inner angle still projects to ~2.6e-3 A, six orders of magnitude
    above the tolerance, so near-linear groups (nitrile, alkyne, azide) are
    measured normally rather than rejected.

    Each argument is shape (B, 3); returns ``(degrees, undefined_mask)``.
    """
    v0, v1, v2 = p1 - p0, p2 - p1, p3 - p2
    norm1 = np.linalg.norm(v1, axis=1)
    # A zero-length v1 is a coincident atom pair -- ASE divides by it and yields
    # NaN rather than raising, so guard it here instead of reproducing that.
    bad_axis = norm1 <= DEGENERATE_TOL
    v1n = v1 / np.where(bad_axis, 1.0, norm1)[:, None]

    v = -v0 - ((-v0) * v1n).sum(axis=1)[:, None] * v1n
    w = v2 - (v2 * v1n).sum(axis=1)[:, None] * v1n
    undefined = (
        bad_axis
        | (np.linalg.norm(v, axis=1) <= DEGENERATE_TOL)
        | (np.linalg.norm(w, axis=1) <= DEGENERATE_TOL)
    )
    angles = np.degrees(np.arctan2((np.cross(v1n, v) * w).sum(axis=1),
                                   (v * w).sum(axis=1))) % 360.0
    return angles, undefined


def _rotate_about_axis_batch(points: np.ndarray, origin: np.ndarray,
                             axis: np.ndarray, angle_rad: np.ndarray) -> np.ndarray:
    """Rodrigues rotation, batched over candidates.

    Args:
        points: (B, M, 3) points to rotate.
        origin: (B, 3) a point on the rotation axis.
        axis: (B, 3) unit rotation axis.
        angle_rad: (B,) rotation angle.
    """
    p = points - origin[:, None, :]
    c = np.cos(angle_rad)[:, None, None]
    s = np.sin(angle_rad)[:, None, None]
    ax = axis[:, None, :]
    dot = (p * ax).sum(axis=2)[:, :, None]
    return p * c + np.cross(ax, p) * s + ax * dot * (1.0 - c) + origin[:, None, :]


# Cap on elements in one block's (block, n_pairs, 3) difference array (~16 MB of
# float64), bounding the clash screen's memory regardless of batch size.
_SCREEN_BLOCK_ELEMENTS = 2_000_000


class ClashDetector:
    """Worst non-bonded contact ratio of a geometry, and the feasibility test.

    Args:
        mol: RDKit molecule with explicit hydrogens, atom order matching the
            geometry being tested.
        threshold: Reject a geometry whose worst ratio falls below this.
        exclude_hbonds: Drop polar-H...(N,O,F,S) pairs from the test. Leave this
            on -- see the module docstring.
    """

    def __init__(self, mol: Chem.Mol, threshold: float = DEFAULT_CLASH_RATIO,
                 exclude_hbonds: bool = True):
        from rdkit import Chem
        from rdkit.Chem import rdmolops

        self.threshold = float(threshold)
        n = mol.GetNumAtoms()
        path = rdmolops.GetDistanceMatrix(mol)
        i, j = np.triu_indices(n, k=1)
        keep = path[i, j] >= MIN_PATH_LENGTH
        i, j = i[keep], j[keep]

        z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
        if exclude_hbonds and len(i):
            polar_h = np.zeros(n, dtype=bool)
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 1:
                    nbrs = atom.GetNeighbors()
                    if nbrs and nbrs[0].GetAtomicNum() in POLAR_ELEMENTS:
                        polar_h[atom.GetIdx()] = True
            polar = np.isin(z, list(POLAR_ELEMENTS))
            hbond = (polar_h[i] & polar[j]) | (polar_h[j] & polar[i])
            i, j = i[~hbond], j[~hbond]

        self.i, self.j = i, j
        table = Chem.GetPeriodicTable()
        radii = np.array([table.GetRvdw(int(zi)) for zi in z])
        self.vdw_sums = radii[i] + radii[j] if len(i) else np.empty(0)
        self._inv_vdw_sums_sq = (
            1.0 / (self.vdw_sums * self.vdw_sums) if len(i) else np.empty(0)
        )

    @property
    def n_pairs(self) -> int:
        return len(self.i)

    def min_ratio(self, positions: np.ndarray) -> float:
        """Worst non-bonded ``dist / (r_i + r_j)``; ``inf`` when no pairs qualify.

        A molecule too small or too tightly connected to have any pair at least
        ``MIN_PATH_LENGTH`` bonds apart cannot clash by this definition, so it is
        reported as maximally feasible rather than as an error.
        """
        return float(self.min_ratio_batch(np.asarray(positions)[None, ...])[0])

    def min_ratio_batch(self, positions: np.ndarray) -> np.ndarray:
        """Batched :meth:`min_ratio`; ``positions`` is (B, n_atoms, 3) -> (B,)."""
        if self.n_pairs == 0:
            return np.full(len(positions), np.inf)
        # Process candidates in blocks so the (block, n_pairs, 3) temporaries stay
        # bounded instead of scaling with the whole batch (acquisition pools are
        # hundreds of candidates x thousands of pairs). Per-candidate minima are
        # independent, so the result is identical to the unblocked reduction.
        out = np.empty(len(positions))
        block = max(1, _SCREEN_BLOCK_ELEMENTS // (3 * self.n_pairs))
        for s in range(0, len(positions), block):
            p = positions[s:s + block]
            diff = p[:, self.i, :] - p[:, self.j, :]
            d2 = (diff * diff).sum(axis=2)
            out[s:s + block] = (d2 * self._inv_vdw_sums_sq).min(axis=1)
        return np.sqrt(out)

    def is_feasible(self, positions: np.ndarray) -> bool:
        return self.min_ratio(positions) >= self.threshold


class TorsionFeasibility:
    """Callable feasibility test over batches of dihedral vectors.

    Binds a :class:`TorsionGeometry` to a :class:`ClashDetector` so the
    acquisition optimizer can screen candidates without touching ASE or the
    calculator. Counts how many candidates it has screened and rejected, for the
    run log.
    """

    def __init__(self, geometry: TorsionGeometry, detector: ClashDetector):
        self.geometry = geometry
        self.detector = detector
        self.n_screened = 0
        self.n_rejected = 0

    def ratios(self, angles: np.ndarray) -> np.ndarray:
        """Worst-contact ratio per candidate, shape (n_candidates,).

        Candidates whose dihedrals are undefined get a ratio of 0, so they are
        rejected by any threshold. Screening must never raise: one degenerate
        candidate in a pool of hundreds should cost that candidate, not the step.
        """
        angles = np.atleast_2d(np.asarray(angles, dtype=float))
        positions, degenerate = self.geometry.build_batch_flagged(angles)
        ratios = self.detector.min_ratio_batch(positions)
        return np.where(degenerate, 0.0, ratios)

    def __call__(self, angles: np.ndarray) -> np.ndarray:
        """Feasibility mask for ``angles`` (degrees), shape (n_candidates, n_dihedrals)."""
        mask = self.ratios(angles) >= self.detector.threshold
        self.n_screened += len(mask)
        self.n_rejected += int((~mask).sum())
        return mask


def build_feasibility(
    atoms: Atoms,
    mol: Chem.Mol,
    dihedrals: Sequence[DihedralInfo],
    threshold: float = DEFAULT_CLASH_RATIO,
    exclude_hbonds: bool = True,
) -> TorsionFeasibility | None:
    """Feasibility test for ``mol``'s torsion domain, or None if it cannot clash.

    Returns None when the molecule has no non-bonded pair to test (nothing to
    constrain), so callers can skip the screen entirely rather than carry a
    predicate that always answers True.
    """
    detector = ClashDetector(mol, threshold=threshold, exclude_hbonds=exclude_hbonds)
    if detector.n_pairs == 0:
        return None
    return TorsionFeasibility(TorsionGeometry(atoms, dihedrals), detector)
