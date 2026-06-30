"""Low-energy collective-coordinate subspace extraction for high-dimensional search.

This is the shared abstraction for Phase 2/3 of the HDBO plan (see
``bouquet_hdbo_plan.md``). On flexible foldamers the low-energy basin is reached only
by a *correlated* rotation of many dihedrals at once (a "fold" / soft collective
coordinate), which axis-wise acquisition never finds. :class:`LowEnergySubspace`
extracts that coordinate from the conformers already sampled and exposes it as a
reduced search space.

Design
------
It is a **selector + projector over the existing observation buffers**, not a second
copy of the data. Feed it the run's observations (degrees, eV, eV/rad -- the
``OptimizationState`` layout) and it:

1. selects a dedup'd, capacity-capped low-energy "elite" set (periodic-torsion dedup,
   so the set isn't dominated by near-duplicates of the incumbent basin);
2. recenters on the incumbent (lowest-energy elite point) in a **wrapped-Δθ tangent
   space** -- linear methods on a torus need a local chart, and Δθ keeps gradients in
   their native eV/rad units (no sin/cos chain rule);
3. builds two collective-coordinate bases in that same space, so they are directly
   comparable:

   * :meth:`pca_basis` -- the **top** eigenvectors of the (energy-weighted) position
     covariance: the directions the good conformers spread along (the valley floor);
   * :meth:`gradient_modes` -- the **bottom** (soft) eigenvectors of the
     (energy-weighted) gradient covariance ``C = Σ wᵢ gᵢgᵢᵀ`` (Constantine's active
     subspace): travelling along a soft mode barely changes the energy = the valley.

   ⚠ Soft vs stiff: the fold is a *soft* mode. ``pca_basis`` takes the TOP position
   eigenvectors but ``gradient_modes(soft=True)`` takes the BOTTOM gradient
   eigenvectors. Taking the top gradient modes would search the stiff walls -- the
   opposite of the fold. :meth:`overlap` cross-checks that the two agree.

Why energy-*weighted* covariance (not hard top-K): in high d the lifted covariance is
up to 2d×2d, so you need *many* points for a full-rank, well-conditioned estimate --
hence the large ``capacity``. But a wide set drags in high-energy "wall" motion. A soft
Boltzmann weight ``exp(-ΔE/kT)`` lets the whole set fill out the rank while the valley
floor dominates the leading components. One temperature knob instead of a brittle
window.

The class also serves Phase 3 (the informed-embedding Σ is :meth:`weighted_covariance`)
and level-set ensembles (:meth:`level_set`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# 1 kcal/mol in eV (matches bouquet.config.KCAL_TO_EV = 1/23.0605).
_KCAL_TO_EV = 1.0 / 23.0605


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap radians to (-π, π]."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class LowEnergySubspace:
    """Collective-coordinate subspace over a run's low-energy conformers.

    Args:
        n_dihedrals: Torsion dimension ``d``.
        capacity: Max size of the dedup'd elite set kept for covariance estimation.
            Want it comfortably larger than ``2*d`` so the lifted covariance is
            full-rank; the high search budget makes 200-500 reasonable.
        dedup_thresh_deg: Two conformers are duplicates if their periodic-torsion RMS
            angle difference is below this (degrees). Set to 0 to disable dedup.
        temperature_kcal: Boltzmann temperature (kcal/mol) for the covariance weights
            ``exp(-ΔE/kT)``. Smaller = sharper emphasis on the valley floor; ``None``
            uses uniform weights. ΔE is measured from the elite-set minimum.

    Populate with :meth:`update` (or :meth:`from_state`), then query
    :meth:`pca_basis` / :meth:`gradient_modes` / :meth:`level_set`. Bases are returned
    as ``(mean, V)`` with ``V`` of shape ``(ambient, k)``; round-trip a reduced point
    ``z`` (length ``k``) back to torsion degrees with :meth:`unlift`.
    """

    n_dihedrals: int
    capacity: int = 400
    dedup_thresh_deg: float = 15.0
    temperature_kcal: float | None = 5.0

    def __post_init__(self) -> None:
        self._theta: np.ndarray | None = None      # elite torsions, radians (m, d)
        self._energy: np.ndarray | None = None     # elite energies, eV (m,)
        self._grad: np.ndarray | None = None        # elite gradients, eV/rad (m, d) or NaN
        self._incumbent: np.ndarray | None = None   # radians (d,)
        self._tangent: np.ndarray | None = None     # wrapped Δθ about incumbent (m, d)

    # ---- ingestion -----------------------------------------------------------
    @classmethod
    def from_state(cls, state, **kwargs) -> LowEnergySubspace:
        """Build directly from an ``OptimizationState`` (duck-typed: needs
        ``observed_coords`` [deg], ``observed_energies`` [eV], and optionally
        ``observed_gradients`` [eV/rad])."""
        coords = np.asarray(state.observed_coords.detach().cpu().numpy(), dtype=float)
        energies = np.asarray(state.observed_energies.detach().cpu().numpy(), dtype=float)
        grads = None
        if getattr(state, "observed_gradients", None) is not None:
            grads = np.asarray(state.observed_gradients.detach().cpu().numpy(), dtype=float)
        obj = cls(n_dihedrals=coords.shape[1], **kwargs)
        obj.update(coords, energies, grads)
        return obj

    def update(
        self,
        coords_deg: np.ndarray,
        energies_eV: np.ndarray,
        gradients: np.ndarray | None = None,
    ) -> LowEnergySubspace:
        """Select the dedup'd low-energy elite set and recenter on the incumbent.

        ``coords_deg`` (n, d) degrees, ``energies_eV`` (n,) relative eV, ``gradients``
        (n, d) eV/rad with NaN where unavailable. Drops failed/sentinel evaluations
        (non-finite energy) before selecting.
        """
        coords_deg = np.asarray(coords_deg, dtype=float)
        energies_eV = np.asarray(energies_eV, dtype=float)
        finite = np.isfinite(energies_eV)
        # Guard against the ~1000 eV failure sentinel poisoning the selection.
        finite &= energies_eV < 100.0
        coords_deg, energies_eV = coords_deg[finite], energies_eV[finite]
        grads = None
        if gradients is not None:
            grads = np.asarray(gradients, dtype=float)[finite]

        if coords_deg.shape[0] == 0:
            self._theta = self._energy = self._grad = None
            self._incumbent = self._tangent = None
            return self

        theta = np.deg2rad(coords_deg)
        keep = self._select_elite(theta, energies_eV)
        self._theta = theta[keep]
        self._energy = energies_eV[keep]
        self._grad = grads[keep] if grads is not None else None
        # Incumbent = lowest-energy elite point; tangent = wrapped Δθ about it.
        self._incumbent = self._theta[np.argmin(self._energy)].copy()
        self._tangent = _wrap_to_pi(self._theta - self._incumbent)
        return self

    def _select_elite(self, theta: np.ndarray, energy: np.ndarray) -> np.ndarray:
        """Greedy best-first selection: lowest energy first, skipping any conformer
        within ``dedup_thresh_deg`` (periodic-torsion RMS) of one already kept; stop
        at ``capacity``. Returns indices into the input arrays."""
        order = np.argsort(energy)
        if self.dedup_thresh_deg <= 0:
            return order[: self.capacity]
        thresh = np.deg2rad(self.dedup_thresh_deg)
        kept: list[int] = []
        kept_theta: list[np.ndarray] = []
        for i in order:
            ti = theta[i]
            if kept_theta:
                d = _wrap_to_pi(np.asarray(kept_theta) - ti)  # (len, d)
                rms = np.sqrt((d ** 2).mean(axis=1))
                if rms.min() < thresh:
                    continue
            kept.append(int(i))
            kept_theta.append(ti)
            if len(kept) >= self.capacity:
                break
        return np.asarray(kept, dtype=int)

    # ---- weights -------------------------------------------------------------
    @property
    def n_elite(self) -> int:
        return 0 if self._energy is None else self._energy.shape[0]

    @property
    def incumbent_deg(self) -> np.ndarray:
        """The lowest-energy elite conformer (torsion degrees, [0, 360))."""
        self._require_elite(1)
        return np.rad2deg(self._incumbent) % 360.0

    def line(self, direction: np.ndarray, alphas: np.ndarray) -> np.ndarray:
        """Points along the incumbent-anchored line ``incumbent + α·direction`` in the
        tangent space, returned as torsion degrees ``(len(alphas), d)``, wrapped to
        [0, 360). ``direction`` is a tangent (radian) vector, e.g. a column of
        :meth:`pca_basis` or :meth:`gradient_modes`. The line-restricted acquisition
        search of Phase 2c scans LogEI over these points."""
        self._require_elite(1)
        theta = self._incumbent[None, :] + np.asarray(alphas)[:, None] * np.asarray(direction)[None, :]
        return np.rad2deg(theta) % 360.0

    def _weights(self) -> np.ndarray:
        """Boltzmann (or uniform) weights over the elite set, summing to 1."""
        e = self._energy
        if self.temperature_kcal is None:
            w = np.ones_like(e)
        else:
            kT = self.temperature_kcal * _KCAL_TO_EV
            w = np.exp(-(e - e.min()) / max(kT, 1e-12))
        return w / w.sum()

    # ---- collective-coordinate bases -----------------------------------------
    def weighted_covariance(self, space: str = "tangent") -> tuple[np.ndarray, np.ndarray]:
        """Energy-weighted position covariance and weighted mean.

        ``space="tangent"`` (d-dim, wrapped Δθ about the incumbent) or ``"sincos"``
        (2d-dim, the global periodicity-safe lift). Returns ``(mean, cov)`` with mean
        of length ``ambient`` and cov ``(ambient, ambient)``.
        """
        X = self._ambient(space)            # (m, ambient)
        w = self._weights()
        mean = w @ X
        Xc = X - mean
        cov = (Xc * w[:, None]).T @ Xc      # weighted, biased
        return mean, cov

    def pca_basis(
        self, k: int, space: str = "tangent"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-``k`` energy-weighted PCA directions of the low-energy set.

        Returns ``(mean, V)`` with ``V`` of shape ``(ambient, k)`` whose columns are
        the leading eigenvectors of :meth:`weighted_covariance` (descending). These
        span the valley floor; search in their span and :meth:`unlift` to evaluate.
        """
        self._require_elite(k)
        mean, cov = self.weighted_covariance(space)
        vals, vecs = np.linalg.eigh(cov)            # ascending
        V = vecs[:, ::-1][:, :k]                     # top-k (descending)
        return mean, V

    def gradient_modes(self, k: int, soft: bool = True) -> np.ndarray:
        """``k`` eigenvectors of the energy-weighted gradient covariance
        ``C = Σ wᵢ gᵢgᵢᵀ`` in the tangent space (always tangent: gradients are
        native eV/rad there). ``soft=True`` returns the **smallest**-eigenvalue
        directions -- the fold/valley coordinate; ``soft=False`` the active (stiff)
        directions. Returns ``V`` of shape ``(d, k)``.

        Raises if no gradients were provided. Rows with NaN gradients are dropped
        (they still counted toward the value-based PCA).
        """
        self._require_elite(k)
        if self._grad is None:
            raise ValueError("gradient_modes requires gradients; none were provided.")
        g = self._grad
        valid = np.isfinite(g).all(axis=1)
        if valid.sum() < k:
            raise ValueError(
                f"only {int(valid.sum())} valid gradients for {k} modes."
            )
        g = g[valid]
        w = self._weights()[valid]
        w = w / w.sum()
        C = (g * w[:, None]).T @ g                   # (d, d), uncentered active subspace
        vals, vecs = np.linalg.eigh(C)               # ascending = soft first
        return vecs[:, :k] if soft else vecs[:, ::-1][:, :k]

    def overlap(self, k: int) -> float:
        """Subspace agreement in [0, 1] between the position-PCA top-``k`` and the
        gradient soft-``k`` (both tangent). 1.0 = identical subspaces; low values flag
        multiple basins or too few low-energy points -- investigate before trusting
        either basis. Requires gradients."""
        _, Vp = self.pca_basis(k, space="tangent")
        Vg = self.gradient_modes(k, soft=True)
        # ||Vpᵀ Vg||_F² / k : mean squared cosine of the principal angles.
        M = Vp.T @ Vg
        return float((M ** 2).sum() / k)

    # ---- level set / coordinate transforms -----------------------------------
    def level_set(
        self, window_kcal: float | None = None, max_k: int | None = None
    ) -> np.ndarray:
        """Indices (into the elite set, ascending energy) of conformers within
        ``window_kcal`` of the incumbent, capped at ``max_k``. Feeds level-set
        ensembles and the informed-embedding covariance. ``window_kcal=None`` returns
        the whole elite set."""
        self._require_elite(1)
        order = np.argsort(self._energy)
        if window_kcal is not None:
            emin = self._energy.min()
            order = order[self._energy[order] - emin <= window_kcal * _KCAL_TO_EV]
        return order[:max_k] if max_k is not None else order

    def unlift(self, z: np.ndarray, mean: np.ndarray, V: np.ndarray, space: str = "tangent") -> np.ndarray:
        """Map a reduced-space point ``z`` (length k) back to torsion angles (degrees,
        [0, 360)). ``mean``/``V`` come from :meth:`pca_basis` (use the matching
        ``space``). Tangent: ``θ = incumbent + (mean + V z)``. Sincos: recover each
        angle with ``atan2`` from its (cos, sin) pair."""
        u = mean + V @ np.asarray(z, dtype=float)
        if space == "tangent":
            theta = self._incumbent + u
        elif space == "sincos":
            d = self.n_dihedrals
            theta = np.arctan2(u[d:], u[:d])
        else:
            raise ValueError(f"space must be 'tangent' or 'sincos', got {space!r}")
        return np.rad2deg(theta) % 360.0

    # ---- internals -----------------------------------------------------------
    def _ambient(self, space: str) -> np.ndarray:
        """Elite points in the requested ambient representation (m, ambient)."""
        if space == "tangent":
            return self._tangent
        if space == "sincos":
            return np.concatenate([np.cos(self._theta), np.sin(self._theta)], axis=1)
        raise ValueError(f"space must be 'tangent' or 'sincos', got {space!r}")

    def _require_elite(self, k: int) -> None:
        if self.n_elite == 0:
            raise ValueError("no elite points; call update() / from_state() first.")
        if k > self.n_elite:
            raise ValueError(f"requested {k} modes but only {self.n_elite} elite points.")
