"""State-free building blocks for ensemble exploration and harvesting.

These helpers do not touch :class:`bouquet.solver.OptimizationState`, so they live
apart from the exploration/selection/harvest *orchestration* (which does, and stays
in :mod:`bouquet.solver`):

* the exploration acquisitions -- ``_LevelSetAcquisition`` / ``_BoundaryAcquisition``
  on the minimization-sense selection GP -- plus the torsion-space diversity
  penalization and per-step schedule (``_exploration_plan``, ``_resolve_ensemble_budget``);
* the harvest primitives -- rotation/permutation-invariant ``_rmsd``, greedy
  ``_dedup``, and ``_boltzmann_weights``.
"""

from __future__ import annotations

import numpy as np
import torch
from ase import Atoms
from ase.build import minimize_rotation_and_translation
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform

from bouquet.config import (
    ENSEMBLE_BASIN_DEG,
    ENSEMBLE_BOUNDARY_HI_KCAL,
    ENSEMBLE_BOUNDARY_KAPPA,
    ENSEMBLE_BOUNDARY_LO_KCAL,
    KB_EV_PER_K,
    KCAL_TO_EV,
)

# iRMSD (rotation- and permutation-invariant RMSD) is an optional dependency:
# it ships binary wheels only for some platforms, so we use it when a real
# install is present and otherwise fall back to a Kabsch-aligned RMSD. The
# hasattr guard also rejects the empty PyPI placeholder package.
try:
    import irmsd as _irmsd

    _HAVE_IRMSD = hasattr(_irmsd, "get_irmsd_ase")
except ImportError:  # pragma: no cover - exercised only without irmsd installed
    _irmsd = None
    _HAVE_IRMSD = False


# --------------------------------------------------------------------------- #
# Exploration: torsion-space distance, diversity, and acquisitions
# --------------------------------------------------------------------------- #


def _periodic_min_dist(
    x_deg: torch.Tensor, minima_deg: torch.Tensor
) -> torch.Tensor:
    """Wrapped RMS angular distance (degrees) from ``x_deg`` to its nearest basin.

    ``x_deg`` has shape ``(..., d)`` and ``minima_deg`` shape ``(m, d)``; the
    per-dihedral difference is wrapped into ``[0, 180]`` (torsions are periodic on
    360 deg) before the RMS over dihedrals and the min over the ``m`` basins.
    Returns shape ``(...)``. Requires ``m >= 1``.
    """
    diff = (x_deg[..., None, :] - minima_deg) % 360.0
    d = torch.minimum(diff, 360.0 - diff)  # wrap onto [0, 180]
    return torch.sqrt((d**2).mean(-1)).min(-1).values  # (...)


def _diversity_factor(
    X: torch.Tensor, minima_deg: torch.Tensor, diversity_lambda: float
) -> torch.Tensor:
    """Multiplicative local-penalization factor in ``[0, 1]``.

    ``~ 1 - lambda`` at a known basin, ``-> 1`` far from every basin (Gaussian in
    the wrapped RMS angular distance, width ``ENSEMBLE_BASIN_DEG``). ``X`` is the
    ``(b, 1, d)`` normalized-``[0, 1]`` acquisition input; returns ``(b,)``.
    """
    x_deg = X.squeeze(-2) * 360.0  # (b, d)
    dist = _periodic_min_dist(x_deg, minima_deg)  # (b,) degrees
    proximity = torch.exp(-((dist / ENSEMBLE_BASIN_DEG) ** 2))  # (b,)
    return (1.0 - diversity_lambda * proximity).clamp_min(0.0)  # (b,)


class _DiversityAcquisition(AnalyticAcquisitionFunction):
    """Base for the exploration acquisitions on the minimization-sense selection
    GP (``_build_selection_gp``: posterior mean is relative energy in eV, lower =
    better). Owns the shared posterior mean/sigma extraction and the optional
    torsion-space diversity *multiplicative local penalization*: a term is scaled
    by ``1 - lambda * proximity(x)`` where ``proximity(x) = exp(-(min_dist /
    ENSEMBLE_BASIN_DEG)^2)`` is ~1 at a known basin and decays to 0 far away
    (``min_dist`` = wrapped RMS angular distance to the nearest known basin). It
    is scale-free (multiplies rather than competing with the acquisition's units)
    and discourages *re-sampling* known basins rather than penalizing exploration.
    ``lambda`` in ``(0, 1]`` sets the suppression depth (0.5 halves it, 1 zeroes).
    """

    def _init_diversity(
        self, minima_deg: torch.Tensor | None, diversity_lambda: float
    ) -> None:
        self.diversity_lambda = float(diversity_lambda)
        self._penalize = (
            minima_deg is not None
            and minima_deg.shape[0] > 0
            and self.diversity_lambda > 0
        )
        if self._penalize:
            self.register_buffer("minima_deg", minima_deg)

    def _mean_sigma(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Posterior mean and sigma (eV) at ``X`` (b, 1, d), each shaped (b,)."""
        posterior = self.model.posterior(X)
        mean = posterior.mean.squeeze(-1).squeeze(-1)
        sigma = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1).squeeze(-1)
        return mean, sigma

    def _apply_diversity(self, term: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        if self._penalize:
            term = term * _diversity_factor(X, self.minima_deg, self.diversity_lambda)
        return term


class _LevelSetAcquisition(_DiversityAcquisition):
    """Level-set acquisition ``P(E(x) <= threshold) * sigma(x)`` (maximized).

    Threshold is ``E_min + delta``; the whole (non-negative) acquisition carries
    the diversity penalization. See :class:`_DiversityAcquisition`.
    """

    def __init__(
        self,
        model,
        threshold_eV: float,
        minima_deg: torch.Tensor | None = None,
        diversity_lambda: float = 0.0,
    ):
        super().__init__(model)
        self.register_buffer("threshold", torch.as_tensor(float(threshold_eV)))
        self._init_diversity(minima_deg, diversity_lambda)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (b, 1, d), normalized to [0, 1] (optimize_acqf bounds).
        mean, sigma = self._mean_sigma(X)  # (b,)
        z = (self.threshold - mean) / sigma
        return self._apply_diversity(torch.special.ndtr(z) * sigma, X)


class _BoundaryAcquisition(_DiversityAcquisition):
    """Boundary-focused acquisition ``-|mu(x) - target| + kappa * sigma(x)``.

    Seeks points whose predicted energy sits ON a target isosurface
    ``target = E_min + offset`` (plus an uncertainty bonus), rather than anywhere
    inside a window. Sweeping ``target`` upward across the exploration budget
    (threshold annealing) marches proposals out through successive energy shells,
    so the higher-energy part of the report window is deliberately populated --
    where the level-set acquisition instead keeps refining the low-energy manifold.

    The diversity penalization suppresses only the (non-negative) uncertainty
    *bonus* near known basins -- not the whole acquisition, which is typically
    negative here, so a multiplicative factor on it would perversely *reward*
    proximity. Leaving the boundary-seeking ``-|mu - target|`` term untouched
    keeps the target isosurface attractive while still discouraging re-sampling.
    """

    def __init__(
        self,
        model,
        target_eV: float,
        kappa: float,
        minima_deg: torch.Tensor | None = None,
        diversity_lambda: float = 0.0,
    ):
        super().__init__(model)
        self.register_buffer("target", torch.as_tensor(float(target_eV)))
        self.kappa = float(kappa)
        self._init_diversity(minima_deg, diversity_lambda)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (b, 1, d), normalized to [0, 1] (optimize_acqf bounds).
        mean, sigma = self._mean_sigma(X)  # (b,)
        bonus = self._apply_diversity(self.kappa * sigma, X)  # (b,), >= 0
        return -(mean - self.target).abs() + bonus


# --------------------------------------------------------------------------- #
# Exploration policy: budget and per-step schedule
# --------------------------------------------------------------------------- #


def _resolve_ensemble_budget(ensemble_steps: int, n_dihedrals: int) -> int:
    """Resolve the exploration step budget (hard cap).

    ``0`` disables exploration (passive harvest only); a positive value is used
    verbatim; ``-1`` (auto) scales with the rotor count -- more dihedrals means
    more potential basins -- bounded to a sane range.
    """
    if ensemble_steps >= 0:
        return ensemble_steps
    return int(min(250, max(25, 15 * n_dihedrals)))  # auto (-1)


def _exploration_plan(
    mode: str, budget: int, delta_eV: float
) -> list[tuple[str, float, float]]:
    """Per-step exploration schedule as ``(step_mode, offset_eV, kappa)``.

    ``levelset`` holds a fixed level-set threshold every step; ``hybrid`` runs
    level-set for the first half of the budget then a boundary sweep -- the target
    offset annealed LO->HI -- over the second half. The loop body just consumes
    this, so each mode is one schedule rather than branches scattered through it.
    """
    levelset = ("levelset", delta_eV, 0.0)
    if mode == "levelset":
        return [levelset] * budget
    # hybrid: level-set first half, boundary sweep second half.
    start = budget // 2
    span = max(1, budget - start - 1)  # -1 so the last boundary step reaches HI
    lo, hi = ENSEMBLE_BOUNDARY_LO_KCAL, ENSEMBLE_BOUNDARY_HI_KCAL
    plan: list[tuple[str, float, float]] = [levelset] * start
    for step in range(start, budget):
        offset_eV = (lo + (step - start) / span * (hi - lo)) * KCAL_TO_EV
        plan.append(("boundary", offset_eV, ENSEMBLE_BOUNDARY_KAPPA))
    return plan


# --------------------------------------------------------------------------- #
# Harvest: structure comparison, deduplication, and populations
# --------------------------------------------------------------------------- #


def _rmsd(a: Atoms, b: Atoms) -> float:
    """RMSD between two structures, in Angstrom.

    When the iRMSD backend is available this is the rotation- and
    permutation-invariant RMSD (atom ordering is canonicalized, so
    symmetry-equivalent conformers are not counted as distinct). Otherwise it
    falls back to a Kabsch-aligned all-atom RMSD that assumes identical atom
    ordering.
    """
    if _HAVE_IRMSD:
        return float(_irmsd.get_irmsd_ase(a, b)[0])

    b = b.copy()
    minimize_rotation_and_translation(a, b)  # mutates b in place toward a
    d = a.get_positions() - b.get_positions()
    return float(np.sqrt((d**2).sum(axis=1).mean()))


def _dedup(
    pairs: list[tuple[Atoms, float]], rmsd_thr: float, e_tol_eV: float
) -> list[tuple[Atoms, float]]:
    """Deduplicate (atoms, energy_eV) pairs, assumed sorted by energy ascending.

    A structure is a duplicate iff it is BOTH energy-close AND geometry-close to
    an already-kept structure.
    """
    unique: list[tuple[Atoms, float]] = []
    for atoms, energy in pairs:
        if any(
            abs(energy - ue) < e_tol_eV and _rmsd(ua, atoms) < rmsd_thr
            for ua, ue in unique
        ):
            continue
        unique.append((atoms, energy))
    return unique


def _boltzmann_weights(energies_eV: np.ndarray, temperature: float) -> np.ndarray:
    """Boltzmann populations from relative electronic energies (eV)."""
    kT = KB_EV_PER_K * temperature
    e = np.asarray(energies_eV, dtype=float)
    e = e - e.min()
    w = np.exp(-e / kT)
    return w / w.sum()
