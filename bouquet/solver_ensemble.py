"""Boltzmann-ensemble selection, harvest, and active level-set exploration.

The stateful orchestration around the state-free primitives in
:mod:`bouquet.ensemble`:

* passive harvest -- select observed conformers by GP posterior inclusion
  probability, tightly optimize, dedup, and Boltzmann-weight
  (:func:`_perform_ensemble_relaxation`);
* active exploration -- drive a level-set / boundary acquisition to push the
  low-energy manifold outward and discover new basins before the harvest reads the
  enriched ``state`` (:func:`_perform_ensemble_exploration`).

Runs after the main loop; enriches an :class:`bouquet.solver.OptimizationState`.
"""

from __future__ import annotations

# See bouquet/__init__.py for the lazy-import rationale.
__lazy_modules__ = [
    "numpy",
    "torch",
    "botorch.optim",
    "botorch.models",
    "botorch.models.transforms.outcome",
]

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf

from bouquet.assess import relax_structure
from bouquet.config import (
    ENSEMBLE_BASIN_DEG,
    ENSEMBLE_BOUNDARY_HI_KCAL,
    ENSEMBLE_BOUNDARY_KAPPA,
    ENSEMBLE_BOUNDARY_LO_KCAL,
    ENSEMBLE_ENERGY_TOL_KCAL,
    ENSEMBLE_EXPLORE_KCAL,
    ENSEMBLE_P_THRESHOLD,
    ENSEMBLE_RMSD_THRESHOLD,
    ENSEMBLE_SATURATION_ITERS,
    ENSEMBLE_SIGMA_FLOOR_KCAL,
    ENSEMBLE_SIGMA_STOP_KCAL,
    ENSEMBLE_TEMPERATURE,
    ENSEMBLE_WINDOW_KCAL,
    FAILURE_ENERGY_EV,
    KCAL_TO_EV,
    RunOptions,
)
from bouquet.ensemble import (
    _BoundaryAcquisition,
    _LevelSetAcquisition,
    _boltzmann_weights,
    _dedup,
    _exploration_plan,
    _periodic_min_dist,
    _resolve_ensemble_budget,
)
from bouquet.setup import DihedralInfo
from bouquet.surrogate import (
    _cert_sobol_pool,
    _fit_value_gp,
    _periodic_covar_module,
    _suppress_fit_warnings,
)

if TYPE_CHECKING:
    from bouquet.solver import OptimizationState

logger = logging.getLogger(__name__)


def _build_selection_gp(
    train_X_deg: torch.Tensor, train_y_eV: torch.Tensor
) -> SingleTaskGP:
    """Fit a GP for ensemble selection.

    Reuses the acquisition GP's kernel and [0, 1] input normalization, but fits
    energies in their natural (minimization) sense with a ``Standardize`` outcome
    transform, so the posterior is returned directly in eV (relative energies).
    """
    train_x = train_X_deg.clone() / 360.0
    train_y = train_y_eV.clone().unsqueeze(-1)  # (n, 1), eV, lower = better
    return _fit_value_gp(
        train_x,
        train_y,
        covar_module=_periodic_covar_module(train_x.shape[1]),
        outcome_transform=Standardize(m=1),
    )


def _select_ensemble_candidates(
    state: OptimizationState,
    window_eV: float,
    p_threshold: float,
    sigma_floor_eV: float,
    failure_energy_eV: float,
) -> list[tuple[np.ndarray, Atoms]]:
    """Select observed conformers to tightly optimize, ordered by predicted energy.

    A conformer ``i`` is included iff EITHER

    * its own observed energy is already within the window
      (``E_i <= E_min + window``), or
    * its GP posterior gives ``P(E_i <= E_min + window) >= p_threshold``.

    The first (observed-energy) clause is exact and guarantees no known-good
    conformer is discarded: the candidates are all *observed* points, whose
    energies we measured, and the subsequent tight optimization is unconstrained,
    so it can only *lower* the energy -- a point at or below the window now is
    certainly in-window after tight-opt. The GP clause only *adds* speculative
    candidates whose posterior is uncertain; it must never *remove* an observed
    in-window point, since the GP (smoothing + likelihood noise) regresses sharp
    minima upward and the observed energies are the higher, constrained-relaxation
    values -- both of which would otherwise drop genuine low-energy basins,
    especially when the observed set is large (e.g. seeded from an RDKit pool).
    The posterior sigma supplies a per-candidate, data-driven buffer for the GP
    clause: tight where the surface is well sampled, wide where it is sparse. No
    candidate cap is applied.
    """
    assert len(state.observed_atoms) == state.observed_energies.shape[0], (
        "observed_atoms is misaligned with observed_energies"
    )

    energies = state.observed_energies
    coords = state.observed_coords

    # Drop failed evaluations BEFORE fitting (the ~1000 eV sentinel wrecks the GP).
    valid = energies < failure_energy_eV
    idx = torch.nonzero(valid, as_tuple=False).flatten()
    if idx.numel() == 0:
        logger.warning("No valid observations for ensemble selection.")
        return []
    e = energies[idx]
    x = coords[idx]

    # Fit selection GP and evaluate the posterior at the observed coordinates.
    if idx.numel() < 3:
        # Too few points for a meaningful posterior: fall back to a flat window.
        mu, sigma = e, torch.full_like(e, sigma_floor_eV)
    else:
        with _suppress_fit_warnings():
            gp = _build_selection_gp(x, e)
            gp.eval()
            with torch.no_grad():
                post = gp.posterior(x / 360.0)
        mu = post.mean.flatten()
        sigma = post.variance.clamp_min(0.0).sqrt().flatten()
    sigma = sigma.clamp_min(sigma_floor_eV)

    # Inclusion = observed-in-window (exact) OR probabilistic GP inclusion.
    e_min = e.min()
    keep_obs = e <= e_min + window_eV
    z = (e_min + window_eV - mu) / sigma
    keep_gp = torch.special.ndtr(z) >= p_threshold  # standard-normal CDF
    keep = keep_obs | keep_gp

    # Map survivors back to global indices, ordered by the best available energy
    # estimate (observed vs predicted), so the cheapest-looking basins are tight-
    # optimized first. No cap on the number kept.
    order_key = torch.minimum(mu, e)
    sel_global = idx[keep][torch.argsort(order_key[keep])]

    logger.info(
        f"Ensemble selection: {sel_global.numel()} candidates "
        f"(from {idx.numel()} valid observations; "
        f"{int(keep_obs.sum())} observed-in-window, "
        f"{int((keep_gp & ~keep_obs).sum())} GP-only)."
    )
    return [
        (coords[i].cpu().numpy(), state.observed_atoms[i])
        for i in sel_global.tolist()
    ]


def _perform_ensemble_relaxation(
    state: OptimizationState,
    dihedrals: list[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
) -> list[tuple[Atoms, float, float]]:
    """Select -> tight (unconstrained) optimize -> dedup -> Boltzmann weight.

    Returns ``[(atoms, relative_energy_eV, weight)]`` sorted by energy ascending,
    where ``relative_energy_eV`` is measured against the run's start energy.
    """
    window_eV = ENSEMBLE_WINDOW_KCAL * KCAL_TO_EV
    sigma_floor_eV = ENSEMBLE_SIGMA_FLOOR_KCAL * KCAL_TO_EV
    e_tol_eV = ENSEMBLE_ENERGY_TOL_KCAL * KCAL_TO_EV

    candidates = _select_ensemble_candidates(
        state,
        window_eV=window_eV,
        p_threshold=ENSEMBLE_P_THRESHOLD,
        sigma_floor_eV=sigma_floor_eV,
        failure_energy_eV=FAILURE_ENERGY_EV,
    )

    # Tight, UNCONSTRAINED optimization of each candidate (no step limit).
    optimized: list[tuple[Atoms, float]] = []
    for k, (coords, atoms) in enumerate(candidates):
        a = atoms.copy()
        a.set_constraint()  # remove any dihedral constraints
        energy, a = relax_structure(a, calc, relaxCalc, steps=None)
        rel = energy - state.start_energy
        if rel >= FAILURE_ENERGY_EV:  # relative cutoff, as in candidate selection
            continue
        optimized.append((a, rel))
        if state.add_entry is not None:
            state.add_entry(
                np.array([d.get_angle(a) for d in dihedrals]), a, energy
            )
        logger.info(f"Tight opt {k+1}/{len(candidates)}: E-E0 = {rel:12.6f} eV")

    if not optimized:
        return []

    # Post-optimization dedup: pre-opt duplicates can split, distinct points merge.
    optimized.sort(key=lambda t: t[1])
    unique = _dedup(optimized, rmsd_thr=ENSEMBLE_RMSD_THRESHOLD, e_tol_eV=e_tol_eV)

    # Final reporting cut + Boltzmann populations on the deduped set.
    e_min = min(e for _, e in unique)
    unique = [(a, e) for a, e in unique if (e - e_min) <= window_eV]
    energies = np.array([e for _, e in unique])
    weights = _boltzmann_weights(energies, ENSEMBLE_TEMPERATURE)

    logger.info(
        f"Final ensemble: {len(unique)} unique conformer(s) within "
        f"{ENSEMBLE_WINDOW_KCAL} kcal/mol."
    )
    return [(a, e, float(w)) for (a, e), w in zip(unique, weights)]


# ---------------------------------------------------------------------------
# Active level-set exploration (ensemble mode)
#
# The passive harvest above can only report basins the minimum-seeking search
# happened to visit. This phase runs AFTER the main loop: it fits the same
# selection GP (energies in eV, minimization sense) and drives a level-set
# acquisition -- P(E(x) <= E_min + delta) * sigma(x) -- to push the boundary of
# the low-energy manifold outward and discover NEW basins. Each proposal is
# evaluated with the same constrained relaxation as a normal BO step and
# appended to ``state``; the passive harvest then finalizes the enriched set for
# free (selection -> tight opt -> dedup -> Boltzmann).
# ---------------------------------------------------------------------------


def _valid_observation_idx(state: OptimizationState) -> torch.Tensor:
    """Indices of observations that are not failure sentinels (rel E below the
    ~1000 eV cutoff), which would otherwise wreck a GP fit or basin seeding."""
    return torch.nonzero(
        state.observed_energies < FAILURE_ENERGY_EV, as_tuple=False
    ).flatten()


def _fit_selection_gp_valid(
    state: OptimizationState,
) -> tuple[SingleTaskGP | None, float | None]:
    """Fit the selection GP on valid (non-failed) observations.

    Returns ``(gp, e_min_eV)`` or ``(None, None)`` when fewer than three valid
    observations exist (too few for a meaningful posterior). Failure sentinels
    are dropped exactly as in ``_select_ensemble_candidates``.
    """
    idx = _valid_observation_idx(state)
    if idx.numel() < 3:
        return None, None
    e = state.observed_energies[idx]
    with _suppress_fit_warnings():
        gp = _build_selection_gp(state.observed_coords[idx], e)
    gp.eval()
    return gp, float(e.min())


def _max_posterior_sigma(
    gp: SingleTaskGP, num_dims: int, dtype: torch.dtype, device: torch.device
) -> float:
    """Max posterior sigma (eV) of the selection GP over the fixed Sobol pool.

    Reuses the certificate's space-filling pool. The selection GP carries a
    ``Standardize`` outcome transform, so the posterior variance is already in
    natural (eV^2) units. Queried in ``dtype`` -- the dtype ``optimize_acqf`` used
    -- so the GP's cached prediction strategy is not re-triggered at a mismatch.
    """
    pool = _cert_sobol_pool(num_dims, dtype, device).unsqueeze(1)  # (M, 1, d)
    with torch.no_grad():
        sd = gp.posterior(pool).variance.clamp_min(0).sqrt()
    return float(sd.max())


def _initial_basins(
    state: OptimizationState, delta_eV: float, basin_deg: float
) -> torch.Tensor:
    """Seed known-basin torsion vectors from low-energy observations.

    Takes the valid observations within ``delta`` of the best, orders them by
    energy, and greedily keeps those at least ``basin_deg`` (wrapped RMS angular
    distance) from an already-kept seed. Returns shape ``(m, d)`` in the coords'
    dtype/device (``m`` may be 0).
    """
    coords = state.observed_coords
    d = coords.shape[1]
    idx = _valid_observation_idx(state)
    if idx.numel() == 0:
        return coords.new_zeros((0, d))
    e = state.observed_energies[idx]
    within = e <= (e.min() + delta_eV)
    xw = coords[idx][within]
    order = torch.argsort(e[within])
    # Greedy dedup into a preallocated buffer (at most one seed per in-window
    # observation), filling the first ``m`` rows -- avoids re-stacking the whole
    # growing seed list on every iteration.
    seeds = xw.new_zeros((xw.shape[0], d))
    m = 0
    for i in order.tolist():
        c = xw[i]
        if m == 0 or float(_periodic_min_dist(c, seeds[:m])) >= basin_deg:
            seeds[m] = c
            m += 1
    return seeds[:m]


def _select_exploration_point(
    state: OptimizationState,
    mode: str,
    offset_eV: float,
    kappa: float,
    minima_deg: torch.Tensor,
    diversity_lambda: float,
    opts: RunOptions,
) -> tuple[np.ndarray | None, SingleTaskGP | None]:
    """Propose one exploration point by optimizing the exploration acquisition.

    ``mode`` selects the acquisition: ``"levelset"`` (``P_in * sigma`` with
    threshold ``E_min + offset``) or ``"boundary"`` (``-|mu - (E_min + offset)| +
    kappa*sigma``). Returns ``(coords_deg, gp)`` -- the proposed torsion vector
    (degrees) and the fitted selection GP (so the caller can read its posterior
    sigma for the stop rule) -- or ``(None, None)`` when there are too few valid
    points to fit.
    """
    gp, e_min = _fit_selection_gp_valid(state)
    if gp is None:
        return None, None
    d = state.observed_coords.shape[1]
    tx = state.observed_coords
    if mode == "boundary":
        acqf = _BoundaryAcquisition(
            gp, e_min + offset_eV, kappa,
            minima_deg=minima_deg, diversity_lambda=diversity_lambda,
        )
    else:
        acqf = _LevelSetAcquisition(
            gp, e_min + offset_eV,
            minima_deg=minima_deg, diversity_lambda=diversity_lambda,
        )
    bounds = torch.zeros(2, d, dtype=tx.dtype, device=tx.device)
    bounds[1, :] = 1.0
    with _suppress_fit_warnings():
        candidate, _ = optimize_acqf(
            acqf,
            bounds=bounds,
            q=1,
            num_restarts=opts.acq_num_restarts,
            raw_samples=opts.acq_raw_samples,
        )
    coords_deg = candidate.detach().cpu().numpy()[0, :] * 360.0
    return coords_deg, gp


def _perform_ensemble_exploration(
    state: OptimizationState,
    dihedrals: list[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
    relax: bool,
    opts: RunOptions,
) -> None:
    """Active exploration; enriches ``state`` in place with new basins.

    Runs a hard budget of q=1 steps (auto-scaled by rotor count) following the
    ``opts.ensemble_explore_mode`` schedule (see :func:`_exploration_plan`). Pure
    ``levelset`` stops early once the search saturates -- no new basin for
    ``ENSEMBLE_SATURATION_ITERS`` steps AND the posterior sigma has collapsed below
    ``ENSEMBLE_SIGMA_STOP_KCAL``; ``boundary``/``hybrid`` run the full budget so the
    annealed target marches through every energy shell. New observations are
    appended exactly like the main loop, so the passive harvest picks them up.
    """
    # _evaluate_point lives in bouquet.solver (the run layer); import it here to
    # avoid a solver <-> solver_ensemble import cycle (bouquet already uses
    # function-level deferred imports for the same reason -- see _low_mode_move).
    from bouquet.solver import _evaluate_point

    n_dihedrals = state.observed_coords.shape[1]
    budget = _resolve_ensemble_budget(opts.ensemble_steps, n_dihedrals)
    if budget <= 0:
        return

    mode = opts.ensemble_explore_mode
    delta_eV = ENSEMBLE_EXPLORE_KCAL * KCAL_TO_EV
    sigma_stop_eV = ENSEMBLE_SIGMA_STOP_KCAL * KCAL_TO_EV
    dtype = state.observed_coords.dtype
    device = state.observed_coords.device

    minima_deg = _initial_basins(state, delta_eV, ENSEMBLE_BASIN_DEG)
    state.reserve(budget)  # loop appends at most one observation per step

    plan = _exploration_plan(mode, budget, delta_eV)
    # Only pure level-set exploration may stop early; boundary/hybrid run the full
    # budget so the annealed target marches through every energy shell.
    allow_early_stop = mode == "levelset"
    if mode == "levelset":
        logger.info(
            f"Ensemble exploration: up to {budget} level-set step(s), "
            f"delta = {ENSEMBLE_EXPLORE_KCAL} kcal/mol, "
            f"{minima_deg.shape[0]} seed basin(s)."
        )
    else:
        n_ls = sum(m == "levelset" for m, _, _ in plan)
        pre = f"{n_ls} level-set + " if n_ls else ""
        logger.info(
            f"Ensemble exploration ({mode}): up to {budget} step(s) "
            f"({pre}boundary target annealed {ENSEMBLE_BOUNDARY_LO_KCAL}->"
            f"{ENSEMBLE_BOUNDARY_HI_KCAL} kcal/mol, kappa={ENSEMBLE_BOUNDARY_KAPPA}), "
            f"{minima_deg.shape[0]} seed basin(s)."
        )

    no_new = 0
    for step, (step_mode, offset_eV, kappa) in enumerate(plan):
        coords_deg, gp = _select_exploration_point(
            state, step_mode, offset_eV, kappa, minima_deg,
            opts.ensemble_diversity, opts,
        )
        if coords_deg is None:
            logger.info(
                "Ensemble exploration: too few valid observations to fit the "
                "selection GP; stopping."
            )
            break

        energy, atoms, gradient = _evaluate_point(
            state, coords_deg, dihedrals, calc, relaxCalc, relax
        )
        rel = energy - state.start_energy
        if state.add_entry is not None:
            state.add_entry(coords_deg, atoms, energy)
        state.append_observation(coords_deg, rel, atoms, gradient)

        # New-basin bookkeeping: only in-window, non-failed points can seed a
        # basin, and only if far enough (in wrapped torsion space) from every
        # known basin -- a coarse in-search analog of the final RMSD dedup.
        e_min = state.observed_energies.min().item()
        is_new = False
        if rel < FAILURE_ENERGY_EV and rel <= e_min + delta_eV:
            c = torch.as_tensor(coords_deg, dtype=dtype, device=device)
            if minima_deg.shape[0] == 0 or (
                float(_periodic_min_dist(c, minima_deg)) >= ENSEMBLE_BASIN_DEG
            ):
                minima_deg = torch.cat([minima_deg, c.unsqueeze(0)], dim=0)
                is_new = True
        no_new = 0 if is_new else no_new + 1

        # The Sobol-pool sigma only gates the (level-set-only) early stop, so skip
        # its 1024-point posterior solve entirely when it can't fire.
        max_sigma = (
            _max_posterior_sigma(gp, n_dihedrals, dtype, device)
            if allow_early_stop else None
        )
        tgt = f"tgt = {offset_eV / KCAL_TO_EV:4.1f}  " if step_mode == "boundary" else ""
        sig = (
            f"  max_sigma = {max_sigma / KCAL_TO_EV:6.2f} kcal/mol"
            if max_sigma is not None else ""
        )
        logger.info(
            f"Explore {step+1: >3}/{budget}: {tgt}E-E0 = {rel:12.6f} eV  "
            f"{'NEW basin ' if is_new else 'known     '}"
            f"basins = {minima_deg.shape[0]: >3}{sig}"
        )

        if (
            allow_early_stop
            and no_new >= ENSEMBLE_SATURATION_ITERS
            and max_sigma < sigma_stop_eV
        ):
            logger.info(
                f"Ensemble exploration converged after {step+1} step(s) "
                f"(no new basin for {no_new} step(s) and sigma collapsed)."
            )
            break

    logger.info(
        f"Ensemble exploration complete: {minima_deg.shape[0]} candidate basin(s)."
    )
