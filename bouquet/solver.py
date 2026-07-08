"""Methods for solving the conformer option problem"""

from __future__ import annotations

# With annotations evaluated as strings (PEP 563), the heavy numeric/BO stack is
# only touched inside function bodies, so defer it until an optimization actually
# runs. This is the dominant `import bouquet` cost (Python 3.15+).
__lazy_modules__ = [
    "numpy",
    "torch",
    "ase",
    "ase.build",
    "ase.calculators.calculator",
    "botorch.acquisition.analytic",
    "botorch.acquisition.prior_guided",
    "botorch.optim.fit",
    "botorch.optim",
    "botorch.models",
    "botorch.models.transforms.outcome",
    "gpytorch",
    "gpytorch.mlls",
    "gpytorch.priors",
]

import itertools
import logging
import math
import time
import warnings
from contextlib import contextmanager
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from ase import Atoms
from ase.build import minimize_rotation_and_translation
from ase.calculators.calculator import Calculator
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.prior_guided import PriorGuidedAcquisitionFunction
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch import kernels as gpykernels
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior, NormalPrior

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

from bouquet.assess import (
    evaluate_energy,
    evaluate_energy_with_gradient,
    relax_structure,
)
from bouquet.config import (
    ACQ_NUM_RESTARTS,
    ACQ_RAW_SAMPLES,
    DEFAULT_CERTIFICATE_BETAS,
    DEFAULT_RELAXATION_STEPS,
    ENSEMBLE_ENERGY_TOL_KCAL,
    ENSEMBLE_P_THRESHOLD,
    ENSEMBLE_RMSD_THRESHOLD,
    ENSEMBLE_SIGMA_FLOOR_KCAL,
    ENSEMBLE_TEMPERATURE,
    ENSEMBLE_WINDOW_KCAL,
    CAT_D_THRESHOLD,
    CAT_MAXSPEC_THRESHOLD,
    FAILURE_ENERGY_EV,
    GP_PERIOD_LENGTH_MEAN,
    GP_PERIOD_LENGTH_STD,
    HIGH_D_DIHEDRAL_THRESHOLD,
    INITIAL_GUESS_STD,
    KB_EV_PER_K,
    KCAL_TO_EV,
    RELAX_FAILURE_ENERGY_EV,
)
from bouquet.gradient_gp import GradientEnhancedPeriodicGP
from bouquet.io import (
    append_xyz_frame,
    create_certificate_logger,
    create_structure_logger,
    initialize_structure_log,
    save_structure,
)
from bouquet.priors import DihedralPriorModule
from bouquet.setup import DihedralInfo, bonds_broken, geometry_bond_set

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_fit_warnings():
    """Silence the expected, noisy botorch/gpytorch fit and acquisition warnings,
    scoped to the wrapped block instead of the whole process (importing this module
    used to install a global ``warnings.filterwarnings('ignore')``)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield

# Marginal-likelihood Adam iterations for a cold GP hyperparameter fit, shared by
# the value-only acquisition GP, the ensemble-selection GP, and the cold
# gradient-GP fit (see _fit_gradient_gp); gradient-GP condition-only updates run 0
# steps. A benchmark found conformer quality is insensitive to fit convergence
# (50/100/200 steps gave statistically indistinguishable minima despite very
# different final NLLs), so 100 trims the cold fits at no quality cost; 50 showed a
# faint degradation and is under-converged for larger molecules.
_GP_FIT_STEPS = 100


def _get_device() -> torch.device:
    """Get the appropriate torch device (CUDA if available, else CPU)."""
    # sadly Apple's MPS doesn't support float64
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(slots=True)
class OptimizationState:
    """Tracks the state of a Bayesian optimization run."""

    start_atoms: Atoms
    start_coords: np.ndarray
    start_energy: float
    observed_coords: torch.Tensor  # Shape: (n_observations, n_dihedrals)
    observed_energies: torch.Tensor  # Shape: (n_observations,)
    # dE/dtheta (eV/rad) per observation, index-aligned with the tensors above;
    # NaN where the gradient is unavailable (failed eval, or use_gradients off).
    observed_gradients: torch.Tensor | None = None  # Shape: (n_observations, n_dihedrals)
    # Per-observation Atoms, aligned index-for-index with the tensors above.
    observed_atoms: list[Atoms] = field(default_factory=list)
    device: torch.device = field(default_factory=_get_device)
    init_steps: int = 0
    best_step: int = 0
    # --retain-bonds: the covalent bond set of the initial structure that every
    # evaluated geometry must preserve (None disables the check). Broken-bond
    # evaluations get a failure energy so they're never selected. n_bond_breaks
    # counts how many were rejected (logged once at the end).
    required_bonds: set | None = None
    n_bond_breaks: int = 0
    add_entry: Callable | None = None
    # Optional per-BO-step stopping-rule certificate logger (see
    # _run_optimization_loop / io.create_certificate_logger). When set, each step
    # logs mu_min/lb/alpha_max alongside e_eval/e_best/n_calls/wall_s. cert_betas is
    # the grid of confidence multipliers for the lower-bound term (one lb per beta).
    cert_log: Callable | None = None
    cert_betas: tuple[float, ...] = DEFAULT_CERTIFICATE_BETAS
    # Optional geometry-trail path (stopping-rule benchmark): when set, the geometry
    # at each best-so-far improvement is appended here (plus the final relaxed best),
    # for the offline RMSD-identity / distinct-conformer analysis. See
    # _log_improvement_geometry and io.append_xyz_frame.
    geom_log_path: Path | None = None
    # When True, evaluations also record dE/dtheta and the acquisition GP uses
    # the gradient-enhanced surrogate (see GradientEnhancedPeriodicGP).
    use_gradients: bool = False
    # Number of leading BO steps that use the gradient-enhanced GP; once the loop
    # passes this many steps it switches to the value-only GP (gradients are still
    # recorded, just not fed to the surrogate). The gradient GP's per-step cost
    # grows as O((n*(1+d))^3), so on large/floppy molecules it becomes intractable
    # late in the run; spending the gradient signal early -- where it helps most --
    # and then dropping to the cheap n*n GP keeps the search tractable. <=0 means
    # never switch (gradient GP for the whole run).
    gradient_steps: int = 0
    # Gradient-GP hyperparameter refit schedule (see _run_optimization_loop). Cold
    # full fits for the first `grad_refit_dense_until` BO steps, then the
    # hyperparameters are frozen and later steps only re-condition; `grad_refit_every`
    # > 0 optionally cold-refreshes them every that many post-dense steps. Default
    # (20, 0) = the "gradfreeze" schedule; set grad_refit_dense_until=0 for a full fit
    # every step. `grad_gp_hypers` holds the last cold-fitted hyperparameters (frozen
    # on condition-only steps).
    grad_refit_dense_until: int = 20
    grad_refit_every: int = 0
    grad_gp_hypers: dict | None = None
    # Acquisition-optimizer effort (optimize_acqf); see Configuration.
    acq_num_restarts: int = ACQ_NUM_RESTARTS
    acq_raw_samples: int = ACQ_RAW_SAMPLES
    # High-leverage gradient subset: keep gradients for only this many points
    # (0 = all). gradient_keep = recent|best|both. See _restrict_gradient_mask.
    gradient_window: int = 0
    gradient_keep: str = "recent"
    # Value-only-GP lengthscale prior: "none" (free fit, historical) or
    # "dim_scaled" (Hvarfner dimensionality-scaled LogNormal). See
    # _periodic_covar_module.
    lengthscale_prior: str = "none"

    # Phase 2.5 low-mode / basin-hopping move. With probability lowmode_prob an eligible
    # step (>= lowmode_warmup evaluations) is replaced by a committed kick along a soft
    # mode + UNCONSTRAINED relaxation (see _low_mode_move) -- the move designed for the
    # curved fold valley. lowmode_kick_deg is the per-dihedral RMS kick amplitude;
    # lowmode_modes is how many leading soft modes to draw a kick from. lowmode_rng is the
    # runtime move-type coin / direction RNG, lowmode_count a tally for the end-of-run log.
    lowmode_prob: float = 0.0
    lowmode_warmup: int = 100
    lowmode_kick_deg: float = 60.0
    lowmode_modes: int = 4
    # Kick-direction source: "pca" (data-derived position-PCA; the default and benchmark
    # winner) or "enm" (data-independent elastic-network soft modes; dormant, lost to PCA
    # -- see bouquet.enm). See _low_mode_move.
    lowmode_kick_dir: str = "pca"
    lowmode_rng: np.random.Generator | None = None
    lowmode_count: int = 0

    # Phase 3 category-tied collective move (chemistry-defined REMBO). With probability
    # category_prob an eligible step is replaced by a low-dimensional move over
    # *per-SMARTS-category* dihedral values: every dihedral sharing a prior category is
    # set to one shared value (a chemistry-defined embedding, available from step 0 --
    # unlike the low-mode move's data-derived PCA, which needs an accumulated elite set).
    # The reduced (n_group-dim) point is chosen by a periodic GP + LogEI over a dedicated
    # buffer of past reduced points, then broadcast and constrained + UNCONSTRAINED relaxed
    # (same clash-cleanup-then-release schedule as the low-mode move). See _category_move.
    #   category_groups   -- the tied-index partition (list of index lists), built at setup
    #                        from the prior's univariate assignments; each dihedral is in
    #                        exactly one group (a real category, or its own singleton).
    #   category_warmup   -- gate on the *outer* buffer size (need an incumbent to anchor).
    #   category_min_moves-- how many prior-seeded reduced points to collect before the
    #                        reduced-space GP is fit (below this, sample z from the prior).
    #   _cat_Z / _cat_Y   -- the reduced-space BO buffer (per-category values in degrees,
    #                        relative eV), grown one row per category move.
    category_prob: float = 0.0
    category_warmup: int = 20
    category_min_moves: int = 6
    category_groups: list | None = None
    category_rng: np.random.Generator | None = None
    category_count: int = 0
    _cat_Z: list = field(default_factory=list)
    _cat_Y: list = field(default_factory=list)

    # PiBO fields
    prior_module: DihedralPriorModule | None = None
    prior_exponent: float = 2.0
    prior_decay: float = 0.9

    def append_observation(
        self,
        coords: np.ndarray,
        energy: float,
        atoms: Atoms,
        gradient: np.ndarray | None = None,
    ) -> None:
        """Append a new observation, keeping observed_atoms index-aligned.

        Args:
            coords: Dihedral coordinates as numpy array
            energy: Relative energy value
            atoms: Structure at this observation (copied for retention)
            gradient: Optional dE/dtheta (eV/rad) for each dihedral; stored as
                NaN when not provided so the gradient tensor stays index-aligned.
        """
        # TODO: each append does three torch.cat reallocations, so accumulating N
        # observations is O(N^2) in copy cost. Fine while N stays in the hundreds;
        # if N grows to thousands, buffer into Python lists and torch.stack once at
        # fit time instead.
        new_coords = torch.tensor(
            coords, dtype=torch.float64, device=self.device
        ).unsqueeze(0)
        new_energy = torch.tensor([energy], dtype=torch.float64, device=self.device)
        if gradient is None:
            gradient = np.full(len(coords), np.nan, dtype=float)
        new_grad = torch.tensor(
            np.asarray(gradient, dtype=float), dtype=torch.float64, device=self.device
        ).unsqueeze(0)
        self.observed_coords = torch.cat([self.observed_coords, new_coords], dim=0)
        self.observed_energies = torch.cat([self.observed_energies, new_energy], dim=0)
        if self.observed_gradients is None:
            self.observed_gradients = new_grad
        else:
            self.observed_gradients = torch.cat(
                [self.observed_gradients, new_grad], dim=0
            )
        self.observed_atoms.append(atoms.copy())


def _periodic_covar_module(
    num_dims: int, lengthscale_prior: str = "none"
) -> gpykernels.ScaleKernel:
    """Build the periodic GP covariance module shared by the acquisition GP and
    the ensemble-selection GP (a scaled product of per-dihedral periodic kernels).

    ``lengthscale_prior`` selects the prior on the (shared) periodic lengthscale:
    - ``"none"`` (default): no lengthscale prior -- the historical behavior, where
      the lengthscale is a free MLL fit.
    - ``"dim_scaled"``: the Hvarfner et al. (ICML 2024) dimensionality-scaled
      LogNormal prior, ``LogNormal(sqrt(2) + 0.5*ln d, sqrt(3))``, biasing the GP
      toward smoother fits as ``d`` grows. NOTE: those constants were calibrated for
      an RBF/Matern kernel; this PeriodicKernel uses ``exp(-2 sin^2(.)/l)`` (l at
      first power over a bounded sin^2 term), so the *location offset* likely needs
      recalibration -- the smoke test measures whether the canonical constants help
      or over-smooth. The lengthscale is also initialized at the prior median.
    """
    base_kwargs = dict(
        period_length_prior=NormalPrior(GP_PERIOD_LENGTH_MEAN, GP_PERIOD_LENGTH_STD)
    )
    loc = None
    if lengthscale_prior == "dim_scaled":
        loc = math.sqrt(2.0) + 0.5 * math.log(num_dims)
        base_kwargs["lengthscale_prior"] = LogNormalPrior(loc, math.sqrt(3.0))
    elif lengthscale_prior != "none":
        raise ValueError(
            f"lengthscale_prior must be 'none' or 'dim_scaled', got {lengthscale_prior!r}"
        )
    base_kernel = gpykernels.PeriodicKernel(**base_kwargs)
    if loc is not None:
        base_kernel.lengthscale = math.exp(loc)  # start at the prior median
    return gpykernels.ScaleKernel(
        gpykernels.ProductStructureKernel(num_dims=num_dims, base_kernel=base_kernel)
    )


def _restrict_gradient_mask(
    mask: torch.Tensor, raw_y: torch.Tensor, window: int, mode: str
) -> torch.Tensor:
    """Keep gradients for only ``window`` high-leverage points (others stay
    value-only), shrinking the augmented GP from n*(1+d) to n + window*d.

    ``mode``: 'recent' (the last ``window`` gradient-valid points -- local
    navigation at the search frontier), 'best' (the ``window`` lowest-energy --
    refining the incumbent basin), or 'both' (half each, union). Points are
    selected only among those already in ``mask`` (gradient-valid, clamp-inactive);
    every point still contributes its value. No-op if <= window valid gradients."""
    valid = torch.nonzero(mask, as_tuple=False).flatten()  # ascending obs order
    if window <= 0 or valid.numel() <= window:
        return mask
    if mode == "recent":
        sel = valid[-window:]
    elif mode == "best":
        sel = valid[torch.argsort(raw_y[valid])[:window]]
    else:  # both: half lowest-energy + half most-recent, unioned
        n_best = window // 2
        best_sel = valid[torch.argsort(raw_y[valid])[:n_best]]
        recent_sel = valid[-(window - n_best):]
        sel = torch.unique(torch.cat([best_sel, recent_sel]))
    new_mask = torch.zeros_like(mask)
    new_mask[sel] = True
    return new_mask


def _fit_gradient_gp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    raw_y: torch.Tensor,
    energy_cap: torch.Tensor,
    observed_gradients: torch.Tensor,
    y_std: torch.Tensor,
    frozen_hypers: dict | None = None,
    gradient_window: int = 0,
    gradient_keep: str = "recent",
) -> GradientEnhancedPeriodicGP:
    """Build and fit the gradient-enhanced periodic GP on standardized data.

    Two modes (see ``_run_optimization_loop``):
    - ``frozen_hypers=None``: a cold fit -- optimize fresh hyperparameters with
      ``_GP_FIT_STEPS`` marginal-likelihood Adam iterations. The caller reads
      ``gp.state_dict()`` afterwards to carry the hyperparameters forward.
    - ``frozen_hypers`` set: a condition-only update -- load those hyperparameters
      and fold in the new data for one Cholesky, running NO Adam steps.

    We deliberately never optimize *from* loaded hyperparameters: warm-starting the
    fit drifts them and degrades the search, so loaded hypers are only ever used for
    conditioning. Collapsing the two knobs into one parameter makes that the only
    expressible behavior.

    The acquisition GP fits standardized values ``y' = (-clamp(E) - mean)/std``
    over inputs ``x' = degrees / 360``. The stored gradients are ``dE/dtheta`` in
    eV/rad, so the chain rule for the matching gradient observation is

        dy'/dx' = -(2*pi / std) * dE/dtheta           [clamp inactive]

    (``x_rad = 2*pi * x'``; the maximization sign flip contributes the minus).
    ``y_std`` is the standardization std the caller applied to the values, so the
    gradients are scaled by exactly the same factor. Points where the energy
    clamp is active or whose gradient is NaN (failed evaluation) are dropped from
    the gradient observations via the mask, but still contribute their value.
    """
    grad = observed_gradients.to(train_x)  # (n, d), eV/rad
    grad_scaled = -(2.0 * math.pi / y_std) * grad  # (n, d)

    clamp_inactive = raw_y <= energy_cap  # (n,)
    grad_valid = ~torch.isnan(grad).any(dim=1)  # (n,)
    mask = clamp_inactive & grad_valid
    # High-leverage subset: optionally keep gradients for only a window of points
    # (caps the augmented matrix at n + window*d -- the main high-d/late cost).
    mask = _restrict_gradient_mask(mask, raw_y, gradient_window, gradient_keep)
    grad_scaled = torch.nan_to_num(grad_scaled, nan=0.0)

    gp = GradientEnhancedPeriodicGP(
        train_x, train_y, grad_scaled, grad_mask=mask, period=1.0
    )
    if frozen_hypers is None:
        gp.fit(steps=_GP_FIT_STEPS, lr=0.05)  # cold fit: optimize fresh hypers
    else:
        # Condition-only: load the frozen hypers and re-condition (steps=0, no Adam).
        gp.load_state_dict(frozen_hypers, strict=True)
        gp.fit(steps=0, lr=0.05)
    return gp


# Size of the Sobol space-filling pool used to estimate the stopping-rule
# certificate each step. The certificate is a smooth function of the posterior, so
# a modest pool localizes its min/max well; observed points and the chosen
# candidate are added on top (see _compute_certificate).
_CERT_POOL_SIZE = 1024
# Fixed scramble seed for the certificate's Sobol pool. Held constant so the pool
# is identical every step and every run: that isolates the certificate's
# step-to-step motion to the GP (and the growing observed set), instead of letting
# fresh Sobol scrambles add sampling noise to lb/alpha_max -- and makes the offline
# stopping-rule replay reproducible.
_CERT_POOL_SEED = 12345


@lru_cache(maxsize=None)
def _cert_sobol_pool(
    num_dims: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """The certificate's fixed Sobol space-filling pool over [0, 1]^num_dims.

    Size and scramble seed are constants, so the pool is identical every step and
    every run (see _CERT_POOL_SEED); caching draws it once per (dims, dtype,
    device) instead of regenerating 1024 points on every BO step. Returned
    read-only -- callers ``cat`` it with the step's observations, never mutate it.
    """
    bounds = torch.stack(
        [
            torch.zeros(num_dims, device=device, dtype=dtype),
            torch.ones(num_dims, device=device, dtype=dtype),
        ]
    )
    return draw_sobol_samples(
        bounds=bounds, n=_CERT_POOL_SIZE, q=1, seed=_CERT_POOL_SEED
    ).squeeze(1)


def _compute_certificate(
    gp,
    base_acqf,
    train_x: torch.Tensor,
    candidate: torch.Tensor,
    y_std: torch.Tensor,
    neg_mean: torch.Tensor,
    betas: tuple[float, ...],
) -> dict:
    """Compute the per-step GP stopping-rule certificate, in run energy units.

    The acquisition GP is fit on the *negated, standardized* energy (minimize ->
    maximize), so a standardized posterior value ``v`` maps back to relative
    energy (eV, the ``e_e0`` convention) as ``E = -(v * y_std + neg_mean)``.
    Returns, all in relative eV:

      * ``mu_min``    -- the predicted global-minimum energy ``min_x mu_t(x)``;
      * ``lb``        -- list of high-probability lower bounds on it, one per beta
                         in ``betas``: ``min_x [mu_t(x) - beta * sigma_t(x)]`` (a
                         UCB on the maximized objective mapped back to energy). The
                         argmax shifts with beta, so the bound is logged across a
                         grid to let the offline replay pick/calibrate beta without
                         re-running;
      * ``alpha_max`` -- the maximum *plain* expected improvement (energy units,
                         not log) over the pool, the quantity the log-EI stopping
                         rule thresholds (beta-independent).

    The pool reuses the observed coordinates and the chosen candidate (so the
    incumbent basin is always represented, which guarantees ``lb <= e_best``)
    plus a fixed Sobol space-filling set over the normalized [0, 1]^d cube.
    """
    device = train_x.device
    num_dims = train_x.shape[1]
    y_std_s = float(y_std.reshape(-1)[0])
    neg_mean_s = float(neg_mean.reshape(-1)[0])

    # Query in the dtype the acquisition optimizer used (the candidate's): the
    # value-only GP caches its prediction strategy from that optimize_acqf pass
    # (typically float32), so a mismatched dtype here errors. The gradient GP casts
    # inputs to its training dtype internally, so this is safe for both paths.
    cand = torch.as_tensor(candidate, device=device).reshape(1, num_dims)
    dtype = cand.dtype
    sobol = _cert_sobol_pool(num_dims, dtype, device)
    pool = torch.cat([sobol, train_x.to(dtype), cand], dim=0).unsqueeze(1)  # (M, 1, d)

    with torch.no_grad():
        posterior = gp.posterior(pool)
        mu_std = posterior.mean.reshape(-1)  # maximize-space mean
        sd_std = posterior.variance.clamp_min(0).sqrt().reshape(-1)
        # base_acqf is the unwrapped LogExpectedImprovement (no PiBO term), so
        # exp() recovers plain EI; scale standardized -> energy units by y_std.
        ei_energy = torch.exp(base_acqf(pool)) * y_std_s

    # min predicted energy = -(max standardized mean) mapped back.
    mu_min = -(float(mu_std.max()) * y_std_s + neg_mean_s)
    # min_x [E_mean - beta*E_sd] = -(neg_mean + y_std * max_x(mu_std + beta*sd_std)),
    # one bound per beta (the argmax point differs per beta, so compute each).
    lb = [
        -(neg_mean_s + y_std_s * float((mu_std + beta * sd_std).max()))
        for beta in betas
    ]
    alpha_max = float(ei_energy.max())
    return {"mu_min": mu_min, "lb": lb, "alpha_max": alpha_max}


def _select_next_points_botorch(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    prior_module: DihedralPriorModule | None = None,
    prior_exponent: float = 0.0,
    observed_gradients: torch.Tensor | None = None,
    use_gradients: bool = False,
    gp_frozen_hypers: dict | None = None,
    gp_hyper_out: dict | None = None,
    cert_out: dict | None = None,
    cert_betas: tuple[float, ...] = DEFAULT_CERTIFICATE_BETAS,
    acq_num_restarts: int = ACQ_NUM_RESTARTS,
    acq_raw_samples: int = ACQ_RAW_SAMPLES,
    gradient_window: int = 0,
    gradient_keep: str = "recent",
    lengthscale_prior: str = "none",
) -> np.ndarray:
    """
    Selects the next dihedral coordinate to evaluate by fitting a Gaussian process to the observed data and optimizing a BOTorch acquisition function.

    Parameters:
        train_X (torch.Tensor): Observed dihedral coordinates, shape (n_observations, n_dims), in degrees.
        train_y (torch.Tensor): Observed energies corresponding to train_X, shape (n_observations,).
        prior_module: Optional DihedralPriorModule for PiBO
        prior_exponent: Prior strength (0 = no prior influence)
        observed_gradients: Optional dE/dtheta (eV/rad), shape (n_observations,
            n_dims), index-aligned with ``train_X``; NaN rows are dropped.
        use_gradients: If True (and gradients are provided), fit the
            gradient-enhanced surrogate instead of the value-only GP.
        gp_frozen_hypers: Optional state-dict of gradient-GP hyperparameters. None
            does a cold fit (optimize fresh hyperparameters); a value loads those
            hyperparameters and conditions only (no Adam). Ignored unless gradients
            are used. (The value-only GP path always uses the standard fit.)
        gp_hyper_out: If provided and gradients are used, this dict is populated
            with ``{"hypers": <fitted state-dict>}`` for the caller to carry forward.
        cert_out: If provided, populated in place with the stopping-rule
            certificate ``{"mu_min", "lb", "alpha_max"}`` (relative eV) computed
            from the just-fitted GP -- see _compute_certificate. ``lb`` is a list
            aligned with ``cert_betas``. None skips the (small) extra work.
        cert_betas: Confidence multipliers for the certificate lower bound ``lb``
            (``mu - beta*sigma``); one bound is computed per beta so the offline
            replay can calibrate beta. Only used when ``cert_out`` is provided.

    Returns:
        np.ndarray: A 1-D array of length n_dims containing the proposed dihedral coordinates in degrees.
    """
    # make a copy of the train_X to standardize
    # we know these are in degrees already
    train_x = train_X.clone() / 360.0

    # Clip the energies if needed
    raw_y = train_y  # relative energies (eV), positive = worse
    energy_cap = 2 + torch.log10(torch.clamp(raw_y, min=1))

    # Negate (minimize -> maximize) and standardize (matching botorch.standardize).
    # Keep the std so the gradient observations can be scaled by exactly the same
    # factor as the values -- see _fit_gradient_gp.
    neg = (-torch.minimum(raw_y, energy_cap))[:, None]
    y_std = neg.std(dim=0, keepdim=True)
    y_std = torch.where(y_std >= 1e-9, y_std, torch.ones_like(y_std))
    neg_mean = neg.mean(dim=0, keepdim=True)  # kept to invert the transform for cert_out
    train_y = (neg - neg_mean) / y_std

    # Time the GP fit/condition and the acquisition optimization separately (logged
    # to the certificate as t_gp_fit / t_acq) so the BO overhead can be attributed.
    _t_fit0 = time.perf_counter()
    with _suppress_fit_warnings():
        if use_gradients and observed_gradients is not None:
            gp = _fit_gradient_gp(
                train_x, train_y, raw_y, energy_cap, observed_gradients, y_std,
                frozen_hypers=gp_frozen_hypers,
                gradient_window=gradient_window, gradient_keep=gradient_keep,
            )
            # Snapshot the fitted hyperparameters only when the caller asks (cold-fit
            # steps), so frozen condition-only steps don't clone them needlessly.
            if gp_hyper_out is not None:
                gp_hyper_out["hypers"] = {
                    k: v.detach().clone() for k, v in gp.state_dict().items()
                }
        else:
            # TODO: make the GP only once and reuse via updates
            gp = SingleTaskGP(
                train_x,
                train_y,
                covar_module=_periodic_covar_module(
                    train_x.shape[1], lengthscale_prior=lengthscale_prior
                ),
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device=train_x.device)
            fit_gpytorch_mll_torch(
                mll,
                step_limit=_GP_FIT_STEPS,
                optimizer=lambda p: torch.optim.Adam(p, lr=0.01),
            )
        t_gp_fit = time.perf_counter() - _t_fit0

        _t_acq0 = time.perf_counter()
        # LogExpectedImprovement has been the best-performing botorch acquisition here.
        base_acqf = LogExpectedImprovement(gp, best_f=torch.max(train_y), maximize=True)

        # Wrap with PiBO if prior is provided and exponent > 0
        if prior_module is not None and prior_exponent > 0:
            # botorch optimizes over normalized [0, 1] bounds (see below), so the
            # prior must interpret its inputs the same way. A module built directly
            # with the DihedralPriorModule default (input_in_degrees=True) would
            # silently mis-scale; require the normalized convention from
            # create_prior_module instead of failing quietly.
            if getattr(prior_module, "input_in_degrees", False):
                raise ValueError(
                    "prior_module expects inputs in degrees, but the acquisition "
                    "optimizer operates in normalized [0, 1] space. Build the "
                    "module with create_prior_module (or input_in_degrees=False)."
                )
            # base_acqf is LogExpectedImprovement (log-scale) and prior_module emits a
            # log probability, so log=True makes botorch combine them additively
            # (logEI + exponent * log_prior) -- the correct PiBO form. Without log=True
            # botorch would multiply logEI by prior**exponent, inverting the prior.
            acqf = PriorGuidedAcquisitionFunction(
                acq_function=base_acqf,
                prior_module=prior_module,
                log=True,
                prior_exponent=prior_exponent,
            )
        else:
            acqf = base_acqf

        # bounds are [0, 1] for each dihedral since we standardized above
        bounds = torch.zeros(2, train_x.shape[1], device=train_x.device)
        bounds[1, :] = 1.0
        candidate, _ = optimize_acqf(
            acqf,
            bounds=bounds,
            q=1,  # q = 1: no batching
            num_restarts=acq_num_restarts,
            raw_samples=acq_raw_samples,
        )
        t_acq = time.perf_counter() - _t_acq0

    # Stopping-rule certificate: evaluate the just-fitted GP (and the unwrapped
    # log-EI) over a candidate pool while it's still in hand -- re-deriving these
    # offline would cost as much as re-running. Uses base_acqf (pure EI, no PiBO
    # term) so the log-EI rule sees the unbiased acquisition.
    if cert_out is not None:
        cert_out["t_gp_fit"] = t_gp_fit  # GP construction + fit/condition
        cert_out["t_acq"] = t_acq        # acquisition build + optimize_acqf
        cert_out.update(
            _compute_certificate(
                gp, base_acqf, train_x, candidate, y_std, neg_mean, cert_betas
            )
        )

    # make sure to convert the candidate back to degrees (.cpu() so a CUDA run
    # can hand the array back to numpy).
    return candidate.detach().cpu().numpy()[0, :] * 360.0


def _setup_initial_state(
    atoms: Atoms,
    dihedrals: list[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
    relax: bool,
    out_dir: Path | None,
    use_gradients: bool = False,
) -> OptimizationState:
    """Perform initial relaxation, evaluate starting point, and set up logging.

    Args:
        atoms: Atoms object with the initial geometry
        dihedrals: List of dihedral angles to modify
        calc: Calculator for energy evaluation
        relaxCalc: Calculator used for geometry relaxation
        relax: Whether to relax non-dihedral degrees of freedom
        out_dir: Output path for logging information
        use_gradients: If True, also record dE/dtheta at the starting point so it
            contributes a gradient observation like every other point.

    Returns:
        OptimizationState with initial values
    """
    if relax:
        logger.info("Initial relaxation")
        _, init_atoms = relax_structure(
            atoms, calc, relaxCalc, DEFAULT_RELAXATION_STEPS
        )
        if out_dir is not None:
            save_structure(out_dir, init_atoms, "relaxed.xyz")
    else:
        init_atoms = atoms

    # Evaluate initial point (with dE/dtheta when gradients are enabled, so the
    # start contributes a gradient observation like every other point).
    start_coords = np.array([d.get_angle(init_atoms) for d in dihedrals])
    logger.info(f"Initial dihedral angles: {start_coords}")
    if use_gradients:
        start_energy, start_atoms, start_gradient = evaluate_energy_with_gradient(
            start_coords, atoms, dihedrals, calc, relaxCalc, relax
        )
    else:
        start_energy, start_atoms = evaluate_energy(
            start_coords, atoms, dihedrals, calc, relaxCalc, relax
        )
        start_gradient = np.full(len(start_coords), np.nan, dtype=float)
    logger.info(f"Computed initial energy: {start_energy}")

    # Set up logging if output directory provided
    add_entry = None
    if out_dir is not None:
        log_path, ens_path = initialize_structure_log(out_dir)
        add_entry = create_structure_logger(log_path, ens_path, start_energy)
        add_entry(start_coords, start_atoms, start_energy)

    device = _get_device()
    state = OptimizationState(
        start_atoms=start_atoms,
        start_coords=start_coords,
        start_energy=start_energy,
        init_steps=0,
        observed_coords=torch.tensor(
            np.asarray([start_coords]), dtype=torch.float64, device=device
        ),
        observed_energies=torch.tensor([0.0], dtype=torch.float64, device=device),
        # Start-point gradient (real dE/dtheta when use_gradients, else NaN),
        # index-aligned with the energy/coord tensors.
        observed_gradients=torch.tensor(
            np.asarray([start_gradient]), dtype=torch.float64, device=device
        ),
        use_gradients=use_gradients,
        device=device,
        best_step=0,
        add_entry=add_entry,
    )
    # Keep observed_atoms aligned with the initial [0.0] observation.
    state.observed_atoms.append(start_atoms.copy())
    return state


def plan_initial_points(
    prior_module: DihedralPriorModule,
    n_dihedrals: int,
    start_coords: np.ndarray,
    init_steps: int,
    grid_budget: int,
    seed: int,
    max_points: int | None = None,
) -> np.ndarray:
    """Plan initial dihedral guesses from the peaks of a dihedral prior.

    Builds either a systematic grid over the prior's peaks (when the full
    Cartesian product of per-axis modes fits within ``grid_budget``) or a
    weighted random sample from those peaks (when it does not, or when
    ``init_steps`` points are wanted from a large space). Dihedrals with a
    uniform prior carry their starting-geometry angle in the systematic grid
    (the start geometry comes from ETKDG or supplied conformers, so it is
    physically realistic) and are drawn uniformly at random when sampling.

    The systematic grid is ordered **best-first** by descending joint prior
    weight (the product of the per-axis mode weights), so the most probable mode
    combinations are evaluated first. This improves anytime behavior and means
    that truncating to ``max_points`` keeps the most probable conformers.

    Args:
        prior_module: Prior whose peaks seed the guesses (see ``peak_modes``).
        n_dihedrals: Number of dihedral dimensions.
        start_coords: Starting dihedral angles (degrees), used to fill
            uniform-prior dimensions in the systematic grid.
        init_steps: Number of guesses to draw when sampling.
        grid_budget: Maximum systematic grid size before falling back to
            sampling.
        seed: Random seed for the sampling fallback.
        max_points: Optional cap on the number of guesses returned. For the
            systematic grid this truncates to the ``max_points`` highest-weight
            mode combinations (leaving budget for later refinement); for sampling
            it caps the number of draws.

    Returns:
        Array of shape ``(n_points, n_dihedrals)`` in degrees [0, 360).
    """
    axes, uniform_dims = prior_module.peak_modes()
    start_coords = np.asarray(start_coords, dtype=float)
    rng = np.random.default_rng(seed)

    grid_size = 1
    for _dims, candidates in axes:
        grid_size *= len(candidates)

    # Systematic grid over every peak combination (uniform dims keep their
    # start angle, so the only variation there comes from the optimizer).
    if axes and grid_size <= grid_budget:
        candidate_lists = [candidates for _dims, candidates in axes]
        # Evaluate the most probable mode combinations first: the joint weight is
        # the product of each axis's mode weight. Best-first ordering makes the
        # early-budget points the likely conformers and makes truncation principled.
        combos = sorted(
            itertools.product(*candidate_lists),
            key=lambda combo: float(np.prod([w for _v, w in combo])),
            reverse=True,
        )
        if max_points is not None:
            combos = combos[:max_points]
        points = []
        for combo in combos:
            pt = start_coords.copy()
            for (dims, _c), (values, _w) in zip(axes, combo):
                for dim, val in zip(dims, values):
                    pt[dim] = val
            points.append(pt % 360.0)
        return np.array(points)

    # Weighted sampling from the peaks. With no peaked axes at all this degrades
    # to uniform-random guesses, so "peaks" still yields a spread of points.
    axis_weights = [
        np.array([w for _v, w in candidates], dtype=float)
        for _dims, candidates in axes
    ]
    axis_weights = [w / w.sum() for w in axis_weights]

    target = init_steps if max_points is None else min(init_steps, max_points)
    points = []
    seen = set()
    max_attempts = max(20 * target, 100)
    for _ in range(max_attempts):
        if len(points) >= target:
            break
        pt = start_coords.copy()
        for d in uniform_dims:
            pt[d] = rng.uniform(0.0, 360.0)
        for (dims, candidates), weights in zip(axes, axis_weights):
            values, _w = candidates[rng.choice(len(candidates), p=weights)]
            for dim, val in zip(dims, values):
                pt[dim] = val
        pt = pt % 360.0
        # Dedup on rounded coordinates: independent draws collide when peaks are
        # few, but distinct uniform-dim values keep otherwise-equal points apart.
        key = tuple(np.round(pt, 3))
        if key in seen:
            continue
        seen.add(key)
        points.append(pt)

    if not points:
        return np.empty((0, n_dihedrals))
    return np.array(points)


def _evaluate_point(
    state: OptimizationState,
    guess: np.ndarray,
    dihedrals: list[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
    relax: bool,
) -> tuple[float, Atoms, np.ndarray | None]:
    """Evaluate a dihedral guess, optionally also returning dE/dtheta.

    When ``state.use_gradients`` is set this projects the calculator's forces
    onto the torsion coordinates (eV/rad); otherwise the gradient is ``None``.
    """
    if state.use_gradients:
        energy, cur_atoms, gradient = evaluate_energy_with_gradient(
            guess, state.start_atoms, dihedrals, calc, relaxCalc, relax
        )
    else:
        energy, cur_atoms = evaluate_energy(
            guess, state.start_atoms, dihedrals, calc, relaxCalc, relax
        )
        gradient = None
    # --retain-bonds: a relaxed geometry that changed connectivity is a different
    # species, not a conformer; give it a failure energy so it can never be picked
    # as best (and drop its gradient so it doesn't bias the GP).
    if (
        state.required_bonds is not None
        and energy < RELAX_FAILURE_ENERGY_EV
        and bonds_broken(cur_atoms, state.required_bonds)
    ):
        state.n_bond_breaks += 1
        energy = RELAX_FAILURE_ENERGY_EV
        gradient = None
    return energy, cur_atoms, gradient


def _evaluate_initial_guesses(
    state: OptimizationState,
    dihedrals: list[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
    relax: bool,
    init_steps: int,
    seed: int,
    initial_conformers: list[Atoms] | None,
    initial_dihedrals: np.ndarray | None = None,
) -> None:
    """Evaluate initial guesses and update state in-place.

    Precedence: provided conformers, then prior-peak guesses
    (``initial_dihedrals``), then random Gaussian guesses around the start.

    Args:
        state: Optimization state to update
        dihedrals: List of dihedral angles to modify
        calc: Calculator for energy evaluation
        relaxCalc: Calculator used for geometry relaxation
        relax: Whether to relax non-dihedral degrees of freedom
        init_steps: Number of random guesses if no conformers/peaks provided
        seed: Random seed for random sampling
        initial_conformers: Optional list of conformer structures
        initial_dihedrals: Optional array of dihedral guesses (degrees),
            shape (n_points, n_dihedrals), e.g. from ``plan_initial_points``
    """
    # Build the list of guess angles (degrees) once, per the precedence above;
    # only the source and a log label differ, so a single evaluation loop follows.
    # Conformer angles are used as reported by get_angle; peak/random guesses are
    # wrapped to [0, 360) so standardization later sees a consistent range.
    if initial_conformers is not None:
        guesses = [
            np.array([d.get_angle(conformer) for d in dihedrals])
            for conformer in initial_conformers
        ]
        label = "provided conformer"
    elif initial_dihedrals is not None:
        guesses = [np.asarray(g, dtype=float) % 360.0 for g in initial_dihedrals]
        label = "prior-peak guess"
    else:
        rng = np.random.default_rng(seed)
        guesses = list(
            rng.normal(
                state.start_coords, INITIAL_GUESS_STD, size=(init_steps, len(dihedrals))
            )
            % 360.0
        )
        label = "random initial guess"

    state.init_steps = len(guesses)
    logger.info(f"Evaluating {state.init_steps} initial guesses ({label}s)")
    for i, guess in enumerate(guesses):
        energy, cur_atoms, gradient = _evaluate_point(
            state, guess, dihedrals, calc, relaxCalc, relax
        )
        rel_energy = energy - state.start_energy
        logger.info(
            f"Evaluated {label} {i+1: >3}/{state.init_steps}. "
            f"Energy-E0: {rel_energy:12.6f}"
        )
        state.append_observation(guess, rel_energy, cur_atoms, gradient)
        if state.add_entry is not None:
            state.add_entry(guess, cur_atoms, energy)


def _grad_gp_refit_decision(
    step: int,
    step_uses_gradients: bool,
    schedule_active: bool,
    dense_until: int,
    refit_every: int,
    grad_gp_hypers: dict | None,
) -> tuple[dict | None, dict | None]:
    """Decide the gradient-GP fit mode for one BO step.

    Returns ``(frozen_hypers, hyper_out)`` for ``_select_next_points_botorch``:
    - ``frozen_hypers`` is None on a cold fit (optimize fresh hyperparameters) or
      the stored hyperparameters on a condition-only update (reuse them, no Adam).
    - ``hyper_out`` is a fresh dict to capture the fitted hyperparameters, but only
      when a cold fit runs under an active schedule (so a later step can freeze on
      them); otherwise None.

    A cold fit happens when there is no active schedule, no frozen hyperparameters
    yet, we are still in the dense phase (``step < dense_until``), or a periodic
    refresh is due. See the schedule notes in ``_run_optimization_loop`` and
    ``_fit_gradient_gp``.
    """
    cold_fit = True
    if step_uses_gradients and schedule_active and grad_gp_hypers is not None:
        on_refit = refit_every > 0 and (step - dense_until) % refit_every == 0
        cold_fit = step < dense_until or on_refit
    frozen_hypers = None if cold_fit else grad_gp_hypers
    hyper_out = {} if step_uses_gradients and cold_fit and schedule_active else None
    return frozen_hypers, hyper_out


def _log_improvement_geometry(
    state: OptimizationState,
    atoms: Atoms,
    kind: str,
    n_calls: int | None = None,
    e_e0_eV: float | None = None,
) -> None:
    """Append one geometry frame to the benchmark geometry trail (no-op if off).

    ``kind`` is ``init_best`` (best after the initial guesses), ``improvement``
    (a new best-so-far during the BO loop), or ``final`` (the unconstrained-relaxed
    best). ``n_calls`` aligns the frame with the certificate CSV (n_calls there is
    the cumulative evaluation count); it defaults to the current observation count.
    Energies are relative eV (the e_e0 convention).
    """
    if state.geom_log_path is None:
        return
    if n_calls is None:
        n_calls = len(state.observed_energies)
    if e_e0_eV is None:
        e_e0_eV = state.observed_energies.min().item()
    comment = f"n_calls={n_calls} e_e0_eV={e_e0_eV:.6f} kind={kind}"
    append_xyz_frame(state.geom_log_path, atoms, comment)


# Phase 2.5 low-mode move relaxation budgets. The constrained pre-relax removes the
# worst clashes at the kicked dihedrals; the UNCONSTRAINED relax then lets every DOF
# (dihedrals included) slide to the nearest local minimum -- the step that bends a
# straight kick onto the curved fold valley (the straight-line dihedral path is
# clash-barrier-blocked; see the Phase 2.4 diagnostics in bouquet_hdbo_plan.md).
_LOWMODE_CONSTRAINED_STEPS = 20
_LOWMODE_FREE_STEPS = 100


def _collective_kick_relax(
    target_deg: np.ndarray,
    anchor: Atoms,
    dihedrals: list[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
) -> tuple[float, Atoms]:
    """Shared collective-move evaluation: set the proposed dihedrals and short CONSTRAINED
    relax (clash cleanup), then release the constraint and relax UNCONSTRAINED to a local
    minimum -- so the geometry can slide along the curved fold valley a standard, dihedral-
    pinned BO step cannot cross. Used by both the low-mode kick and the category broadcast.
    Returns (energy [eV], relaxed atoms)."""
    _, atoms = evaluate_energy(
        target_deg, anchor, dihedrals, calc, relaxCalc,
        relax=True, steps=_LOWMODE_CONSTRAINED_STEPS,
    )
    atoms.set_constraint()
    return relax_structure(atoms, calc, relaxCalc, _LOWMODE_FREE_STEPS)


def _collective_result(
    state: OptimizationState,
    atoms: Atoms,
    energy: float,
    dihedrals: list[DihedralInfo],
) -> tuple[np.ndarray, Atoms, float] | None:
    """Finish a collective move: reject it (return None) if the UNCONSTRAINED relax changed
    connectivity under --retain-bonds, else return (relaxed dihedrals [deg], atoms, energy)."""
    if (
        state.required_bonds is not None
        and energy < RELAX_FAILURE_ENERGY_EV
        and bonds_broken(atoms, state.required_bonds)
    ):
        state.n_bond_breaks += 1
        return None
    new_coords = np.array([di.get_angle(atoms) for di in dihedrals])
    return new_coords, atoms, energy


def _low_mode_move(
    state: OptimizationState,
    dihedrals: list[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
    rng: np.random.Generator,
) -> tuple[np.ndarray, Atoms, float] | None:
    """One low-mode / basin-hopping move: kick the incumbent along a soft (position-PCA)
    direction, then relax UNCONSTRAINED, returning (relaxed dihedrals [deg], relaxed
    atoms, energy [eV]).

    Unlike a standard BO step -- which relaxes only the non-dihedral DOF with the
    dihedrals pinned -- the unconstrained relaxation lets the dihedrals move, so the
    geometry can slide along the curved fold valley instead of stalling on the
    straight-line clash ridge (Kolossvary-Guida low-mode search). Returns ``None`` to
    fall back to a standard step (too few elite points, or a move that broke bonds)."""
    from bouquet.subspace import LowEnergySubspace

    coords = state.observed_coords.detach().cpu().numpy()
    energies = state.observed_energies.detach().cpu().numpy()
    # Clamp to the torsion count: the PCA basis has at most `d` columns, so a larger
    # k would let the `rng.integers(k)` mode pick index past the available columns.
    k = min(state.lowmode_modes, coords.shape[1])
    ss = LowEnergySubspace(n_dihedrals=coords.shape[1])
    ss.update(coords, energies)
    if ss.n_elite < k + 1:
        return None

    # Anchor = the incumbent's actual 3D structure (lowest-energy observation).
    best_idx = int(state.observed_energies.argmin().item())
    anchor = state.observed_atoms[best_idx]

    # Kick direction: a random one of the k soft modes (random sign), amplitude
    # ~ lowmode_kick_deg RMS/dih.
    #  "pca" -- top-k position-PCA of the low-energy set (the default and benchmark
    #           winner: adaptive directions that re-aim as relaxation walks the incumbent).
    #  "enm" -- softest elastic-network modes projected into torsion space (bouquet.enm);
    #           DORMANT: data-independent, lost to PCA in the Phase-C benchmark.
    V = None
    if state.lowmode_kick_dir == "enm":
        from bouquet.enm import enm_dihedral_modes
        chains = [tuple(di.chain) for di in dihedrals]
        V = enm_dihedral_modes(anchor.get_positions(), chains, k)
        if V.shape[1] == 0:                      # no usable ENM mode
            V = None
    if V is None:                                # PCA (default, or ENM fallback)
        _, V = ss.pca_basis(k)                   # (d, k) tangent (radian) directions
    direction = V[:, rng.integers(V.shape[1])]
    direction = direction / np.linalg.norm(direction)
    if rng.random() < 0.5:
        direction = -direction
    # incumbent + amp*direction, wrapped to torsion degrees (reuse LowEnergySubspace.line).
    amp = math.radians(state.lowmode_kick_deg) * math.sqrt(direction.shape[0])
    target_deg = ss.line(direction, np.array([amp]))[0]

    energy, atoms = _collective_kick_relax(target_deg, anchor, dihedrals, calc, relaxCalc)
    return _collective_result(state, atoms, energy, dihedrals)


def _build_category_groups(
    prior_module: DihedralPriorModule | None, n_dihedrals: int
) -> list[list[int]]:
    """Partition dihedral indices into tied categories for the category move.

    Two dihedrals share a group iff the prior's SMARTS matcher assigned them the same
    *specific* fitted-library category (an integer type id) -- i.e. they are the same
    chemical rotor (e.g. the k-th dihedral of every monomer in a foldamer). The generic
    builtin fallbacks (``sp3_sp3``/``sp3_sp2``, string type ids), unassigned dihedrals,
    and bivariate-pair dihedrals are NOT a shared rotor, so each becomes its own singleton
    group -- matching the ``max_spec`` in scripts/cat_suitability.py and the
    ``CAT_MAXSPEC_THRESHOLD`` auto-enable calibration. A homopolymer foldamer thus collapses
    to a handful of groups (the repeat unit); a molecule with no specific repeats stays at
    one group per dihedral (the move degenerates to ordinary REMBO). Returns the groups as
    sorted lists of indices, ordered by first member.
    """
    key_to_members: dict = {}
    assignments = getattr(prior_module, "univariate_assignments", {}) or {}
    for d in range(n_dihedrals):
        t = assignments.get(d)
        # Only an integer (fitted-library) id is a real shared rotor; generic string ids
        # and unassigned/bivariate dihedrals get their own singleton group.
        key = ("cat", t) if isinstance(t, int) else ("solo", d)
        key_to_members.setdefault(key, []).append(d)
    groups = sorted(key_to_members.values(), key=lambda m: m[0])
    return groups


def _sample_category_z_from_prior(
    state: OptimizationState, rng: np.random.Generator
) -> np.ndarray:
    """Draw one reduced-space point (one value per category group, degrees) from the
    prior, for warming up the reduced-space GP. Each group's value is sampled from a
    mode of that category's fitted 1D von Mises mixture (weighted by mixture weight,
    with a small angular jitter); a uniform (no-preference) category draws uniformly.
    """
    groups = state.category_groups or []
    pm = state.prior_module
    uni_dists = getattr(pm, "univariate_dists", {}) if pm is not None else {}
    z = np.empty(len(groups), dtype=float)
    for gi, members in enumerate(groups):
        dist = uni_dists.get(members[0])
        if dist is None:
            z[gi] = rng.uniform(0.0, 360.0)
            continue
        locs = dist.component_distribution.loc.detach().cpu().numpy()      # radians
        probs = dist.mixture_distribution.probs.detach().cpu().numpy()
        probs = probs / probs.sum()
        k = int(rng.choice(len(locs), p=probs))
        z[gi] = (math.degrees(float(locs[k])) + rng.normal(0.0, 15.0)) % 360.0
    return z


def _category_move(
    state: OptimizationState,
    dihedrals: list[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
    rng: np.random.Generator,
) -> tuple[np.ndarray, Atoms, float] | None:
    """One category-tied collective move: choose a per-category value vector (from the
    prior while the reduced buffer is small, else by reduced-space LogEI), set every
    dihedral in a category to its category's value on top of the incumbent, then
    constrained + UNCONSTRAINED relax -- returning (relaxed dihedrals [deg], relaxed
    atoms, energy [eV]).

    This is the chemistry-defined counterpart of :func:`_low_mode_move`: instead of
    kicking along a data-derived PCA direction, it moves along the fixed block-constant
    embedding implied by the SMARTS categories, so a correlated fold is one reduced-space
    point and is reachable from the very first move. The reduced point is recorded in
    ``state._cat_Z/_cat_Y`` (the low-D BO buffer) before the connectivity check, so even a
    rejected move informs the reduced GP. Returns ``None`` to fall back to a standard step
    (no groups, or a move that broke bonds under --retain-bonds).

    NOTE (prototype): the reduced coordinates use period 360 for every category. Aromatic
    categories are physically 180-periodic (theta == theta+180), so their search domain is
    2x larger than necessary -- correct, just not maximally efficient. Per-category period
    is the obvious next refinement.
    """
    groups = state.category_groups
    if not groups:
        return None

    best_idx = int(state.observed_energies.argmin().item())
    anchor = state.observed_atoms[best_idx]
    incumbent_deg = state.observed_coords[best_idx].detach().cpu().numpy()

    # Reduced-space point z (one value per category), in degrees. Below the warmup the
    # prior seeds it; after that the reduced-space GP picks it -- the same value-only
    # periodic-GP + LogEI selector the outer loop uses, just over the n_group-dim buffer.
    if len(state._cat_Z) < state.category_min_moves:
        z = _sample_category_z_from_prior(state, rng)
    else:
        z = _select_next_points_botorch(
            torch.tensor(np.asarray(state._cat_Z), dtype=torch.float64, device=state.device),
            torch.tensor(np.asarray(state._cat_Y), dtype=torch.float64, device=state.device),
            acq_num_restarts=state.acq_num_restarts,
            acq_raw_samples=state.acq_raw_samples,
        )

    # Broadcast: every dihedral takes its group's value (singletons get their own
    # coordinate). The groups partition all indices, so the incumbent copy is just a
    # defensive base -- every entry is overwritten.
    target_deg = incumbent_deg.copy()
    for gi, members in enumerate(groups):
        target_deg[members] = z[gi] % 360.0

    energy, atoms = _collective_kick_relax(target_deg, anchor, dihedrals, calc, relaxCalc)

    # Record the reduced-space observation (energy relative to the run's start) for the
    # low-D BO, before the connectivity check so a rejected z still teaches the GP.
    state._cat_Z.append([float(v) for v in z])
    state._cat_Y.append(float(energy - state.start_energy))

    return _collective_result(state, atoms, energy, dihedrals)


def _run_optimization_loop(
    state: OptimizationState,
    n_steps: int,
    dihedrals: list[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
    relax: bool,
    out_dir: Path | None,
) -> None:
    """Run the Bayesian optimization loop, updating state in-place.

    Args:
        state: Optimization state to update
        n_steps: Number of optimization steps to perform
        dihedrals: List of dihedral angles to modify
        calc: Calculator for energy evaluation
        relaxCalc: Calculator used for geometry relaxation
        relax: Whether to relax non-dihedral degrees of freedom
        out_dir: Output path for saving best structure
    """
    # Optional grad->value handoff: use the gradient-enhanced GP only for the first
    # `gradient_steps` BO steps, then the cheap value-only GP (see OptimizationState).
    switch_at = state.gradient_steps if state.gradient_steps > 0 else None

    # Gradient-GP hyperparameter refit schedule. A full fit runs ~200 Choleskys and
    # its cost grows steeply with the observation count, so fitting every step
    # dominates the run late on. Instead: full (cold) fits for the first
    # `dense_until` BO steps -- hyperparameters move most early and the matrices are
    # still small -- then FREEZE them and only re-condition, which folds each new
    # point in under the frozen hyperparameters for one Cholesky instead of ~200
    # (a cold fit vs. a condition-only update, selected per step below via
    # `gp_frozen_hypers`). `refit_every > 0` optionally refreshes the frozen
    # hyperparameters with a *cold* refit every that many post-dense steps (cold,
    # not warm-started: warm-starting the fit drifts the hyperparameters and
    # degrades the search). The shipped default freezes after `dense_until` steps;
    # the opt-out `dense_until=0` (with `refit_every<=1`) refits every step -- the
    # slow full-gradient reference. See _fit_gradient_gp.
    dense_until = max(0, state.grad_refit_dense_until)
    refit_every = state.grad_refit_every
    schedule_active = dense_until > 0 or refit_every > 1

    # Geometry trail (benchmark): snapshot the best after the initial guesses, so a
    # global minimum already found in the init phase is captured before any BO step.
    if state.geom_log_path is not None and len(state.observed_energies):
        best_idx = state.observed_energies.argmin().item()
        _log_improvement_geometry(
            state, state.observed_atoms[best_idx], "init_best",
            n_calls=best_idx + 1, e_e0_eV=state.observed_energies[best_idx].item(),
        )

    loop_start = time.perf_counter()
    for step in range(n_steps):
        step_uses_gradients = state.use_gradients and (
            switch_at is None or step < switch_at
        )
        if switch_at is not None and state.use_gradients and step == switch_at:
            logger.info(
                f"Switching to value-only GP after {switch_at} gradient-enhanced "
                f"step(s) (gradient GP cost grows with observation count)."
            )

        # Per-step gradient-GP cost: a cold full fit vs. a condition-only update that
        # reuses the frozen hyperparameters and runs no Adam (see the schedule notes
        # above and _grad_gp_refit_decision).
        frozen_hypers, hyper_out = _grad_gp_refit_decision(
            step, step_uses_gradients, schedule_active, dense_until, refit_every,
            state.grad_gp_hypers,
        )

        # Collective moves: with some probability (past a warmup) this step is replaced
        # by a committed move + UNCONSTRAINED relax that combines selection and
        # evaluation, so it short-circuits the standard step. Two flavors, checked in
        # order and mutually exclusive per step:
        #   Phase 3  category move -- tie same-SMARTS-category dihedrals (chemistry-defined
        #             embedding, available from step 0); takes precedence when enabled.
        #   Phase 2.5 low-mode move -- kick along a data-derived PCA soft mode.
        collective_result = None
        t_collective = 0.0
        if (
            state.category_prob > 0
            and len(state.observed_energies) >= state.category_warmup
            and state.category_rng is not None
            and state.category_rng.random() < state.category_prob
        ):
            _t0 = time.perf_counter()
            collective_result = _category_move(
                state, dihedrals, calc, relaxCalc, state.category_rng
            )
            t_collective = time.perf_counter() - _t0
            if collective_result is not None:
                state.category_count += 1

        if (
            collective_result is None
            and state.lowmode_prob > 0
            and len(state.observed_energies) >= state.lowmode_warmup
            and state.lowmode_rng is not None
            and state.lowmode_rng.random() < state.lowmode_prob
        ):
            _t0 = time.perf_counter()
            collective_result = _low_mode_move(
                state, dihedrals, calc, relaxCalc, state.lowmode_rng
            )
            t_collective = time.perf_counter() - _t0
            if collective_result is not None:
                state.lowmode_count += 1

        if collective_result is not None:
            next_coords, cur_atoms, energy = collective_result
            gradient = None
            cert_out = None
            t_select, t_eval = 0.0, t_collective
        else:
            # ---- standard BO step: fit the GP and optimize the acquisition ----
            cert_out = {} if state.cert_log is not None else None
            # Time the two cost centers separately: GP fit/condition + acquisition
            # optimization (t_select) vs the xTB energy evaluation + relaxation (t_eval).
            _t0 = time.perf_counter()
            next_coords = _select_next_points_botorch(
                state.observed_coords, state.observed_energies,
                prior_module=state.prior_module,
                prior_exponent=state.prior_exponent,
                observed_gradients=state.observed_gradients,
                use_gradients=step_uses_gradients,
                gp_frozen_hypers=frozen_hypers,
                gp_hyper_out=hyper_out,
                cert_out=cert_out,
                cert_betas=state.cert_betas,
                acq_num_restarts=state.acq_num_restarts,
                acq_raw_samples=state.acq_raw_samples,
                gradient_window=state.gradient_window,
                gradient_keep=state.gradient_keep,
                lengthscale_prior=state.lengthscale_prior,
            )
            t_select = time.perf_counter() - _t0
            if hyper_out:  # a cold fit; carry its hyperparameters forward to freeze
                state.grad_gp_hypers = hyper_out["hypers"]
            # logger.info(f'Selected next point: {next_coords}')

            _t0 = time.perf_counter()
            energy, cur_atoms, gradient = _evaluate_point(
                state, next_coords, dihedrals, calc, relaxCalc, relax
            )
            t_eval = time.perf_counter() - _t0
            if cert_out is not None:
                cert_out["t_select"] = t_select  # GP + acquisition (incl. certificate)
                cert_out["t_eval"] = t_eval      # xTB energy + relaxation
        rel_energy = energy - state.start_energy
        logger.info(
            f"Evaluated energy in step {step+1: >3}/{n_steps}. Energy-E0: {rel_energy:12.6f}"
        )

        if rel_energy < state.observed_energies.min().item():
            state.best_step = step
            if out_dir is not None:
                save_structure(out_dir, cur_atoms, "current_best.xyz")
            # Geometry trail: this BO point is a new best-so-far. n_calls matches the
            # certificate row about to be logged (post-append observation count).
            _log_improvement_geometry(
                state, cur_atoms, "improvement",
                n_calls=len(state.observed_energies) + 1, e_e0_eV=rel_energy,
            )

        if state.add_entry is not None:
            state.add_entry(next_coords, cur_atoms, energy)

        state.append_observation(next_coords, rel_energy, cur_atoms, gradient)

        # Per-step stopping-rule certificate row. cert_out was filled during
        # selection (GP fit on data through step-1, predicting this step); pair it
        # with this step's realized outcome. n_calls counts cumulative energy
        # evaluations (start + initial guesses + BO steps so far) -- the real cost
        # axis -- which equals the post-append observation count.
        if state.cert_log is not None and cert_out is not None:
            state.cert_log(
                step,
                rel_energy,
                state.observed_energies.min().item(),
                len(state.observed_energies),
                time.perf_counter() - loop_start,
                cert_out,
            )

        # Decay prior exponent
        if state.prior_module is not None:
            state.prior_exponent *= state.prior_decay

    if state.lowmode_prob > 0:
        logger.info(
            f"Phase 2.5: {state.lowmode_count} low-mode (kick + unconstrained-relax) "
            f"move(s) of {n_steps} BO steps."
        )
    if state.category_prob > 0:
        n_groups = len(state.category_groups) if state.category_groups else 0
        logger.info(
            f"Phase 3: {state.category_count} category-tied move(s) of {n_steps} BO "
            f"steps over {n_groups} categor(y/ies) for {len(dihedrals)} dihedrals."
        )


def _perform_final_relaxation(
    state: OptimizationState,
    dihedrals: list[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
) -> Atoms:
    """Perform final relaxation steps and return best atoms.

    Performs two relaxations: first with dihedral constraints, then without.

    Args:
        state: Optimization state with best structure
        dihedrals: List of dihedral angles to modify
        calc: Calculator for energy evaluation
        relaxCalc: Calculator used for geometry relaxation

    Returns:
        Final optimized Atoms structure
    """
    best_idx = state.observed_energies.argmin().item()
    best_coords = state.observed_coords[best_idx].cpu().numpy()

    logger.info(f"Best energy found on step {state.best_step + 1}")

    # Report when the search first entered the best basin (within 10 kcal/mol of the
    # best energy) and first reached it (within 1 kcal/mol). Step numbers are offset
    # by init_steps so they count BO steps rather than initial guesses.
    delta = (state.observed_energies - state.observed_energies[best_idx]).abs()
    near = torch.nonzero(delta < KCAL_TO_EV * 10.0, as_tuple=False)
    good = torch.nonzero(delta < KCAL_TO_EV, as_tuple=False)
    if near.numel():
        logger.info(f"Found low energy on step {near[0].item() - state.init_steps}")
    if good.numel():
        logger.info(
            f"Found first good energy on step {good[0].item() - state.init_steps}"
        )

    # Seed from the actual best observation (aligned with best_idx/best_coords),
    # which may come from the initial point, a seeded conformer, a random guess,
    # or the BO loop -- best_step only tracks BO-loop wins, so don't key off it here.
    best_energy, best_atoms = evaluate_energy(
        best_coords, state.observed_atoms[best_idx], dihedrals, calc, relaxCalc, steps=None
    )
    logger.info(
        f"Performed final relaxation with dihedral constraints. "
        f"E: {best_energy}. E-E0: {best_energy - state.start_energy}"
    )
    if state.add_entry is not None:
        state.add_entry(best_coords, best_atoms, best_energy)

    # Relaxation without dihedral constraints
    constrained_atoms = best_atoms.copy()  # retained-bonds fallback
    constrained_energy = best_energy  # its energy, restored alongside if we revert
    best_atoms.set_constraint()
    best_energy, best_atoms = evaluate_energy(
        best_coords, best_atoms, dihedrals, calc, relaxCalc, steps=None
    )
    logger.info(
        f"Performed final relaxation without dihedral constraints. "
        f"E: {best_energy}. E-E0: {best_energy - state.start_energy}"
    )
    # --retain-bonds: if releasing the dihedral constraints let the geometry
    # rearrange, keep the constrained (bond-preserving) result instead.
    if state.required_bonds is not None and bonds_broken(
        best_atoms, state.required_bonds
    ):
        logger.warning(
            "Final unconstrained relaxation broke a bond; "
            "reverting to the constrained best (--retain-bonds)."
        )
        best_atoms = constrained_atoms
        best_energy = constrained_energy

    best_coords = np.array([d.get_angle(best_atoms) for d in dihedrals])
    if state.add_entry is not None:
        state.add_entry(best_coords, best_atoms, best_energy)

    return best_atoms


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
    gp = SingleTaskGP(
        train_x,
        train_y,
        covar_module=_periodic_covar_module(train_x.shape[1]),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device=train_x.device)
    fit_gpytorch_mll_torch(
        mll, step_limit=_GP_FIT_STEPS, optimizer=lambda p: torch.optim.Adam(p, lr=0.01)
    )
    return gp


def _select_ensemble_candidates(
    state: OptimizationState,
    window_eV: float,
    p_threshold: float,
    sigma_floor_eV: float,
    failure_energy_eV: float,
) -> list[tuple[np.ndarray, Atoms]]:
    """Select observed conformers to tightly optimize, ordered by predicted energy.

    A conformer ``i`` is included iff its GP posterior gives
    ``P(E_i <= E_min + window) >= p_threshold``. The posterior sigma supplies a
    per-candidate, data-driven buffer: tight where the surface is well sampled,
    wide where it is sparse. No candidate cap is applied.
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

    # Probabilistic inclusion: P(E_i <= E_min + window) >= p_threshold.
    e_min = e.min()
    z = (e_min + window_eV - mu) / sigma
    keep = torch.special.ndtr(z) >= p_threshold  # standard-normal CDF

    # Map survivors back to global indices, ordered by predicted energy
    # (no cap on the number kept).
    sel_global = idx[keep][torch.argsort(mu[keep])]

    logger.info(
        f"Ensemble selection: {sel_global.numel()} candidates "
        f"(from {idx.numel()} valid observations)."
    )
    return [
        (coords[i].cpu().numpy(), state.observed_atoms[i])
        for i in sel_global.tolist()
    ]


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


def run_optimization(
    atoms: Atoms,
    dihedrals: list[DihedralInfo],
    n_steps: int,
    calc: Calculator,
    relaxCalc: Calculator,
    init_steps: int,
    out_dir: Path | None,
    relax: bool = True,
    seed: int = 0,
    initial_conformers: list[Atoms] | None = None,
    initial_dihedrals: np.ndarray | None = None,
    # New PiBO parameters
    prior_module: DihedralPriorModule | None = None,
    initial_prior_exponent: float = 2.0,
    prior_exponent_decay: float = 0.9,
    return_ensemble: bool = False,
    use_gradients: bool = False,
    gradient_steps: int = 0,
    grad_refit_dense_until: int = 20,
    grad_refit_every: int = 0,
    acq_num_restarts: int = ACQ_NUM_RESTARTS,
    acq_raw_samples: int = ACQ_RAW_SAMPLES,
    gradient_window: int = 0,
    gradient_keep: str = "recent",
    lengthscale_prior: str = "auto",
    lowmode_prob: float | None = None,
    lowmode_warmup: int = 100,
    lowmode_kick_deg: float = 60.0,
    lowmode_modes: int = 4,
    lowmode_kick_dir: str = "pca",
    category_prob: float | None = None,
    category_warmup: int = 20,
    category_min_moves: int = 6,
    cert_log_path: Path | None = None,
    cert_betas: tuple[float, ...] = DEFAULT_CERTIFICATE_BETAS,
    geom_log_path: Path | None = None,
    retain_bonds: bool = False,
) -> Atoms | tuple[Atoms, list[tuple[Atoms, float, float]]]:
    """Optimize the structure of a molecule by iteratively changing the dihedral angles.

    Args:
        atoms: Atoms object with the initial geometry
        dihedrals: List of dihedral angles to modify
        n_steps: Number of optimization steps to perform
        init_steps: Number of initial guesses to evaluate (ignored if initial_conformers provided)
        calc: Calculator to pick the energy
        relaxCalc: Calculator used for geometry relaxation
        out_dir: Output path for logging information
        relax: Whether to relax non-dihedral degrees of freedom each step
        seed: Random seed to use for initial sampling
        initial_conformers: Optional list of conformer structures to use as initial guesses
                           instead of random sampling
        initial_dihedrals: Optional array of dihedral guesses (degrees) to use as
                           initial points (e.g. from plan_initial_points). Used
                           when no conformers are provided; falls back to random.
        return_ensemble: If True, also select and tightly optimize a Boltzmann
            ensemble of low-energy conformers and return it alongside the best
            structure.
        use_gradients: If True, record dE/dtheta at each evaluation and use the
            gradient-enhanced periodic GP surrogate for acquisition. With
            ``relax=True`` the projected gradient is only consistent with the
            energy objective when ``calc`` and ``relaxCalc`` are the same surface
            (the envelope theorem needs the geometry to be a constrained minimum
            of the energy calculator); pass matching calculators in that case.
        gradient_steps: If > 0 (and ``use_gradients``), use the gradient-enhanced
            GP only for the first ``gradient_steps`` BO steps, then switch to the
            value-only GP for the remainder. The gradient GP's per-step cost grows
            steeply with the observation count, so this caps it on large molecules
            while keeping the early-search benefit. <=0 keeps gradients for the
            whole run; a budget smaller than ``gradient_steps`` never switches.
        grad_refit_dense_until: Number of leading BO steps that do a full
            gradient-GP hyperparameter fit. Refitting is the dominant cost (~200
            Choleskys, growing with observation count), so beyond this the
            hyperparameters are frozen and later steps only re-condition (one
            Cholesky). Hyperparameters move most early, when the matrices are still
            small, so the cold fits there are both useful and cheap. Default 20
            (validated quality-neutral vs full refitting for 5-11 dihedrals); 0
            refits every step (the slow full-gradient reference).
        grad_refit_every: After the dense phase, optionally do a *cold* refit of the
            frozen hyperparameters every this many BO steps (<=0 or 1 with no dense
            phase keeps the original full-fit-every-step behavior; >0 with a dense
            phase refreshes periodically, 0 freezes for the rest of the run). Cold,
            not warm-started: warm-starting the fit drifts the hyperparameters and
            degrades the search.

    Returns:
        If ``return_ensemble`` is False, the optimized lowest-energy geometry as
        an Atoms object. If True, a tuple ``(best_atoms, ensemble)`` where
        ``ensemble`` is a list of ``(atoms, relative_energy_eV, weight)`` sorted
        by energy ascending (``best_atoms`` is the lowest-energy ensemble member,
        or the single-best relaxation if the ensemble is empty).
    """
    # Validate the gradient-windowing knobs up front: _restrict_gradient_mask
    # silently treats window <= 0 as keep-all and any unknown mode as "both", so
    # an out-of-range value would otherwise change the surrogate without warning.
    if gradient_keep not in ("recent", "best", "both"):
        raise ValueError(
            f"gradient_keep must be one of 'recent', 'best', 'both', got {gradient_keep!r}"
        )
    if gradient_window < 0:
        raise ValueError(
            f"gradient_window must be a non-negative integer (0 = all), got {gradient_window}"
        )

    # High-d auto defaults: unless explicitly overridden, turn on the dimensionality-
    # scaled lengthscale prior and low-mode search once the dihedral count crosses the
    # high-d threshold -- both help there (see the HDBO benchmarks) and are off at low d.
    # An explicit prior name, or a concrete lowmode_prob, always wins over "auto"/None.
    n_dihedrals = len(dihedrals)
    if lengthscale_prior == "auto":
        lengthscale_prior = (
            "dim_scaled" if n_dihedrals >= HIGH_D_DIHEDRAL_THRESHOLD else "none"
        )
    if lowmode_prob is None:
        lowmode_prob = 0.5 if n_dihedrals >= HIGH_D_DIHEDRAL_THRESHOLD else 0.0

    # Category-tied move auto default. Like low-mode it turns on at high d, but ALSO gated
    # on repeat structure: the benchmark shows the win requires a real tie to exploit
    # (max_spec = the largest tied SMARTS category), and that on large-but-irregular
    # molecules (high d, low max_spec) the tie mildly HURTS -- so raw d is not enough.
    # Needs a prior_module (no categories without one). Build the groups once here and
    # reuse them below. An explicit category_prob always wins over None/auto.
    category_groups = (
        _build_category_groups(prior_module, n_dihedrals)
        if prior_module is not None else None
    )
    if category_prob is None:
        max_spec = max((len(g) for g in category_groups), default=0) if category_groups else 0
        category_prob = 0.5 if (
            n_dihedrals > CAT_D_THRESHOLD and max_spec > CAT_MAXSPEC_THRESHOLD
        ) else 0.0

    # A low-mode move is a kick followed by a constrained + UNCONSTRAINED relaxation
    # (see _low_mode_move); it is meaningless without relaxation and would silently
    # relax structures in a run that asked not to. Reject the combination up front.
    if lowmode_prob > 0 and not relax:
        raise ValueError(
            "lowmode_prob > 0 requires relax=True: low-mode moves are kick + relax steps"
        )

    # A category-tied move broadcasts a per-category value then relaxes (see
    # _category_move); like the low-mode move it is meaningless without relaxation, and
    # it needs the prior's SMARTS categories to know which dihedrals to tie.
    if category_prob > 0:
        if not relax:
            raise ValueError(
                "category_prob > 0 requires relax=True: category moves are broadcast + relax steps"
            )
        if prior_module is None:
            raise ValueError(
                "category_prob > 0 requires a prior_module: the SMARTS categories define "
                "which dihedrals are tied. Pass --priors / build one with create_prior_module."
            )

    # Seed every RNG the run touches from `seed`, so a run is reproducible and the
    # seed actually controls the search. The acquisition optimizer (optimize_acqf)
    # draws its restart points from torch's global RNG at every BO step; without
    # this that dominant stochasticity is uncontrolled (and config.seed would only
    # affect the numpy-drawn initial guesses). Seeding torch also enables
    # common-random-number paired comparisons across arms at a fixed seed.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup initial state (relaxation, starting point, logging)
    state = _setup_initial_state(
        atoms, dihedrals, calc, relaxCalc, relax, out_dir, use_gradients=use_gradients
    )
    state.gradient_steps = gradient_steps
    state.grad_refit_dense_until = grad_refit_dense_until
    state.grad_refit_every = grad_refit_every
    state.acq_num_restarts = acq_num_restarts
    state.acq_raw_samples = acq_raw_samples
    state.gradient_window = gradient_window
    state.gradient_keep = gradient_keep
    state.lengthscale_prior = lengthscale_prior
    state.lowmode_prob = lowmode_prob
    state.lowmode_warmup = lowmode_warmup
    state.lowmode_kick_deg = lowmode_kick_deg
    state.lowmode_modes = lowmode_modes
    state.lowmode_kick_dir = lowmode_kick_dir
    # Dedicated RNG for the low-mode-move coin/direction, offset from the global seed so
    # it doesn't co-vary with the torch/numpy streams but stays reproducible (paired
    # comparisons across arms at a fixed seed).
    if lowmode_prob > 0:
        state.lowmode_rng = np.random.default_rng(seed + 99991)

    # Category-tied move setup: partition the dihedrals into SMARTS categories once, and
    # give the move its own reproducible RNG (offset from the seed, distinct from the
    # low-mode stream) for the move coin and prior-seeded warmup draws.
    state.category_prob = category_prob
    state.category_warmup = category_warmup
    state.category_min_moves = category_min_moves
    if category_prob > 0:
        # Reuse the partition built for the auto-default decision above (prior_module is
        # non-None here: category_prob > 0 was validated to require it).
        state.category_groups = category_groups
        state.category_rng = np.random.default_rng(seed + 88883)
        logger.info(
            f"Phase 3 category move enabled: {len(state.category_groups)} categor(y/ies) "
            f"for {len(dihedrals)} dihedrals (embedding dim "
            f"{len(state.category_groups)})."
        )

    # --retain-bonds: adopt the (relaxed) start structure's covalent bond set as the
    # reference every later evaluation must preserve. The start is computed inside
    # _setup_initial_state above and defines the molecule's connectivity.
    if retain_bonds:
        state.required_bonds = geometry_bond_set(state.start_atoms)

    # Optional per-step stopping-rule certificate log (calibration benchmark).
    if cert_log_path is not None:
        state.cert_betas = tuple(cert_betas)
        state.cert_log = create_certificate_logger(cert_log_path, state.cert_betas)

    # Optional geometry trail (benchmark): start a fresh file; frames are appended
    # at each best-so-far improvement (and the final relaxed best) for the offline
    # RMSD-identity / distinct-conformer analysis.
    if geom_log_path is not None:
        state.geom_log_path = Path(geom_log_path)
        state.geom_log_path.open("w").close()

    # Add prior settings to state
    state.prior_module = prior_module
    state.prior_exponent = initial_prior_exponent
    state.prior_decay = prior_exponent_decay

    # Evaluate initial guesses (conformers, prior peaks, or random)
    _evaluate_initial_guesses(
        state, dihedrals, calc, relaxCalc, relax, init_steps, seed,
        initial_conformers, initial_dihedrals,
    )

    # Run Bayesian optimization loop
    _run_optimization_loop(state, n_steps, dihedrals, calc, relaxCalc, relax, out_dir)

    if state.required_bonds is not None and state.n_bond_breaks:
        logger.info(
            f"--retain-bonds: rejected {state.n_bond_breaks} evaluation(s) that "
            f"changed connectivity."
        )

    if return_ensemble:
        # Select and tightly optimize the low-energy ensemble. The best member
        # is the lowest-energy conformer; fall back to single-best relaxation if
        # the ensemble comes back empty (e.g. all evaluations failed).
        ensemble = _perform_ensemble_relaxation(state, dihedrals, calc, relaxCalc)
        if ensemble:
            best_atoms = ensemble[0][0]
        else:
            best_atoms = _perform_final_relaxation(state, dihedrals, calc, relaxCalc)
        _log_improvement_geometry(state, best_atoms, "final")
        return best_atoms, ensemble

    # Final relaxation and return best structure
    best_atoms = _perform_final_relaxation(state, dihedrals, calc, relaxCalc)
    _log_improvement_geometry(state, best_atoms, "final")
    return best_atoms
