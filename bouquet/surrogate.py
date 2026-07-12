"""Gaussian-process surrogates and acquisition for the conformer BO search.

The value-only and gradient-enhanced periodic GPs, their shared fit config, the
BOTorch acquisition step (:func:`_select_next_points_botorch`), and the per-step
stopping-rule certificate. This layer is state-free -- it takes tensors, not an
``OptimizationState`` -- so it is shared by the main loop, the category move, and
the ensemble-selection GP without importing back into :mod:`bouquet.solver`.
"""

from __future__ import annotations

# See bouquet/__init__.py: defer the heavy numeric/BO stack (Python 3.15+) so the
# import stays cheap until an optimization actually runs.
__lazy_modules__ = [
    "numpy",
    "torch",
    "botorch.acquisition.analytic",
    "botorch.acquisition.prior_guided",
    "botorch.optim.fit",
    "botorch.optim",
    "botorch.models",
    "gpytorch",
    "gpytorch.mlls",
    "gpytorch.priors",
]

import math
import time
import warnings
from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.prior_guided import PriorGuidedAcquisitionFunction
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch import kernels as gpykernels
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior, NormalPrior

from bouquet.config import (
    ACQ_NUM_RESTARTS,
    ACQ_RAW_SAMPLES,
    DEFAULT_CERTIFICATE_BETAS,
    GP_PERIOD_LENGTH_MEAN,
    GP_PERIOD_LENGTH_STD,
)
from bouquet.gradient_gp import GradientEnhancedPeriodicGP

if TYPE_CHECKING:
    from bouquet.priors import DihedralPriorModule


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


def _fit_value_gp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    covar_module: gpykernels.ScaleKernel,
    outcome_transform=None,
) -> SingleTaskGP:
    """Build and fit a value-only ``SingleTaskGP`` with the shared fit config.

    Both the acquisition GP and the ensemble-selection GP fit a periodic
    ``SingleTaskGP`` with ``fit_gpytorch_mll_torch`` (Adam, lr 0.01, capped at
    ``_GP_FIT_STEPS``); centralizing the construction and fit here keeps the two
    from drifting apart. ``covar_module`` carries the (optionally prior-equipped)
    kernel and ``outcome_transform`` is passed through (e.g. ``Standardize(m=1)``
    for the selection GP fitting energies in eV).
    """
    gp = SingleTaskGP(
        train_x,
        train_y,
        covar_module=covar_module,
        outcome_transform=outcome_transform,
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device=train_x.device)
    fit_gpytorch_mll_torch(
        mll,
        step_limit=_GP_FIT_STEPS,
        optimizer=lambda p: torch.optim.Adam(p, lr=0.01),
    )
    return gp


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
            gp = _fit_value_gp(
                train_x,
                train_y,
                covar_module=_periodic_covar_module(
                    train_x.shape[1], lengthscale_prior=lengthscale_prior
                ),
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
