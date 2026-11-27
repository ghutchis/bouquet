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
from collections.abc import Sequence
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
    ENERGY_CLIP_OFFSET,
    GP_PERIOD_LENGTH_MEAN,
    GP_PERIOD_LENGTH_STD,
    KERNEL_ARMS,
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


def _periodic_kernel_kwargs(
    num_dims: int, lengthscale_prior: str
) -> tuple[dict, float | None]:
    """Shared PeriodicKernel kwargs and optional LogNormal prior median."""
    kwargs: dict = dict(
        period_length_prior=NormalPrior(GP_PERIOD_LENGTH_MEAN, GP_PERIOD_LENGTH_STD)
    )
    loc = None
    if lengthscale_prior == "dim_scaled":
        loc = math.sqrt(2.0) + 0.5 * math.log(num_dims)
        kwargs["lengthscale_prior"] = LogNormalPrior(loc, math.sqrt(3.0))
    elif lengthscale_prior != "none":
        raise ValueError(
            f"lengthscale_prior must be 'none' or 'dim_scaled', got {lengthscale_prior!r}"
        )
    return kwargs, loc


class _SymmetryPeriodicKernel(gpykernels.PeriodicKernel):
    """PeriodicKernel with per-dimension ``m_j^2`` lengthscale compensation.

    In symmetry-folded coordinates ``u_j = (m_j * theta_j / 360) mod 1``, sharing
    one lengthscale ``l`` magnifies the local angular curvature of a symmetric
    rotor by ``m_j^2`` (9x for a C3 methyl, 4x for a C2 phenyl), causing low-
    barrier methyls to dominate the GP distance metric over backbone torsions.
    Scaling dimension ``j``'s effective lengthscale to ``m_j^2 * l`` preserves
    the exact physical angular curvature locally while keeping exact
    ``360 / m_j`` periodicity globally. Works with both shared ``l`` and ARD.
    """

    def __init__(self, symmetry_orders: Sequence[int], **kwargs):
        super().__init__(**kwargs)
        self.register_buffer(
            "_sym_sq",
            torch.tensor([float(m * m) for m in symmetry_orders], dtype=torch.float64),
        )

    def forward(self, x1, x2, diag=False, **params):
        last_dim_is_batch = params.pop("last_dim_is_batch", False)
        x1_ = x1.div(self.period_length / math.pi)
        x2_ = x2.div(self.period_length / math.pi)
        diff = self.covar_dist(x1_, x2_, diag=diag, last_dim_is_batch=True, **params)

        sym_sq = self._sym_sq.to(diff)
        lengthscale = self.lengthscale[..., 0, :] * sym_sq  # (*batch, d)
        if diag:
            lengthscale = lengthscale[..., :, None]
        else:
            lengthscale = lengthscale[..., :, None, None]
        exp_term = diff.sin().pow(2.0).div(lengthscale).mul(-2.0)

        if not last_dim_is_batch:
            exp_term = exp_term.sum(dim=(-2 if diag else -3))

        return exp_term.exp()


def _periodic_covar_module(
    num_dims: int,
    lengthscale_prior: str = "none",
    symmetry_orders: Sequence[int] | None = None,
    ard: bool = False,
) -> gpykernels.ScaleKernel:
    """Build the periodic GP covariance module shared by the acquisition GP and
    the ensemble-selection GP (a scaled product of per-dihedral periodic kernels).

    ``lengthscale_prior`` selects the prior on the (shared) periodic lengthscale:
    - ``"none"`` (default): no lengthscale prior -- the historical behavior, where
      the lengthscale is a free MLL fit.
    - ``"dim_scaled"``: the Hvarfner et al. (ICML 2024) dimensionality-scaled
      LogNormal prior, ``LogNormal(sqrt(2) + 0.5*ln d, sqrt(3))``, biasing the GP
      toward smoother fits as ``d`` grows. The lengthscale is also initialized at
      the prior median.

    When ``symmetry_orders`` contains any ``m_j > 1``, uses
    :class:`_SymmetryPeriodicKernel` so dimension ``j`` has effective lengthscale
    ``m_j^2 * l`` (preserving physical angular curvature under ``u_j = m_j * x_j``).

    ``ard=True`` gives each torsion its own lengthscale. ``PeriodicKernel`` with
    ``ard_num_dims=d`` is already the product form
    ``exp(-2 sum_m sin^2(pi dx_m / p) / l_m)``; without ARD, the non-symmetric
    path wraps it in ``ProductStructureKernel`` to share one ``l`` across torsions.
    """
    base_kwargs, loc = _periodic_kernel_kwargs(num_dims, lengthscale_prior)
    if ard:
        base_kwargs["ard_num_dims"] = num_dims
    symmetric = symmetry_orders is not None and any(m > 1 for m in symmetry_orders)
    if symmetric:
        # Folded inputs span exactly one irreducible period per dimension, so the
        # period is a structural constant (1.0), not a fitted hyperparameter --
        # matching the gradient GP. A fitted period would break the exact 360/m_j
        # periodicity the folding relies on.
        base_kwargs.pop("period_length_prior", None)
        base_kernel = _SymmetryPeriodicKernel(symmetry_orders, **base_kwargs)
        base_kernel.period_length = 1.0
        base_kernel.raw_period_length.requires_grad_(False)
    else:
        base_kernel = gpykernels.PeriodicKernel(**base_kwargs)
    if loc is not None:
        base_kernel.lengthscale = math.exp(loc)  # start at the prior median
    if symmetric or ard:
        return gpykernels.ScaleKernel(base_kernel)
    return gpykernels.ScaleKernel(
        gpykernels.ProductStructureKernel(num_dims=num_dims, base_kernel=base_kernel)
    )


# Covariance arms. The shipped kernel is a product of per-torsion periodic kernels
# sharing ONE lengthscale ("product"); "ard" is the same product form with a
# lengthscale per torsion.
#
# A first-order ADDITIVE component and two pairwise ("trellis") variants were also
# built and measured, and all were removed. They are worth not re-deriving:
#   * pairwise psi+- terms lost to a plain no-pairwise control (-0.119 nats), and the
#     fitted sum/difference weights came out symmetric, i.e. no directional structure
#     was ever used;
#   * the additive term won held-out prediction decisively (+0.25 to +0.30 nats, 93%
#     of 509 trials, uniform in d and n) and then made the SEARCH WORSE end-to-end
#     (+0.41 kcal/mol vs product, 63% of paired trials worse), with the deficit
#     GROWING with budget -- the over-exploitation signature, since a sharper
#     posterior shrinks sigma and EI stops exploring.
# Predictive accuracy and search utility turned out anti-correlated here. Keep that
# in mind before trusting a cross-validation result about this surrogate again.
COVAR_ARMS = KERNEL_ARMS


def make_covar_module(
    num_dims: int,
    lengthscale_prior: str = "none",
    arm: str = "product",
    symmetry_orders: Sequence[int] | None = None,
):
    """Build the surrogate covariance for one arm (see :data:`COVAR_ARMS`).

    ``"product"`` shares one lengthscale across every torsion; ``"ard"`` gives each
    torsion its own lengthscale (for ``num_dims >= 2``; 1D falls back to shared).
    """
    if arm not in COVAR_ARMS:
        raise ValueError(f"arm must be one of {COVAR_ARMS}, got {arm!r}")
    return _periodic_covar_module(
        num_dims, lengthscale_prior, symmetry_orders=symmetry_orders,
        ard=(arm == "ard" and num_dims >= 2),
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


def resolve_gradient_window(
    n_points: int, num_dims: int, window: int, matrix_cap: int, window_min: int
) -> int:
    """Effective gradient window W for an augmented GP of size ``N = n + W*d``.

    Capping N rather than W is what actually bounds cost: a fixed W leaves N growing
    linearly in d, and measured per-step acquisition time is ~linear in N. The rule is

        W = clamp((matrix_cap - n) / d,  window_min,  n)

    with two escapes. ``window > 0`` pins W explicitly and ignores the cap (benchmark
    arms and anyone who wants a fixed W). ``matrix_cap <= 0`` disables capping, the
    pre-cap keep-all behaviour.

    Returns 0 -- meaning "keep every gradient", the no-op ``_restrict_gradient_mask``
    understands -- whenever the budget is already wide enough to cover all n points, so
    low-d runs are untouched. When ``n >= matrix_cap`` the budget goes negative and
    ``window_min`` takes over; that is not a failure but the regime where the step
    budget simply exceeds the cap (e.g. C(12) = 5215 > 4000), and no window can honour
    it.
    """
    if window > 0:
        return window
    if matrix_cap <= 0 or num_dims <= 0:
        return 0
    budget = (matrix_cap - n_points) // num_dims
    if budget >= n_points:
        return 0  # cap not binding at this n: every gradient fits
    return max(window_min, int(budget))


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


class _SymmetryScaledPrior(torch.nn.Module):
    """Score symmetry-folded ``u in [0, 1]^d`` under a ``DihedralPriorModule`` built
    in the ``[0, 1] = [0, 360 deg)`` convention.

    A folded ``u_j`` stands for all ``m_j`` equivalent full-turn angles
    ``(u_j + k) / m_j``; scoring only one of them (``u / m``) would ignore prior mass
    at the others and make the boundaries u=0 and u=1 -- the same physical angle --
    score differently. The prior module therefore combines every representative per
    factor (jointly for bivariate pairs); order-1 dimensions are unchanged."""

    def __init__(self, prior_module: torch.nn.Module, sym_orders: Sequence[int]):
        super().__init__()
        self.prior_module = prior_module
        self.sym_orders = tuple(int(m) for m in sym_orders)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.prior_module(X, symmetry_orders=self.sym_orders)


def _periods_deg_array(
    num_dims: int, symmetry_orders: Sequence[int] | None
) -> np.ndarray:
    """Per-dihedral irreducible period in degrees (``360 / m_j``)."""
    if symmetry_orders is None:
        return np.full(num_dims, 360.0, dtype=float)
    orders = np.asarray(symmetry_orders, dtype=float)
    if orders.shape != (num_dims,):
        raise ValueError(
            f"symmetry_orders length {orders.shape} != num_dims ({num_dims},)"
        )
    return 360.0 / orders


def _deg_to_unit(x_deg: torch.Tensor, periods_deg: torch.Tensor | float) -> torch.Tensor:
    """Map torsions in degrees into the symmetry-reduced unit cube.

    With a per-dihedral period tensor, ``u_j = (theta_j mod period_j) / period_j``;
    with the scalar 360.0 (no symmetry) the input is only rescaled, not wrapped.
    """
    if torch.is_tensor(periods_deg):
        return torch.remainder(x_deg, periods_deg) / periods_deg
    return x_deg / float(periods_deg)


def _fit_gradient_gp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    raw_y: torch.Tensor,
    energy_cap: torch.Tensor,
    observed_gradients: torch.Tensor,
    y_std: torch.Tensor,
    frozen_hypers: dict | None = None,
    gradient_window: int = 0,
    gradient_matrix_cap: int = 0,
    gradient_window_min: int = 1,
    gradient_keep: str = "recent",
    kernel_arm: str = "product",
    symmetry_orders: Sequence[int] | None = None,
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
    over symmetry-folded inputs ``u_j = (m_j * degrees / 360) mod 1`` where
    ``m_j`` is the dihedral's rotational symmetry order (1 for asymmetric rotors).
    The stored gradients are ``dE/dtheta`` in eV/rad, so the chain rule for the
    matching gradient observation is

        dy'/du_j = -(2*pi / (m_j * std)) * dE/dtheta_j   [clamp inactive]

    (``theta_rad_j = (2*pi / m_j) * u_j``; the maximization sign flip contributes
    the minus). ``y_std`` is the standardization std the caller applied to the
    values, so the gradients are scaled by exactly the same factor. Points where
    the energy clamp is active or whose gradient is NaN (failed evaluation) are
    dropped from the gradient observations via the mask, but still contribute
    their value.
    """
    grad = observed_gradients.to(train_x)  # (n, d), eV/rad
    if symmetry_orders is not None:
        sym = torch.as_tensor(symmetry_orders, dtype=train_x.dtype, device=train_x.device)
        grad_scaled = -(2.0 * math.pi / (y_std * sym)) * grad  # (n, d)
    else:
        grad_scaled = -(2.0 * math.pi / y_std) * grad  # (n, d)

    clamp_inactive = raw_y <= energy_cap  # (n,)
    grad_valid = ~torch.isnan(grad).any(dim=1)  # (n,)
    mask = clamp_inactive & grad_valid
    # High-leverage subset: keep gradients for only a window of points (caps the
    # augmented matrix at n + window*d -- the main high-d/late cost). The window is
    # resolved PER FIT, not once at setup, because it depends on the current n: the
    # cap keeps everything early and tightens as observations accumulate.
    window = resolve_gradient_window(
        train_x.shape[0], train_x.shape[1],
        gradient_window, gradient_matrix_cap, gradient_window_min,
    )
    mask = _restrict_gradient_mask(mask, raw_y, window, gradient_keep)
    grad_scaled = torch.nan_to_num(grad_scaled, nan=0.0)

    gp = GradientEnhancedPeriodicGP(
        train_x, train_y, grad_scaled, grad_mask=mask, period=1.0,
        kernel_arm=kernel_arm, condition=False,
        symmetry_orders=symmetry_orders,
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


# Feasible-pool sizing for the clash-constrained acquisition. The pool is drawn in
# chunks until enough clash-free points are found (or the draw budget runs out),
# because the feasible fraction of the box collapses with dihedral count -- roughly
# 40% at d=5 but a few percent by d=12 (see bouquet.clash).
_FEASIBLE_CHUNK = 512
_FEASIBLE_MAX_CHUNKS = 8
_FEASIBLE_POOL_SEED = 4242


def _draw_feasible_pool(
    feasible,
    num_dims: int,
    n_target: int,
    dtype: torch.dtype,
    device: torch.device,
    seed_offset: int = 0,
    periods_deg: np.ndarray | float = 360.0,
) -> torch.Tensor | None:
    """Sobol points in [0, 1]^num_dims whose 3D geometry is clash-free.

    Draws in chunks until at least ``n_target`` feasible points are in hand or the
    chunk budget is exhausted. ``seed_offset`` advances the Sobol scramble across
    BO steps so each step explores a fresh feasible pool (while staying
    reproducible at a fixed run seed). ``periods_deg`` scales ``[0, 1]`` to the
    irreducible symmetry period ``[0, 360 / m_j)`` for each dihedral. Returns None
    if nothing feasible turned up, which tells the caller to fall back to the
    unconstrained optimizer rather than stall.
    """
    bounds = torch.stack(
        [
            torch.zeros(num_dims, device=device, dtype=dtype),
            torch.ones(num_dims, device=device, dtype=dtype),
        ]
    )
    kept: list[torch.Tensor] = []
    found = 0
    base_seed = _FEASIBLE_POOL_SEED + seed_offset
    for chunk in range(_FEASIBLE_MAX_CHUNKS):
        x = draw_sobol_samples(
            bounds=bounds, n=_FEASIBLE_CHUNK, q=1, seed=base_seed + chunk
        ).squeeze(1)
        mask = feasible(x.detach().cpu().numpy() * periods_deg)
        n_ok = int(mask.sum())
        if n_ok:
            kept.append(x[torch.as_tensor(mask, device=x.device)])
            found += n_ok
        if found >= n_target:
            break
    return torch.cat(kept) if kept else None


def _optimize_acqf_clash_free(
    acqf,
    bounds: torch.Tensor,
    num_restarts: int,
    raw_samples: int,
    feasible,
    seed_offset: int = 0,
    periods_deg: np.ndarray | float = 360.0,
) -> torch.Tensor:
    """``optimize_acqf`` restricted to the clash-free region of the torsion box.

    botorch's ``nonlinear_inequality_constraints`` path wants a differentiable
    constraint and switches the inner solver to SLSQP; the clash test is a
    non-differentiable geometry build, so it is applied by construction instead:

      1. draw a Sobol pool and keep only the clash-free points,
      2. seed the L-BFGS-B restarts from the best-acquisition ones,
      3. accept the refined candidate only if it is *still* clash-free, otherwise
         return the best feasible pool point.

    Step 3 matters: seeding from a feasible point does not keep the local
    refinement inside the feasible set, and the constraint is worthless if the
    optimizer can walk back out of it.

    Falls back to the unconstrained optimizer when ``feasible`` is None or no
    feasible point is found, so a molecule whose box is (by this criterion)
    entirely clashed still makes progress.
    """
    num_dims = bounds.shape[1]
    pool = None if feasible is None else _draw_feasible_pool(
        feasible, num_dims, num_restarts, bounds.dtype, bounds.device,
        seed_offset=seed_offset, periods_deg=periods_deg,
    )
    if pool is None:
        candidate, _ = optimize_acqf(
            acqf, bounds=bounds, q=1,
            num_restarts=num_restarts, raw_samples=raw_samples,
        )
        return candidate

    with torch.no_grad():
        values = acqf(pool.unsqueeze(1))
    n_ic = min(num_restarts, len(pool))
    top = torch.topk(values, n_ic).indices
    best_pool_point = pool[top[0]].unsqueeze(0)

    candidate, _ = optimize_acqf(
        acqf, bounds=bounds, q=1,
        num_restarts=n_ic,
        batch_initial_conditions=pool[top].unsqueeze(1),
    )
    if bool(feasible(candidate.detach().cpu().numpy() * periods_deg)[0]):
        return candidate
    return best_pool_point


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
    gradient_matrix_cap: int = 0,
    gradient_window_min: int = 1,
    gradient_keep: str = "both",
    lengthscale_prior: str = "none",
    feasible=None,
    kernel_arm: str = "product",
    feasible_seed_offset: int = 0,
    symmetry_orders: Sequence[int] | None = None,
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
        kernel_arm: Which covariance to use (see :data:`COVAR_ARMS`), on both the
            value-only and gradient-enhanced paths. "product" is the shipped
            kernel: a product of per-torsion periodic kernels sharing ONE
            lengthscale; "ard" gives each torsion its own lengthscale.
        feasible: Optional steric-clash feasibility test restricting the
            acquisition domain to the clash-free region. Called with a batch of
            dihedral vectors in DEGREES, shape (n_candidates, n_dims), and must
            return a boolean mask of the same length. None searches the full box.
            See :mod:`bouquet.clash`.

    Returns:
        np.ndarray: A 1-D array of length n_dims containing the proposed dihedral coordinates in degrees.
    """
    # Normalize degrees into the symmetry-reduced irreducible unit cube [0, 1)^d:
    # each dihedral j with symmetry order m_j has period 360 / m_j degrees, so
    # u_j = (theta_j mod period_j) / period_j has exact unit period 1.0 in the GP.
    num_dims = train_X.shape[1]
    has_symmetry = symmetry_orders is not None and any(m > 1 for m in symmetry_orders)
    periods_deg = _periods_deg_array(num_dims, symmetry_orders) if has_symmetry else 360.0
    train_x = _deg_to_unit(
        train_X,
        torch.as_tensor(periods_deg, dtype=train_X.dtype, device=train_X.device)
        if has_symmetry else 360.0,
    )

    # Clip the energies if needed
    raw_y = train_y  # relative energies (eV), positive = worse
    energy_cap = ENERGY_CLIP_OFFSET + torch.log10(torch.clamp(raw_y, min=1))

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
                gradient_window=gradient_window,
                gradient_matrix_cap=gradient_matrix_cap,
                gradient_window_min=gradient_window_min,
                gradient_keep=gradient_keep,
                kernel_arm=kernel_arm,
                symmetry_orders=symmetry_orders if has_symmetry else None,
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
                covar_module=make_covar_module(
                    train_x.shape[1],
                    lengthscale_prior=lengthscale_prior,
                    arm=kernel_arm,
                    symmetry_orders=symmetry_orders if has_symmetry else None,
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
            effective_prior = (
                _SymmetryScaledPrior(prior_module, symmetry_orders)
                if has_symmetry
                else prior_module
            )
            acqf = PriorGuidedAcquisitionFunction(
                acq_function=base_acqf,
                prior_module=effective_prior,
                log=True,
                prior_exponent=prior_exponent,
            )
        else:
            acqf = base_acqf

        # bounds are [0, 1] for each dihedral since we standardized above
        bounds = torch.zeros(2, train_x.shape[1], dtype=train_x.dtype, device=train_x.device)
        bounds[1, :] = 1.0
        candidate = _optimize_acqf_clash_free(
            acqf, bounds, acq_num_restarts, acq_raw_samples, feasible,
            seed_offset=feasible_seed_offset, periods_deg=periods_deg,
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

    # Convert the candidate from [0, 1]^d back to degrees in [0, 360 / m_j)
    # (.cpu() so a CUDA run can hand the array back to numpy).
    return candidate.detach().cpu().numpy()[0, :] * periods_deg
