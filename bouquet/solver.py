"""Methods for solving the conformer option problem"""

import itertools
import logging
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

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
from gpytorch import kernels as gpykernels
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import NormalPrior

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

warnings.filterwarnings("ignore")

from bouquet.assess import (
    evaluate_energy,
    evaluate_energy_with_gradient,
    relax_structure,
)
from bouquet.config import (
    ACQ_NUM_RESTARTS,
    ACQ_RAW_SAMPLES,
    DEFAULT_RELAXATION_STEPS,
    ENSEMBLE_ENERGY_TOL_KCAL,
    ENSEMBLE_P_THRESHOLD,
    ENSEMBLE_RMSD_THRESHOLD,
    ENSEMBLE_SIGMA_FLOOR_KCAL,
    ENSEMBLE_TEMPERATURE,
    ENSEMBLE_WINDOW_KCAL,
    FAILURE_ENERGY_EV,
    GP_PERIOD_LENGTH_MEAN,
    GP_PERIOD_LENGTH_STD,
    INITIAL_GUESS_STD,
    KB_EV_PER_K,
    KCAL_TO_EV,
)
from bouquet.gradient_gp import GradientEnhancedPeriodicGP
from bouquet.io import create_structure_logger, initialize_structure_log, save_structure
from bouquet.priors import DihedralPriorModule
from bouquet.setup import DihedralInfo

logger = logging.getLogger(__name__)

# Marginal-likelihood Adam iterations for a cold gradient-GP hyperparameter fit
# (see _fit_gradient_gp). Condition-only updates run 0 steps. A benchmark found
# conformer quality is insensitive to fit convergence (50/100/200 steps gave
# statistically indistinguishable minima despite very different final NLLs), so 100
# trims the dense-phase cold fits at no quality cost; 50 showed a faint degradation
# and is under-converged for larger molecules.
_GRAD_GP_FIT_STEPS = 100


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
    observed_gradients: torch.Tensor = None  # Shape: (n_observations, n_dihedrals)
    # Per-observation Atoms, aligned index-for-index with the tensors above.
    observed_atoms: List[Atoms] = field(default_factory=list)
    device: torch.device = field(default_factory=_get_device)
    init_steps: int = 0
    best_atoms: Optional[Atoms] = None
    best_step: int = 0
    add_entry: Optional[Callable] = None
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
    grad_gp_hypers: Optional[dict] = None

    # PiBO fields
    prior_module: Optional[DihedralPriorModule] = None
    prior_exponent: float = 2.0
    prior_decay: float = 0.9

    def append_observation(
        self,
        coords: np.ndarray,
        energy: float,
        atoms: Atoms,
        gradient: Optional[np.ndarray] = None,
    ) -> None:
        """Append a new observation, keeping observed_atoms index-aligned.

        Args:
            coords: Dihedral coordinates as numpy array
            energy: Relative energy value
            atoms: Structure at this observation (copied for retention)
            gradient: Optional dE/dtheta (eV/rad) for each dihedral; stored as
                NaN when not provided so the gradient tensor stays index-aligned.
        """
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


def _periodic_covar_module(num_dims: int) -> gpykernels.ScaleKernel:
    """Build the periodic GP covariance module shared by the acquisition GP and
    the ensemble-selection GP (a scaled product of per-dihedral periodic kernels)."""
    return gpykernels.ScaleKernel(
        gpykernels.ProductStructureKernel(
            num_dims=num_dims,
            base_kernel=gpykernels.PeriodicKernel(
                period_length_prior=NormalPrior(
                    GP_PERIOD_LENGTH_MEAN, GP_PERIOD_LENGTH_STD
                )
            ),
        )
    )


def _fit_gradient_gp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    raw_y: torch.Tensor,
    energy_cap: torch.Tensor,
    observed_gradients: torch.Tensor,
    y_std: torch.Tensor,
    frozen_hypers: Optional[dict] = None,
) -> GradientEnhancedPeriodicGP:
    """Build and fit the gradient-enhanced periodic GP on standardized data.

    Two modes (see ``_run_optimization_loop``):
    - ``frozen_hypers=None``: a cold fit -- optimize fresh hyperparameters with
      ``_GRAD_GP_FIT_STEPS`` marginal-likelihood Adam iterations. The caller reads
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
    grad_scaled = torch.nan_to_num(grad_scaled, nan=0.0)

    gp = GradientEnhancedPeriodicGP(
        train_x, train_y, grad_scaled, grad_mask=mask, period=1.0
    )
    if frozen_hypers is None:
        gp.fit(steps=_GRAD_GP_FIT_STEPS, lr=0.05)  # cold fit: optimize fresh hypers
    else:
        # Condition-only: load the frozen hypers and re-condition (steps=0, no Adam).
        gp.load_state_dict(frozen_hypers, strict=False)
        gp.fit(steps=0, lr=0.05)
    return gp


def _select_next_points_botorch(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    prior_module: Optional[DihedralPriorModule] = None,
    prior_exponent: float = 0.0,
    observed_gradients: Optional[torch.Tensor] = None,
    use_gradients: bool = False,
    gp_frozen_hypers: Optional[dict] = None,
    gp_hyper_out: Optional[dict] = None,
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
    neg = (-1 * torch.minimum(raw_y, energy_cap))[:, None]
    y_std = neg.std(dim=0, keepdim=True)
    y_std = torch.where(y_std >= 1e-9, y_std, torch.ones_like(y_std))
    train_y = (neg - neg.mean(dim=0, keepdim=True)) / y_std

    if use_gradients and observed_gradients is not None:
        gp = _fit_gradient_gp(
            train_x, train_y, raw_y, energy_cap, observed_gradients, y_std,
            frozen_hypers=gp_frozen_hypers,
        )
        # Snapshot the fitted hyperparameters only when the caller asks (cold-fit
        # steps), so frozen condition-only steps don't clone them needlessly.
        if gp_hyper_out is not None:
            gp_hyper_out["hypers"] = {
                k: v.detach().clone() for k, v in gp.state_dict().items()
            }
    else:
        # Make the GP
        # TODO: make the GP only once and reuse via updates
        gp = SingleTaskGP(
            train_x,
            train_y,
            covar_module=_periodic_covar_module(train_x.shape[1]),
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device=train_x.device)
        fit_gpytorch_mll_torch(
            mll,
            step_limit=200,
            optimizer=lambda p: torch.optim.Adam(p, lr=0.01),
        )

    # Solve the optimization problem
    n_sampled, n_dim = train_x.shape
    # So far, this seems to be the best of the functions in botorch
    # Create base acquisition function
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
    bounds = torch.zeros(2, train_x.shape[1])
    bounds[1, :] = 1.0
    candidate, acq_value = optimize_acqf(
        # at the moment use q = 1 for no batching
        acqf,
        bounds=bounds,
        q=1,
        num_restarts=ACQ_NUM_RESTARTS,
        raw_samples=ACQ_RAW_SAMPLES,
    )
    # make sure to convert the candidate back to degrees
    return candidate.detach().numpy()[0, :] * 360.0


def _setup_initial_state(
    atoms: Atoms,
    dihedrals: List[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
    relax: bool,
    out_dir: Optional[Path],
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
            [start_coords], dtype=torch.float64, device=device
        ),
        observed_energies=torch.tensor([0.0], dtype=torch.float64, device=device),
        # Start-point gradient (real dE/dtheta when use_gradients, else NaN),
        # index-aligned with the energy/coord tensors.
        observed_gradients=torch.tensor(
            [start_gradient], dtype=torch.float64, device=device
        ),
        use_gradients=use_gradients,
        device=device,
        best_atoms=start_atoms.copy(),
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
    max_points: Optional[int] = None,
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
    dihedrals: List[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
    relax: bool,
) -> Tuple[float, Atoms, Optional[np.ndarray]]:
    """Evaluate a dihedral guess, optionally also returning dE/dtheta.

    When ``state.use_gradients`` is set this projects the calculator's forces
    onto the torsion coordinates (eV/rad); otherwise the gradient is ``None``.
    """
    if state.use_gradients:
        return evaluate_energy_with_gradient(
            guess, state.start_atoms, dihedrals, calc, relaxCalc, relax
        )
    energy, cur_atoms = evaluate_energy(
        guess, state.start_atoms, dihedrals, calc, relaxCalc, relax
    )
    return energy, cur_atoms, None


def _evaluate_initial_guesses(
    state: OptimizationState,
    dihedrals: List[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
    relax: bool,
    init_steps: int,
    seed: int,
    initial_conformers: Optional[List[Atoms]],
    initial_dihedrals: Optional[np.ndarray] = None,
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
    if initial_conformers is not None:
        state.init_steps = len(initial_conformers)
        logger.info(f"Using {state.init_steps} provided conformers as initial guesses")
        for i, conformer in enumerate(initial_conformers):
            guess = np.array([d.get_angle(conformer) for d in dihedrals])
            energy, cur_atoms, gradient = _evaluate_point(
                state, guess, dihedrals, calc, relaxCalc, relax
            )
            rel_energy = energy - state.start_energy
            logger.info(
                f"Evaluated conformer {i+1: >3}/{len(initial_conformers)}. Energy-E0: {rel_energy:12.6f}"
            )

            state.append_observation(guess, rel_energy, cur_atoms, gradient)

            if state.add_entry is not None:
                state.add_entry(guess, cur_atoms, energy)
    elif initial_dihedrals is not None:
        state.init_steps = len(initial_dihedrals)
        logger.info(
            f"Using {state.init_steps} prior-peak initial guesses"
        )
        for i, guess in enumerate(initial_dihedrals):
            guess = np.asarray(guess, dtype=float) % 360.0
            energy, cur_atoms, gradient = _evaluate_point(
                state, guess, dihedrals, calc, relaxCalc, relax
            )
            rel_energy = energy - state.start_energy
            logger.info(
                f"Evaluated peak guess {i+1: >3}/{len(initial_dihedrals)}. Energy-E0: {rel_energy:12.6f}"
            )

            state.append_observation(guess, rel_energy, cur_atoms, gradient)

            if state.add_entry is not None:
                state.add_entry(guess, cur_atoms, energy)
    else:
        state.init_steps = init_steps
        logger.info(f"Generating {init_steps} random initial guesses")
        rng = np.random.default_rng(seed)
        init_guesses = rng.normal(
            state.start_coords, INITIAL_GUESS_STD, size=(init_steps, len(dihedrals))
        )

        for i, guess in enumerate(init_guesses):
            # make sure angles are between 0 and 360
            # to standardize later
            guess = guess % 360.0
            energy, cur_atoms, gradient = _evaluate_point(
                state, guess, dihedrals, calc, relaxCalc, relax
            )
            rel_energy = energy - state.start_energy
            logger.info(
                f"Evaluated initial guess {i+1: >3}/{init_steps}. Energy-E0: {rel_energy:12.6f}"
            )

            state.append_observation(guess, rel_energy, cur_atoms, gradient)

            if state.add_entry is not None:
                state.add_entry(guess, cur_atoms, energy)


def _run_optimization_loop(
    state: OptimizationState,
    n_steps: int,
    dihedrals: List[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
    relax: bool,
    out_dir: Optional[Path],
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

    for step in range(n_steps):
        step_uses_gradients = state.use_gradients and (
            switch_at is None or step < switch_at
        )
        if switch_at is not None and state.use_gradients and step == switch_at:
            logger.info(
                f"Switching to value-only GP after {switch_at} gradient-enhanced "
                f"step(s) (gradient GP cost grows with observation count)."
            )

        # Per-step gradient-GP cost: a cold full fit (no schedule, dense phase,
        # periodic refresh, or the mandatory first fit) vs. a condition-only update
        # that reuses the frozen hyperparameters and runs no Adam. Only a cold fit
        # produces new hyperparameters, so only then do we snapshot them (via a
        # non-None gp_hyper_out) to freeze on later steps.
        cold_fit = True
        if step_uses_gradients and schedule_active and state.grad_gp_hypers is not None:
            on_refit = refit_every > 0 and (step - dense_until) % refit_every == 0
            cold_fit = step < dense_until or on_refit
        # Snapshot fitted hypers only on a cold fit under an active schedule -- i.e.
        # only when a later step will actually freeze on them.
        hyper_out = {} if step_uses_gradients and cold_fit and schedule_active else None

        next_coords = _select_next_points_botorch(
            state.observed_coords, state.observed_energies,
            prior_module=state.prior_module,
            prior_exponent=state.prior_exponent,
            observed_gradients=state.observed_gradients,
            use_gradients=step_uses_gradients,
            gp_frozen_hypers=None if cold_fit else state.grad_gp_hypers,
            gp_hyper_out=hyper_out,
        )
        if hyper_out:  # a cold fit; carry its hyperparameters forward to freeze
            state.grad_gp_hypers = hyper_out["hypers"]
        # logger.info(f'Selected next point: {next_coords}')

        energy, cur_atoms, gradient = _evaluate_point(
            state, next_coords, dihedrals, calc, relaxCalc, relax
        )
        rel_energy = energy - state.start_energy
        logger.info(
            f"Evaluated energy in step {step+1: >3}/{n_steps}. Energy-E0: {rel_energy:12.6f}"
        )

        if rel_energy < state.observed_energies.min().item():
            state.best_step = step
            state.best_atoms = cur_atoms.copy()
            if out_dir is not None:
                save_structure(out_dir, cur_atoms, "current_best.xyz")

        if state.add_entry is not None:
            state.add_entry(next_coords, cur_atoms, energy)

        state.append_observation(next_coords, rel_energy, cur_atoms, gradient)

        # Decay prior exponent
        if state.prior_module is not None:
            state.prior_exponent *= state.prior_decay


def _perform_final_relaxation(
    state: OptimizationState,
    dihedrals: List[DihedralInfo],
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

    # go through the energies and find the first one within 0.001 of the best energy
    # .. we'll need to subtract the number of initial guesses from the step count
    first_low_energy = False
    for i in range(len(state.observed_energies)):
        if (
            not first_low_energy
            and abs(
                state.observed_energies[i].item()
                - state.observed_energies[best_idx].item()
            )
            < KCAL_TO_EV * 10.0
        ):
            first_low_energy = True
            logger.info(f"Found low energy on step {i - state.init_steps}")
        if (
            abs(
                state.observed_energies[i].item()
                - state.observed_energies[best_idx].item()
            )
            < KCAL_TO_EV
        ):
            logger.info(f"Found first good energy on step {i - state.init_steps}")
            break

    # Seed from the actual best observation (aligned with best_idx/best_coords),
    # which may come from the initial point, a seeded conformer, a random guess,
    # or the BO loop -- not from state.best_atoms, which only tracks BO-loop wins.
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
    best_atoms.set_constraint()
    best_energy, best_atoms = evaluate_energy(
        best_coords, best_atoms, dihedrals, calc, relaxCalc, steps=None
    )
    logger.info(
        f"Performed final relaxation without dihedral constraints. "
        f"E: {best_energy}. E-E0: {best_energy - state.start_energy}"
    )

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
        mll, step_limit=200, optimizer=lambda p: torch.optim.Adam(p, lr=0.01)
    )
    return gp


def _select_ensemble_candidates(
    state: OptimizationState,
    window_eV: float,
    p_threshold: float,
    sigma_floor_eV: float,
    failure_energy_eV: float,
) -> List[Tuple[np.ndarray, Atoms]]:
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
    pairs: List[Tuple[Atoms, float]], rmsd_thr: float, e_tol_eV: float
) -> List[Tuple[Atoms, float]]:
    """Deduplicate (atoms, energy_eV) pairs, assumed sorted by energy ascending.

    A structure is a duplicate iff it is BOTH energy-close AND geometry-close to
    an already-kept structure.
    """
    unique: List[Tuple[Atoms, float]] = []
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
    dihedrals: List[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
) -> List[Tuple[Atoms, float, float]]:
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
    optimized: List[Tuple[Atoms, float]] = []
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
    dihedrals: List[DihedralInfo],
    n_steps: int,
    calc: Calculator,
    relaxCalc: Calculator,
    init_steps: int,
    out_dir: Optional[Path],
    relax: bool = True,
    seed: int = 0,
    initial_conformers: Optional[List[Atoms]] = None,
    initial_dihedrals: Optional[np.ndarray] = None,
    # New PiBO parameters
    prior_module: Optional[DihedralPriorModule] = None,
    initial_prior_exponent: float = 2.0,
    prior_exponent_decay: float = 0.9,
    return_ensemble: bool = False,
    use_gradients: bool = False,
    gradient_steps: int = 0,
    grad_refit_dense_until: int = 20,
    grad_refit_every: int = 0,
) -> Union[Atoms, Tuple[Atoms, List[Tuple[Atoms, float, float]]]]:
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

    if return_ensemble:
        # Select and tightly optimize the low-energy ensemble. The best member
        # is the lowest-energy conformer; fall back to single-best relaxation if
        # the ensemble comes back empty (e.g. all evaluations failed).
        ensemble = _perform_ensemble_relaxation(state, dihedrals, calc, relaxCalc)
        if ensemble:
            best_atoms = ensemble[0][0]
        else:
            best_atoms = _perform_final_relaxation(state, dihedrals, calc, relaxCalc)
        return best_atoms, ensemble

    # Final relaxation and return best structure
    return _perform_final_relaxation(state, dihedrals, calc, relaxCalc)
