"""Methods for solving the conformer option problem"""

import itertools
import logging
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
from botorch.utils import standardize
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

from bouquet.assess import evaluate_energy, relax_structure
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
from bouquet.io import create_structure_logger, initialize_structure_log, save_structure
from bouquet.priors import DihedralPriorModule
from bouquet.setup import DihedralInfo

logger = logging.getLogger(__name__)


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
    # Per-observation Atoms, aligned index-for-index with the tensors above.
    observed_atoms: List[Atoms] = field(default_factory=list)
    device: torch.device = field(default_factory=_get_device)
    init_steps: int = 0
    best_atoms: Optional[Atoms] = None
    best_step: int = 0
    add_entry: Optional[Callable] = None

    # PiBO fields
    prior_module: Optional[DihedralPriorModule] = None
    prior_exponent: float = 2.0
    prior_decay: float = 0.9

    def append_observation(
        self, coords: np.ndarray, energy: float, atoms: Atoms
    ) -> None:
        """Append a new observation, keeping observed_atoms index-aligned.

        Args:
            coords: Dihedral coordinates as numpy array
            energy: Relative energy value
            atoms: Structure at this observation (copied for retention)
        """
        new_coords = torch.tensor(
            coords, dtype=torch.float64, device=self.device
        ).unsqueeze(0)
        new_energy = torch.tensor([energy], dtype=torch.float64, device=self.device)
        self.observed_coords = torch.cat([self.observed_coords, new_coords], dim=0)
        self.observed_energies = torch.cat([self.observed_energies, new_energy], dim=0)
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


def _select_next_points_botorch(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    prior_module: Optional[DihedralPriorModule] = None,
    prior_exponent: float = 0.0,
) -> np.ndarray:
    """
    Selects the next dihedral coordinate to evaluate by fitting a Gaussian process to the observed data and optimizing a BOTorch acquisition function.

    Parameters:
        train_X (torch.Tensor): Observed dihedral coordinates, shape (n_observations, n_dims), in degrees.
        train_y (torch.Tensor): Observed energies corresponding to train_X, shape (n_observations,).
        prior_module: Optional DihedralPriorModule for PiBO
        prior_exponent: Prior strength (0 = no prior influence)

    Returns:
        np.ndarray: A 1-D array of length n_dims containing the proposed dihedral coordinates in degrees.
    """
    # make a copy of the train_X to standardize
    # we know these are in degrees already
    train_x = train_X.clone() / 360.0

    # Clip the energies if needed
    train_y = torch.clamp(train_y, max=2 + torch.log10(torch.clamp(train_y, min=1)))

    # Reshape and standardize for GP (minimize -> maximize by negating)
    train_y = train_y[:, None]
    train_y = standardize(-1 * train_y)  # make this a maximization problem

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
        acqf = PriorGuidedAcquisitionFunction(
            acq_function=base_acqf,
            prior_module=prior_module,
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
) -> OptimizationState:
    """Perform initial relaxation, evaluate starting point, and set up logging.

    Args:
        atoms: Atoms object with the initial geometry
        dihedrals: List of dihedral angles to modify
        calc: Calculator for energy evaluation
        relaxCalc: Calculator used for geometry relaxation
        relax: Whether to relax non-dihedral degrees of freedom
        out_dir: Output path for logging information

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

    # Evaluate initial point
    start_coords = np.array([d.get_angle(init_atoms) for d in dihedrals])
    logger.info(f"Initial dihedral angles: {start_coords}")
    start_energy, start_atoms = evaluate_energy(
        start_coords, atoms, dihedrals, calc, relaxCalc, relax
    )
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
) -> np.ndarray:
    """Plan initial dihedral guesses from the peaks of a dihedral prior.

    Builds either a systematic grid over the prior's peaks (when the full
    Cartesian product of per-axis modes fits within ``grid_budget``) or a
    weighted random sample from those peaks (when it does not, or when
    ``init_steps`` points are wanted from a large space). Dihedrals with a
    uniform prior carry their starting-geometry angle in the systematic grid
    (the start geometry comes from ETKDG or supplied conformers, so it is
    physically realistic) and are drawn uniformly at random when sampling.

    Args:
        prior_module: Prior whose peaks seed the guesses (see ``peak_modes``).
        n_dihedrals: Number of dihedral dimensions.
        start_coords: Starting dihedral angles (degrees), used to fill
            uniform-prior dimensions in the systematic grid.
        init_steps: Number of guesses to draw when sampling.
        grid_budget: Maximum systematic grid size before falling back to
            sampling.
        seed: Random seed for the sampling fallback.

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
        points = []
        candidate_lists = [candidates for _dims, candidates in axes]
        for combo in itertools.product(*candidate_lists):
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

    points = []
    seen = set()
    max_attempts = max(20 * init_steps, 100)
    for _ in range(max_attempts):
        if len(points) >= init_steps:
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
            energy, cur_atoms = evaluate_energy(
                guess, state.start_atoms, dihedrals, calc, relaxCalc, relax
            )
            rel_energy = energy - state.start_energy
            logger.info(
                f"Evaluated conformer {i+1: >3}/{len(initial_conformers)}. Energy-E0: {rel_energy:12.6f}"
            )

            state.append_observation(guess, rel_energy, cur_atoms)

            if state.add_entry is not None:
                state.add_entry(guess, cur_atoms, energy)
    elif initial_dihedrals is not None:
        state.init_steps = len(initial_dihedrals)
        logger.info(
            f"Using {state.init_steps} prior-peak initial guesses"
        )
        for i, guess in enumerate(initial_dihedrals):
            guess = np.asarray(guess, dtype=float) % 360.0
            energy, cur_atoms = evaluate_energy(
                guess, state.start_atoms, dihedrals, calc, relaxCalc, relax
            )
            rel_energy = energy - state.start_energy
            logger.info(
                f"Evaluated peak guess {i+1: >3}/{len(initial_dihedrals)}. Energy-E0: {rel_energy:12.6f}"
            )

            state.append_observation(guess, rel_energy, cur_atoms)

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
            energy, cur_atoms = evaluate_energy(
                guess, state.start_atoms, dihedrals, calc, relaxCalc, relax
            )
            rel_energy = energy - state.start_energy
            logger.info(
                f"Evaluated initial guess {i+1: >3}/{init_steps}. Energy-E0: {rel_energy:12.6f}"
            )

            state.append_observation(guess, rel_energy, cur_atoms)

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
    for step in range(n_steps):
        next_coords = _select_next_points_botorch(
            state.observed_coords, state.observed_energies,
            prior_module=state.prior_module,
            prior_exponent=state.prior_exponent,
        )
        # logger.info(f'Selected next point: {next_coords}')

        energy, cur_atoms = evaluate_energy(
            next_coords, state.start_atoms, dihedrals, calc, relaxCalc, relax
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

        state.append_observation(next_coords, rel_energy, cur_atoms)

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

    Returns:
        If ``return_ensemble`` is False, the optimized lowest-energy geometry as
        an Atoms object. If True, a tuple ``(best_atoms, ensemble)`` where
        ``ensemble`` is a list of ``(atoms, relative_energy_eV, weight)`` sorted
        by energy ascending (``best_atoms`` is the lowest-energy ensemble member,
        or the single-best relaxation if the ensemble is empty).
    """
    # Setup initial state (relaxation, starting point, logging)
    state = _setup_initial_state(atoms, dihedrals, calc, relaxCalc, relax, out_dir)

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
