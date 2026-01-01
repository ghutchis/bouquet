"""Methods for solving the conformer option problem"""
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from botorch.acquisition.analytic import *
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils import standardize
from gpytorch import kernels as gpykernels
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import NormalPrior

warnings.filterwarnings("ignore")

from bouquet.assess import evaluate_energy, relax_structure
from bouquet.config import (
    ACQ_NUM_RESTARTS,
    ACQ_RAW_SAMPLES,
    DEFAULT_RELAXATION_STEPS,
    GP_PERIOD_LENGTH_MEAN,
    GP_PERIOD_LENGTH_STD,
    INITIAL_GUESS_STD,
)
from bouquet.io import create_structure_logger, initialize_structure_log, save_structure
from bouquet.setup import DihedralInfo

logger = logging.getLogger(__name__)

kcal = 1.0 / 627.5094740631 # kcal/mol in Hartree

def _get_device() -> torch.device:
    """Get the appropriate torch device (CUDA if available, else CPU)."""
    # sadly Apple's MPS doesn't support float64
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class OptimizationState:
    """Tracks the state of a Bayesian optimization run."""
    start_atoms: Atoms
    start_coords: np.ndarray
    start_energy: float
    observed_coords: torch.Tensor  # Shape: (n_observations, n_dihedrals)
    observed_energies: torch.Tensor  # Shape: (n_observations,)
    device: torch.device = field(default_factory=_get_device)
    init_steps: int = 0
    best_atoms: Optional[Atoms] = None
    best_step: int = 0
    add_entry: Optional[Callable] = None

    def append_observation(self, coords: np.ndarray, energy: float) -> None:
        """Append a new observation to the tracked data.

        Args:
            coords: Dihedral coordinates as numpy array
            energy: Relative energy value
        """
        new_coords = torch.tensor(coords, dtype=torch.float64, device=self.device).unsqueeze(0)
        new_energy = torch.tensor([energy], dtype=torch.float64, device=self.device)
        self.observed_coords = torch.cat([self.observed_coords, new_coords], dim=0)
        self.observed_energies = torch.cat([self.observed_energies, new_energy], dim=0)


def _select_next_points_botorch(train_X: torch.Tensor, train_y: torch.Tensor) -> np.ndarray:
    """Generate the next sample to evaluate with the energy calculator

    Uses BOTorch to pick the next sample using Expected Improvement

    Args:
        train_X: Observed coordinates as torch tensor (n_obs, n_dims)
        train_y: Observed energies as torch tensor (n_obs,)
    Returns:
        Next coordinates to try (in dihedral space)
    """
    # Clip the energies if needed
    train_y = torch.clamp(train_y, max=2 + torch.log10(torch.clamp(train_y, min=1)))

    # Reshape and standardize for GP (minimize -> maximize by negating)
    train_y = train_y[:, None]
    train_y = standardize(-1 * train_y)  # make this a maximization problem

    # Make the GP
    # TODO: make the GP only once and reuse
    gp = SingleTaskGP(train_X, train_y,
        covar_module=gpykernels.ScaleKernel(gpykernels.ProductStructureKernel(
        num_dims=train_X.shape[1],
        base_kernel=gpykernels.PeriodicKernel(period_length_prior=NormalPrior(GP_PERIOD_LENGTH_MEAN, GP_PERIOD_LENGTH_STD))
    )))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device=train_X.device)
    fit_gpytorch_mll(mll)

    # Solve the optimization problem
    n_sampled, n_dim = train_X.shape
    acqf = LogExpectedImprovement(gp, best_f=torch.max(train_y), maximize=True)
    # alternative acquisition functions, e.g.
    # TODO: See if different acqf give better results
    # Following boss, we use Eq. 5 of https://arxiv.org/pdf/1012.2599.pdf
    #    with delta=0.1
    #kappa = np.sqrt(2 * np.log10(
    #    np.power(n_sampled, n_dim / 2 + 2) * np.pi ** 2 / (3.0 * 0.1)
    #))  # Results in more exploration over time
    # kappa = 1.2
    # acqf = UpperConfidenceBound(gp, kappa)
    bounds = torch.zeros(2, train_X.shape[1])
    bounds[1, :] = 360
    candidate, acq_value = optimize_acqf(
        # at the moment use q = 1 for no batching
        acqf, bounds=bounds, q=1, num_restarts=ACQ_NUM_RESTARTS, raw_samples=ACQ_RAW_SAMPLES
    )
    return candidate.detach().numpy()[0, :]


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
    logger.info('Initial relaxation')
    _, init_atoms = relax_structure(atoms, calc, relaxCalc, DEFAULT_RELAXATION_STEPS)
    if out_dir is not None:
        save_structure(out_dir, init_atoms, 'relaxed.xyz')

    # Evaluate initial point
    start_coords = np.array([d.get_angle(init_atoms) for d in dihedrals])
    start_energy, start_atoms = evaluate_energy(start_coords, atoms, dihedrals, calc, relaxCalc, relax)
    logger.info(f'Computed initial energy: {start_energy}')

    # Set up logging if output directory provided
    add_entry = None
    if out_dir is not None:
        log_path, ens_path = initialize_structure_log(out_dir)
        add_entry = create_structure_logger(log_path, ens_path, start_energy)
        add_entry(start_coords, start_atoms, start_energy)

    device = _get_device()
    return OptimizationState(
        start_atoms=start_atoms,
        start_coords=start_coords,
        start_energy=start_energy,
        init_steps=0,
        observed_coords=torch.tensor([start_coords], dtype=torch.float64, device=device),
        observed_energies=torch.tensor([0.0], dtype=torch.float64, device=device),
        device=device,
        best_atoms=start_atoms.copy(),
        best_step=0,
        add_entry=add_entry,
    )


def _evaluate_initial_guesses(
    state: OptimizationState,
    dihedrals: List[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
    relax: bool,
    init_steps: int,
    seed: int,
    initial_conformers: Optional[List[Atoms]],
) -> None:
    """Evaluate initial guesses and update state in-place.

    Uses provided conformers if available, otherwise generates random guesses.

    Args:
        state: Optimization state to update
        dihedrals: List of dihedral angles to modify
        calc: Calculator for energy evaluation
        relaxCalc: Calculator used for geometry relaxation
        relax: Whether to relax non-dihedral degrees of freedom
        init_steps: Number of random guesses if no conformers provided
        seed: Random seed for random sampling
        initial_conformers: Optional list of conformer structures
    """
    if initial_conformers is not None:
        state.init_steps = len(initial_conformers)
        logger.info(f'Using {state.init_steps} provided conformers as initial guesses')
        for i, conformer in enumerate(initial_conformers):
            guess = np.array([d.get_angle(conformer) for d in dihedrals])
            energy, cur_atoms = evaluate_energy(guess, state.start_atoms, dihedrals, calc, relaxCalc, relax)
            rel_energy = energy - state.start_energy
            logger.info(f'Evaluated conformer {i+1: >3}/{len(initial_conformers)}. Energy-E0: {rel_energy:12.6f}')

            state.append_observation(guess, rel_energy)

            if state.add_entry is not None:
                state.add_entry(guess, cur_atoms, energy)
    else:
        state.init_steps = init_steps
        logger.info(f'Generating {init_steps} random initial guesses')
        rng = np.random.default_rng(seed)
        init_guesses = rng.normal(state.start_coords, INITIAL_GUESS_STD, size=(init_steps, len(dihedrals)))

        for i, guess in enumerate(init_guesses):
            energy, cur_atoms = evaluate_energy(guess, state.start_atoms, dihedrals, calc, relaxCalc, relax)
            rel_energy = energy - state.start_energy
            logger.info(f'Evaluated initial guess {i+1: >3}/{init_steps}. Energy-E0: {rel_energy:12.6f}')

            state.append_observation(guess, rel_energy)

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
        next_coords = _select_next_points_botorch(state.observed_coords, state.observed_energies)

        energy, cur_atoms = evaluate_energy(next_coords, state.start_atoms, dihedrals, calc, relaxCalc, relax)
        rel_energy = energy - state.start_energy
        logger.info(f'Evaluated energy in step {step+1: >3}/{n_steps}. Energy-E0: {rel_energy:12.6f}')

        if rel_energy < state.observed_energies.min().item():
            state.best_step = step
            state.best_atoms = cur_atoms.copy()
            if out_dir is not None:
                save_structure(out_dir, cur_atoms, 'current_best.xyz')

        if state.add_entry is not None:
            state.add_entry(next_coords, cur_atoms, energy)

        state.append_observation(next_coords, rel_energy)


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

    logger.info(f'Best energy found on step {state.best_step + 1}')

    # go through the energies and find the first one within 0.001 of the best energy
    # .. we'll need to subtract the number of initial guesses from the step count
    first_low_energy = False
    for i in range(len(state.observed_energies)):
        if not first_low_energy and abs(state.observed_energies[i].item() - state.observed_energies[best_idx].item()) < kcal * 10.0:
            first_low_energy = True
            logger.info(f"Found low energy on step {i - state.init_steps}")
        if abs(state.observed_energies[i].item() - state.observed_energies[best_idx].item()) < kcal:
            logger.info(f"Found first good energy on step {i - state.init_steps}")
            break

    best_energy, best_atoms = evaluate_energy(best_coords, state.best_atoms, dihedrals, calc, relaxCalc)
    logger.info(
        f'Performed final relaxation with dihedral constraints. '
        f'E: {best_energy}. E-E0: {best_energy - state.start_energy}'
    )
    if state.add_entry is not None:
        state.add_entry(best_coords, best_atoms, best_energy)

    # Relaxation without dihedral constraints
    best_atoms.set_constraint()
    best_energy, best_atoms = evaluate_energy(best_coords, best_atoms, dihedrals, calc, relaxCalc)
    logger.info(
        f'Performed final relaxation without dihedral constraints. '
        f'E: {best_energy}. E-E0: {best_energy - state.start_energy}'
    )

    best_coords = np.array([d.get_angle(best_atoms) for d in dihedrals])
    if state.add_entry is not None:
        state.add_entry(best_coords, best_atoms, best_energy)

    return best_atoms


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
) -> Atoms:
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

    Returns:
        Optimized geometry as an Atoms object
    """
    # Setup initial state (relaxation, starting point, logging)
    state = _setup_initial_state(atoms, dihedrals, calc, relaxCalc, relax, out_dir)

    # Evaluate initial guesses (conformers or random)
    _evaluate_initial_guesses(
        state, dihedrals, calc, relaxCalc, relax, init_steps, seed, initial_conformers
    )

    # Run Bayesian optimization loop
    _run_optimization_loop(state, n_steps, dihedrals, calc, relaxCalc, relax, out_dir)

    # Final relaxation and return best structure
    return _perform_final_relaxation(state, dihedrals, calc, relaxCalc)
