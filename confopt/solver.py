"""Methods for solving the conformer option problem"""
import logging
from csv import DictWriter
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import List, Optional

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io.xyz import simple_write_xyz
import torch

from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch import kernels as gpykernels
from gpytorch.priors import NormalPrior

import numpy as np

from confopt.assess import evaluate_energy, relax_structure
from confopt.setup import DihedralInfo


logger = logging.getLogger(__name__)


def select_next_points_botorch(observed_X: List[List[float]], observed_y: List[float]) -> np.ndarray:
    """Generate the next sample to evaluate with XTB

    Uses BOTorch to pick the next sample using Expected Improvement

    Args:
        observed_X: Observed coordinates
        observed_y: Observed energies
    Returns:
        Next coordinates to try
    """

    # Clip the energies if needed
    observed_y = np.clip(observed_y, -np.inf, 2 + np.log10(np.clip(observed_y, 1, np.inf)))

    # we should really track the torch device
    #  .. unfortuantely "MPS" for Apple Silicon doesn't support float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert inputs to torch arrays
    train_X = torch.tensor(observed_X, dtype=torch.float64, device=device)
    train_y = torch.tensor(observed_y, dtype=torch.float64, device=device)
    train_y = train_y[:, None]
    train_y = standardize(-1 * train_y)

    # Make the GP
    gp = SingleTaskGP(train_X, train_y, covar_module=gpykernels.ScaleKernel(gpykernels.ProductStructureKernel(
        num_dims=train_X.shape[1],
        base_kernel=gpykernels.PeriodicKernel(period_length_prior=NormalPrior(360, 0.1))
    )))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device=device)
    fit_gpytorch_model(mll)

    # Solve the optimization problem
    #  Following boss, we use Eq. 5 of https://arxiv.org/pdf/1012.2599.pdf with delta=0.1
    n_sampled, n_dim = train_X.shape
    kappa = np.sqrt(2 * np.log10(
        np.power(n_sampled, n_dim / 2 + 2) * np.pi ** 2 / (3.0 * 0.1)
    ))  # Results in more exploration over time
    ei = UpperConfidenceBound(gp, kappa)
    bounds = torch.zeros(2, train_X.shape[1])
    bounds[1, :] = 360
    candidate, acq_value = optimize_acqf(
        ei, bounds=bounds, q=1, num_restarts=64, raw_samples=64
    )
    return candidate.detach().numpy()[0, :]


def run_optimization(atoms: Atoms, dihedrals: List[DihedralInfo], n_steps: int, calc: Calculator,
                     init_steps: int, out_dir: Optional[Path], relax: bool = True) -> Atoms:
    """Optimize the structure of a molecule by iteratively changing the dihedral angles

    Args:
        atoms: Atoms object with the initial geometry
        dihedrals: List of dihedral angles to modify
        n_steps: Number of optimization steps to perform
        init_steps: Number of initial guesses to evaluate
        calc: Calculator to pick the energy
        out_dir: Output path for logging information
        relax: Whether to relax non-dihedral degrees of freedom each step
    Returns:
        (Atoms) optimized geometry
    """
    # Perform an initial relaxation
    logger.info('Initial relaxation')
    _, init_atoms = relax_structure(atoms, calc, 50)
    if out_dir is not None:
        with open(out_dir.joinpath('relaxed.xyz'), 'w') as fp:
            simple_write_xyz(fp, [init_atoms])

    # Evaluate initial point
    start_coords = np.array([d.get_angle(init_atoms) for d in dihedrals])
    start_energy, start_atoms = evaluate_energy(start_coords, atoms, dihedrals, calc, relax)
    logger.info(f'Computed initial energy: {start_energy}')

    # Begin a structure log, if output available
    if out_dir is not None:
        log_path = out_dir.joinpath('structures.csv')
        ens_path = out_dir.joinpath('ensemble.xyz')
        with log_path.open('w') as fp:
            writer = DictWriter(fp, ['time', 'xyz', 'energy', 'ediff'])
            writer.writeheader()

        def add_entry(coords, atoms, energy):
            with log_path.open('a') as fp:
                writer = DictWriter(fp, ['time', 'coords', 'xyz', 'energy', 'ediff'])
                xyz = StringIO()
                simple_write_xyz(xyz, [atoms])
                writer.writerow({
                    'time': datetime.now().timestamp(),
                    'coords': coords.tolist(),
                    'xyz': xyz.getvalue(),
                    'energy': energy,
                    'ediff': energy - start_energy
                })
            with ens_path.open('a') as fp:
                simple_write_xyz(fp, [atoms], comment=f'\t{energy}')
        add_entry(start_coords, start_atoms, start_energy)

    # Make some initial guesses
    init_guesses = np.random.normal(start_coords, 30, size=(init_steps, len(dihedrals)))
    init_energies = []
    for i, guess in enumerate(init_guesses):
        energy, cur_atoms = evaluate_energy(guess, start_atoms, dihedrals, calc, relax)
        init_energies.append(energy - start_energy)
        logger.info(f'Evaluated initial guess {i+1}/{init_steps}. Energy-E0: {energy-start_energy}')

        if out_dir is not None:
            add_entry(guess, cur_atoms, energy)

    # Save the initial guesses
    observed_coords = np.array([start_coords, *init_guesses.tolist()])
    observed_energies = [0.] + init_energies

    # Loop over many steps
    cur_atoms = start_atoms.copy()
    for step in range(n_steps):
        # Make a new search space
        best_coords = select_next_points_botorch(observed_coords, observed_energies)

        # Compute the energies of those points
        energy, cur_atoms = evaluate_energy(best_coords, cur_atoms, dihedrals, calc, relax)
        logger.info(f'Evaluated energy in step {step+1}/{n_steps}. Energy-E0: {energy-start_energy}')
        if energy - start_energy < np.min(observed_energies) and out_dir is not None:
            with open(out_dir.joinpath('current_best.xyz'), 'w') as fp:
                simple_write_xyz(fp, [cur_atoms])

        # Update the log
        if out_dir is not None:
            add_entry(start_coords, cur_atoms, energy)

        # Update the search space
        observed_coords = np.vstack([observed_coords, best_coords])
        observed_energies.append(energy - start_energy)

    # Final relaxations
    best_atoms = cur_atoms.copy()
    best_coords = observed_coords[np.argmin(observed_energies)]
    best_energy, best_atoms = evaluate_energy(best_coords, best_atoms, dihedrals, calc)
    logger.info('Performed final relaxation with dihedral constraints.'
                f'E: {best_energy}. E-E0: {best_energy - start_energy}')
    if out_dir is not None:
        add_entry(np.array(best_coords), best_atoms, best_energy)

    # Relaxations
    best_atoms.set_constraint()
    best_energy, best_atoms = relax_structure(best_atoms, calc, None)
    logger.info('Performed final relaxation without dihedral constraints.'
                f' E: {best_energy}. E-E0: {best_energy - start_energy}')
    best_coords = np.array([d.get_angle(best_atoms) for d in dihedrals])
    if out_dir is not None:
        add_entry(best_coords, best_atoms, best_energy)
    return best_atoms
