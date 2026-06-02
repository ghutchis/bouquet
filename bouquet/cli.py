import logging
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
from rdkit import Chem

from bouquet.calculator import CalculatorFactory
from bouquet.config import (
    DEFAULT_ENERGY_METHOD,
    DEFAULT_INIT_GRID_BUDGET,
    DEFAULT_INIT_METHOD,
    DEFAULT_INIT_STEPS,
    DEFAULT_NUM_STEPS,
    DEFAULT_OPTIMIZER_METHOD,
    DEFAULT_PRIOR_DECAY,
    DEFAULT_PRIOR_EXPONENT,
    Configuration,
)
from bouquet.io import (
    create_output_directory,
    save_ensemble,
    save_run_parameters,
    save_structure,
    setup_logging,
)
from bouquet.setup import (
    detect_dihedrals,
    get_conformers_from_file,
    get_initial_structure,
    get_initial_structure_from_file,
)
from bouquet.solver import plan_initial_points, run_optimization


def main():
    # Parse the command line arguments
    parser = ArgumentParser(
        description="Optimize molecular conformers using Bayesian optimization"
    )
    parser.add_argument(
        "--seed", type=int, default=datetime.now().microsecond, help="Random seed"
    )
    parser.add_argument(
        "--smiles", type=str, help="SMILES string of molecule to optimize"
    )
    parser.add_argument(
        "--file", type=str, help="File containing the structure to optimize"
    )
    parser.add_argument(
        "--name", type=str, help="Output name (defaults to SMILES or input file name)"
    )
    parser.add_argument(
        "--conformer-file",
        type=str,
        help="File containing multiple conformers to use as initial guesses "
        "(instead of random sampling). Overrides --init-steps.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Set the number of steps based on the number of dihedrals",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=DEFAULT_NUM_STEPS,
        help="Number of optimization steps to take",
    )
    parser.add_argument(
        "--init-steps",
        type=int,
        default=DEFAULT_INIT_STEPS,
        help="Number of initial guesses to make (ignored if --conformer-file is provided)",
    )
    parser.add_argument(
        "--init-method",
        choices=["random", "peaks"],
        default=DEFAULT_INIT_METHOD,
        help="How to generate initial guesses: 'random' (Gaussian around the "
        "start) or 'peaks' (systematic grid / weighted sampling from the "
        "dihedral prior peaks). 'peaks' uses built-in priors if --priors is "
        "not given, and is ignored when --conformer-file is provided.",
    )
    parser.add_argument(
        "--init-grid-budget",
        type=int,
        default=DEFAULT_INIT_GRID_BUDGET,
        help="Maximum systematic peak-grid size for --init-method peaks before "
        "falling back to weighted sampling of --init-steps points.",
    )
    parser.add_argument(
        "--energy",
        choices=sorted(CalculatorFactory.SUPPORTED_METHODS),
        default=DEFAULT_ENERGY_METHOD,
        help="Energy method",
    )
    parser.add_argument(
        "--optimizer",
        choices=sorted(CalculatorFactory.SUPPORTED_METHODS),
        default=DEFAULT_OPTIMIZER_METHOD,
        help="Optimizer method",
    )
    parser.add_argument(
        "--relax",
        action="store_true",
        help="Relax the non-dihedral degrees of freedom before computing energy",
    )
    parser.add_argument(
        "--priors",
        type=str,
        help="JSON file with dihedral prior definitions",
    )
    parser.add_argument(
        "--prior-exponent",
        type=float,
        default=DEFAULT_PRIOR_EXPONENT,
        help="Initial prior exponent for PiBO (0 to disable)",
    )
    parser.add_argument(
        "--prior-decay",
        type=float,
        default=DEFAULT_PRIOR_DECAY,
        help="Prior exponent decay rate per iteration",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Select a Boltzmann ensemble of low-energy conformers, tightly "
        "optimize them all, and write ensemble_final.xyz + ensemble.csv",
    )
    args = parser.parse_args()

    name = args.name or args.smiles or (Path(args.file).stem if args.file else None)

    # Create configuration from parsed arguments
    config = Configuration(
        smiles=args.smiles,
        input_file=Path(args.file) if args.file else None,
        name=name,
        conformer_file=Path(args.conformer_file) if args.conformer_file else None,
        energy_method=args.energy,
        optimizer_method=args.optimizer,
        num_steps=args.num_steps,
        init_steps=args.init_steps,
        init_method=args.init_method,
        init_grid_budget=args.init_grid_budget,
        auto_steps=args.auto,
        relax=args.relax,
        seed=args.seed,
        priors_file=Path(args.priors) if args.priors else None,
        initial_prior_exponent=args.prior_exponent,
        prior_exponent_decay=args.prior_decay,
        ensemble=args.ensemble,
    )

    # Make an output directory
    out_dir = create_output_directory(
        name, config.seed, config.energy_method, args.__dict__
    )
    config.out_dir = out_dir

    save_run_parameters(out_dir, args.__dict__)

    # Set up the logging
    logger = setup_logging(out_dir)
    logger.info(f"Started optimizing the conformers for {config.name}")

    # Make the initial guess
    if config.input_file is None:
        # this will do some initial cleanup from the SMILES string
        init_atoms, mol = get_initial_structure(config.smiles)
    else:
        # this will just read the geometry from the file
        # and parse using Pybel
        init_atoms, mol = get_initial_structure_from_file(str(config.input_file))
    logger.info(f"Determined initial structure with {len(init_atoms)} atoms")

    # charge
    config.charge = Chem.GetFormalCharge(mol)

    # Detect the dihedral angles
    dihedrals = detect_dihedrals(mol)
    logger.info(f"Detected {len(dihedrals)} dihedral angles")

    # Load conformers if provided
    initial_conformers = None
    if config.conformer_file is not None:
        initial_conformers = get_conformers_from_file(str(config.conformer_file))
        logger.info(
            f"Loaded {len(initial_conformers)} conformers from {config.conformer_file}"
        )
        # Validate that conformers have the same number of atoms
        for i, conf in enumerate(initial_conformers):
            if len(conf) != len(init_atoms):
                raise ValueError(
                    f"Conformer {i} has {len(conf)} atoms, expected {len(init_atoms)}"
                )

    # Create prior module if priors file provided. This drives PiBO acquisition
    # steering and is only built when --priors is given.
    prior_module = None
    if config.priors_file is not None:
        from bouquet.priors import create_prior_module

        # Get dihedral atom tuples
        dihedral_tuples = [d.chain for d in dihedrals]
        prior_module = create_prior_module(
            mol=mol,
            dihedrals=dihedral_tuples,
            univariate_file=config.priors_file,
        )
        logger.info(f"Created prior module from {config.priors_file}")
        logger.info(prior_module.describe())

    # Plan prior-peak initial guesses (opt-in via --init-method peaks).
    # Conformers take precedence. Peak-seeding is decoupled from PiBO
    # acquisition: when no --priors file is given we build a built-ins-only
    # module purely for seeding, leaving acquisition steering off.
    initial_dihedrals = None
    if config.init_method == "peaks" and initial_conformers is None:
        from bouquet.priors import create_prior_module

        planning_module = prior_module
        if planning_module is None:
            planning_module = create_prior_module(
                mol=mol, dihedrals=[d.chain for d in dihedrals]
            )
        start_coords = np.array([d.get_angle(init_atoms) for d in dihedrals])
        initial_dihedrals = plan_initial_points(
            planning_module,
            len(dihedrals),
            start_coords,
            config.init_steps,
            config.init_grid_budget,
            config.seed,
        )
        logger.info(
            f"Planned {len(initial_dihedrals)} prior-peak initial guesses "
            f"(grid budget {config.init_grid_budget})"
        )

    # Compute the number of optimization steps (handles auto mode)
    if initial_conformers is not None:
        initial_count = len(initial_conformers)
    elif initial_dihedrals is not None:
        initial_count = len(initial_dihedrals)
    else:
        initial_count = config.init_steps
    num_steps = config.compute_auto_steps(len(dihedrals), initial_count)

    # Save the initial guess
    save_structure(out_dir, init_atoms, "initial.xyz")

    # Create calculators using the factory
    calc = CalculatorFactory.from_config(config, for_optimizer=False, mol=mol)
    relaxCalc = CalculatorFactory.from_config(config, for_optimizer=True, mol=mol)

    result = run_optimization(
        init_atoms,
        dihedrals,
        num_steps,
        calc,
        relaxCalc,
        config.init_steps,
        out_dir,
        relax=config.relax,
        seed=config.seed,
        initial_conformers=initial_conformers,
        initial_dihedrals=initial_dihedrals,
        prior_module=prior_module,
        initial_prior_exponent=config.initial_prior_exponent,
        prior_exponent_decay=config.prior_exponent_decay,
        return_ensemble=config.ensemble,
    )

    if config.ensemble:
        final_atoms, ensemble = result
        save_ensemble(out_dir, ensemble)
        logger.info(f"Wrote {len(ensemble)}-conformer ensemble to {out_dir}")
    else:
        final_atoms = result

    # Save the final structure
    save_structure(out_dir, final_atoms, "final.xyz")
    logger.info(f"Done. Files are stored in {str(out_dir)}")


if __name__ == "__main__":
    main()
