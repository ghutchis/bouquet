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
    DEFAULT_MIN_AUTO_BO_STEPS,
    DEFAULT_NUM_STEPS,
    DEFAULT_OPTIMIZER_METHOD,
    DEFAULT_PRIOR_BACKGROUND_WEIGHT,
    DEFAULT_PRIOR_DECAY,
    DEFAULT_PRIOR_EXPONENT,
    DEFAULT_PRIOR_MAX_CONCENTRATION,
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
        "--use-gradients",
        action="store_true",
        help="Use the gradient-enhanced GP surrogate: project the calculator's "
        "forces onto each torsion (dE/dtheta) and feed them to the acquisition "
        "GP, so each energy evaluation also contributes a gradient observation.",
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=0,
        help="With --use-gradients, use the gradient-enhanced GP only for the "
        "first N BO steps, then switch to the value-only GP. The gradient GP's "
        "per-step cost grows steeply with the observation count, so this caps it "
        "on large/flexible molecules while keeping the early-search benefit. "
        "0 (default) keeps gradients for the whole run.",
    )
    parser.add_argument(
        "--grad-refit-dense-until",
        type=int,
        default=20,
        help="Gradient-GP refit schedule ('gradfreeze'): do a full hyperparameter "
        "fit for the first N BO steps, then FREEZE the hyperparameters and only "
        "re-condition (one Cholesky/step instead of ~200). Refitting is the dominant "
        "gradient-GP cost and grows with observation count, so freezing keeps the "
        "full-gradient run affordable on large molecules. Default 20 (validated "
        "quality-neutral vs full refitting, 5-11 dihedrals); set 0 to refit every "
        "step (the slow reference).",
    )
    parser.add_argument(
        "--grad-refit-every",
        type=int,
        default=0,
        help="After the dense phase, optionally cold-refresh the frozen "
        "hyperparameters every Nth BO step (0 freezes for the rest of the run; the "
        "refresh is a cold fit -- warm-starting drifts the hyperparameters and "
        "degrades the search). 0 (default); with no dense phase, fits every step.",
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
        "--prior-max-concentration",
        type=float,
        default=DEFAULT_PRIOR_MAX_CONCENTRATION,
        help="Cap on fitted von Mises concentration (kappa) used as a search prior. "
        "Raw histogram fits can be near-delta (kappa ~1e4); capping keeps the prior "
        "smooth enough for the acquisition optimizer to follow (<=0 disables the cap).",
    )
    parser.add_argument(
        "--prior-background-weight",
        type=float,
        default=DEFAULT_PRIOR_BACKGROUND_WEIGHT,
        help="Weight in [0,1) of a uniform background mixed into each univariate "
        "prior: (1-w)*vonMises + w*uniform. Bounds how strongly any single mode can "
        "dominate the acquisition and gives a smooth floor. 0 disables it. Try 0.05-0.2.",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Select a Boltzmann ensemble of low-energy conformers, tightly "
        "optimize them all, and write ensemble_final.xyz + ensemble.csv",
    )
    args = parser.parse_args()

    # Create configuration from parsed arguments
    config = Configuration(
        smiles=args.smiles,
        input_file=Path(args.file) if args.file else None,
        name=args.name,
        conformer_file=Path(args.conformer_file) if args.conformer_file else None,
        energy_method=args.energy,
        optimizer_method=args.optimizer,
        num_steps=args.num_steps,
        init_steps=args.init_steps,
        init_method=args.init_method,
        init_grid_budget=args.init_grid_budget,
        auto_steps=args.auto,
        relax=args.relax,
        use_gradients=args.use_gradients,
        gradient_steps=args.gradient_steps,
        grad_refit_dense_until=args.grad_refit_dense_until,
        grad_refit_every=args.grad_refit_every,
        seed=args.seed,
        priors_file=Path(args.priors) if args.priors else None,
        initial_prior_exponent=args.prior_exponent,
        prior_exponent_decay=args.prior_decay,
        prior_max_concentration=args.prior_max_concentration,
        prior_background_weight=args.prior_background_weight,
        ensemble=args.ensemble,
    )

    # Gradient labels are only consistent with the energy objective when the
    # geometry is a constrained minimum of the *energy* calculator. With
    # relaxation the geometry is minimized on the optimizer surface, so the
    # projected torsion gradient equals dE*/dtheta only if the two calculators
    # are the same surface. Refuse the mismatched combination rather than feeding
    # the GP biased gradient labels. (Without --relax the rigid-scan gradient is
    # always consistent, so the check is limited to the relaxed case.)
    if (
        config.use_gradients
        and config.relax
        and config.energy_method != config.optimizer_method
    ):
        raise ValueError(
            "--use-gradients with --relax requires --energy and --optimizer to be "
            f"the same method (got energy={config.energy_method!r}, "
            f"optimizer={config.optimizer_method!r}). The torsion gradient is only "
            "dE*/dtheta at a constrained minimum of the energy calculator, but the "
            "geometry is relaxed on the optimizer surface. Use "
            f"--optimizer {config.energy_method}, or drop --use-gradients / --relax."
        )

    # Make an output directory
    out_dir = create_output_directory(
        config.name, config.seed, config.energy_method, args.__dict__
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
        # <=0 disables the cap (use the raw fitted concentrations).
        max_conc = (
            config.prior_max_concentration
            if config.prior_max_concentration > 0
            else None
        )
        prior_module = create_prior_module(
            mol=mol,
            dihedrals=dihedral_tuples,
            univariate_file=config.priors_file,
            max_concentration=max_conc,
            background_weight=config.prior_background_weight,
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
        # Under --auto, cap the number of seeded points so a large systematic grid
        # can't consume the whole budget and leave zero BO refinement (worst on
        # small molecules, where grid size >= the auto total). The grid is
        # best-first, so the cap keeps the most probable mode combinations.
        max_init = None
        if config.auto_steps:
            total = config.auto_total(len(dihedrals))
            max_init = max(1, total - DEFAULT_MIN_AUTO_BO_STEPS)
        initial_dihedrals = plan_initial_points(
            planning_module,
            len(dihedrals),
            start_coords,
            config.init_steps,
            config.init_grid_budget,
            config.seed,
            max_points=max_init,
        )
        logger.info(
            f"Planned {len(initial_dihedrals)} prior-peak initial guesses "
            f"(grid budget {config.init_grid_budget}"
            + (f", capped at {max_init} to keep >= {DEFAULT_MIN_AUTO_BO_STEPS} "
               f"BO steps" if max_init is not None else "")
            + ")"
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
        use_gradients=config.use_gradients,
        gradient_steps=config.gradient_steps,
        grad_refit_dense_until=config.grad_refit_dense_until,
        grad_refit_every=config.grad_refit_every,
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
