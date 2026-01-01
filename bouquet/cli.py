import logging
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from bouquet.calculator import CalculatorFactory
from bouquet.config import (
    DEFAULT_ENERGY_METHOD,
    DEFAULT_INIT_STEPS,
    DEFAULT_NUM_STEPS,
    DEFAULT_OPTIMIZER_METHOD,
    Configuration,
)
from bouquet.io import (
    create_output_directory,
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
from bouquet.solver import run_optimization


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
        "--energy",
        choices=["ani", "b3lyp", "b97", "gfn0", "gfn2", "gfnff"],
        default=DEFAULT_ENERGY_METHOD,
        help="Energy method",
    )
    parser.add_argument(
        "--optimizer",
        choices=["ani", "b3lyp", "b97", "gfn0", "gfn2", "gfnff"],
        default=DEFAULT_OPTIMIZER_METHOD,
        help="Optimizer method",
    )
    parser.add_argument(
        "--relax",
        action="store_true",
        help="Relax the non-dihedral degrees of freedom before computing energy",
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
        auto_steps=args.auto,
        relax=args.relax,
        seed=args.seed,
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

    # TODO: have optional cleanups

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

    # Compute the number of optimization steps (handles auto mode)
    initial_count = len(initial_conformers) if initial_conformers else config.init_steps
    num_steps = config.compute_auto_steps(len(dihedrals), initial_count)

    # Save the initial guess
    save_structure(out_dir, init_atoms, "initial.xyz")

    # Create calculators using the factory
    calc = CalculatorFactory.from_config(config, for_optimizer=False)
    relaxCalc = CalculatorFactory.from_config(config, for_optimizer=True)

    final_atoms = run_optimization(
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
    )

    # Save the final structure
    save_structure(out_dir, final_atoms, "final.xyz")
    logger.info(f"Done. Files are stored in {str(out_dir)}")


if __name__ == "__main__":
    main()
