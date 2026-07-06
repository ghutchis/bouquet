import logging
from argparse import SUPPRESS, ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
from rdkit import Chem

from bouquet.calculator import CalculatorFactory
from bouquet.config import (
    ACQ_NUM_RESTARTS,
    ACQ_RAW_SAMPLES,
    DEFAULT_CERTIFICATE_BETAS,
    DEFAULT_ENERGY_METHOD,
    DEFAULT_INIT_CONFORMER_CAP,
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
    format_certificate_betas,
    parse_certificate_betas,
)
from bouquet.io import (
    create_output_directory,
    save_ensemble,
    save_run_parameters,
    save_structure,
    setup_logging,
)
from bouquet.setup import (
    apply_charge_spin,
    default_multiplicity,
    detect_dihedrals,
    get_conformers_from_file,
    get_initial_candidates,
    get_initial_structure_from_file,
    select_initial_structure,
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
        "--init-conformers",
        type=int,
        default=DEFAULT_INIT_CONFORMER_CAP,
        help="Maximum number of ETKDG embeddings to generate from SMILES and "
        "score with the energy method, keeping the lowest-energy one as the "
        "starting structure. The actual count scales with ring flexibility; 1 "
        "disables the search (single embedding). Ignored for file/conformer "
        "input.",
    )
    parser.add_argument(
        "--energy",
        choices=CalculatorFactory.available_methods(),
        default=DEFAULT_ENERGY_METHOD,
        help="Energy method (only methods whose dependencies are installed are listed)",
    )
    parser.add_argument(
        "--optimizer",
        choices=CalculatorFactory.available_methods(),
        default=DEFAULT_OPTIMIZER_METHOD,
        help="Optimizer method (only methods whose dependencies are installed are listed)",
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=None,
        help="Total molecular charge. Default: the SMILES/structure's formal charge.",
    )
    parser.add_argument(
        "--spin",
        "--multiplicity",
        dest="spin",
        type=int,
        default=None,
        help="Spin multiplicity (2S+1). Default: the lowest spin consistent with "
        "the electron count (singlet for an even number of electrons, doublet for "
        "odd). Used by all calculators: psi4 via its constructor, xtb via "
        "uhf=multiplicity-1 on the atoms.",
    )
    parser.add_argument(
        "--relax",
        action="store_true",
        help="Relax the non-dihedral degrees of freedom before computing energy",
    )
    parser.add_argument(
        "--solvent",
        type=str,
        default=None,
        help="Implicit solvent model (e.g. 'water'). Default: gas phase (no "
        "solvent). Supported by xtb methods (gfn2/gfn0/gfnff, via their native "
        "GBSA/ALPB solvent keyword) and psi4 methods (via DDX continuum "
        "solvation); not supported by ani, aimnet2, mmff, or uff.",
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
        "--lengthscale-prior",
        choices=["auto", "none", "dim_scaled"],
        default="auto",
        help="Prior on the value-only GP's periodic lengthscale. 'auto' (default): "
        "'dim_scaled' once the dihedral count reaches the high-d threshold, else 'none'. "
        "'none': free MLL fit (historical). 'dim_scaled': Hvarfner et al. (ICML 2024) "
        "dimensionality-scaled LogNormal prior, biasing the GP toward smoother fits "
        "as the dihedral count grows -- helps high-d search.",
    )
    parser.add_argument(
        "--lowmode-prob",
        type=float,
        default=None,
        help="Phase 2.5 low-mode search: probability that an eligible BO step (past "
        "--lowmode-warmup evaluations) is replaced by a committed kick along a soft "
        "mode followed by an UNCONSTRAINED relaxation (letting the dihedrals move, so "
        "the geometry can slide along a curved fold valley a standard BO step cannot "
        "cross). Default: auto (0.5 once the dihedral count reaches the high-d "
        "threshold, else 0). Set 0 to disable, or a probability in (0, 1].",
    )
    parser.add_argument(
        "--lowmode-warmup",
        type=int,
        default=100,
        help="With --lowmode-prob, only start low-mode moves after this many "
        "evaluations (default 100); align it past the gradient Phase A.",
    )
    parser.add_argument(
        "--lowmode-kick-deg",
        type=float,
        default=60.0,
        help="With --lowmode-prob, per-dihedral RMS kick amplitude (degrees, "
        "default 60) along the chosen soft mode.",
    )
    parser.add_argument(
        "--lowmode-kick-dir",
        choices=["pca", "enm"],
        default="pca",
        help="Kick-direction source for low-mode moves: 'pca' (data-derived position "
        "PCA of the low-energy set) or 'enm' (data-independent elastic-network soft "
        "modes -- global bend/compaction = folding, projected to torsion space). "
        "Default 'pca'.",
    )
    parser.add_argument(
        "--priors",
        type=str,
        help="JSON file with dihedral prior definitions",
    )
    # Initial prior exponent for PiBO (0 to disable, default is 0.5)
    parser.add_argument(
        "--prior-exponent",
        type=float,
        default=DEFAULT_PRIOR_EXPONENT,
        help=SUPPRESS,
    )
    # Prior exponent decay rate per iteration, default is 0.5
    parser.add_argument(
        "--prior-decay",
        type=float,
        default=DEFAULT_PRIOR_DECAY,
        help=SUPPRESS,
    )
    # Cap on fitted von Mises concentration (kappa) used as a search prior.
    # Raw histogram fits might be near-delta (kappa ~1e4); capping keeps the prior
    # smooth enough for the acquisition optimizer to follow (<=0 disables the cap).
    # default is 50.0
    parser.add_argument(
        "--prior-max-concentration",
        type=float,
        default=DEFAULT_PRIOR_MAX_CONCENTRATION,
        help=SUPPRESS,
    )
    # Weight in [0,1) of a uniform background mixed into each univariate
    # prior: (1-w)*vonMises + w*uniform. Bounds how strongly any single mode can "
    # dominate the acquisition and gives a smooth floor. 0 disables it. Try 0.05-0.2."
    # default is 0.05
    parser.add_argument(
        "--prior-background-weight",
        type=float,
        default=DEFAULT_PRIOR_BACKGROUND_WEIGHT,
        help=SUPPRESS,
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Select a Boltzmann ensemble of low-energy conformers, tightly "
        "optimize them all, and write ensemble_final.xyz + ensemble.csv",
    )
    parser.add_argument(
        "--certificate-log",
        type=str,
        default=None,
        help="Path to write a per-BO-step stopping-rule certificate CSV "
        "(mu_min/lb/alpha_max + e_eval/e_best/n_calls/wall_s). Used by the "
        "various benchmarks; off by default.",
    )
    # Comma-separated confidence multipliers for the certificate lower
    # bound (mu - beta*sigma); one lb_b<beta> column is logged per value so the
    # offline replay can calibrate beta. Only used with --certificate-log.
    parser.add_argument(
        "--certificate-betas",
        type=str,
        default=format_certificate_betas(DEFAULT_CERTIFICATE_BETAS),
        help=SUPPRESS,
    )
    # Path to write a multi-frame XYZ of the geometry at each best-so-far
    # improvement (plus the final relaxed best), for the benchmark's
    # RMSD-identity / distinct-conformer analysis. Off by default.
    parser.add_argument(
        "--geometry-log",
        type=str,
        default=None,
        help=SUPPRESS,
    )
    # Hidden (SUPPRESS): acq24 (24 restarts / 24 raw samples) is the validated
    # default -- a paired sweep vs the old 64/64 showed no quality change at ~2x
    # speed (see scripts/acq_sweep.py) -- so it is not a knob worth exposing. Kept
    # functional, though, so scripts/acq_sweep.py can still vary it across arms via
    # the subprocess CLI.
    parser.add_argument(
        "--acq-num-restarts",
        type=int,
        default=ACQ_NUM_RESTARTS,
        help=SUPPRESS,
    )
    parser.add_argument(
        "--acq-raw-samples",
        type=int,
        default=ACQ_RAW_SAMPLES,
        help=SUPPRESS,
    )
    parser.add_argument(
        "--gradient-window",
        type=int,
        default=150,  # 0 = no window, default for now
        help="Gradient GP: keep gradients for only this many high-leverage points "
        "(0 = all). Shrinks the augmented GP to n + window*d -- a high-d speedup "
        "that keeps gradients in the active region (unlike value-only-late).",
    )
    parser.add_argument(
        "--gradient-keep",
        choices=["recent", "best", "both"],
        default="both",
        help=SUPPRESS,
    )
    parser.add_argument(
        "--retain-bonds",
        action="store_true",
        help="Reject any evaluated geometry whose covalent bond graph differs from "
        "the initial structure (the optimizer can rearrange/dissociate strained or "
        "unusual species into a spurious lower minimum). Such points get a failure "
        "energy so they're never selected; a final relaxation that breaks bonds "
        "reverts to the constrained best.",
    )
    args = parser.parse_args()

    # Multiplicity (2S+1) must be a positive integer; anything <= 0 yields an
    # invalid unpaired-electron count (uhf = multiplicity - 1) when stamped onto
    # the atoms in apply_charge_spin. Reject it before touching any structure.
    if args.spin is not None and args.spin < 1:
        parser.error(f"--spin/--multiplicity must be a positive integer, got {args.spin}")

    # Gradient window is a count of points to keep gradients for (0 = all); a
    # negative value is meaningless and would corrupt the GP windowing logic.
    if args.gradient_window < 0:
        parser.error(
            f"--gradient-window must be >= 0 (0 = all), got {args.gradient_window}"
        )

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
        init_conformer_cap=args.init_conformers,
        auto_steps=args.auto,
        relax=args.relax,
        solvent=args.solvent,
        use_gradients=args.use_gradients,
        gradient_steps=args.gradient_steps,
        grad_refit_dense_until=args.grad_refit_dense_until,
        grad_refit_every=args.grad_refit_every,
        lengthscale_prior=args.lengthscale_prior,
        lowmode_prob=args.lowmode_prob,
        lowmode_warmup=args.lowmode_warmup,
        lowmode_kick_deg=args.lowmode_kick_deg,
        lowmode_kick_dir=args.lowmode_kick_dir,
        seed=args.seed,
        priors_file=Path(args.priors) if args.priors else None,
        initial_prior_exponent=args.prior_exponent,
        prior_exponent_decay=args.prior_decay,
        prior_max_concentration=args.prior_max_concentration,
        prior_background_weight=args.prior_background_weight,
        ensemble=args.ensemble,
        certificate_log=Path(args.certificate_log) if args.certificate_log else None,
        certificate_betas=parse_certificate_betas(args.certificate_betas),
        geometry_log=Path(args.geometry_log) if args.geometry_log else None,
        retain_bonds=args.retain_bonds,
        acq_num_restarts=args.acq_num_restarts,
        acq_raw_samples=args.acq_raw_samples,
        gradient_window=args.gradient_window,
        gradient_keep=args.gradient_keep,
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
        # this will do some initial cleanup from the SMILES string. Seed the ETKDG
        # embedding from the run seed so different seeds start from different 3D
        # geometries. We generate several embeddings (count scales with ring
        # flexibility) and pick the lowest-energy one below -- this is the only
        # way to sample distinct ring puckers, since the BO loop perturbs
        # rotatable dihedrals only, never ring bonds.
        candidate_cap = (
            1 if config.conformer_file is not None else config.init_conformer_cap
        )
        candidates, mol = get_initial_candidates(
            config.smiles, seed=config.seed, max_confs=candidate_cap
        )
    else:
        # this will just read the geometry from the file (single candidate)
        init_atoms, mol = get_initial_structure_from_file(str(config.input_file))
        candidates = [init_atoms]
    logger.info(f"Determined initial structure with {len(candidates[0])} atoms")

    # Charge + spin. Charge defaults to the molecule's formal charge (--charge
    # overrides); multiplicity comes from --multiplicity. psi4 reads these from its
    # constructor (CalculatorFactory.from_config -> config.charge/multiplicity);
    # xtb reads them off the Atoms (sum of initial charges / magnetic moments), so
    # stamp them here too -- they propagate through atoms.copy() into every
    # evaluation. uhf = multiplicity - 1 (number of unpaired electrons). Stamp
    # every candidate so the calculator reads the right state when scoring them
    # (all candidates share composition, so the multiplicity is the same).
    config.charge = args.charge if args.charge is not None else Chem.GetFormalCharge(mol)
    config.multiplicity = (
        args.spin if args.spin is not None
        else default_multiplicity(candidates[0], config.charge)
    )
    for cand in candidates:
        apply_charge_spin(cand, config.charge, config.multiplicity)

    # Energy calculator, built early so we can score the initial-structure
    # candidates with the run's energy method (relaxCalc is built later, just
    # before the BO loop, since scoring needs single-point energies only).
    calc = CalculatorFactory.from_config(config, for_optimizer=False, mol=mol)

    # Pick the lowest-energy ETKDG embedding as the starting structure.
    init_atoms, _ = select_initial_structure(candidates, calc)

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
        # Stamp charge/spin on conformer atoms too, so xtb reads them per-eval.
        for conf in initial_conformers:
            apply_charge_spin(conf, config.charge, config.multiplicity)
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

    # Optimizer calculator for the BO relaxations (the energy calc was built
    # earlier to score the initial-structure candidates).
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
        lengthscale_prior=config.lengthscale_prior,
        lowmode_prob=config.lowmode_prob,
        lowmode_warmup=config.lowmode_warmup,
        lowmode_kick_deg=config.lowmode_kick_deg,
        lowmode_kick_dir=config.lowmode_kick_dir,
        acq_num_restarts=config.acq_num_restarts,
        acq_raw_samples=config.acq_raw_samples,
        gradient_window=config.gradient_window,
        gradient_keep=config.gradient_keep,
        cert_log_path=config.certificate_log,
        cert_betas=config.certificate_betas,
        geom_log_path=config.geometry_log,
        retain_bonds=config.retain_bonds,
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
