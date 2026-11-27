from argparse import BooleanOptionalAction, SUPPRESS, ArgumentParser
import secrets
from pathlib import Path
from typing import Optional

from bouquet.calculator import CalculatorFactory
from bouquet.clash import DEFAULT_CLASH_RATIO
from bouquet.config import (
    ACQ_NUM_RESTARTS,
    ACQ_RAW_SAMPLES,
    DEFAULT_CERTIFICATE_BETAS,
    DEFAULT_ENERGY_METHOD,
    DEFAULT_ENSEMBLE_DIVERSITY,
    DEFAULT_ENSEMBLE_EXPLORE_MODE,
    DEFAULT_ENSEMBLE_STEPS,
    DEFAULT_ETKDG_CLUSTER_DEG,
    DEFAULT_ETKDG_FALLBACK_CLUSTER_DEG,
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
    ENSEMBLE_EXPLORE_MODES,
    GRADIENT_MATRIX_CAP,
    GRADIENT_WINDOW_MIN,
    INIT_METHODS,
    KERNEL_ARMS,
    Configuration,
    RunOptions,
    format_certificate_betas,
    parse_certificate_betas,
)


def _resolve_priors_arg(value: Optional[str]) -> Optional[Path]:
    """Resolve the ``--priors`` argument to a JSON path (or ``None`` = priors off).

    ``None`` (flag omitted) leaves PiBO steering off. The bare-flag sentinel selects
    the bundled fitted priors; anything else is treated as a user-supplied path.
    """
    if value is None:
        return None
    if value == "__bundled__":
        from bouquet.priors import default_priors_path

        return default_priors_path()
    return Path(value)


def build_parser() -> ArgumentParser:
    """Build the bouquet CLI argument parser (kept free of heavy imports)."""
    available_methods = CalculatorFactory.available_methods()

    parser = ArgumentParser(
        description="Optimize molecular conformers using Bayesian optimization"
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="List installed energy/optimizer methods and exit",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: cryptographically random)",
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
        "--out-dir",
        type=str,
        default=None,
        help="Explicit output directory (default: solutions/<name>-<seed>-<method>-<hash>)",
    )
    parser.add_argument(
        "--conformer-file",
        type=str,
        help="File containing multiple conformers to use as initial guesses "
        "(instead of random sampling). Overrides --init-steps.",
    )
    parser.add_argument(
        "--max-conformers",
        type=int,
        default=0,
        help="Cap on conformers taken from --conformer-file (0 = all). A ring-seed "
        ".json is cut with --select-rule; other formats are truncated.",
    )
    parser.add_argument(
        "--select-rule",
        default="energy",
        help="How to choose which ring-seed conformers to keep under --max-conformers "
        "(ring-seed .json only): energy, coverage, novelty, diversity, population, "
        "geometry, hybrid. Default: energy.",
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
        default=None,
        help="Number of initial guesses to make (default: auto-scaled "
        f"min({12}, max({DEFAULT_INIT_STEPS}, d + 2)) from the active dihedral count; "
        "pass an explicit integer to override; ignored if --conformer-file is provided)",
    )
    parser.add_argument(
        "--init-method",
        choices=list(INIT_METHODS),
        default=DEFAULT_INIT_METHOD,
        help="How to generate initial guesses (ignored if --conformer-file is "
        "provided): 'etkdg' (default; dimension-scaled ETKDGv3 + MMFF94 pool "
        "clustered in symmetry-reduced torsion space), 'random' (Gaussian draws "
        "around the start geometry), or 'peaks' (prior-peak grid / sampling).",
    )
    # "Maximum systematic peak-grid size for --init-method peaks before "
    # "falling back to weighted sampling of --init-steps points."
    parser.add_argument(
        "--init-grid-budget",
        type=int,
        default=DEFAULT_INIT_GRID_BUDGET,
        help=SUPPRESS,
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
        "--method",
        choices=available_methods,
        default=None,
        help="Shorthand that sets both --energy and --optimizer to the same "
        "method. An explicit --energy or --optimizer overrides it for that role.",
    )
    parser.add_argument(
        "--energy",
        choices=available_methods,
        default=None,
        help="Energy method (only methods whose dependencies are installed are "
        f"listed). Default: {DEFAULT_ENERGY_METHOD}, or --method if given.",
    )
    parser.add_argument(
        "--optimizer",
        choices=available_methods,
        default=None,
        help="Optimizer method (only methods whose dependencies are installed "
        f"are listed). Default: {DEFAULT_OPTIMIZER_METHOD}, or --method if given.",
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
        action=BooleanOptionalAction,
        default=True,
        help="Relax non-dihedral degrees of freedom before computing energy "
        "(default: on; pass --no-relax for rigid torsion scans)",
    )
    parser.add_argument(
        "--drop-h-rotors",
        dest="drop_h_c3",
        action=BooleanOptionalAction,
        default=True,
        help="Exclude trivial terminal H-only C3 rotors (-CH3, -NH3+, -SiH3) "
        "from the BO dihedral search when --relax is active so local relaxation "
        "places them instead of spending BO dimensions on methyl hydrogens "
        "(default: on; pass --no-drop-h-rotors to include all rotatable bonds).",
    )
    parser.add_argument(
        "--solvent",
        type=str,
        default=None,
        help="Implicit solvent model (e.g. 'water'). Default: gas phase (no "
        "solvent). Supported by gfn2 and gfnff (via xtb's native GBSA/ALPB "
        "solvent keyword) and psi4 methods (via DDX continuum solvation); not "
        "supported by gfn0 (no fitted GBSA parameters), ani, aimnet2, mmff, "
        "or uff.",
    )
    parser.add_argument(
        "--use-gradients",
        action="store_true",
        help="Use the gradient-enhanced GP surrogate: project the calculator's "
        "forces onto each torsion (dE/dtheta) and feed them to the acquisition "
        "GP, so each energy evaluation also contributes a gradient observation.",
    )
    # "With --use-gradients, use the gradient-enhanced GP only for the "
    # "first N BO steps, then switch to the value-only GP. The gradient GP's "
    # "per-step cost grows steeply with the observation count, so this caps it "
    # "on large/flexible molecules while keeping the early-search benefit. "
    # "0 (default) keeps gradients for the whole run."
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=0,
        help=SUPPRESS,
    )
    # "Gradient-GP refit schedule ('gradfreeze'): do a full hyperparameter "
    # "fit for the first N BO steps, then FREEZE the hyperparameters and only "
    # "re-condition (one Cholesky/step instead of ~200). Refitting is the dominant "
    # "gradient-GP cost and grows with observation count, so freezing keeps the "
    # "full-gradient run affordable on large molecules. Default 20 (validated "
    # "quality-neutral vs full refitting, 5-11 dihedrals); set 0 to refit every "
    # "step (the slow reference)."
    parser.add_argument(
        "--grad-refit-dense-until",
        type=int,
        default=20,
        help=SUPPRESS,
    )
    # "After the dense phase, optionally cold-refresh the frozen "
    # "hyperparameters every Nth BO step (0 freezes for the rest of the run; the "
    # "refresh is a cold fit -- warm-starting drifts the hyperparameters and "
    # "degrades the search). 0 (default); with no dense phase, fits every step."
    parser.add_argument(
        "--grad-refit-every",
        type=int,
        default=0,
        help=SUPPRESS,
    )
    # "Prior on the value-only GP's periodic lengthscale. 'auto' (default): "
    # "'dim_scaled' once the dihedral count reaches the high-d threshold, else 'none'. "
    # "'none': free MLL fit (historical). 'dim_scaled': Hvarfner et al. (ICML 2024) "
    # "dimensionality-scaled LogNormal prior, biasing the GP toward smoother fits "
    # "as the dihedral count grows -- helps high-d search."
    parser.add_argument(
        "--lengthscale-prior",
        choices=["auto", "none", "dim_scaled"],
        default="auto",
        help=SUPPRESS,
    )
    # "Low-mode search: probability that an eligible BO step (past "
    # "--lowmode-warmup evaluations) is replaced by a committed kick along a soft "
    # "mode followed by an UNCONSTRAINED relaxation (letting the dihedrals move, so "
    # "the geometry can slide along a curved fold valley a standard BO step cannot "
    # "cross). Default: auto (0.5 once the dihedral count reaches the high-d "
    # "threshold, else 0). Set 0 to disable, or a probability in (0, 1]."
    parser.add_argument(
        "--lowmode-prob",
        type=float,
        default=None,
        help=SUPPRESS,
    )
    # "With --lowmode-prob, only start low-mode moves after this many "
    # "evaluations (default 100); align it past the gradient Phase A."
    parser.add_argument(
        "--lowmode-warmup",
        type=int,
        default=100,
        help=SUPPRESS,
    )
    # "With --lowmode-prob, per-dihedral RMS kick amplitude (degrees, "
    # "default 60) along the chosen soft mode."
    parser.add_argument(
        "--lowmode-kick-deg",
        type=float,
        default=60.0,
        help=SUPPRESS,
    )
    # "Kick-direction source for low-mode moves: 'pca' (data-derived position "
    # "PCA of the low-energy set) or 'enm' (data-independent elastic-network soft "
    # "modes -- global bend/compaction = folding, projected to torsion space). "
    #"Default 'pca'."
    parser.add_argument(
        "--lowmode-kick-dir",
        choices=["pca", "enm"],
        default="pca",
        help=SUPPRESS,
    )
    # "Category-tied search: probability that an eligible BO step (past "
    # "--category-warmup evaluations) is replaced by a collective move that sets every "
    #    "dihedral sharing a SMARTS prior category to one shared value (a chemistry-"
    #    "defined low-dimensional embedding), chosen by a periodic GP + LogEI over the reduced space, then "
    #    "relaxed UNCONSTRAINED. Requires --priors and --relax. Default: auto (0.5 when "
    #    "--priors is given and the molecule is large with real repeat structure -- "
    #    "n_dihedrals > 14 and max category > 4 -- else 0). Set 0 to disable, or a "
    #    "probability in (0, 1]. Takes precedence over a low-mode move on a shared step.",
    parser.add_argument(
        "--category-prob",
        type=float,
        default=None,
        help=SUPPRESS
    )
    # "With --category-prob, only start category moves after this many "
    # "evaluations (default 20); the incumbent anchors each move."
    parser.add_argument(
        "--category-warmup",
        type=int,
        default=20,
        help=SUPPRESS,
    )
    # "With --category-prob, number of prior-seeded reduced points to collect "
    #  "before fitting the reduced-space GP (default 6); below this, the per-category "
    #  "value is drawn from the prior."
    parser.add_argument(
        "--category-min-moves",
        type=int,
        default=6,
        help=SUPPRESS,
    )
    parser.add_argument(
        "--priors",
        type=str,
        nargs="?",
        const="__bundled__",
        default=None,
        help="Enable PiBO priors. Bare --priors uses the bundled fitted priors "
        "(gfn2_priors.json); pass a path for a custom JSON. Omit to leave priors off.",
    )
    parser.add_argument(
        "--bivariate-priors",
        type=str,
        default=None,
        help="Path to a JSON file of correlated (2D) dihedral priors. Each entry "
        "carries a SMARTS pattern, a topology, and a bivariate von Mises mixture; "
        "matched dihedral pairs are steered jointly instead of independently. "
        "May be given with or without --priors.",
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
        "--ensemble-steps",
        type=int,
        default=DEFAULT_ENSEMBLE_STEPS,
        help="With --ensemble: budget of active level-set exploration steps run "
        "after the main search to discover additional basins. -1 = auto "
        "(scale with rotor count), 0 = passive harvest only, >0 = fixed cap.",
    )
    # Torsion-space diversity penalty subtracted from the level-set acquisition
    # during ensemble exploration (0 = off).
    parser.add_argument(
        "--ensemble-diversity",
        type=float,
        default=DEFAULT_ENSEMBLE_DIVERSITY,
        help=SUPPRESS,
    )
    # Exploration mode: "levelset" (P_in*sigma, best low-energy recall) or
    # "hybrid" (level-set then an annealed boundary sweep for the high-energy shell).
    parser.add_argument(
        "--ensemble-explore-mode",
        choices=list(ENSEMBLE_EXPLORE_MODES),
        default=DEFAULT_ENSEMBLE_EXPLORE_MODE,
        help=SUPPRESS,
    )
    # "Path to write a per-BO-step stopping-rule certificate CSV "
    # "(mu_min/lb/alpha_max + e_eval/e_best/n_calls/wall_s). Used by the "
    #"various benchmarks; off by default."
    parser.add_argument(
        "--certificate-log",
        type=str,
        default=None,
        help=SUPPRESS,
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
    # Hidden (SUPPRESS): benchmark-only. Redirects the per-evaluation
    # (coords, energy) trail so a sweep can collect every trial in one place.
    parser.add_argument(
        "--structure-log",
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
        default=0,  # 0 = no window, default for now
        help="Gradient GP: keep gradients for only this many high-leverage points. "
        "0 (default) derives the window from --gradient-matrix-cap instead; a "
        "positive value pins it and ignores the cap. Shrinks the augmented GP to "
        "n + window*d -- a high-d speedup that keeps gradients in the active region.",
    )
    parser.add_argument(
        "--gradient-matrix-cap",
        type=int,
        default=GRADIENT_MATRIX_CAP,
        help="Gradient GP: cap the augmented covariance at N = n + window*d points, "
        f"deriving the window as (cap - n)/d (default {GRADIENT_MATRIX_CAP}, "
        "0 = uncapped). Capping N rather than the point count is what bounds cost, "
        "since per-step acquisition time is ~linear in N and a fixed window still "
        "grows with d.",
    )
    parser.add_argument(
        "--gradient-window-min",
        type=int,
        default=GRADIENT_WINDOW_MIN,
        help=SUPPRESS,
    )
    parser.add_argument(
        "--gradient-keep",
        choices=["recent", "best", "both"],
        default="both",
        help=SUPPRESS,
    )
    parser.add_argument(
        "--kernel",
        choices=KERNEL_ARMS,
        default="product",
        help="GP covariance. 'product' (default) is a product of per-torsion "
        "periodic kernels sharing ONE lengthscale; 'ard' gives each torsion its own. "
        "ARD was the only alternative never worse than product in the end-to-end "
        "search benchmark (54-59%% of paired trials better at every budget) but was "
        "not significant at the endpoint, so it stays opt-in.",
    )
    parser.add_argument(
        "--retain-bonds",
        action="store_true",
        help="Also reject any evaluated geometry that BROKE a bond relative to the "
        "initial structure (the optimizer can dissociate strained or unusual species "
        "into a spurious lower minimum). Such points get a failure energy so they're "
        "never selected; a final relaxation that breaks bonds reverts to the "
        "constrained best. Bond *formation* is already rejected by default -- see "
        "--no-species-guard.",
    )
    parser.add_argument(
        "--no-species-guard",
        dest="species_guard",
        action="store_false",
        help="Disable the default rejection of geometries whose relaxation formed a "
        "new bond. Such a collapse reports a spuriously huge negative energy, becomes "
        "the incumbent, and traps the search there for the rest of the budget.",
    )
    parser.add_argument(
        "--no-clash-constraint",
        dest="clash_constraint",
        action="store_false",
        help="Search the full torsion box instead of restricting the acquisition "
        "optimizer to sterically feasible geometries.",
    )
    parser.add_argument(
        "--clash-threshold",
        type=float,
        default=DEFAULT_CLASH_RATIO,
        metavar="RATIO",
        help="Worst permitted non-bonded contact, as a fraction of the summed van der "
        "Waals radii, for the acquisition feasibility region. Applied to the rigid "
        "geometry before relaxation, so it must be looser than a relaxed conformer "
        f"would show (default {DEFAULT_CLASH_RATIO}; 0.8 is far too tight and "
        "discards most of the good conformers).",
    )
    parser.add_argument(
        "--clash-include-hbonds",
        dest="clash_exclude_hbonds",
        action="store_false",
        help="Include polar-H...(N,O,F,S) pairs in the clash test. Off by default "
        "because a real intramolecular H-bond sits near 0.5x the summed vdW radii, so "
        "screening those pairs rejects exactly the folded conformers worth finding.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_methods:
        for name, desc in sorted(CalculatorFactory.describe_methods().items()):
            print(f"{name:10s} {desc}")
        return

    if args.seed is None:
        args.seed = secrets.randbelow(2**31)

    # --method is shorthand for setting both roles at once. An explicit
    # --energy/--optimizer wins for its role; otherwise fall back to --method,
    # then to the per-role default.
    if args.energy is None:
        args.energy = args.method if args.method is not None else DEFAULT_ENERGY_METHOD
    if args.optimizer is None:
        args.optimizer = (
            args.method if args.method is not None else DEFAULT_OPTIMIZER_METHOD
        )

    # Multiplicity (2S+1) must be a positive integer; anything <= 0 yields an
    # invalid unpaired-electron count (uhf = multiplicity - 1) when stamped onto
    # the atoms in apply_charge_spin. Reject it before touching any structure.
    if args.spin is not None and args.spin < 1:
        parser.error(f"--spin/--multiplicity must be a positive integer, got {args.spin}")

    # Search / surrogate / benchmark knobs handed to run_optimization (lowmode_modes
    # has no CLI flag, so it keeps its RunOptions default). Validation lives in
    # RunOptions.__post_init__ and Configuration.__post_init__.
    try:
        run_opts = RunOptions(
            use_gradients=args.use_gradients,
            gradient_steps=args.gradient_steps,
            grad_refit_dense_until=args.grad_refit_dense_until,
            grad_refit_every=args.grad_refit_every,
            gradient_window=args.gradient_window,
            gradient_matrix_cap=args.gradient_matrix_cap,
            gradient_window_min=args.gradient_window_min,
            gradient_keep=args.gradient_keep,
            acq_num_restarts=args.acq_num_restarts,
            acq_raw_samples=args.acq_raw_samples,
            lengthscale_prior=args.lengthscale_prior,
            lowmode_prob=args.lowmode_prob,
            lowmode_warmup=args.lowmode_warmup,
            lowmode_kick_deg=args.lowmode_kick_deg,
            lowmode_kick_dir=args.lowmode_kick_dir,
            category_prob=args.category_prob,
            category_warmup=args.category_warmup,
            category_min_moves=args.category_min_moves,
            initial_prior_exponent=args.prior_exponent,
            prior_exponent_decay=args.prior_decay,
            cert_log_path=Path(args.certificate_log) if args.certificate_log else None,
            cert_betas=parse_certificate_betas(args.certificate_betas),
            geom_log_path=Path(args.geometry_log) if args.geometry_log else None,
            structure_log_path=Path(args.structure_log) if args.structure_log else None,
            retain_bonds=args.retain_bonds,
            kernel_arm=args.kernel,
            species_guard=args.species_guard,
            clash_constraint=args.clash_constraint,
            clash_threshold=args.clash_threshold,
            clash_exclude_hbonds=args.clash_exclude_hbonds,
            ensemble_steps=args.ensemble_steps,
            ensemble_diversity=args.ensemble_diversity,
            ensemble_explore_mode=args.ensemble_explore_mode,
        )
        config = Configuration(
            smiles=args.smiles,
            input_file=Path(args.file) if args.file else None,
            name=args.name,
            out_dir=Path(args.out_dir) if args.out_dir else None,
            conformer_file=Path(args.conformer_file) if args.conformer_file else None,
            max_conformers=args.max_conformers,
            max_conformer_rule=args.select_rule,
            energy_method=args.energy,
            optimizer_method=args.optimizer,
            num_steps=args.num_steps,
            init_steps=args.init_steps,
            init_method=args.init_method,
            init_grid_budget=args.init_grid_budget,
            init_conformer_cap=args.init_conformers,
            auto_steps=args.auto,
            relax=args.relax,
            drop_h_c3=args.drop_h_c3,
            solvent=args.solvent,
            seed=args.seed,
            priors_file=_resolve_priors_arg(args.priors),
            bivariate_priors_file=(
                Path(args.bivariate_priors) if args.bivariate_priors else None
            ),
            prior_max_concentration=args.prior_max_concentration,
            prior_background_weight=args.prior_background_weight,
            ensemble=args.ensemble,
            run=run_opts,
        )
    except ValueError as exc:
        parser.error(str(exc))

    # Defer heavy scientific-stack imports (numpy, rdkit, torch, botorch, ase)
    # until after argument parsing and validation succeed.
    import numpy as np
    from rdkit import Chem

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
        plan_etkdg_initial_points,
        select_initial_structure,
    )
    from bouquet.solver import plan_initial_points, run_optimization

    # Gradient labels are only consistent with the energy objective when the
    # geometry is a constrained minimum of the *energy* calculator. With
    # relaxation the geometry is minimized on the optimizer surface, so the
    # projected torsion gradient equals dE*/dtheta only if the two calculators
    # are the same surface. Refuse the mismatched combination rather than feeding
    # the GP biased gradient labels. (Without --relax the rigid-scan gradient is
    # always consistent, so the check is limited to the relaxed case.)
    if (
        config.run.use_gradients
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

    # Relaxation drives the optimizer calculator through FIRE2/LBFGS, which need
    # forces (atoms.get_forces()) at every step. _psi4_calculator's DDX solvation
    # branch (calculator.py) is only exercised for single-point energies in
    # practice -- gradients under DDX are not validated here -- so reject the
    # combination outright rather than risk silently wrong or crashing forces.
    if (
        config.solvent is not None
        and config.relax
        and CalculatorFactory.uses_psi4(config.optimizer_method)
    ):
        raise ValueError(
            "--solvent with --relax and a psi4-based --optimizer is not supported "
            f"(got optimizer={config.optimizer_method!r}). Relaxation needs forces "
            "from the optimizer calculator, and psi4 DDX solvation gradients are "
            "not validated. Drop --solvent, drop --relax, or use a non-psi4 "
            "--optimizer (e.g. gfn2/gfnff, which support --solvent directly)."
        )

    # Make an output directory (respecting explicit --out-dir if provided)
    if config.out_dir is not None:
        out_dir = config.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
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

    # Detect the dihedral angles (drop trivial H-only C3 tops when relaxing
    # unless opted out via --no-drop-h-rotors).
    dihedrals = detect_dihedrals(mol, drop_h_c3=config.drop_h_c3 and config.relax)
    logger.info(f"Detected {len(dihedrals)} active dihedral angles")

    # Load conformers if provided
    initial_conformers = None
    if config.conformer_file is not None:
        initial_conformers = get_conformers_from_file(
            str(config.conformer_file),
            max_conformers=config.max_conformers,
            select_rule=config.max_conformer_rule,
        )
        picked = (f" (capped at {config.max_conformers} by '{config.max_conformer_rule}')"
                  if config.max_conformers else "")
        logger.info(
            f"Loaded {len(initial_conformers)} conformers from {config.conformer_file}{picked}"
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
    if dihedrals and (
        config.priors_file is not None or config.bivariate_priors_file is not None
    ):
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
            bivariate_file=config.bivariate_priors_file,
            max_concentration=max_conc,
            background_weight=config.prior_background_weight,
        )
        sources = [
            str(f)
            for f in (config.priors_file, config.bivariate_priors_file)
            if f is not None
        ]
        logger.info(f"Created prior module from {', '.join(sources)}")
        logger.info(prior_module.describe())

    # Assign torsion-library SMARTS categories to every dihedral. This ties chemically-
    # equivalent rotors for the category-tied collective move and is independent of the
    # fitted priors above -- so category moves are available even with PiBO steering off
    # (the auto-default fires on large molecules with real repeat structure).
    from bouquet.priors import assign_categories

    category_assignments = assign_categories(mol, [d.chain for d in dihedrals])
    logger.info(
        f"Assigned torlib categories to {len(category_assignments)} of "
        f"{len(dihedrals)} dihedrals"
    )

    # Plan initial guesses from ETKDG+MMFF94 distinct clusters (default) or
    # prior peaks (--init-method peaks). Conformers take precedence; --init-method
    # random leaves initial_dihedrals=None for Gaussian draws around the start.
    init_steps = config.resolve_init_steps(len(dihedrals))
    initial_dihedrals = None
    if dihedrals and config.init_method == "etkdg" and initial_conformers is None:
        from bouquet.clash import build_feasibility

        feasible = (
            build_feasibility(
                init_atoms,
                mol,
                dihedrals,
                threshold=config.run.clash_threshold,
                exclude_hbonds=config.run.clash_exclude_hbonds,
            )
            if config.run.clash_constraint
            else None
        )
        max_init = init_steps
        if config.auto_steps:
            total = config.auto_total(len(dihedrals))
            max_init = min(max_init, max(1, total - DEFAULT_MIN_AUTO_BO_STEPS))
        planned = plan_etkdg_initial_points(
            mol,
            init_atoms,
            dihedrals,
            max_points=max_init,
            seed=config.seed,
            cluster_deg=DEFAULT_ETKDG_CLUSTER_DEG,
            fallback_cluster_deg=DEFAULT_ETKDG_FALLBACK_CLUSTER_DEG,
            feasible=feasible,
        )
        if len(planned) > 0:
            initial_dihedrals = planned
        else:
            logger.info(
                "ETKDG cluster seeding found no distinct clash-free clusters; "
                "falling back to random initial guesses."
            )
    elif dihedrals and config.init_method == "peaks" and initial_conformers is None:
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
            init_steps,
            config.init_grid_budget,
            config.seed,
            max_points=max_init,
            symmetry_orders=[d.symmetry_order for d in dihedrals],
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
        initial_count = init_steps
    num_steps = config.compute_auto_steps(len(dihedrals), initial_count)

    # Save the initial guess
    save_structure(out_dir, init_atoms, "initial.xyz")

    # Optimizer calculator for the BO relaxations (the energy calc was built
    # earlier to score the initial-structure candidates).
    relaxCalc = CalculatorFactory.from_config(config, for_optimizer=True, mol=mol)

    # Full geometries whose RING PUCKERS should be able to enter the ensemble: the
    # BO loop turns rotatable bonds only, so every evaluated point inherits the start
    # structure's ring geometry -- ring conformations can only come from these
    # initial full structures. Feed the scored ETKDG embeddings (all of them, not
    # just the lowest-energy start we keep) plus any --conformer-file set into the
    # harvest's tight-opt (see _perform_ensemble_relaxation). Only used in --ensemble.
    ensemble_seed_geometries = list(candidates)
    if initial_conformers is not None:
        ensemble_seed_geometries += initial_conformers

    result = run_optimization(
        init_atoms,
        dihedrals,
        num_steps,
        calc,
        relaxCalc,
        init_steps,
        out_dir,
        relax=config.relax,
        seed=config.seed,
        initial_conformers=initial_conformers,
        initial_dihedrals=initial_dihedrals,
        prior_module=prior_module,
        category_assignments=category_assignments,
        return_ensemble=config.ensemble,
        ensemble_seed_geometries=ensemble_seed_geometries,
        mol=mol,
        opts=config.run,
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
