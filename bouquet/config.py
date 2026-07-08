"""Constants and default configuration values for Bouquet"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Method names are registry-driven in calculator.py; MethodType is just the str
# alias used for the energy/optimizer config field hints. The selectable set is
# CalculatorFactory.available_methods() (the installed subset).
from bouquet.calculator import MethodType

__all__ = ["Configuration", "MethodType"]


# Raw energy (eV) returned by assess.evaluate_energy when an energy evaluation
# or relaxation fails; callers treat any energy at or above this as a failure.
RELAX_FAILURE_ENERGY_EV = 1000.0

# Optimization defaults
DEFAULT_NUM_STEPS = 32
DEFAULT_INIT_STEPS = 5
DEFAULT_RELAXATION_STEPS = 10
DEFAULT_FMAX = 5e-2
TIGHT_FMAX = 1e-3

# Auto-scaling thresholds for optimization steps based on dihedral count
AUTO_STEPS_THRESHOLDS = {
    3: 25,  # <= 3 dihedrals
    5: 50,  # <= 5 dihedrals
    7: 100,  # <= 7 dihedrals
}
AUTO_STEPS_DEFAULT = 200  # > 7 dihedrals

# Default confidence-multiplier grid for the stopping-rule certificate lower bound
# (mu - beta*sigma). Logged per step (one lb_b<beta> column each) so the offline
# replay can pick/calibrate beta without re-running. See solver._compute_certificate.
DEFAULT_CERTIFICATE_BETAS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def format_certificate_betas(betas: tuple) -> str:
    """Serialize a certificate beta grid to the comma-separated CLI form."""
    return ",".join(f"{b:g}" for b in betas)


def parse_certificate_betas(s: str) -> tuple:
    """Parse the comma-separated CLI form of a certificate beta grid."""
    return tuple(float(b) for b in s.split(",") if b.strip())



# Under --auto, reserve at least this many BO refinement steps when seeding many
# initial points (e.g. a systematic peak grid): the seeded points are capped to
# total - this, so a large grid can't consume the whole budget and leave zero
# refinement. Targets small molecules where grid size >= the auto total.
DEFAULT_MIN_AUTO_BO_STEPS = 10

# Dihedral count at/above which "auto" defaults switch on the high-d machinery
# (dimensionality-scaled lengthscale prior + low-mode search). Below it those default
# off. See solver.run_optimization.
# Set from the crossover benchmark (scripts/threshold_bench.py on stopbench d=6-20): the
# gated combo (dim_scaled prior + low-mode) significantly HURTS at d<=11 (the prior over-
# smooths the GP -> trapping), then WINS from d=12 up with the gain growing monotonically
# with d (paired median gain +0.027 @ 12-14 -> +0.094 @ >=18; Spearman rho +0.39, p<1e-4).
# Low-mode also slashes step-1 trapping (26.8%/43.2% -> 6.2%). 12 is exactly the crossover.
HIGH_D_DIHEDRAL_THRESHOLD = 12

# Category-tied move (Phase 3) auto-enable thresholds. Unlike low-mode, the category
# move helps only when there is a repeat the tie can exploit, so it is gated on BOTH the
# dihedral count AND max_spec (the largest tied fitted-library category = repeat units).
# From the high_d_lowmode benchmark (scripts/cat_stratified.py on the 103-mol set): cat
# gain by max_spec is ~0 at <=4 (0.009 eV, coin-flip win) and rises with it; the d>=20
# bins gain ~0.6 eV while the 14-20 bin is ~0.09 (near the 0.11 eV seed noise floor). The
# CONTROL (d>=20 & max_spec<=4, large-but-irregular) has median gain -0.11: cat mildly
# HURTS with no repeat to exploit. So max_spec>4 is the necessary gate; d>14 is inclusive
# (bump toward 20 for wins clearly above noise). Needs a prior_module for the categories.
CAT_D_THRESHOLD = 14        # auto-enable only when n_dihedrals > this
CAT_MAXSPEC_THRESHOLD = 4   # ...AND the largest tied category > this (real repeat)

# Energy clipping for Bayesian optimization
ENERGY_CLIP_OFFSET = 2.0

# Gaussian process settings.
# The acquisition/ensemble GPs operate on dihedral inputs normalized by /360
# (see solver.py), so a full turn is 1.0 and the periodic kernel period is
# exactly 1.0. (A period of 360 on [0, 1] inputs makes the kernel argument
# pi*dx/360 tiny for all pairs -> near-degenerate, near-constant covariance,
# e.g. k(0deg, 180deg) ~= 1.)
GP_PERIOD_LENGTH_MEAN = 1.0
GP_PERIOD_LENGTH_STD = 0.1

# Acquisition function optimization. acq24 (24 restarts / 24 raw samples) is the
# validated default: a paired sweep vs the old 64/64 showed no quality change at
# ~2x speed, while smaller (acq8) regressed. num_restarts is the real speed lever
# (each is an L-BFGS multi-start); raw_samples is matched to it for fidelity to the
# measured config. See scripts/acq_sweep.py (analyze + paired).
ACQ_NUM_RESTARTS = 24
ACQ_RAW_SAMPLES = 24

# Initial guess sampling
INITIAL_GUESS_STD = 90
# How initial guesses are generated: "random" (Gaussian around the start) or
# "peaks" (systematic grid / weighted sampling from the dihedral prior peaks).
DEFAULT_INIT_METHOD = "random"
# Maximum systematic peak-grid size for --init-method peaks before falling back
# to weighted sampling (e.g. 3 modes x 3 dihedrals = 27 fits; x4 = 81 does not).
DEFAULT_INIT_GRID_BUDGET = 64
# Upper bound on the number of ETKDG embeddings generated and scored (with the
# energy calculator) when picking the initial 3D structure from SMILES. The
# actual count scales with ring flexibility (see setup.num_initial_embeddings);
# this caps the cost on highly flexible polycyclic systems. 1 disables the
# multi-embedding search and restores the single-conformer behavior.
DEFAULT_INIT_CONFORMER_CAP = 16

# Prior (PiBO) defaults. The exponent weights the prior against logEI in the
# additive PiBO objective. Since priors.DihedralPriorModule.forward now *averages*
# the per-dihedral log-prior (was a sum), its magnitude is O(1) in dihedral count,
# so the exponent means the same thing across molecule sizes. The prior sweep
# selected exponent 0.5 with a per-step decay of 0.5: strong early guidance that
# fades quickly so the data-driven acquisition takes over within a few steps.
# (Was 2.0, calibrated for the old summed prior, which froze the search after the
# mean change.)
DEFAULT_PRIOR_EXPONENT = 0.5
DEFAULT_PRIOR_DECAY = 0.5
# Cap on von Mises concentration when fitted priors are used for search; see
# bouquet.priors.DEFAULT_MAX_CONCENTRATION.
DEFAULT_PRIOR_MAX_CONCENTRATION = 50.0
# Uniform background weight mixed into each univariate prior; see
# bouquet.priors.DEFAULT_BACKGROUND_WEIGHT. 0.0 disables it.
DEFAULT_PRIOR_BACKGROUND_WEIGHT = 0.05

KCAL_TO_EV = 1.0 / 23.0605

# --- Ensemble selection ---
# Reporting window W (also used at selection); sized for ZPE / level-of-theory
# reordering of the final ensemble.
ENSEMBLE_WINDOW_KCAL = 6.0
# Minimum probability that a candidate's true energy lands inside the window.
ENSEMBLE_P_THRESHOLD = 0.01
# Numerical floor on the posterior sigma used for the inclusion test.
ENSEMBLE_SIGMA_FLOOR_KCAL = 0.1
# Geometric deduplication (CREST-like): RMSD in Angstrom, paired with an
# energy gate so only structures that are BOTH close are merged.
ENSEMBLE_RMSD_THRESHOLD = 0.125
ENSEMBLE_ENERGY_TOL_KCAL = 0.1
# Temperature (K) for Boltzmann populations.
ENSEMBLE_TEMPERATURE = 298.15
# Relative energies above this (eV) indicate a failed evaluation and are dropped
# before fitting the selection GP and before tight optimization.
FAILURE_ENERGY_EV = 100.0
# Boltzmann constant in eV/K.
KB_EV_PER_K = 8.617333262e-5

# Default methods
DEFAULT_ENERGY_METHOD = "gfn2"
DEFAULT_OPTIMIZER_METHOD = "gfnff"

# multithreading (e.g., Psi4)
NUM_THREADS = 4


@dataclass(slots=True)
class Configuration:
    """Configuration for a Bouquet optimization run."""

    # Input specification (one of these must be provided)
    smiles: Optional[str] = None
    input_file: Optional[Path] = None
    conformer_file: Optional[Path] = None

    # Output name
    name: Optional[str] = None

    # Calculation methods
    energy_method: MethodType = DEFAULT_ENERGY_METHOD
    optimizer_method: MethodType = DEFAULT_OPTIMIZER_METHOD

    # Optimization parameters
    num_steps: int = DEFAULT_NUM_STEPS
    init_steps: int = DEFAULT_INIT_STEPS
    init_method: str = DEFAULT_INIT_METHOD
    init_grid_budget: int = DEFAULT_INIT_GRID_BUDGET
    init_conformer_cap: int = DEFAULT_INIT_CONFORMER_CAP
    auto_steps: bool = False
    relax: bool = True
    # Acquisition-optimizer effort (optimize_acqf). num_restarts L-BFGS runs seeded
    # from raw_samples random points. These dominate per-step BO cost (each eval
    # queries the GP posterior ~n(1+d)), so they are the main speed lever; lowering
    # them trades search quality for speed (see scripts/acq_sweep / the timing study).
    acq_num_restarts: int = ACQ_NUM_RESTARTS
    acq_raw_samples: int = ACQ_RAW_SAMPLES
    # High-leverage gradient subset (gradient GP): keep gradients for only this many
    # points (0 = all), shrinking the augmented GP from n*(1+d) to n + window*d -- a
    # high-d speedup that, unlike value-only-late, keeps gradients in the active
    # region. gradient_keep selects which: recent | best | both. See
    # solver._restrict_gradient_mask.
    gradient_window: int = 0
    gradient_keep: str = "both"
    # Use the gradient-enhanced periodic GP surrogate: record dE/dtheta at each
    # evaluation and feed it to the acquisition GP (see GradientEnhancedPeriodicGP).
    use_gradients: bool = False
    # Cap on the number of leading BO steps that use the gradient-enhanced GP; the
    # remaining steps use the value-only GP. The gradient GP's per-step cost grows
    # steeply with observation count, so this bounds it on large molecules while
    # keeping the early benefit. <=0 keeps gradients for the whole run.
    gradient_steps: int = 0
    # Gradient-GP hyperparameter refit schedule (see solver._run_optimization_loop):
    # cold full fits for the first `grad_refit_dense_until` BO steps, then freeze the
    # hyperparameters and only re-condition; `grad_refit_every` > 0 cold-refreshes
    # them every that many post-dense steps. Default (20, 0) = the "gradfreeze"
    # schedule: cold-fit 20 steps then freeze (validated quality-neutral vs full
    # refitting across 5-11 dihedrals). Set grad_refit_dense_until=0 to refit every
    # step (the slow reference).
    grad_refit_dense_until: int = 20
    grad_refit_every: int = 0
    # Value-only-GP lengthscale prior: "auto" (dim_scaled once d >= the high-d
    # threshold, else none), "none" (free fit, historical), or "dim_scaled" (Hvarfner
    # dimensionality-scaled LogNormal). See solver._periodic_covar_module / run_optimization.
    lengthscale_prior: str = "auto"
    # Phase 2.5 low-mode / basin-hopping moves (see solver._low_mode_move). None = auto
    # (0.5 once d >= the high-d threshold, else 0); a float sets it explicitly (0 disables).
    # With prob lowmode_prob (past lowmode_warmup evals) a step is a committed kick +
    # UNCONSTRAINED relax along a soft mode. lowmode_kick_dir = "pca" (default) | "enm".
    lowmode_prob: Optional[float] = None
    lowmode_warmup: int = 100
    lowmode_kick_deg: float = 60.0
    lowmode_kick_dir: str = "pca"
    # Phase 3 category-tied collective move (see solver._category_move). With prob
    # category_prob (past category_warmup evals) a step sets every dihedral in a SMARTS
    # prior category to one shared value (chemistry-defined embedding) and relaxes
    # UNCONSTRAINED; requires priors_file. category_min_moves reduced points are
    # prior-seeded before the reduced-space GP is fit. None = auto (0.5 when priors are
    # given and n_dihedrals > CAT_D_THRESHOLD and max_spec > CAT_MAXSPEC_THRESHOLD, i.e.
    # a large molecule with real repeat structure; else 0); a float sets it explicitly.
    category_prob: Optional[float] = None
    category_warmup: int = 20
    category_min_moves: int = 6
    seed: int = field(default_factory=lambda: datetime.now().microsecond)

    # Prior settings
    priors_file: Optional[Path] = None
    initial_prior_exponent: float = DEFAULT_PRIOR_EXPONENT
    prior_exponent_decay: float = DEFAULT_PRIOR_DECAY
    prior_max_concentration: float = DEFAULT_PRIOR_MAX_CONCENTRATION
    prior_background_weight: float = DEFAULT_PRIOR_BACKGROUND_WEIGHT

    # Ensemble selection
    ensemble: bool = False

    # When set, reject any evaluated geometry whose covalent bond graph differs
    # from the initial structure's (the optimizer can rearrange/dissociate strained
    # or unusual species into a spuriously low, non-conformer minimum). Such points
    # get a failure energy so they are never selected; a final relaxation that
    # breaks bonds reverts to the constrained best. See bouquet.setup.connectivity.
    retain_bonds: bool = False

    # Stopping-rule calibration benchmark: when set, the solver logs a per-BO-step
    # certificate (mu_min/alpha_max/lb + e_eval/e_best/n_calls/wall_s) to this CSV.
    # certificate_betas is the grid of confidence multipliers for the lower-bound
    # term -- one lb_b<beta> column per beta, so the offline replay can calibrate
    # beta without re-running.
    certificate_log: Optional[Path] = None
    certificate_betas: tuple = DEFAULT_CERTIFICATE_BETAS

    # Stopping-rule benchmark, geometry trail: when set, the solver writes the
    # geometry at each best-so-far improvement (constrained-relaxed, the geometry
    # actually visited) plus the final unconstrained-relaxed best to this
    # multi-frame XYZ. Written incrementally so a timed-out (censored) run still
    # keeps its improvements. Enables the RMSD-identity success criterion and
    # distinct-conformer (basin) analysis offline -- the torsion vector alone can't,
    # since it omits ring pucker and other non-dihedral DOF.
    geometry_log: Optional[Path] = None

    # Output
    out_dir: Optional[Path] = None

    # Psi4 settings (for DFT methods)
    num_threads: int = NUM_THREADS
    charge: int = 0
    multiplicity: int = 1

    # Implicit solvent model. gfn2/gfnff pass this straight through as xtb's
    # native GBSA/ALPB "solvent" keyword; psi4 methods use it to turn on DDX
    # continuum solvation (see calculator._psi4_calculator). None = gas phase.
    # Not supported by gfn0 (no fitted GBSA parameters) or by ml/forcefield
    # methods (ani, aimnet2, mmff, uff).
    solvent: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.smiles is None and self.input_file is None:
            raise ValueError("Must specify either smiles or input_file")

        # Convert string paths to Path objects if needed
        if isinstance(self.input_file, str):
            self.input_file = Path(self.input_file)
        if isinstance(self.conformer_file, str):
            self.conformer_file = Path(self.conformer_file)
        if isinstance(self.out_dir, str):
            self.out_dir = Path(self.out_dir)
        if isinstance(self.priors_file, str):
            self.priors_file = Path(self.priors_file)
        if isinstance(self.certificate_log, str):
            self.certificate_log = Path(self.certificate_log)

        # Acquisition-optimizer effort must be positive: 0 or negative would reach
        # optimize_acqf as an opaque failure (or silently skip the search).
        if self.acq_num_restarts < 1 or self.acq_raw_samples < 1:
            raise ValueError(
                "acq_num_restarts and acq_raw_samples must be positive integers; got "
                f"acq_num_restarts={self.acq_num_restarts}, "
                f"acq_raw_samples={self.acq_raw_samples}."
            )

        # Derive a run name from the input if one wasn't provided: prefer the
        # SMILES string, otherwise fall back to the input file's stem.
        if self.name is None:
            if self.smiles is not None:
                self.name = self.smiles
            elif self.input_file is not None:
                self.name = self.input_file.stem

    def auto_total(self, num_dihedrals: int) -> int:
        """Total --auto evaluation budget (initial guesses + BO steps).

        Args:
            num_dihedrals: Number of dihedral angles detected

        Returns:
            The tiered total evaluation budget for this dihedral count.
        """
        total = AUTO_STEPS_DEFAULT
        for threshold, steps in sorted(AUTO_STEPS_THRESHOLDS.items()):
            if num_dihedrals <= threshold:
                total = steps
                break
        return total

    def compute_auto_steps(self, num_dihedrals: int, num_initial: int) -> int:
        """Compute the number of optimization steps based on dihedral count.

        Args:
            num_dihedrals: Number of dihedral angles detected
            num_initial: Number of initial conformers/guesses

        Returns:
            Number of optimization steps to perform
        """
        if not self.auto_steps:
            return self.num_steps

        return max(0, self.auto_total(num_dihedrals) - num_initial)
