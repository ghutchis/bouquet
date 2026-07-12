"""Constants and default configuration values for Bouquet"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Method names are registry-driven in calculator.py; MethodType is just the str
# alias used for the energy/optimizer config field hints. The selectable set is
# CalculatorFactory.available_methods() (the installed subset).
from bouquet.calculator import MethodType

__all__ = ["Configuration", "MethodType", "RunOptions"]


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

# --- Ensemble level-set exploration ---
# Active exploration budget after the main search: -1 = auto (scale by dihedral
# count), 0 = passive harvest only (historical --ensemble behavior), >0 = fixed
# hard cap on exploration steps.
DEFAULT_ENSEMBLE_STEPS = -1
# Discovery window: the level-set acquisition pushes toward conformers within
# this energy of the running best. Wider than ENSEMBLE_WINDOW_KCAL so basins
# just outside the (narrower) reporting window are still found and can relax in.
ENSEMBLE_EXPLORE_KCAL = 10.0
# Torsion-space diversity strength for the level-set acquisition (0 = off).
# Multiplicative local penalization: the acquisition near a known basin is scaled
# by (1 - diversity), so 0.5 halves it there and 1.0 fully suppresses re-sampling
# it, while regions far from every basin are left untouched (see
# _LevelSetAcquisition). Values in (0, 1]; the basin width is ENSEMBLE_BASIN_DEG.
DEFAULT_ENSEMBLE_DIVERSITY = 0.5
# Two conformers count as the same basin when their wrapped RMS angular distance
# is below this (degrees); the coarse in-search analog of the final RMSD dedup.
ENSEMBLE_BASIN_DEG = 40.0
# Adaptive stop: require this many consecutive steps with no newly discovered
# basin AND a collapsed posterior before ending exploration early.
ENSEMBLE_SATURATION_ITERS = 5
# Adaptive stop: max posterior sigma (kcal/mol) over a Sobol pool below which the
# surface is considered well-characterized.
ENSEMBLE_SIGMA_STOP_KCAL = 1.0
# Exploration mode:
#   "levelset" -- P_in * sigma over the whole window; best low-energy recall but
#                 under-covers the high-energy shell.
#   "hybrid"   -- level-set for the first half of the budget (lock in the
#                 low-energy manifold), then a boundary sweep for the second half
#                 -- the boundary acquisition (-|mu - target| + kappa*sigma) with
#                 the target energy annealed upward -- to fill the high-energy
#                 shell: captures both instead of trading one for the other.
# (A pure-boundary mode was benchmarked and dropped -- it under-covered the
# low-energy manifold; the boundary sweep is only useful as hybrid's second half.)
DEFAULT_ENSEMBLE_EXPLORE_MODE = "levelset"
ENSEMBLE_EXPLORE_MODES = ("levelset", "hybrid")
# Hybrid boundary sweep: uncertainty-bonus weight and the target-offset sweep (kcal/mol,
# relative to the running best) marched from LO to HI. HI intentionally OVERSHOOTS
# the report window (ENSEMBLE_WINDOW_KCAL): each proposal is constrained-relaxed
# then tight-optimized, which lowers its energy, so to *populate* a 4-6 kcal
# minimum the acquisition must *target* ~6-10 kcal and let relaxation drop it into
# the shell. Capping HI at the window instead starves the high-energy shell
# (empirically worse recall there -- see the cysteine benchmark).
ENSEMBLE_BOUNDARY_KAPPA = 2.0
ENSEMBLE_BOUNDARY_LO_KCAL = 1.5
ENSEMBLE_BOUNDARY_HI_KCAL = ENSEMBLE_EXPLORE_KCAL  # 10.0 (overshoots report window)

# Default methods
DEFAULT_ENERGY_METHOD = "gfn2"
DEFAULT_OPTIMIZER_METHOD = "gfnff"

# multithreading (e.g., Psi4)
NUM_THREADS = 4


@dataclass(slots=True)
class RunOptions:
    """Tuning knobs for :func:`bouquet.solver.run_optimization`.

    Groups the ~two-dozen scalar search/surrogate/benchmark knobs that used to be
    threaded through ``run_optimization`` as individual keyword arguments, so the
    solver's public entry point takes the runtime objects (atoms, calculators,
    ...) plus one ``opts`` object. :class:`Configuration` embeds one as its
    ``run`` field (so these defaults are declared in exactly one place); the CLI
    populates it from argparse and the solver reads it back as ``opts``.
    Direct/library callers construct it directly (all fields default, so
    ``RunOptions()`` reproduces the historical defaults).
    """

    # Prior (PiBO) weighting
    initial_prior_exponent: float = DEFAULT_PRIOR_EXPONENT
    prior_exponent_decay: float = DEFAULT_PRIOR_DECAY

    # Gradient-enhanced surrogate + its cost controls
    use_gradients: bool = False
    gradient_steps: int = 0
    grad_refit_dense_until: int = 20
    grad_refit_every: int = 0
    # High-leverage gradient subset (0 = keep all); keep = recent | best | both.
    gradient_window: int = 0
    gradient_keep: str = "both"

    # Acquisition-optimizer effort (optimize_acqf)
    acq_num_restarts: int = ACQ_NUM_RESTARTS
    acq_raw_samples: int = ACQ_RAW_SAMPLES

    # Value-only-GP lengthscale prior: "auto" | "none" | "dim_scaled"
    lengthscale_prior: str = "auto"

    # Phase 2.5 low-mode / basin-hopping move (None = auto by dihedral count)
    lowmode_prob: Optional[float] = None
    lowmode_warmup: int = 100
    lowmode_kick_deg: float = 60.0
    lowmode_modes: int = 4
    lowmode_kick_dir: str = "pca"

    # Phase 3 category-tied collective move (None = auto; uses torlib SMARTS categories,
    # independent of the fitted priors)
    category_prob: Optional[float] = None
    category_warmup: int = 20
    category_min_moves: int = 6

    # Benchmark-only logging (hidden from the normal CLI happy path)
    cert_log_path: Optional[Path] = None
    cert_betas: tuple = DEFAULT_CERTIFICATE_BETAS
    geom_log_path: Optional[Path] = None

    # Reject evaluations that change the initial covalent bond graph
    retain_bonds: bool = False

    # Ensemble active level-set exploration (only used when Configuration.ensemble
    # is set). ensemble_steps: -1 = auto, 0 = passive harvest only, >0 = hard cap.
    # ensemble_diversity: torsion-space diversity penalty strength (0 = off).
    ensemble_steps: int = DEFAULT_ENSEMBLE_STEPS
    ensemble_diversity: float = DEFAULT_ENSEMBLE_DIVERSITY
    ensemble_explore_mode: str = DEFAULT_ENSEMBLE_EXPLORE_MODE

    def __post_init__(self):
        # Coerce string log paths (direct callers may pass str; the CLI already
        # wraps with Path).
        if isinstance(self.cert_log_path, str):
            self.cert_log_path = Path(self.cert_log_path)
        if isinstance(self.geom_log_path, str):
            self.geom_log_path = Path(self.geom_log_path)

        # Validate the gradient-windowing knobs up front: solver._restrict_gradient_mask
        # silently treats window <= 0 as keep-all and any unknown mode as "both", so an
        # out-of-range value would otherwise change the surrogate without warning.
        if self.gradient_keep not in ("recent", "best", "both"):
            raise ValueError(
                "gradient_keep must be one of 'recent', 'best', 'both', got "
                f"{self.gradient_keep!r}"
            )
        if self.gradient_window < 0:
            raise ValueError(
                "gradient_window must be a non-negative integer (0 = all), got "
                f"{self.gradient_window}"
            )

        # Acquisition-optimizer effort must be positive: 0 or negative would reach
        # optimize_acqf as an opaque failure (or silently skip the search).
        if self.acq_num_restarts < 1 or self.acq_raw_samples < 1:
            raise ValueError(
                "acq_num_restarts and acq_raw_samples must be positive integers; got "
                f"acq_num_restarts={self.acq_num_restarts}, "
                f"acq_raw_samples={self.acq_raw_samples}."
            )

        # Kick-direction source must be recognized: solver._lowmode_move silently
        # falls back to the PCA path for any unknown value, so an out-of-range
        # value would otherwise change the move source without warning.
        if self.lowmode_kick_dir not in ("pca", "enm"):
            raise ValueError(
                "lowmode_kick_dir must be one of 'pca', 'enm', got "
                f"{self.lowmode_kick_dir!r}"
            )

        # Ensemble exploration budget: -1 (auto) and 0 (off) are sentinels;
        # anything below -1 is a typo, not a meaningful step cap.
        if self.ensemble_steps < -1:
            raise ValueError(
                "ensemble_steps must be -1 (auto), 0 (passive harvest), or a "
                f"positive step cap; got {self.ensemble_steps}."
            )
        if self.ensemble_diversity < 0:
            raise ValueError(
                "ensemble_diversity must be non-negative (0 = off), got "
                f"{self.ensemble_diversity}."
            )
        if self.ensemble_explore_mode not in ENSEMBLE_EXPLORE_MODES:
            raise ValueError(
                f"ensemble_explore_mode must be one of {ENSEMBLE_EXPLORE_MODES}, "
                f"got {self.ensemble_explore_mode!r}."
            )

    # ---- auto-default resolution (the "auto"/None sentinels -> concrete values) ----
    # Kept next to the fields and thresholds they read; run_optimization calls these
    # once it knows the dihedral count (and, for categories, the SMARTS partition).

    def resolve_lengthscale_prior(self, n_dihedrals: int) -> str:
        """Resolve the ``"auto"`` value-GP lengthscale prior to a concrete choice.

        ``"auto"`` turns on the dimensionality-scaled prior once the dihedral count
        crosses the high-d threshold (it helps there, and is off at low d); an
        explicit ``"none"`` / ``"dim_scaled"`` passes through unchanged.
        """
        if self.lengthscale_prior != "auto":
            return self.lengthscale_prior
        return "dim_scaled" if n_dihedrals >= HIGH_D_DIHEDRAL_THRESHOLD else "none"

    def resolve_lowmode_prob(self, n_dihedrals: int) -> float:
        """Resolve the ``None`` (auto) low-mode move probability by dihedral count.

        Auto enables the low-mode move at high d; an explicit float passes through.
        """
        if self.lowmode_prob is not None:
            return self.lowmode_prob
        return 0.5 if n_dihedrals >= HIGH_D_DIHEDRAL_THRESHOLD else 0.0

    def resolve_category_prob(
        self, n_dihedrals: int, category_groups: Optional[list] = None
    ) -> float:
        """Resolve the ``None`` (auto) category-tied move probability.

        Auto enables the move only when the molecule is both high-d AND has a real
        repeat to exploit -- the largest tied SMARTS category (``max_spec``) exceeds
        ``CAT_MAXSPEC_THRESHOLD``. On large-but-irregular molecules (high d, low
        max_spec) the tie mildly hurts, so raw dihedral count is not enough. An
        explicit float passes through unchanged.
        """
        if self.category_prob is not None:
            return self.category_prob
        max_spec = (
            max((len(g) for g in category_groups), default=0)
            if category_groups else 0
        )
        return 0.5 if (
            n_dihedrals > CAT_D_THRESHOLD and max_spec > CAT_MAXSPEC_THRESHOLD
        ) else 0.0


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
    seed: int = field(default_factory=lambda: datetime.now().microsecond)

    # Search / surrogate / benchmark tuning knobs handed to run_optimization.
    # These live on a single nested RunOptions so their defaults are declared in
    # exactly one place (RunOptions); the CLI builds one from argparse and the
    # solver reads it back as ``opts``. See RunOptions for per-field docs.
    run: RunOptions = field(default_factory=RunOptions)

    # Prior settings. priors_file / concentration / background build the PiBO
    # prior_module in the CLI; the exponent/decay that weight it at search time
    # live on ``run`` (initial_prior_exponent / prior_exponent_decay).
    priors_file: Optional[Path] = None
    prior_max_concentration: float = DEFAULT_PRIOR_MAX_CONCENTRATION
    prior_background_weight: float = DEFAULT_PRIOR_BACKGROUND_WEIGHT

    # Ensemble selection
    ensemble: bool = False

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
