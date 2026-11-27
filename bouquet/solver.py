"""Methods for solving the conformer option problem"""

from __future__ import annotations

# With annotations evaluated as strings (PEP 563), the heavy numeric stack is only
# touched inside function bodies, so defer it until an optimization actually runs
# (Python 3.15+); see bouquet/__init__.py. The GP/acquisition, collective-move, and
# ensemble machinery now live in the split modules imported below, each of which
# defers its own heavy imports the same way.
__lazy_modules__ = [
    "numpy",
    "torch",
    "ase",
    "ase.calculators.calculator",
]

import itertools
import logging
import time
from dataclasses import dataclass, field, InitVar
from pathlib import Path
from typing import Callable, TYPE_CHECKING

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator

from bouquet.assess import (
    _reset_calculators,
    evaluate_energy,
    evaluate_energy_with_gradient,
    relax_structure,
)
from bouquet.config import (
    ACQ_NUM_RESTARTS,
    GRADIENT_MATRIX_CAP,
    GRADIENT_WINDOW_MIN,
    ACQ_RAW_SAMPLES,
    DEFAULT_CERTIFICATE_BETAS,
    DEFAULT_GC_WARMUP,
    DEFAULT_RELAXATION_STEPS,
    DEFAULT_PRIOR_DECAY,
    DEFAULT_PRIOR_EXPONENT,
    INITIAL_GUESS_STD,
    KCAL_TO_EV,
    RELAX_FAILURE_ENERGY_EV,
    RunOptions,
)
from bouquet.io import (
    append_xyz_frame,
    create_certificate_logger,
    create_structure_logger,
    initialize_structure_log,
    save_structure,
)
from bouquet.clash import build_feasibility
from bouquet.priors import DihedralPriorModule
from bouquet.setup import (
    DihedralInfo, bonds_broken, geometry_bond_set,
)

if TYPE_CHECKING:
    from rdkit import Chem

# The GP/acquisition, collective-move, and ensemble machinery split out of this
# file. Imported here because the loop/orchestrator below call into them (plus a
# minimal set of historical re-exports still imported via ``solver.*`` by tests).
from bouquet.surrogate import (
    _FEASIBLE_MAX_CHUNKS,
    _fit_gradient_gp,
    _periodic_covar_module,
    _periods_deg_array,
    _select_next_points_botorch,
)
from bouquet.solver_moves import (
    _build_category_groups,
    _category_move,
    _evaluate_ring_grafts,
    _low_mode_move,
    _propose_gc_crossover,
    _species_guard,
    build_graph_cuts,
    build_ring_graft_candidates,
)
from bouquet.solver_ensemble import (
    _build_selection_gp,
    _fit_selection_gp_valid,
    _initial_basins,
    _max_posterior_sigma,
    _perform_ensemble_exploration,
    _perform_ensemble_relaxation,
    _select_ensemble_candidates,
    tight_optimize_and_weight,
)

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
    # Observations live in pre-sized buffers grown once per phase (see reserve /
    # append_observation) so accumulating N points is O(N) copy cost, not O(N^2).
    # They are exposed as index-aligned tensors via the observed_coords /
    # observed_energies / observed_gradients properties below; the constructor
    # takes the initial rows through these InitVar parameters (kwarg names differ
    # from the properties to avoid shadowing them).
    initial_coords: InitVar[torch.Tensor]  # Shape: (n_initial, n_dihedrals)
    initial_energies: InitVar[torch.Tensor]  # Shape: (n_initial,)
    # dE/dtheta (eV/rad) per observation, index-aligned with the tensors above;
    # NaN where the gradient is unavailable (failed eval, or use_gradients off).
    # None keeps gradients unrecorded entirely (never appended to in that case).
    initial_gradients: InitVar[torch.Tensor | None] = None  # (n_initial, n_dihedrals)
    # Per-observation Atoms, aligned index-for-index with the observed_* tensors.
    observed_atoms: list[Atoms] = field(default_factory=list)
    device: torch.device = field(default_factory=_get_device)
    init_steps: int = 0
    best_step: int = 0
    # Species guard: the covalent bond set of the initial structure, used as the
    # reference for both directions of connectivity change (None disables both
    # checks). Rejected evaluations get a failure energy so they're never selected.
    #
    # check_bond_forms (default on) rejects geometries that gained a bond -- a
    # relaxation that drove atoms through each other. This is not a rare edge
    # case: in the d=8-12 benchmark it was an absorbing state, one run entering
    # it at step 0 and spending 1159 of 1232 steps there, and one such run was
    # reported as a *success* at -4856 kcal/mol.
    #
    # check_bond_breaks (--retain-bonds, default off) additionally rejects
    # geometries that lost a bond. Left opt-in because dissociation is sometimes
    # legitimate chemistry on these surfaces (zwitterion proton transfer), so
    # forcing it on would change answers for reasons unrelated to the collapse.
    required_bonds: set | None = None
    check_bond_forms: bool = True
    check_bond_breaks: bool = False
    n_bond_breaks: int = 0
    n_bond_forms: int = 0
    n_collective_clash_rejects: int = 0
    # Steric-clash feasibility test restricting the acquisition domain to the
    # clash-free region (None searches the full torsion box). See bouquet.clash.
    clash_feasibility: Callable | None = None
    add_entry: Callable | None = None
    # Optional per-BO-step stopping-rule certificate logger (see
    # _run_optimization_loop / io.create_certificate_logger). When set, each step
    # logs mu_min/lb/alpha_max alongside e_eval/e_best/n_calls/wall_s. cert_betas is
    # the grid of confidence multipliers for the lower-bound term (one lb per beta).
    cert_log: Callable | None = None
    cert_betas: tuple[float, ...] = DEFAULT_CERTIFICATE_BETAS
    # Optional geometry-trail path (stopping-rule benchmark): when set, the geometry
    # at each best-so-far improvement is appended here (plus the final relaxed best),
    # for the offline RMSD-identity / distinct-conformer analysis. See
    # _log_improvement_geometry and io.append_xyz_frame.
    geom_log_path: Path | None = None
    # When True, evaluations also record dE/dtheta and the acquisition GP uses
    # the gradient-enhanced surrogate (see GradientEnhancedPeriodicGP).
    use_gradients: bool = False
    # Number of leading BO steps that use the gradient-enhanced GP; once the loop
    # passes this many steps it switches to the value-only GP (gradients are still
    # recorded, just not fed to the surrogate). The gradient GP's per-step cost
    # grows as O((n*(1+d))^3), so on large/floppy molecules it becomes intractable
    # late in the run; spending the gradient signal early -- where it helps most --
    # and then dropping to the cheap n*n GP keeps the search tractable. <=0 means
    # never switch (gradient GP for the whole run).
    gradient_steps: int = 0
    # Gradient-GP hyperparameter refit schedule (see _run_optimization_loop). Cold
    # full fits for the first `grad_refit_dense_until` BO steps, then the
    # hyperparameters are frozen and later steps only re-condition; `grad_refit_every`
    # > 0 optionally cold-refreshes them every that many post-dense steps. Default
    # (20, 0) = the "gradfreeze" schedule; set grad_refit_dense_until=0 for a full fit
    # every step. `grad_gp_hypers` holds the last cold-fitted hyperparameters (frozen
    # on condition-only steps).
    grad_refit_dense_until: int = 20
    grad_refit_every: int = 0
    grad_gp_hypers: dict | None = None
    # Acquisition-optimizer effort (optimize_acqf); see Configuration.
    acq_num_restarts: int = ACQ_NUM_RESTARTS
    acq_raw_samples: int = ACQ_RAW_SAMPLES
    kernel_arm: str = "product"
    # High-leverage gradient subset: keep gradients for only this many points
    # (0 = all). gradient_keep = recent|best|both. See _restrict_gradient_mask.
    gradient_window: int = 0
    gradient_matrix_cap: int = GRADIENT_MATRIX_CAP
    gradient_window_min: int = GRADIENT_WINDOW_MIN
    gradient_keep: str = "both"
    # Value-only-GP lengthscale prior: "none" (free fit, historical) or
    # "dim_scaled" (Hvarfner dimensionality-scaled LogNormal). See
    # _periodic_covar_module.
    lengthscale_prior: str = "none"

    # Phase 2.5 low-mode / basin-hopping move. With probability lowmode_prob an eligible
    # step (>= lowmode_warmup evaluations) is replaced by a committed kick along a soft
    # mode + UNCONSTRAINED relaxation (see _low_mode_move) -- the move designed for the
    # curved fold valley. lowmode_kick_deg is the per-dihedral RMS kick amplitude;
    # lowmode_modes is how many leading soft modes to draw a kick from. lowmode_rng is the
    # runtime move-type coin / direction RNG, lowmode_count a tally for the end-of-run log.
    lowmode_prob: float = 0.0
    lowmode_warmup: int = 100
    lowmode_kick_deg: float = 60.0
    lowmode_modes: int = 4
    # Kick-direction source: "pca" (data-derived position-PCA; the default and benchmark
    # winner) or "enm" (data-independent elastic-network soft modes; dormant, lost to PCA
    # -- see bouquet.enm). See _low_mode_move.
    lowmode_kick_dir: str = "pca"
    lowmode_rng: np.random.Generator | None = None
    lowmode_count: int = 0

    # Phase 3 category-tied collective move (chemistry-defined REMBO). With probability
    # category_prob an eligible step is replaced by a low-dimensional move over
    # *per-SMARTS-category* dihedral values: every dihedral sharing a prior category is
    # set to one shared value (a chemistry-defined embedding, available from step 0 --
    # unlike the low-mode move's data-derived PCA, which needs an accumulated elite set).
    # The reduced (n_group-dim) point is chosen by a periodic GP + LogEI over a dedicated
    # buffer of past reduced points, then broadcast and constrained + UNCONSTRAINED relaxed
    # (same clash-cleanup-then-release schedule as the low-mode move). See _category_move.
    #   category_groups   -- the tied-index partition (list of index lists), built at setup
    #                        from the prior's univariate assignments; each dihedral is in
    #                        exactly one group (a real category, or its own singleton).
    #   category_warmup   -- gate on the *outer* buffer size (need an incumbent to anchor).
    #   category_min_moves-- how many prior-seeded reduced points to collect before the
    #                        reduced-space GP is fit (below this, sample z from the prior).
    #   _cat_Z / _cat_Y   -- the reduced-space BO buffer (per-category values in degrees,
    #                        relative eV), grown one row per category move.
    category_prob: float = 0.0
    category_warmup: int = 20
    category_min_moves: int = 6
    category_groups: list | None = None
    category_rng: np.random.Generator | None = None
    category_count: int = 0
    _cat_Z: list = field(default_factory=list)
    _cat_Y: list = field(default_factory=list)

    # Graph-aware genetic subtree crossover (Phase 3): every gc_every BO steps
    # (starting at step >= gc_warmup), proposes an MMFF-screened novel clash-free
    # subtree cross of top distinct basins and evaluates it on the same BO surface.
    gc_every: int = 0
    gc_warmup: int = DEFAULT_GC_WARMUP
    gc_cuts: list | None = None
    gc_mol: object | None = None
    gc_geom: object | None = None
    gc_rng: np.random.Generator | None = None
    gc_count: int = 0
    gc_best_count: int = 0

    # PiBO fields
    prior_module: DihedralPriorModule | None = None
    prior_exponent: float = DEFAULT_PRIOR_EXPONENT
    prior_decay: float = DEFAULT_PRIOR_DECAY
    # Exact per-dihedral rotational symmetry orders (m_j in {1, 2, 3, 6});
    # each dihedral j has irreducible period 360 / m_j degrees.
    symmetry_orders: tuple[int, ...] | None = None

    # Observation buffers: pre-sized tensors whose first `_n` rows are the logical
    # observations (exposed via the observed_* properties). init=False -- populated
    # from the InitVar args in __post_init__ and grown by reserve().
    _coords_buf: torch.Tensor = field(init=False, repr=False, default=None)
    _energies_buf: torch.Tensor = field(init=False, repr=False, default=None)
    _grads_buf: torch.Tensor | None = field(init=False, repr=False, default=None)
    _n: int = field(init=False, repr=False, default=0)

    def __post_init__(
        self,
        initial_coords: torch.Tensor,
        initial_energies: torch.Tensor,
        initial_gradients: torch.Tensor | None,
    ) -> None:
        # Adopt the initial rows as the starting buffers; reserve() grows them in
        # place (exact, single allocation per phase) as observations are appended.
        self._n = int(initial_coords.shape[0])
        self._coords_buf = initial_coords.clone()
        self._energies_buf = initial_energies.clone()
        self._grads_buf = (
            None if initial_gradients is None else initial_gradients.clone()
        )

    @property
    def observed_coords(self) -> torch.Tensor:
        """Observed dihedral coordinates, shape (n_observations, n_dihedrals)."""
        return self._coords_buf[: self._n]

    @property
    def observed_energies(self) -> torch.Tensor:
        """Observed relative energies, shape (n_observations,)."""
        return self._energies_buf[: self._n]

    @property
    def observed_gradients(self) -> torch.Tensor | None:
        """Observed dE/dtheta (n_observations, n_dihedrals), or None if unrecorded."""
        return None if self._grads_buf is None else self._grads_buf[: self._n]

    def reserve(self, additional: int) -> None:
        """Ensure room for ``additional`` more observations without reallocating.

        Grows the observation buffers once, to exactly ``_n + additional`` rows (a
        single copy), when they are too small; a no-op otherwise. Call it at the
        start of each append phase (the initial guesses and the BO loop, whose
        sizes are known up front) so the per-append writes never reallocate --
        keeping accumulation O(N) instead of the O(N^2) of a per-append torch.cat.
        """
        need = self._n + additional
        if self._coords_buf.shape[0] >= need:
            return
        d = self._coords_buf.shape[1]

        def _grown(buf: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
            new = buf.new_empty(shape)
            new[: self._n] = buf[: self._n]
            return new

        self._coords_buf = _grown(self._coords_buf, (need, d))
        self._energies_buf = _grown(self._energies_buf, (need,))
        if self._grads_buf is not None:
            self._grads_buf = _grown(self._grads_buf, (need, d))

    def append_observation(
        self,
        coords: np.ndarray,
        energy: float,
        atoms: Atoms,
        gradient: np.ndarray | None = None,
    ) -> None:
        """Append a new observation, keeping observed_atoms index-aligned.

        Writes into the pre-sized buffers (see reserve); it only reallocates if the
        caller under-reserved, so the common path -- a phase-level reserve followed
        by that phase's appends -- does no per-append copy.

        Args:
            coords: Dihedral coordinates as numpy array
            energy: Relative energy value
            atoms: Structure at this observation (copied for retention)
            gradient: Optional dE/dtheta (eV/rad) for each dihedral; stored as
                NaN when not provided so the gradient tensor stays index-aligned.
                Ignored when gradients are unrecorded (``observed_gradients`` None).
        """
        self.reserve(1)  # no-op after a phase-level reserve; safety net otherwise
        i = self._n
        self._coords_buf[i] = torch.as_tensor(
            coords, dtype=torch.float64, device=self.device
        )
        self._energies_buf[i] = energy
        if self._grads_buf is not None:
            if gradient is None:
                self._grads_buf[i] = torch.nan
            else:
                self._grads_buf[i] = torch.as_tensor(
                    np.asarray(gradient, dtype=float),
                    dtype=torch.float64,
                    device=self.device,
                )
        self._n = i + 1
        self.observed_atoms.append(atoms.copy())

    def apply_run_options(
        self,
        opts: RunOptions,
        *,
        lengthscale_prior: str,
        lowmode_prob: float,
        category_prob: float,
        category_groups: list | None,
        seed: int,
    ) -> None:
        """Copy the resolved search/surrogate/move knobs from ``opts`` onto self.

        The three ``auto``-resolved values (``lengthscale_prior``, ``lowmode_prob``,
        ``category_prob``) are passed in already resolved (see the ``RunOptions.
        resolve_*`` methods); every other field is a straight copy. Also seeds the
        two collective-move RNGs when their move is enabled. Prior (PiBO) settings
        are set by the caller since ``prior_module`` is not a ``RunOptions`` field.
        """
        self.gradient_steps = opts.gradient_steps
        self.grad_refit_dense_until = opts.grad_refit_dense_until
        self.grad_refit_every = opts.grad_refit_every
        self.acq_num_restarts = opts.acq_num_restarts
        self.acq_raw_samples = opts.acq_raw_samples
        self.kernel_arm = opts.kernel_arm
        self.gradient_window = opts.gradient_window
        self.gradient_matrix_cap = opts.gradient_matrix_cap
        self.gradient_window_min = opts.gradient_window_min
        self.gradient_keep = opts.gradient_keep
        self.lengthscale_prior = lengthscale_prior

        # Phase 2.5 low-mode move. Its coin/direction RNG is offset from the global
        # seed so it doesn't co-vary with the torch/numpy streams but stays
        # reproducible (paired comparisons across arms at a fixed seed).
        self.lowmode_prob = lowmode_prob
        self.lowmode_warmup = opts.lowmode_warmup
        self.lowmode_kick_deg = opts.lowmode_kick_deg
        self.lowmode_modes = opts.lowmode_modes
        self.lowmode_kick_dir = opts.lowmode_kick_dir
        if lowmode_prob > 0:
            self.lowmode_rng = np.random.default_rng(seed + 99991)

        # Phase 3 category-tied move. Its RNG is a distinct offset stream (move coin
        # + prior-seeded warmup draws). category_groups is the SMARTS partition built
        # by the caller; only stored when the move is actually enabled.
        self.category_prob = category_prob
        self.category_warmup = opts.category_warmup
        self.category_min_moves = opts.category_min_moves
        if category_prob > 0:
            self.category_groups = category_groups
            self.category_rng = np.random.default_rng(seed + 88883)


def _setup_initial_state(
    atoms: Atoms,
    dihedrals: list[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
    relax: bool,
    out_dir: Path | None,
    use_gradients: bool = False,
    structure_log_path: Path | None = None,
) -> OptimizationState:
    """Perform initial relaxation, evaluate starting point, and set up logging.

    Args:
        atoms: Atoms object with the initial geometry
        dihedrals: List of dihedral angles to modify
        calc: Calculator for energy evaluation
        relaxCalc: Calculator used for geometry relaxation
        relax: Whether to relax non-dihedral degrees of freedom
        out_dir: Output path for logging information
        use_gradients: If True, also record dE/dtheta at the starting point so it
            contributes a gradient observation like every other point.

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

    # Evaluate initial point (with dE/dtheta when gradients are enabled, so the
    # start contributes a gradient observation like every other point). When
    # relax=True we already relaxed init_atoms above, so evaluate directly at
    # its geometry without repeating a redundant 10-step relaxation.
    start_coords = np.array([d.get_angle(init_atoms) for d in dihedrals])
    logger.info(f"Initial dihedral angles: {start_coords}")
    if use_gradients:
        start_energy, start_atoms, start_gradient = evaluate_energy_with_gradient(
            start_coords, init_atoms, dihedrals, calc, relaxCalc, relax=False
        )
    else:
        start_energy, start_atoms = evaluate_energy(
            start_coords, init_atoms, dihedrals, calc, relaxCalc, relax=False
        )
        start_gradient = np.full(len(start_coords), np.nan, dtype=float)
    logger.info(f"Computed initial energy: {start_energy}")

    # Set up logging if an output directory (or an explicit structure-log path,
    # used by the kernel benchmarks to collect every trial's trail) is provided.
    add_entry = None
    if out_dir is not None or structure_log_path is not None:
        log_path, ens_path = initialize_structure_log(out_dir, structure_log_path)
        add_entry = create_structure_logger(log_path, ens_path, start_energy)
        add_entry(start_coords, start_atoms, start_energy)

    device = _get_device()
    state = OptimizationState(
        start_atoms=start_atoms,
        start_coords=start_coords,
        start_energy=start_energy,
        init_steps=0,
        initial_coords=torch.tensor(
            np.asarray([start_coords]), dtype=torch.float64, device=device
        ),
        initial_energies=torch.tensor([0.0], dtype=torch.float64, device=device),
        # Start-point gradient (real dE/dtheta when use_gradients, else NaN),
        # index-aligned with the energy/coord tensors.
        initial_gradients=torch.tensor(
            np.asarray([start_gradient]), dtype=torch.float64, device=device
        ),
        use_gradients=use_gradients,
        device=device,
        best_step=0,
        add_entry=add_entry,
        symmetry_orders=tuple(d.symmetry_order for d in dihedrals),
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
    max_points: int | None = None,
    symmetry_orders: tuple[int, ...] | list[int] | None = None,
) -> np.ndarray:
    """Plan initial dihedral guesses from the peaks of a dihedral prior.

    Builds either a systematic grid over the prior's peaks (when the full
    Cartesian product of per-axis modes fits within ``grid_budget``) or a
    weighted random sample from those peaks (when it does not, or when
    ``init_steps`` points are wanted from a large space). Dihedrals with a
    uniform prior carry their starting-geometry angle in the systematic grid
    (the start geometry comes from ETKDG or supplied conformers, so it is
    physically realistic) and are drawn uniformly at random when sampling.

    The systematic grid is ordered **best-first** by descending joint prior
    weight (the product of the per-axis mode weights), so the most probable mode
    combinations are evaluated first. This improves anytime behavior and means
    that truncating to ``max_points`` keeps the most probable conformers.

    Args:
        prior_module: Prior whose peaks seed the guesses (see ``peak_modes``).
        n_dihedrals: Number of dihedral dimensions.
        start_coords: Starting dihedral angles (degrees), used to fill
            uniform-prior dimensions in the systematic grid.
        init_steps: Number of guesses to draw when sampling.
        grid_budget: Maximum systematic grid size before falling back to
            sampling.
        seed: Random seed for the sampling fallback.
        max_points: Optional cap on the number of guesses returned. For the
            systematic grid this truncates to the ``max_points`` highest-weight
            mode combinations (leaving budget for later refinement); for sampling
            it caps the number of draws.

    Returns:
        Array of shape ``(n_points, n_dihedrals)`` in degrees [0, 360).
    """
    axes, uniform_dims = prior_module.peak_modes()
    start_coords = np.asarray(start_coords, dtype=float)
    periods_deg = _periods_deg_array(n_dihedrals, symmetry_orders)
    rng = np.random.default_rng(seed)

    grid_size = 1
    for _dims, candidates in axes:
        grid_size *= len(candidates)

    # Systematic grid over every peak combination (uniform dims keep their
    # start angle, so the only variation there comes from the optimizer).
    if axes and grid_size <= grid_budget:
        candidate_lists = [candidates for _dims, candidates in axes]
        # Evaluate the most probable mode combinations first: the joint weight is
        # the product of each axis's mode weight. Best-first ordering makes the
        # early-budget points the likely conformers and makes truncation principled.
        combos = sorted(
            itertools.product(*candidate_lists),
            key=lambda combo: float(np.prod([w for _v, w in combo])),
            reverse=True,
        )
        points = []
        seen = set()
        for combo in combos:
            pt = start_coords.copy()
            for (dims, _c), (values, _w) in zip(axes, combo):
                for dim, val in zip(dims, values):
                    pt[dim] = val
            pt = pt % periods_deg
            key = tuple(np.round(pt, 3))
            if key in seen:
                continue
            seen.add(key)
            points.append(pt)
            if max_points is not None and len(points) >= max_points:
                break
        return np.array(points)

    # Weighted sampling from the peaks. With no peaked axes at all this degrades
    # to uniform-random guesses, so "peaks" still yields a spread of points.
    axis_weights = [
        np.array([w for _v, w in candidates], dtype=float)
        for _dims, candidates in axes
    ]
    axis_weights = [w / w.sum() for w in axis_weights]

    target = init_steps if max_points is None else min(init_steps, max_points)
    points = []
    seen = set()
    max_attempts = max(20 * target, 100)
    for _ in range(max_attempts):
        if len(points) >= target:
            break
        pt = start_coords.copy()
        for d in uniform_dims:
            pt[d] = rng.uniform(0.0, float(periods_deg[d]))
        for (dims, candidates), weights in zip(axes, axis_weights):
            values, _w = candidates[rng.choice(len(candidates), p=weights)]
            for dim, val in zip(dims, values):
                pt[dim] = val
        pt = pt % periods_deg
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


def _evaluate_point(
    state: OptimizationState,
    guess: np.ndarray,
    dihedrals: list[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
    relax: bool,
) -> tuple[float, Atoms, np.ndarray | None]:
    """Evaluate a dihedral guess, optionally also returning dE/dtheta.

    When ``state.use_gradients`` is set this projects the calculator's forces
    onto the torsion coordinates (eV/rad); otherwise the gradient is ``None``.
    """
    if state.use_gradients:
        energy, cur_atoms, gradient = evaluate_energy_with_gradient(
            guess, state.start_atoms, dihedrals, calc, relaxCalc, relax
        )
    else:
        energy, cur_atoms = evaluate_energy(
            guess, state.start_atoms, dihedrals, calc, relaxCalc, relax
        )
        gradient = None
    # Species guard: a relaxed geometry that changed connectivity is a different
    # species, not a conformer; give it a failure energy so it can never be picked
    # as best (and drop its gradient so it doesn't bias the GP). Without this a
    # collapsed geometry reports a hugely negative energy, becomes the incumbent,
    # and the search never leaves it. The guard also resets the calculators on a
    # rejected or failed evaluation.
    if _species_guard(state, cur_atoms, energy, calc, relaxCalc):
        energy = RELAX_FAILURE_ENERGY_EV
        gradient = None
    return energy, cur_atoms, gradient


# Perturbation widths tried, in order, when resampling clashed initial guesses.
# Starts at the ordinary INITIAL_GUESS_STD so a molecule whose guesses are already
# feasible behaves exactly as before, then narrows. Narrowing is guaranteed to
# succeed eventually because the start geometry is itself feasible -- and it has to
# narrow, because on a crowded molecule the feasible fraction collapses with width:
# measured on m279_d12 (d=12) it is 4.8% at 90 degrees but 72% at 15, and on the
# bulkier m278_d12, 0.2% at 90 versus 72% at 15. Resampling at a fixed 90 degrees
# therefore starves the initial design exactly where the search can least afford it.
_INIT_RESAMPLE_WIDTHS = (INITIAL_GUESS_STD, 60.0, 45.0, 30.0, 20.0, 15.0, 10.0)
# Oversampling factor per round: draw more than the shortfall so a round with a low
# acceptance rate can still fill the quota.
_INIT_RESAMPLE_OVERSAMPLE = 4


def _screen_initial_guesses(
    state: OptimizationState,
    guesses: list[np.ndarray],
    rng_seed: int,
    label: str,
) -> list[np.ndarray]:
    """Replace clashed initial guesses with feasible ones drawn near the start.

    Keeps the requested number of guesses: infeasible ones are resampled rather
    than dropped, so the initial design does not silently shrink. Resampling
    ANNEALS the perturbation width (see ``_INIT_RESAMPLE_WIDTHS``) instead of
    retrying at a fixed one -- a narrow feasible guess is worth far more than no
    guess, and infinitely more than a clashed one, which would be rejected anyway
    and can leave the calculator in a broken state.

    The trade is diversity: a guess drawn at 15 degrees explores less than one at
    90. That is the right trade only because the alternative on these molecules is
    not a wider guess, it is no guess at all.
    """
    feasible = state.clash_feasibility
    guess_array = np.asarray(guesses, dtype=float)
    keep = feasible(guess_array)
    n_wanted = len(guess_array)
    kept = list(guess_array[keep])

    rng = np.random.default_rng(rng_seed + 1)
    widths_used = []
    for width in _INIT_RESAMPLE_WIDTHS:
        if len(kept) >= n_wanted:
            break
        shortfall = n_wanted - len(kept)
        draw = rng.normal(
            state.start_coords, width,
            size=(shortfall * _INIT_RESAMPLE_OVERSAMPLE, guess_array.shape[1]),
        ) % 360.0
        accepted = draw[feasible(draw)]
        if len(accepted):
            kept.extend(accepted[:shortfall])
            widths_used.append(width)

    n_rejected = n_wanted - int(keep.sum())
    kept = kept[:n_wanted]
    if n_rejected:
        detail = (f" (resampled at {'/'.join(f'{w:g}' for w in widths_used)} deg)"
                  if widths_used else "")
        logger.info(
            f"Clash constraint: {n_rejected} of {n_wanted} initial guesses "
            f"({label}) were sterically infeasible; {len(kept) - int(keep.sum())} "
            f"resampled{detail}, {len(kept)} usable."
        )
    if len(kept) < n_wanted:
        # Only screened points are returned: a rejected guess is never sent to the
        # calculator (it could poison the SCF restart). With none left, the search
        # simply continues from the start observation already in the state.
        logger.warning(
            f"Clash constraint: only {len(kept)} of {n_wanted} initial guesses "
            f"({label}) are feasible"
            + ("; continuing from the start structure alone." if not kept else ".")
        )
    return kept


def _evaluate_initial_guesses(
    state: OptimizationState,
    dihedrals: list[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
    relax: bool,
    init_steps: int,
    seed: int,
    initial_conformers: list[Atoms] | None,
    initial_dihedrals: np.ndarray | None = None,
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
    # Build the list of guess angles (degrees) once, per the precedence above;
    # only the source and a log label differ, so a single evaluation loop follows.
    # Conformer angles are used as reported by get_angle; peak/random guesses are
    # wrapped to [0, 360) so standardization later sees a consistent range.
    periods_deg = _periods_deg_array(len(dihedrals), state.symmetry_orders)
    if initial_conformers is not None:
        guesses = [
            np.array([d.get_angle(conformer) for d in dihedrals])
            for conformer in initial_conformers
        ]
        label = "provided conformer"
    elif initial_dihedrals is not None:
        guesses = [np.asarray(g, dtype=float) % periods_deg for g in initial_dihedrals]
        label = "prior-peak guess"
    else:
        rng = np.random.default_rng(seed)
        guess_std = np.minimum(INITIAL_GUESS_STD, periods_deg / 2.0)
        guesses = list(
            rng.normal(
                state.start_coords, guess_std, size=(init_steps, len(dihedrals))
            )
            % periods_deg
        )
        label = "random initial guess"

    # Screen the initial guesses through the same feasibility region as the BO
    # candidates. This is not a nicety: the random init design is drawn blind, so
    # it is the most likely source of a badly clashed geometry, and evaluating one
    # can leave the calculator in a broken state that corrupts the whole run (see
    # _reset_calculators). Conformer guesses are screened too: only their TORSIONS
    # transfer -- _evaluate_point rebuilds every guess on state.start_atoms -- so a
    # vetted conformer can still clash once its angles land on the start geometry.
    # User-provided seeds are never replaced by random draws; a clashed one is
    # dropped.
    if state.clash_feasibility is not None and guesses:
        if initial_conformers is not None:
            keep = np.asarray(state.clash_feasibility(np.asarray(guesses)), dtype=bool)
            if not keep.all():
                logger.warning(
                    f"Clash constraint: dropped {int((~keep).sum())} of {len(guesses)} "
                    f"provided conformers whose torsions clash on the start structure."
                )
            guesses = [g for g, ok in zip(guesses, keep) if ok]
        else:
            guesses = _screen_initial_guesses(state, guesses, rng_seed=seed, label=label)

    state.init_steps = len(guesses)
    state.reserve(len(guesses))  # size the buffers once for this phase's appends
    logger.info(f"Evaluating {state.init_steps} initial guesses ({label}s)")
    for i, guess in enumerate(guesses):
        energy, cur_atoms, gradient = _evaluate_point(
            state, guess, dihedrals, calc, relaxCalc, relax
        )
        rel_energy = energy - state.start_energy
        logger.info(
            f"Evaluated {label} {i+1: >3}/{state.init_steps}. "
            f"Energy-E0: {rel_energy:12.6f}"
        )
        state.append_observation(guess, rel_energy, cur_atoms, gradient)
        if state.add_entry is not None:
            state.add_entry(guess, cur_atoms, energy)


def _grad_gp_refit_decision(
    step: int,
    step_uses_gradients: bool,
    schedule_active: bool,
    dense_until: int,
    refit_every: int,
    grad_gp_hypers: dict | None,
) -> tuple[dict | None, dict | None]:
    """Decide the gradient-GP fit mode for one BO step.

    Returns ``(frozen_hypers, hyper_out)`` for ``_select_next_points_botorch``:
    - ``frozen_hypers`` is None on a cold fit (optimize fresh hyperparameters) or
      the stored hyperparameters on a condition-only update (reuse them, no Adam).
    - ``hyper_out`` is a fresh dict to capture the fitted hyperparameters, but only
      when a cold fit runs under an active schedule (so a later step can freeze on
      them); otherwise None.

    A cold fit happens when there is no active schedule, no frozen hyperparameters
    yet, we are still in the dense phase (``step < dense_until``), or a periodic
    refresh is due. See the schedule notes in ``_run_optimization_loop`` and
    ``_fit_gradient_gp``.
    """
    cold_fit = True
    if step_uses_gradients and schedule_active and grad_gp_hypers is not None:
        on_refit = refit_every > 0 and (step - dense_until) % refit_every == 0
        cold_fit = step < dense_until or on_refit
    frozen_hypers = None if cold_fit else grad_gp_hypers
    hyper_out = {} if step_uses_gradients and cold_fit and schedule_active else None
    return frozen_hypers, hyper_out


def _log_improvement_geometry(
    state: OptimizationState,
    atoms: Atoms,
    kind: str,
    n_calls: int | None = None,
    e_e0_eV: float | None = None,
) -> None:
    """Append one geometry frame to the benchmark geometry trail (no-op if off).

    ``kind`` is ``init_best`` (best after the initial guesses), ``improvement``
    (a new best-so-far during the BO loop), or ``final`` (the unconstrained-relaxed
    best). ``n_calls`` aligns the frame with the certificate CSV (n_calls there is
    the cumulative evaluation count); it defaults to the current observation count.
    Energies are relative eV (the e_e0 convention).
    """
    if state.geom_log_path is None:
        return
    if n_calls is None:
        n_calls = len(state.observed_energies)
    if e_e0_eV is None:
        e_e0_eV = state.observed_energies.min().item()
    comment = f"n_calls={n_calls} e_e0_eV={e_e0_eV:.6f} kind={kind}"
    append_xyz_frame(state.geom_log_path, atoms, comment)


def _run_optimization_loop(
    state: OptimizationState,
    n_steps: int,
    dihedrals: list[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
    relax: bool,
    out_dir: Path | None,
    seed: int = 0,
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
    # Optional grad->value handoff: use the gradient-enhanced GP only for the first
    # `gradient_steps` BO steps, then the cheap value-only GP (see OptimizationState).
    switch_at = state.gradient_steps if state.gradient_steps > 0 else None

    # Gradient-GP hyperparameter refit schedule. A full fit runs ~200 Choleskys and
    # its cost grows steeply with the observation count, so fitting every step
    # dominates the run late on. Instead: full (cold) fits for the first
    # `dense_until` BO steps -- hyperparameters move most early and the matrices are
    # still small -- then FREEZE them and only re-condition, which folds each new
    # point in under the frozen hyperparameters for one Cholesky instead of ~200
    # (a cold fit vs. a condition-only update, selected per step below via
    # `gp_frozen_hypers`). `refit_every > 0` optionally refreshes the frozen
    # hyperparameters with a *cold* refit every that many post-dense steps (cold,
    # not warm-started: warm-starting the fit drifts the hyperparameters and
    # degrades the search). The shipped default freezes after `dense_until` steps;
    # the opt-out `dense_until=0` (with `refit_every<=1`) refits every step -- the
    # slow full-gradient reference. See _fit_gradient_gp.
    dense_until = max(0, state.grad_refit_dense_until)
    refit_every = state.grad_refit_every
    schedule_active = dense_until > 0 or refit_every > 1

    # Geometry trail (benchmark): snapshot the best after the initial guesses, so a
    # global minimum already found in the init phase is captured before any BO step.
    if state.geom_log_path is not None and len(state.observed_energies):
        best_idx = state.observed_energies.argmin().item()
        _log_improvement_geometry(
            state, state.observed_atoms[best_idx], "init_best",
            n_calls=best_idx + 1, e_e0_eV=state.observed_energies[best_idx].item(),
        )

    # The loop appends exactly one observation per step, so size the buffers once.
    state.reserve(n_steps)

    loop_start = time.perf_counter()
    for step in range(n_steps):
        step_uses_gradients = state.use_gradients and (
            switch_at is None or step < switch_at
        )
        if switch_at is not None and state.use_gradients and step == switch_at:
            logger.info(
                f"Switching to value-only GP after {switch_at} gradient-enhanced "
                f"step(s) (gradient GP cost grows with observation count)."
            )

        # Per-step gradient-GP cost: a cold full fit vs. a condition-only update that
        # reuses the frozen hyperparameters and runs no Adam (see the schedule notes
        # above and _grad_gp_refit_decision).
        frozen_hypers, hyper_out = _grad_gp_refit_decision(
            step, step_uses_gradients, schedule_active, dense_until, refit_every,
            state.grad_gp_hypers,
        )

        # Collective moves: with some probability (past a warmup) this step is replaced
        # by a committed move + UNCONSTRAINED relax that combines selection and
        # evaluation, so it short-circuits the standard step. Two flavors, checked in
        # order and mutually exclusive per step:
        #   Phase 3  category move -- tie same-SMARTS-category dihedrals (chemistry-defined
        #             embedding, available from step 0); takes precedence when enabled.
        #   Phase 2.5 low-mode move -- kick along a data-derived PCA soft mode.
        collective_result = None
        used_gc = False
        t_collective = 0.0
        if (
            state.category_prob > 0
            and len(state.observed_energies) >= state.category_warmup
            and state.category_rng is not None
            and state.category_rng.random() < state.category_prob
        ):
            _t0 = time.perf_counter()
            collective_result = _category_move(
                state, dihedrals, calc, relaxCalc, state.category_rng
            )
            t_collective += time.perf_counter() - _t0
            if collective_result is not None:
                state.category_count += 1

        if (
            collective_result is None
            and state.lowmode_prob > 0
            and len(state.observed_energies) >= state.lowmode_warmup
            and state.lowmode_rng is not None
            and state.lowmode_rng.random() < state.lowmode_prob
        ):
            _t0 = time.perf_counter()
            collective_result = _low_mode_move(
                state, dihedrals, calc, relaxCalc, state.lowmode_rng
            )
            t_collective += time.perf_counter() - _t0
            if collective_result is not None:
                state.lowmode_count += 1

        if collective_result is not None:
            next_coords, cur_atoms, energy = collective_result
            gradient = None
            t_select, t_eval = 0.0, 0.0
            # Collective (low-mode / category) step: no GP fit or acquisition ran, so
            # there is no posterior to certify (mu_min/alpha_max/lb stay blank). The
            # whole cost is the committed kick + UNCONSTRAINED relax (GFN2); record it
            # in its own t_collective bucket so the row is still logged. Previously
            # cert_out was None here, which dropped the row and hid this time entirely.
            cert_out = ({"t_select": 0.0, "t_eval": 0.0, "t_collective": t_collective}
                        if state.cert_log is not None else None)
        else:
            # ---- standard BO step (or scheduled graph-subtree crossover) ----
            next_coords = None
            cert_out = {} if state.cert_log is not None else None
            _t0 = time.perf_counter()
            if (
                state.gc_every > 0
                and step >= state.gc_warmup
                and (step % state.gc_every == 0)
            ):
                next_coords = _propose_gc_crossover(state, dihedrals)
                if next_coords is not None:
                    # No GP fit ran on a GC step, so cert_out keeps no posterior
                    # fields; the timings are filled in below as for a BO step.
                    used_gc = True
                    state.gc_count += 1
            if next_coords is None:
                # Time the two cost centers separately: GP fit/condition + acquisition
                # optimization (t_select) vs the xTB energy evaluation + relaxation (t_eval).
                next_coords = _select_next_points_botorch(
                    state.observed_coords, state.observed_energies,
                    prior_module=state.prior_module,
                    prior_exponent=state.prior_exponent,
                    observed_gradients=state.observed_gradients,
                    use_gradients=step_uses_gradients,
                    gp_frozen_hypers=frozen_hypers,
                    gp_hyper_out=hyper_out,
                    cert_out=cert_out,
                    cert_betas=state.cert_betas,
                    acq_num_restarts=state.acq_num_restarts,
                    acq_raw_samples=state.acq_raw_samples,
                    gradient_window=state.gradient_window,
                    gradient_matrix_cap=state.gradient_matrix_cap,
                    gradient_window_min=state.gradient_window_min,
                    gradient_keep=state.gradient_keep,
                    lengthscale_prior=state.lengthscale_prior,
                    feasible=state.clash_feasibility,
                    kernel_arm=state.kernel_arm,
                    feasible_seed_offset=seed + step * _FEASIBLE_MAX_CHUNKS,
                    symmetry_orders=state.symmetry_orders,
                )
                if hyper_out:  # a cold fit; carry its hyperparameters forward to freeze
                    state.grad_gp_hypers = hyper_out["hypers"]
            t_select = time.perf_counter() - _t0

            _t0 = time.perf_counter()
            energy, cur_atoms, gradient = _evaluate_point(
                state, next_coords, dihedrals, calc, relaxCalc, relax
            )
            t_eval = time.perf_counter() - _t0
            if cert_out is not None:
                cert_out["t_select"] = t_select  # GP + acquisition (or GC proposal)
                cert_out["t_eval"] = t_eval      # xTB energy + relaxation
                # Any collective attempt(s) this step returned None (failed), so we fell
                # through to the standard BO step. Their attempt time still elapsed, so
                # record it (0.0 when no collective move was tried) to reconcile wall_s.
                cert_out["t_collective"] = t_collective
        rel_energy = energy - state.start_energy
        logger.info(
            f"Evaluated energy in step {step+1: >3}/{n_steps}. Energy-E0: {rel_energy:12.6f}"
        )

        if rel_energy < state.observed_energies.min().item():
            state.best_step = step
            if used_gc:
                state.gc_best_count += 1
            if out_dir is not None:
                save_structure(out_dir, cur_atoms, "current_best.xyz")
            # Geometry trail: this BO point is a new best-so-far. n_calls matches the
            # certificate row about to be logged (post-append observation count).
            _log_improvement_geometry(
                state, cur_atoms, "improvement",
                n_calls=len(state.observed_energies) + 1, e_e0_eV=rel_energy,
            )

        if state.add_entry is not None:
            state.add_entry(next_coords, cur_atoms, energy)

        state.append_observation(next_coords, rel_energy, cur_atoms, gradient)

        # Per-step stopping-rule certificate row. cert_out was filled during
        # selection (GP fit on data through step-1, predicting this step); pair it
        # with this step's realized outcome. n_calls counts cumulative energy
        # evaluations (start + initial guesses + BO steps so far) -- the real cost
        # axis -- which equals the post-append observation count.
        if state.cert_log is not None and cert_out is not None:
            state.cert_log(
                step,
                rel_energy,
                state.observed_energies.min().item(),
                len(state.observed_energies),
                time.perf_counter() - loop_start,
                cert_out,
            )

        # Decay prior exponent
        if state.prior_module is not None:
            state.prior_exponent *= state.prior_decay

    if state.lowmode_prob > 0:
        logger.info(
            f"Phase 2.5: {state.lowmode_count} low-mode (kick + unconstrained-relax) "
            f"move(s) of {n_steps} BO steps."
        )
    if state.category_prob > 0:
        n_groups = len(state.category_groups) if state.category_groups else 0
        logger.info(
            f"Phase 3: {state.category_count} category-tied move(s) of {n_steps} BO "
            f"steps over {n_groups} categor(y/ies) for {len(dihedrals)} dihedrals."
        )
    if state.gc_every > 0:
        logger.info(
            f"Graph-subtree crossover: {state.gc_count} move(s) of {n_steps} BO "
            f"steps ({state.gc_best_count} set new best)."
        )


def _perform_final_relaxation(
    state: OptimizationState,
    dihedrals: list[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
) -> tuple[Atoms, float]:
    """Perform final relaxation steps and return the best atoms and its e_e0.

    Performs two relaxations: first with dihedral constraints, then without.

    Args:
        state: Optimization state with best structure
        dihedrals: List of dihedral angles to modify
        calc: Calculator for energy evaluation
        relaxCalc: Calculator used for geometry relaxation

    Returns:
        (Final optimized Atoms structure, its energy relative to state.start_energy)
    """
    best_idx = state.observed_energies.argmin().item()
    best_coords = state.observed_coords[best_idx].cpu().numpy()

    logger.info(f"Best energy found on step {state.best_step + 1}")

    # Report when the search first entered the best basin (within 10 kcal/mol of the
    # best energy) and first reached it (within 1 kcal/mol). Step numbers are offset
    # by init_steps so they count BO steps rather than initial guesses.
    delta = (state.observed_energies - state.observed_energies[best_idx]).abs()
    near = torch.nonzero(delta < KCAL_TO_EV * 10.0, as_tuple=False)
    good = torch.nonzero(delta < KCAL_TO_EV, as_tuple=False)
    if near.numel():
        logger.info(f"Found low energy on step {near[0].item() - state.init_steps}")
    if good.numel():
        logger.info(
            f"Found first good energy on step {good[0].item() - state.init_steps}"
        )

    # Seed from the actual best observation (aligned with best_idx/best_coords),
    # which may come from the initial point, a seeded conformer, a random guess,
    # or the BO loop -- best_step only tracks BO-loop wins, so don't key off it here.
    observed_atoms = state.observed_atoms[best_idx].copy()
    # observed_energies stores E-E0 (relative); convert to absolute energy so
    # comparisons and E/E-E0 bookkeeping stay consistent.
    observed_energy = state.observed_energies[best_idx].item() + state.start_energy
    observed_bonds = geometry_bond_set(observed_atoms)

    constrained_energy, constrained_atoms = evaluate_energy(
        best_coords, observed_atoms, dihedrals, calc, relaxCalc, steps=None
    )
    # Guard the constrained tight relaxation against calculator failures,
    # connectivity changes, or energy regressions (e.g. FixInternals + BFGS
    # stepping onto a higher-energy surface while still returning a finite energy
    # below RELAX_FAILURE_ENERGY_EV). Reverting BEFORE releasing constraints lets
    # the unconstrained relaxation start from the clean best observed geometry.
    if not np.isfinite(constrained_energy) or constrained_energy >= RELAX_FAILURE_ENERGY_EV:
        revert_reason = (
            "Final constrained relaxation failed (calculator error, e.g. SCF "
            "non-convergence); seeding unconstrained relaxation from best observed."
        )
    elif constrained_energy > observed_energy + 1e-6:
        revert_reason = (
            f"Final constrained relaxation regressed in energy "
            f"({constrained_energy - observed_energy:+.4f} eV vs best observed); "
            f"reverting to best observed geometry."
        )
    elif geometry_bond_set(constrained_atoms) != observed_bonds:
        revert_reason = (
            "Final constrained relaxation changed connectivity vs best observed; "
            "reverting to best observed geometry."
        )
    else:
        revert_reason = None
    if revert_reason is not None:
        logger.warning(revert_reason)
        constrained_atoms = observed_atoms.copy()
        constrained_energy = observed_energy

    # (E-E0 is logged once at the end, after any failed-relax fallback, so the value
    # the sweep log-parser reads is always the trustworthy final energy.)
    if state.add_entry is not None:
        state.add_entry(best_coords, constrained_atoms, constrained_energy)

    # Relaxation without dihedral constraints
    best_atoms = constrained_atoms.copy()
    best_atoms.set_constraint()
    # Relax the released geometry directly. Going through evaluate_energy would
    # reapply FixInternals (relax=True re-pins the dihedrals via
    # set_dihedral/FixInternals), leaving final.xyz still constrained and making
    # the --retain-bonds release check test a pinned geometry rather than a real
    # released one.
    best_energy, best_atoms = relax_structure(best_atoms, calc, relaxCalc, None)
    # If the final tight relaxation walked into a geometry the calculator cannot
    # evaluate (SCF non-convergence -> sentinel failure energy) or regressed to a
    # higher finite energy than the geometry it started from, revert to the
    # lower-energy constrained/observed geometry.
    if not np.isfinite(best_energy) or best_energy >= RELAX_FAILURE_ENERGY_EV:
        logger.warning(
            "Final unconstrained relaxation failed (calculator error, e.g. SCF "
            "non-convergence); reverting to the constrained/observed best."
        )
        best_atoms, best_energy = constrained_atoms, constrained_energy
    elif best_energy > constrained_energy + 1e-6:
        logger.warning(
            f"Final unconstrained relaxation regressed in energy "
            f"({best_energy - constrained_energy:+.4f} eV vs constrained/observed best); "
            f"reverting to the lower-energy geometry."
        )
        best_atoms, best_energy = constrained_atoms, constrained_energy

    # Species guard (always on): if releasing the dihedral constraints let the
    # geometry FORM or BREAK any bond relative to the constrained final geometry
    # (same fold, trusted species), the released structure is a different chemical
    # species -- proton transfer, ring closure, H2 collapse -- which xTB can place
    # hundreds of kcal below the true conformers, so it would be recorded as a bogus
    # global minimum (and poison any reference anchored on this trail). Revert to the
    # constrained best. This is broader than the --retain-bonds check below (that only
    # catches BROKEN bonds, vs the START structure, and only when requested); a
    # formed-bond species is never a valid conformer answer, so this is unconditional.
    if (
        best_energy < RELAX_FAILURE_ENERGY_EV
        and constrained_energy < RELAX_FAILURE_ENERGY_EV
        and geometry_bond_set(best_atoms) != geometry_bond_set(constrained_atoms)
    ):
        logger.warning(
            "Final unconstrained relaxation changed connectivity (formed/broke a "
            "bond) vs the constrained geometry; reverting to the constrained best."
        )
        best_atoms, best_energy = constrained_atoms, constrained_energy

    # --retain-bonds: if releasing the dihedral constraints let the geometry
    # rearrange, keep the constrained (bond-preserving) result instead (only if
    # the constrained relaxation actually succeeded).
    if (
        state.check_bond_breaks
        and state.required_bonds is not None
        and constrained_energy < RELAX_FAILURE_ENERGY_EV
        and bonds_broken(best_atoms, state.required_bonds)
    ):
        logger.warning(
            "Final unconstrained relaxation broke a bond; "
            "reverting to the constrained best (--retain-bonds)."
        )
        best_atoms = constrained_atoms
        best_energy = constrained_energy

    # Log the FINAL energies here -- after any failed-relax fallback above -- so the
    # values the sweep log-parser scrapes are always trustworthy and never the sentinel
    # failure energy. If the constrained final relax itself failed, report the reverted
    # best in its place rather than the sentinel.
    report_constrained = (
        constrained_energy if constrained_energy < RELAX_FAILURE_ENERGY_EV else best_energy
    )
    logger.info(
        f"Performed final relaxation with dihedral constraints. "
        f"E: {report_constrained}. E-E0: {report_constrained - state.start_energy}"
    )
    logger.info(
        f"Performed final relaxation without dihedral constraints. "
        f"E: {best_energy}. E-E0: {best_energy - state.start_energy}"
    )

    best_coords = np.array([d.get_angle(best_atoms) for d in dihedrals])
    if state.add_entry is not None:
        state.add_entry(best_coords, best_atoms, best_energy)

    return best_atoms, best_energy - state.start_energy


def run_optimization(
    atoms: Atoms,
    dihedrals: list[DihedralInfo],
    n_steps: int,
    calc: Calculator,
    relaxCalc: Calculator,
    init_steps: int,
    out_dir: Path | None,
    relax: bool = True,
    seed: int = 0,
    initial_conformers: list[Atoms] | None = None,
    initial_dihedrals: np.ndarray | None = None,
    prior_module: DihedralPriorModule | None = None,
    category_assignments: dict | None = None,
    return_ensemble: bool = False,
    ensemble_seed_geometries: list[Atoms] | None = None,
    opts: RunOptions | None = None,
    *,
    mol: Chem.Mol | None = None,
) -> Atoms | tuple[Atoms, list[tuple[Atoms, float, float]]]:
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
        prior_module: Optional PiBO dihedral prior guiding acquisition. Independent
                           of the category-tied collective move (see
                           ``category_assignments``).
        category_assignments: Optional ``{dihedral_index: torlib category id}`` map
                           (from :func:`bouquet.priors.assign_categories`) that ties
                           chemically-equivalent rotors for the category-tied
                           collective move. Decoupled from ``prior_module`` so category
                           moves work with PiBO steering off. When ``None``, falls back
                           to ``prior_module``'s univariate assignments (legacy callers).
        return_ensemble: If True, also select and tightly optimize a Boltzmann
            ensemble of low-energy conformers and return it alongside the best
            structure.
        opts: Search/surrogate/benchmark tuning knobs (see :class:`RunOptions`).
            ``None`` uses the defaults. The gradient-enhanced surrogate
            (``opts.use_gradients``) needs ``calc`` and ``relaxCalc`` to be the
            same surface when ``relax=True`` (the envelope theorem makes the
            projected torsion gradient equal ``dE/dtheta`` only at a constrained
            minimum of the *energy* calculator); pass matching calculators then.

    Returns:
        If ``return_ensemble`` is False, the optimized lowest-energy geometry as
        an Atoms object. If True, a tuple ``(best_atoms, ensemble)`` where
        ``ensemble`` is a list of ``(atoms, relative_energy_eV, weight)`` sorted
        by energy ascending (``best_atoms`` is the lowest-energy ensemble member,
        or the single-best relaxation if the ensemble is empty).
    """
    # RunOptions.__post_init__ has already validated the gradient-windowing knobs.
    opts = opts if opts is not None else RunOptions()

    n_dihedrals = len(dihedrals)

    # No-rotor short-circuit: with zero dihedrals the BO machinery has nothing to
    # search (SobolEngine requires dim >= 1 and the GP has no features), so relax
    # once, write the same outputs a normal run would, and return.
    if n_dihedrals == 0:
        logger.info("No rotatable dihedrals detected; relaxing once and returning.")
        if relax:
            energy, best_atoms = relax_structure(atoms, calc, relaxCalc, None)
        else:
            best_atoms = atoms.copy()
            best_atoms.calc = calc
            try:
                energy = calc.get_potential_energy(best_atoms)
            except Exception:
                energy = RELAX_FAILURE_ENERGY_EV
        logger.info(f"Energy (no rotatable dihedrals): {energy}")
        # Ring-pucker seeds are the ONLY conformational diversity a zero-rotor
        # molecule has: the BO loop turns rotors and there are none, so the seed
        # full geometries (distinct ring puckers from ETKDG / --conformer-file) are
        # the sole source of an ensemble. Tight-optimize/dedup/weight them just as
        # the normal path does; with no seeds we keep the singleton behavior below.
        ensemble = None
        if return_ensemble and ensemble_seed_geometries:
            ensemble = tight_optimize_and_weight(
                [best_atoms] + list(ensemble_seed_geometries),
                calc, relaxCalc, start_energy=energy,
            )
            if ensemble:
                # ensemble[0] is the lowest-energy pucker; adopt it as the best.
                best_atoms, best_e_e0, _weight = ensemble[0]
                energy = energy + best_e_e0  # absolute energy of the new best
        # Produce the same artifacts a normal run would. There is a single
        # (zero-dihedral) structure, so it is both the start and the best: mirror
        # _setup_initial_state's relaxed.xyz + structure log and the final
        # current_best.xyz / geometry-trail frame that _perform_final_relaxation's
        # caller emits.
        coords = np.zeros(n_dihedrals, dtype=float)
        if out_dir is not None or opts.structure_log_path is not None:
            log_path, ens_path = initialize_structure_log(out_dir, opts.structure_log_path)
            add_entry = create_structure_logger(log_path, ens_path, energy)
            add_entry(coords, best_atoms, energy)
        if out_dir is not None:
            if relax:
                save_structure(out_dir, best_atoms, "relaxed.xyz")
            save_structure(out_dir, best_atoms, "current_best.xyz")
        # Benchmark geometry trail (no-op unless --geom-log is set): one 'final'
        # frame, matching _log_improvement_geometry(state, best_atoms, "final").
        if opts.geom_log_path is not None:
            geom_log_path = Path(opts.geom_log_path)
            geom_log_path.parent.mkdir(parents=True, exist_ok=True)
            geom_log_path.open("w").close()
            append_xyz_frame(
                geom_log_path, best_atoms, "n_calls=1 e_e0_eV=0.000000 kind=final"
            )
        if return_ensemble:
            # Singleton fallback when no seeds were supplied (or all seeds failed).
            return best_atoms, ensemble if ensemble else [(best_atoms, 0.0, 1.0)]
        return best_atoms

    # Resolve the "auto"/None search knobs now that the dihedral count is known
    # (the detailed high-d rules live on the RunOptions.resolve_* methods). The
    # category move also needs the SMARTS-category partition, built once here and
    # reused for both its auto-enable decision and the move itself. Categories come
    # from torlib SMARTS (category_assignments), independent of the fitted priors;
    # legacy callers that pass only a prior_module still work via its assignments.
    lengthscale_prior = opts.resolve_lengthscale_prior(n_dihedrals)
    lowmode_prob = opts.resolve_lowmode_prob(n_dihedrals, relax=relax)
    if category_assignments is None and prior_module is not None:
        category_assignments = getattr(prior_module, "univariate_assignments", None)
    category_groups = (
        _build_category_groups(category_assignments, n_dihedrals)
        if category_assignments is not None else None
    )
    category_prob = opts.resolve_category_prob(
        n_dihedrals, category_groups, relax=relax
    )

    # A low-mode move is a kick followed by a constrained + UNCONSTRAINED relaxation
    # (see _low_mode_move); it is meaningless without relaxation and would silently
    # relax structures in a run that asked not to. Reject the combination up front.
    if lowmode_prob > 0 and not relax:
        raise ValueError(
            "lowmode_prob > 0 requires relax=True: low-mode moves are kick + relax steps"
        )

    # A category-tied move broadcasts a per-category value then relaxes (see
    # _category_move); like the low-mode move it is meaningless without relaxation, and
    # it needs SMARTS category assignments to know which dihedrals to tie.
    if category_prob > 0:
        if not relax:
            raise ValueError(
                "category_prob > 0 requires relax=True: category moves are broadcast + relax steps"
            )
        if not category_assignments:
            raise ValueError(
                "category_prob > 0 requires category_assignments: the torlib SMARTS "
                "categories define which dihedrals are tied. Build them with "
                "bouquet.priors.assign_categories (or pass a prior_module)."
            )

    # Seed every RNG the run touches from `seed`, so a run is reproducible and the
    # seed actually controls the search. The acquisition optimizer (optimize_acqf)
    # draws its restart points from torch's global RNG at every BO step; without
    # this that dominant stochasticity is uncontrolled (and config.seed would only
    # affect the numpy-drawn initial guesses). Seeding torch also enables
    # common-random-number paired comparisons across arms at a fixed seed.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup initial state (relaxation, starting point, logging)
    state = _setup_initial_state(
        atoms, dihedrals, calc, relaxCalc, relax, out_dir,
        use_gradients=opts.use_gradients,
        structure_log_path=opts.structure_log_path,
    )
    # Wire the resolved search/surrogate/move knobs onto the state and seed the
    # collective-move RNGs (see OptimizationState.apply_run_options).
    state.apply_run_options(
        opts,
        lengthscale_prior=lengthscale_prior,
        lowmode_prob=lowmode_prob,
        category_prob=category_prob,
        category_groups=category_groups,
        seed=seed,
    )
    if category_prob > 0:
        logger.info(
            f"Phase 3 category move enabled: {len(state.category_groups)} categor(y/ies) "
            f"for {len(dihedrals)} dihedrals (embedding dim "
            f"{len(state.category_groups)})."
        )

    # Species guard: adopt the (relaxed) start structure's covalent bond set as the
    # reference every later evaluation is compared against. The start is computed
    # inside _setup_initial_state above and defines the molecule's connectivity.
    # The bond-formation half runs by default; --retain-bonds adds the breaking half.
    state.check_bond_forms = opts.species_guard
    state.check_bond_breaks = opts.retain_bonds
    if state.check_bond_forms or state.check_bond_breaks:
        state.required_bonds = geometry_bond_set(state.start_atoms)

    # Steric-clash feasibility region for the acquisition optimizer. Needs the
    # RDKit mol for exact topological distances and polar-H perception; callers
    # that don't supply one simply search the unconstrained box.
    if opts.clash_constraint and mol is not None:
        state.clash_feasibility = build_feasibility(
            state.start_atoms, mol, dihedrals,
            threshold=opts.clash_threshold,
            exclude_hbonds=opts.clash_exclude_hbonds,
        )
        if state.clash_feasibility is not None:
            logger.info(
                f"Clash constraint active: acquisition restricted to geometries "
                f"with no non-bonded contact below {opts.clash_threshold:.2f} x "
                f"sum(vdW) ({state.clash_feasibility.detector.n_pairs} pairs "
                f"screened, H-bond pairs "
                f"{'excluded' if opts.clash_exclude_hbonds else 'included'})."
            )
    elif opts.clash_constraint:
        logger.info(
            "Clash constraint requested but no RDKit mol was supplied to "
            "run_optimization; searching the unconstrained torsion box."
        )

    # Graph-aware genetic subtree crossover (Phase 3): auto-enables at d >= 6
    # when mol is available (or forced via opts.gc_every > 0; 0 disables).
    gc_every = opts.resolve_gc_every(n_dihedrals, has_mol=mol is not None)
    if gc_every > 0:
        gc_cuts = build_graph_cuts(mol, dihedrals)
        if gc_cuts:
            state.gc_every = gc_every
            state.gc_warmup = opts.gc_warmup
            state.gc_cuts = gc_cuts
            state.gc_mol = mol
            state.gc_rng = np.random.default_rng(seed + 77773)
            logger.info(
                f"Graph-subtree crossover enabled: every {gc_every} BO step(s) "
                f"(warmup {opts.gc_warmup}, {len(gc_cuts)} graph cut(s))."
            )

    # Optional per-step stopping-rule certificate log (calibration benchmark).
    if opts.cert_log_path is not None:
        state.cert_betas = tuple(opts.cert_betas)
        state.cert_log = create_certificate_logger(opts.cert_log_path, state.cert_betas)

    # Optional geometry trail (benchmark): start a fresh file; frames are appended
    # at each best-so-far improvement (and the final relaxed best) for the offline
    # RMSD-identity / distinct-conformer analysis.
    if opts.geom_log_path is not None:
        state.geom_log_path = Path(opts.geom_log_path)
        state.geom_log_path.parent.mkdir(parents=True, exist_ok=True)
        state.geom_log_path.open("w").close()

    # Add prior settings to state
    state.prior_module = prior_module
    state.prior_exponent = opts.initial_prior_exponent
    state.prior_decay = opts.prior_exponent_decay

    # Evaluate initial guesses (conformers, prior peaks, or random)
    _evaluate_initial_guesses(
        state, dihedrals, calc, relaxCalc, relax, init_steps, seed,
        initial_conformers, initial_dihedrals,
    )

    # Run Bayesian optimization loop
    _run_optimization_loop(
        state, n_steps, dihedrals, calc, relaxCalc, relax, out_dir, seed=seed
    )

    if state.n_bond_forms:
        logger.info(
            f"Species guard: rejected {state.n_bond_forms} evaluation(s) whose "
            f"relaxation formed a new bond (collapsed geometry)."
        )
    if state.n_bond_breaks:
        logger.info(
            f"--retain-bonds: rejected {state.n_bond_breaks} evaluation(s) that "
            f"broke a bond."
        )
    if state.n_collective_clash_rejects:
        logger.info(
            f"Clash constraint: rejected {state.n_collective_clash_rejects} collective "
            f"move(s) before evaluation."
        )
    if state.clash_feasibility is not None and state.clash_feasibility.n_screened:
        f = state.clash_feasibility
        logger.info(
            f"Clash constraint: screened {f.n_screened} candidate geometries, "
            f"rejected {f.n_rejected} ({100.0 * f.n_rejected / f.n_screened:.1f}%) "
            f"before any energy evaluation."
        )

    # Post-BO ring-exocyclic grafting (Plan 2 Phase 2): on flexible-ring
    # molecules with alternate ring puckers in `mol`'s conformer pool, graft
    # BO's top exocyclic folds onto alternate rings so single-scaffold BO isn't
    # trapped on a suboptimal ring pucker.
    ring_grafts: list[Atoms] = []
    if (
        opts.ring_grafts > 0
        and relax
        and mol is not None
        and mol.GetNumConformers() >= 2
    ):
        ring_grafts = build_ring_graft_candidates(
            state,
            mol,
            dihedrals,
            max_grafts=opts.ring_grafts,
            clash_constraint=opts.clash_constraint,
            clash_threshold=opts.clash_threshold,
            clash_exclude_hbonds=opts.clash_exclude_hbonds,
        )

    if return_ensemble:
        # Active level-set exploration (on by default; opts.ensemble_steps == 0
        # keeps the historical passive-harvest behavior). Enriches `state` with
        # newly discovered basins before the harvest reads it.
        if opts.ensemble_steps != 0:
            _perform_ensemble_exploration(
                state, dihedrals, calc, relaxCalc, relax, opts, seed=seed
            )
        extra_ens = list(ensemble_seed_geometries or [])
        # Ring grafts get the same guards as _evaluate_ring_grafts: after the tight
        # opt, connectivity must match the START structure (not just the graft's own
        # pre-opt bonds) and stereochemistry must be retained.
        graft_validator = None
        if ring_grafts:
            from bouquet.rings.harvest import stereo_retained, stereo_signature

            start_sig = stereo_signature(mol, state.start_atoms)
            ref_bonds = geometry_bond_set(state.start_atoms)

            def graft_validator(rel_a: Atoms) -> str | None:
                if geometry_bond_set(rel_a) != ref_bonds:
                    return "ring graft connectivity changed vs start structure"
                if not stereo_retained(mol, rel_a, start_sig):
                    return "ring graft stereochemistry inverted"
                return None

        # Select and tightly optimize the low-energy ensemble. The best member
        # is the lowest-energy conformer; fall back to single-best relaxation if
        # the ensemble comes back empty (e.g. all evaluations failed).
        ensemble = _perform_ensemble_relaxation(
            state, dihedrals, calc, relaxCalc,
            extra_candidates=extra_ens or None,
            graft_candidates=ring_grafts or None,
            graft_validator=graft_validator,
        )
        if ensemble:
            best_atoms, best_e_e0, _weight = ensemble[0]
        else:
            best_atoms, best_e_e0 = _perform_final_relaxation(state, dihedrals, calc, relaxCalc)
        _log_improvement_geometry(state, best_atoms, "final", e_e0_eV=best_e_e0)
        return best_atoms, ensemble

    # Final relaxation, optional post-BO alternate-ring grafts, and return best
    best_atoms, best_e_e0 = _perform_final_relaxation(state, dihedrals, calc, relaxCalc)
    if ring_grafts:
        best_atoms, best_e_e0 = _evaluate_ring_grafts(
            ring_grafts, best_atoms, best_e_e0, state, mol, dihedrals, calc, relaxCalc
        )
    _log_improvement_geometry(state, best_atoms, "final", e_e0_eV=best_e_e0)
    return best_atoms
