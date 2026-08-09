"""Ring-conformer metadynamics driver: molecule in, ranked ring-state seeds out.

``run_ring_mtd`` ties the pieces together: perceive ring systems -> (gate) -> seed a few
walkers from distinct ETKDG ring states -> run RMSD-biased Langevin MD on each (the bias
pushes out of the starting pucker basins) -> harvest into stereo-checked, ranked ring-state
seeds. Walkers are independent, so they run in a process pool.

The biased MD itself is the mechanism validated in the ring-MTD proof-of-concept: hydrogen
masses inflated to allow a 2 fs step, per-group Gaussian widths calibrated from the thermal
ring fluctuation, and one RMSD bias per ring system so spiro systems explore the product of
their ring states.

The hill HEIGHT is adapted per molecule (``_verdict`` / ``_adapt_k_hill``). A single global
``k_hill`` provably cannot serve the whole ring population: on the Stage-0 benchmark it
destroyed stereochemistry on 6 of 17 molecules (two of them to zero seeds) while leaving 3
others stuck in their starting pucker. The controller bisects the hill height between those
two failures, and a molecule that never yields a usable seed raises :class:`RingMTDError`
rather than returning an empty set that downstream code cannot distinguish from "no ring
seeding needed".
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import numpy as np

from bouquet.rings.harvest import (
    HarvestResult, harvest, quench, stereo_retained, stereo_signature,
)
from bouquet.rings.systems import perceive_ring_systems, ring_state_distance


@dataclass
class RingMTDConfig:
    """Knobs for :func:`run_ring_mtd` (defaults follow RING_MTD_v2 §6.4)."""

    engine: str = "gfnff"
    temperature_K: float = 500.0
    timestep_fs: float = 2.0
    h_mass: float = 4.0
    friction_inv_fs: float = 0.01
    t_equil_ps: float = 2.0            # unbiased run that calibrates alpha
    t_run_ps: float = 20.0            # biased run, per walker
    n_walkers: int = 4
    n_processes: int = 1              # 1 = sequential; >1 uses a process pool
    deposit_interval_ps: float = 1.0
    snapshot_interval_ps: float = 0.2
    k_hill_kcal: float = 0.5
    max_refs: int = 25
    max_embeddings: int = 32
    alpha_clamp: tuple[float, float] = (1.0, 30.0)
    # MUST stay below fine_dedup_A: the coarse metric (ring heavy atoms, proper-Kabsch) is
    # strictly weaker than the fine one (ring + first shell, iRMSD), so a wider coarse
    # tolerance discards ring states before they are classified -- which is what made the
    # 5-ring molecules harvest a single state (RING_MTD_BENCH §9.7).
    coarse_dedup_A: float = 0.10
    fine_dedup_A: float = 0.20
    # Quench budget, farthest-point sampled: cost, not fidelity. 60 was measurably too tight --
    # a paired ablation (same trajectory, both budgets) on four state-rich molecules x 3 draws
    # found 60 -> 180 helped 9 of 12 and hurt 0, mean -0.10 A ring-iRMSD, for ~24 s on a ~190 s
    # run. Saturation above 180 was not probed. RING_MTD_BENCH §9.11.
    max_coarse_frames: int = 180
    energy_window_kcal: float = 12.0
    harvest_temperature_K: float = 298.15
    quench_fmax: float = 0.02
    quench_maxsteps: int = 300
    topology_check: bool = False      # set True for bond-breaking engines (gfn2, aimnet2)
    seed: int = 0

    # -- adaptive bias controller (see `_adapt_k_hill`) -----------------------------------
    adaptive_bias: bool = True
    max_bias_rounds: int = 3          # 1 disables retries (== the old single-shot behaviour)
    stereo_reject_ceiling: float = 0.25   # ACTION threshold; harvest's 0.05 alarm only warns
    min_states_target: int = 2        # fewer distinct ring states than this == the bias stalled
    k_hill_bounds: tuple[float, float] = (0.05, 4.0)   # kcal/mol, per hill


def should_run(mol) -> bool:
    """Whether the ring-MTD is worth running: the molecule has a puckerable ring system.

    The gate keeps bouquet cheap on the easy majority -- acyclic / purely-aromatic molecules
    (and anything ETKDG already covers) should not pay for a mini-CREST. A stronger gate
    (cheap ETKDG ring-diversity check) can layer on later; ring presence is the floor.
    """
    return len(perceive_ring_systems(mol)) > 0


def _steps(ps: float, timestep_fs: float) -> int:
    return max(1, int(round(ps * 1000.0 / timestep_fs)))


def _inflate_h(atoms, h_mass: float) -> None:
    m = atoms.get_masses()
    m[atoms.get_atomic_numbers() == 1] = h_mass
    atoms.set_masses(m)


def _group_rmsds(pos, pos0, systems) -> list[float]:
    from bouquet.rings.bias import rmsd_sq_and_grad
    return [float(np.sqrt(rmsd_sq_and_grad(pos[s.bias_idx], pos0[s.bias_idx])[0]))
            for s in systems]


def calibrate_alpha(atoms, calc, systems, cfg: RingMTDConfig) -> np.ndarray:
    """Per-group Gaussian width from a short unbiased run.

    ``alpha_g = 1/(2 sigma_g^2)`` with ``sigma_g = max(2 * sigma_therm, 0.1 A)``, clamped.
    Ring-only RMSD lives on a smaller scale than CREST's whole-molecule RMSD, so the hill
    must be calibrated (a fixed CREST-scale alpha would fill the neighbouring basin it is
    meant to reveal).
    """
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution, Stationary,
                                             ZeroRotation)
    a = atoms.copy()
    a.calc = calc
    _inflate_h(a, cfg.h_mass)
    MaxwellBoltzmannDistribution(a, temperature_K=cfg.temperature_K)
    Stationary(a); ZeroRotation(a)
    dyn = Langevin(a, timestep=cfg.timestep_fs * units.fs, temperature_K=cfg.temperature_K,
                   friction=cfg.friction_inv_fs / units.fs)
    pos0 = a.get_positions()
    series: list[list[float]] = []
    dyn.attach(lambda: series.append(_group_rmsds(a.get_positions(), pos0, systems)),
               interval=10)
    dyn.run(_steps(cfg.t_equil_ps, cfg.timestep_fs))
    sig_therm = (np.std(np.array(series), axis=0) if series
                 else np.full(len(systems), 0.1))
    sigma = np.maximum(2.0 * sig_therm, 0.10)
    return np.clip(1.0 / (2.0 * sigma ** 2), *cfg.alpha_clamp)


def _run_walker(task: dict):
    """One independent biased-MD walker; returns its list of snapshot coordinate arrays.

    Module-level and self-contained so it is picklable for the process pool: it rebuilds
    the molecule/engine from the SMILES and starts from the supplied geometry.
    """
    from ase import Atoms, units
    from ase.calculators.mixing import SumCalculator
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution, Stationary,
                                             ZeroRotation)
    from rdkit import Chem

    from bouquet.calculator import CalculatorFactory
    from bouquet.rings.bias import KCAL_TO_EV, RMSDBias
    from bouquet.setup import apply_charge_spin

    cfg: RingMTDConfig = task["cfg"]
    systems = task["systems"]
    # RDKit binary, not a SMILES: it round-trips the atom order exactly (a SMILES does not,
    # see seed_walkers), and engines that actually read `mol` -- mmff, uff -- would otherwise
    # get a topology indexed differently from the coordinates. xtb ignores `mol`, which is the
    # only reason the SMILES path did not corrupt the GFN-FF/GFN2 dynamics.
    molh = Chem.Mol(task["mol_binary"])
    atoms = Atoms(numbers=task["numbers"], positions=task["positions"])
    apply_charge_spin(atoms, task["charge"], task["mult"])
    _inflate_h(atoms, cfg.h_mass)

    engine = CalculatorFactory.create(method=cfg.engine, mol=molh, charge=task["charge"],
                                      multiplicity=task["mult"])
    alpha = calibrate_alpha(atoms, engine, systems, cfg)
    bias = RMSDBias([s.bias_idx for s in systems], k=cfg.k_hill_kcal * KCAL_TO_EV,
                    alpha=alpha, max_refs=cfg.max_refs)
    atoms.calc = SumCalculator([engine, bias])

    MaxwellBoltzmannDistribution(atoms, temperature_K=cfg.temperature_K,
                                 rng=np.random.default_rng(task["vseed"]))
    Stationary(atoms); ZeroRotation(atoms)
    dyn = Langevin(atoms, timestep=cfg.timestep_fs * units.fs,
                   temperature_K=cfg.temperature_K,
                   friction=cfg.friction_inv_fs / units.fs)
    bias.deposit(atoms)  # reference 0 = the starting basin
    frames: list[np.ndarray] = []
    dyn.attach(lambda: bias.deposit(atoms),
               interval=_steps(cfg.deposit_interval_ps, cfg.timestep_fs))
    dyn.attach(lambda: frames.append(atoms.get_positions().copy()),
               interval=_steps(cfg.snapshot_interval_ps, cfg.timestep_fs))
    dyn.run(_steps(cfg.t_run_ps, cfg.timestep_fs))
    return frames


class RingMTDError(RuntimeError):
    """The metadynamics ran but produced no usable seeds.

    Distinct from the gate declining (an acyclic/aromatic molecule, which legitimately yields
    an empty seed set): this means the bias destroyed every frame it generated. Raised rather
    than returned because an empty seed FILE is indistinguishable downstream from "this
    molecule needs no ring seeding", and silently shipping one is how a 100%-stereo-reject
    run reads as a success.
    """


def _verdict(diag: dict, cfg: RingMTDConfig) -> str:
    """Classify a round's harvest: ``'ok'``, ``'too_strong'``, or ``'too_weak'``.

    Two failures, one knob. Measured on the Stage-0 benchmark (RING_MTD_BENCH §9.2): with a
    single global ``k_hill``, 6 of 17 molecules destroyed stereochemistry (up to 100% of
    frames rejected, two harvesting to zero seeds) while 3 others never left the starting
    pucker (404 frames -> 1 ring state). ``k_hill = 0.5`` is tuned to the cyclohexane chair
    flip; strained/quaternary systems invert well below that, floppy small rings need more.
    """
    if diag.get("stereo_reject_rate", 0.0) > cfg.stereo_reject_ceiling:
        return "too_strong"
    if diag.get("n_clusters", 0) < cfg.min_states_target:
        return "too_weak"
    return "ok"


def _adapt_k_hill(k: float, verdict: str, lo: float, hi: float,
                  cfg: RingMTDConfig) -> tuple[float, float, float]:
    """Next ``k_hill`` plus the updated bracket, by bisection on the verdict.

    Each round brackets the usable hill height from one side -- ``too_strong`` caps it,
    ``too_weak`` floors it -- so once both sides are known the next probe is their geometric
    mean (the scale is multiplicative). Bisecting rather than stepping is what stops a
    molecule that is too strong at 0.5 and too weak at 0.2 from oscillating between them,
    which a fixed multiplicative back-off would do.
    """
    if verdict == "too_strong":
        hi = min(hi, k)
        nxt = np.sqrt(lo * hi) if lo > 0 else k * 0.4
    else:  # too_weak
        lo = max(lo, k)
        nxt = np.sqrt(lo * hi) if np.isfinite(hi) else k * 2.5
    return float(np.clip(nxt, *cfg.k_hill_bounds)), lo, hi


def seed_walkers(molh, systems, cfg: RingMTDConfig, charge: int):
    """Distinct ETKDG ring states as walker start geometries, plus the multiplicity.

    Embeds ``max_embeddings`` conformers (MMFF-optimized), keeps the lowest-energy
    representative of each distinct ring state (by :func:`ring_state_distance`), and reuses
    them cyclically if there are fewer than ``n_walkers``. Returns ``(list[Atoms], mult)``
    with charge/spin stamped; the first entry is the lowest-energy start (bouquet's start).

    Takes the H-added molecule ITSELF, never a SMILES to rebuild from. ``systems``' ring and
    dedup indices, and the stereo signature the caller reads off these geometries, are all
    indices into ``molh``'s atom order -- so the embedded conformers must carry that exact
    order. Rebuilding via ``MolFromSmiles(MolToSmiles(mol, canonical=False))`` does NOT
    preserve it: the SMILES writer emits a DFS traversal, and reparsing yields traversal
    order, which diverges from input order whenever ring-closure bonds reorder the walk
    (measured: m077/m095/m109/m120 all permute, m047 happens not to). Every index-based read
    then lands on the wrong atom -- which is what produced their k-independent stereo
    rejections (RING_MTD_BENCH §9.4).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from bouquet.setup import apply_charge_spin, default_multiplicity, mol_to_ase_atoms

    molh = Chem.Mol(molh)  # copy: we add conformers to it
    params = AllChem.ETKDGv3()
    params.randomSeed = cfg.seed
    params.useSmallRingTorsions = True
    cids = list(AllChem.EmbedMultipleConfs(molh, numConfs=cfg.max_embeddings, params=params))
    if not cids:
        raise ValueError(f"ETKDG could not embed {Chem.MolToSmiles(molh)}")
    ff = AllChem.MMFFOptimizeMoleculeConfs(molh)
    energy = {c: (ff[i][1] if ff[i][0] == 0 else float("inf")) for i, c in enumerate(cids)}
    mult = default_multiplicity(mol_to_ase_atoms(molh, cids[0]), charge)

    reps = []
    for c in sorted(cids, key=lambda c: energy[c]):
        atoms = mol_to_ase_atoms(molh, c)
        pos = atoms.get_positions()
        if all(max(ring_state_distance(s, pos, r.get_positions()) for s in systems)
               >= cfg.fine_dedup_A for r in reps):
            apply_charge_spin(atoms, charge, mult)
            reps.append(atoms)
    starts = (reps * cfg.n_walkers)[: cfg.n_walkers]
    return starts, mult


def run_ring_mtd(mol_or_smiles, cfg: RingMTDConfig | None = None) -> HarvestResult:
    """Discover ring-conformer seeds for ``mol_or_smiles`` via RMSD-biased metadynamics.

    Returns a :class:`bouquet.rings.harvest.HarvestResult` (ranked ring-state seeds +
    diagnostics). An acyclic / purely-aromatic molecule yields an empty result -- the
    caller then just uses a single ETKDG seed.
    """
    from rdkit import Chem

    from bouquet.calculator import CalculatorFactory

    cfg = cfg or RingMTDConfig()
    mol = Chem.MolFromSmiles(mol_or_smiles) if isinstance(mol_or_smiles, str) else mol_or_smiles
    # Keep the INPUT atom order end-to-end: `systems` (dedup_idx), the walker starts, the
    # stereo signature, and harvest all index into `molh = AddHs(mol)`, and the seeds come out
    # in that order too -- so they line up atom-for-atom with the molecule bouquet builds as
    # AddHs(MolFromSmiles(smiles)) and reads dihedrals off by index. Nothing downstream may
    # rebuild the molecule from a SMILES: NEITHER canonical nor canonical=False round-trips
    # the atom order (a SMILES is a DFS traversal; reparsing yields traversal order). The
    # molecule travels to the process pool as an RDKit binary instead.
    molh = Chem.AddHs(mol)
    charge = Chem.GetFormalCharge(mol)

    systems = perceive_ring_systems(molh)
    if not systems:
        return HarvestResult(seeds=[], diagnostics={"skipped": "no puckerable ring system"})

    starts, mult = seed_walkers(molh, systems, cfg, charge)

    calc = CalculatorFactory.create(method=cfg.engine, mol=molh, charge=charge,
                                    multiplicity=mult)

    # Quench the ETKDG/MMFF starts onto the RUN ENGINE's surface before anything reads their
    # stereochemistry. Defensive, not a bug fix: the stereo filter asks "did the MD change the
    # stereochemistry relative to where we started", and every structure it judges has been
    # engine-quenched, so measuring the reference on the same surface removes a whole class of
    # cross-surface disagreement by construction. (GFN-FF was NOT in fact found to move these
    # assignments -- a plain quench retains them on all of m047/m077/m095/m109/m120 once the
    # atom order is right; the apparent flips were the seed_walkers ordering bug.) Starting the
    # MD from an engine minimum is also better dynamics -- no initial relaxation transient
    # dumping energy into the ring.
    quenched = []
    for s in starts:
        q = quench(s.copy(), calc, fmax=cfg.quench_fmax, steps=cfg.quench_maxsteps)
        if q is not None:
            quenched.append(q[0])
    starts = quenched or starts

    # Stereo reference = the lowest-energy start (bouquet's start). Keep only walkers on the
    # SAME diastereomer, so ETKDG concretizing an unspecified spiro centre differently in
    # some embedding cannot leak a different molecule into the seed set.
    start_sig = stereo_signature(molh, starts[0])
    starts = [s for s in starts if stereo_retained(molh, s, start_sig)] or [starts[0]]

    def _round(round_cfg: RingMTDConfig) -> HarvestResult:
        """One full walkers-then-harvest pass at ``round_cfg``'s hill height."""
        tasks = [{
            "mol_binary": molh.ToBinary(),
            "numbers": s.get_atomic_numbers(), "positions": s.get_positions(),
            "charge": charge, "mult": mult, "cfg": round_cfg, "systems": systems,
            "vseed": round_cfg.seed * 1000 + w,
        } for w, s in enumerate(starts)]

        if round_cfg.n_processes > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=round_cfg.n_processes) as ex:
                frame_lists = list(ex.map(_run_walker, tasks))
        else:
            frame_lists = [_run_walker(t) for t in tasks]
        frames = [f for fl in frame_lists for f in fl]

        return harvest(
            frames, molh, systems, calc, start_signature=start_sig,
            coarse_tol=round_cfg.coarse_dedup_A, fine_tol=round_cfg.fine_dedup_A,
            max_coarse=round_cfg.max_coarse_frames,
            temperature=round_cfg.harvest_temperature_K,
            energy_window_kcal=round_cfg.energy_window_kcal,
            fmax=round_cfg.quench_fmax, steps=round_cfg.quench_maxsteps,
            topology_check=round_cfg.topology_check,
        )

    # -- adaptive bias loop --------------------------------------------------------------
    # Rounds REPLACE rather than accumulate: a too-strong round's frames are mostly invalid
    # and a too-weak round's are all the starting basin, so neither is worth the second
    # quench pass that merging them would cost. A healthy molecule accepts on round 1 and
    # pays nothing (8 of 17 in Stage-0 did).
    k = cfg.k_hill_kcal
    lo, hi = cfg.k_hill_bounds[0], float("inf")
    rounds, best, best_verdict = [], None, None
    n_rounds = cfg.max_bias_rounds if cfg.adaptive_bias else 1
    for _ in range(n_rounds):
        result = _round(replace(cfg, k_hill_kcal=k))
        verdict = _verdict(result.diagnostics, cfg)
        rounds.append({"k_hill_kcal": round(k, 4), "verdict": verdict,
                       "n_seeds": len(result.seeds),
                       "n_clusters": result.diagnostics.get("n_clusters"),
                       "stereo_reject_rate": result.diagnostics.get("stereo_reject_rate")})
        # Keep the round with the most ring states among healthy ones; failing that, the
        # least-rejected round, so a molecule that never converges still returns its best.
        if best is None or _better(result, verdict, best, best_verdict):
            best, best_verdict, best_k = result, verdict, k
        if verdict == "ok":
            break
        k, lo, hi = _adapt_k_hill(k, verdict, lo, hi, cfg)

    best.diagnostics.update(n_walkers=len(starts), n_ring_systems=len(systems),
                            bias_rounds=rounds, k_hill_final=round(best_k, 4),
                            bias_verdict=best_verdict)
    best.systems = systems

    if not best.seeds:
        raise RingMTDError(
            f"ring MTD produced no usable seeds after {len(rounds)} round(s) "
            f"(last verdict {best_verdict!r}, stereo reject rate "
            f"{best.diagnostics.get('stereo_reject_rate')}); rounds={rounds}")
    return best


def _better(result: HarvestResult, verdict: str, best: HarvestResult, best_verdict: str) -> bool:
    """Is ``result`` a better round than the incumbent ``best``?

    A healthy round always beats an unhealthy one; among equals, more distinct ring states
    wins, then a lower stereo-reject rate.
    """
    if (verdict == "ok") != (best_verdict == "ok"):
        return verdict == "ok"
    if len(result.seeds) != len(best.seeds):
        return len(result.seeds) > len(best.seeds)
    return (result.diagnostics.get("stereo_reject_rate", 1.0)
            < best.diagnostics.get("stereo_reject_rate", 1.0))
