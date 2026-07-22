"""Ring-conformer metadynamics driver: molecule in, ranked ring-state seeds out.

``run_ring_mtd`` ties the pieces together: perceive ring systems -> (gate) -> seed a few
walkers from distinct ETKDG ring states -> run RMSD-biased Langevin MD on each (the bias
pushes out of the starting pucker basins) -> harvest into stereo-checked, ranked ring-state
seeds. Walkers are independent, so they run in a process pool.

The biased MD itself is the mechanism validated in the ring-MTD proof-of-concept: hydrogen
masses inflated to allow a 2 fs step, per-group Gaussian widths calibrated from the thermal
ring fluctuation, and one RMSD bias per ring system so spiro systems explore the product of
their ring states.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from bouquet.rings.harvest import HarvestResult, harvest, stereo_retained, stereo_signature
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
    coarse_dedup_A: float = 0.5
    fine_dedup_A: float = 0.20
    energy_window_kcal: float = 12.0
    harvest_temperature_K: float = 298.15
    quench_fmax: float = 0.02
    quench_maxsteps: int = 300
    topology_check: bool = False      # set True for bond-breaking engines (gfn2, aimnet2)
    seed: int = 0


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
    molh = Chem.AddHs(Chem.MolFromSmiles(task["smiles"]))
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


def seed_walkers(smiles: str, systems, cfg: RingMTDConfig, charge: int):
    """Distinct ETKDG ring states as walker start geometries, plus the multiplicity.

    Embeds ``max_embeddings`` conformers (MMFF-optimized), keeps the lowest-energy
    representative of each distinct ring state (by :func:`ring_state_distance`), and reuses
    them cyclically if there are fewer than ``n_walkers``. Returns ``(list[Atoms], mult)``
    with charge/spin stamped; the first entry is the lowest-energy start (bouquet's start).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from bouquet.setup import apply_charge_spin, default_multiplicity, mol_to_ase_atoms

    molh = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = cfg.seed
    params.useSmallRingTorsions = True
    cids = list(AllChem.EmbedMultipleConfs(molh, numConfs=cfg.max_embeddings, params=params))
    if not cids:
        raise ValueError(f"ETKDG could not embed {smiles}")
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
    smiles = Chem.MolToSmiles(mol)
    molh = Chem.AddHs(mol)
    charge = Chem.GetFormalCharge(mol)

    systems = perceive_ring_systems(molh)
    if not systems:
        return HarvestResult(seeds=[], diagnostics={"skipped": "no puckerable ring system"})

    starts, mult = seed_walkers(smiles, systems, cfg, charge)

    # Stereo reference = the lowest-energy start (bouquet's start). Keep only walkers on the
    # SAME diastereomer, so ETKDG concretizing an unspecified spiro centre differently in
    # some embedding cannot leak a different molecule into the seed set.
    start_sig = stereo_signature(molh, starts[0])
    starts = [s for s in starts if stereo_retained(molh, s, start_sig)] or [starts[0]]

    tasks = [{
        "smiles": smiles, "numbers": s.get_atomic_numbers(), "positions": s.get_positions(),
        "charge": charge, "mult": mult, "cfg": cfg, "systems": systems,
        "vseed": cfg.seed * 1000 + w,
    } for w, s in enumerate(starts)]

    if cfg.n_processes > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=cfg.n_processes) as ex:
            frame_lists = list(ex.map(_run_walker, tasks))
    else:
        frame_lists = [_run_walker(t) for t in tasks]
    frames = [f for fl in frame_lists for f in fl]

    calc = CalculatorFactory.create(method=cfg.engine, mol=molh, charge=charge,
                                    multiplicity=mult)
    result = harvest(
        frames, molh, systems, calc, start_signature=start_sig,
        coarse_tol=cfg.coarse_dedup_A, fine_tol=cfg.fine_dedup_A,
        temperature=cfg.harvest_temperature_K, energy_window_kcal=cfg.energy_window_kcal,
        fmax=cfg.quench_fmax, steps=cfg.quench_maxsteps, topology_check=cfg.topology_check,
    )
    result.diagnostics.update(n_walkers=len(starts), n_ring_systems=len(systems))
    return result
