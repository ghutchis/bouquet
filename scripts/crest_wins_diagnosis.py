#!/usr/bin/env python
"""Diagnose molecules where CREST beats bouquet: ring-conformer miss vs dihedral miss.

For each molecule whose reference E* is set by the CREST pool (E_star_crest lower
than bouquet's E_star_BO by ``--gap-kcal``), classify *why* bouquet lost:

  * DIHEDRAL_MISS          -- bouquet's SEARCH could reach the CREST minimum from its
    start by turning rotatable dihedrals alone (or the molecule has no flexible ring),
    so the loss is a search failure, not a ring problem. (Reported with the max
    rotatable-dihedral difference to the CREST minimum.)  Actionable: budget / acquisition.

  * RING_FLIP_MISS         -- the start had the right ring SHAPE (ring-heavy iRMSD
    matched) but bouquet's move still cannot reach the CREST minimum: the WRONG ring
    flip / axial-vs-equatorial attachment, which needs a ring-bond change bouquet never
    makes. Only distinguishable with the operational test (default on).  Actionable:
    ensemble must carry the flipped ring (ETKDG usually samples both) / start selection.

  * RING_MISS__IN_SEED_POOL -- the start's ring shape was wrong, but the CREST minimum
    IS operationally reachable from another of the run's own ETKDG embeddings (the
    correct ring conformer was sampled, just not chosen as the search start). The
    ensemble harvest should surface it. Actionable: start selection / ensemble coverage.

  * RING_MISS__ETKDG_REACHABLE -- the run's embeddings can't reach it, but an expanded
    ETKDG pool (more seeds/confs) can. Actionable: raise --init-conformers / more seeds.

  * RING_MISS__ETKDG_UNREACHABLE -- no ETKDG embedding reaches the CREST minimum at all;
    distance geometry cannot produce this ring conformer. Actionable: CREST / systematic
    ring enumeration is required (bouquet cannot get there by construction).

Method (all decisions confirmed with the user):
  * Ring identity = iRMSD (the project's rotation- AND permutation-invariant RMSD,
    ``irmsd.get_irmsd_ase``) over the flexible (non-aromatic) ring HEAVY atoms only;
    "same pucker" below ``--ring-rmsd`` (default 0.125 A, the ensemble-dedup tol).
    iRMSD's permutation invariance is essential here: a chair and the SAME chair
    embedded as its mirror image relabel onto each other, so a plain Kabsch RMSD
    spuriously reports ~0.4 A between identical puckers -- iRMSD collapses them to
    ~0. Restricting to ring heavy atoms isolates the pucker from the pendant
    dihedrals bouquet can turn (matches ensemble_bench/probe_sampling.py's ring mode).
  * "Other ETKDG conformers" is probed on BOTH the run's own seed pool and an
    expanded capability pool (``--cap-seeds`` x ``--cap-confs``).
  * Everything (CREST cell minima, ETKDG embeddings, bouquet trail finals) is
    re-optimized on GFN2 first so all puckers live in the same basins; the CREST
    winner is defined exactly as reference.py defines E_star_crest.
  * The operational reachability test (ON by default; --no-operational to skip) is
    bouquet's OWN move used as the authoritative signal: from each embedding, dial the
    rotatable dihedrals to the CREST minimum's optimized values, constrained-relax then
    release, and measure full-molecule iRMSD to CREST. Ring-shape iRMSD cannot tell an
    axial from an equatorial chair (same ring shape); this move can, because bouquet can
    turn dihedrals but never flip the ring -- from the wrong chair it relaxes back to a
    different basin and stays far from CREST. It decides the DIHEDRAL_MISS / RING_MISS
    ladder and, combined with the ring-shape match, isolates RING_FLIP_MISS.

Atom ordering is the RDKit ``AddHs`` order for every geometry (CREST cells, BO
trails, and regenerated ETKDG embeddings alike) -- the same assumption reference.py
relies on -- so ring/dihedral atom indices are shared without remapping.

Usage (paths mirror reference.py; point them at wherever the run data lives):
    pixi run python scripts/crest_wins_diagnosis.py \
        --reference-dir  runs/reference \
        --geom-dir       runs/geom \
        --crest-dir      runs/ensemble_bench \
        --manifest       smiles/autosteps-calib.csv \
        --method gfn2 --workers 8 \
        --out runs/analysis/crest_wins.csv --dump-dir runs/analysis/crest_wins_geom
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from ase import Atoms

# scripts/ is importable as a flat dir when run from the repo root (reference.py sits
# beside this file); reuse its GFN2 relaxation + CREST-cell + trail readers verbatim.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference as ref  # noqa: E402  (_gfnff_calc, _optimize, _read_refined_xyz, _read_trail, KCAL)
from bouquet.ensemble import _rmsd as full_irmsd  # noqa: E402  (iRMSD, Kabsch fallback)

KCAL = ref.KCAL

# The project's rotation+permutation-invariant RMSD backend, used here on ring-atom
# substructures. Mirrors bouquet.ensemble's guarded import so this degrades to a
# Kabsch fallback if the optional wheel is missing (with a loud caveat at runtime).
try:
    import irmsd as _irmsd  # noqa: E402
    _HAVE_IRMSD = hasattr(_irmsd, "get_irmsd_ase")
except ImportError:  # pragma: no cover
    _irmsd = None
    _HAVE_IRMSD = False


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def flexible_ring_atoms(mol) -> list[int]:
    """Heavy-atom indices sitting in a non-aromatic (puckerable) ring.

    These carry the ring-pucker DOF the BO loop can never reach. A molecule with
    none has no ring problem by construction -> any CREST win there is dihedral."""
    ri = mol.GetRingInfo()
    idx: set[int] = set()
    for ring in ri.AtomRings():
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if all(a.GetIsAromatic() for a in atoms):
            continue  # fully aromatic ring is planar; no pucker
        for a in atoms:
            if a.GetAtomicNum() > 1:
                idx.add(a.GetIdx())
    return sorted(idx)


def _subset(atoms, idx: list[int]) -> Atoms:
    """Ring-atom substructure as a standalone Atoms (geometry-only; iRMSD perceives
    the ring graph from elements + coordinates)."""
    sym = atoms.get_chemical_symbols()
    return Atoms(symbols=[sym[i] for i in idx], positions=atoms.get_positions()[idx])


def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Rotation-only RMSD; the fallback when the iRMSD wheel is unavailable. Note
    this CANNOT superpose a pucker onto its mirror-image embedding, so ring calls
    may over-report differences without iRMSD -- hence the runtime caveat."""
    Pc, Qc = P - P.mean(axis=0), Q - Q.mean(axis=0)
    V, _, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1.0, 1.0, d]) @ Wt
    diff = Pc @ R - Qc
    return float(np.sqrt((diff * diff).sum() / len(P)))


def ring_rmsd(atoms_a, atoms_b, ring_idx: list[int]) -> float:
    """iRMSD over the flexible-ring HEAVY atoms of two geometries of the same
    molecule (shared atom indexing). Permutation invariance collapses mirror-image
    embeddings of one pucker to ~0; falls back to Kabsch only without the wheel."""
    sa, sb = _subset(atoms_a, ring_idx), _subset(atoms_b, ring_idx)
    if _HAVE_IRMSD:
        return float(_irmsd.get_irmsd_ase(sa, sb)[0])
    return _kabsch_rmsd(sa.get_positions(), sb.get_positions())


def circular_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two angles (degrees)."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def operational_irmsd(embedding, crest_min, crest_angles, dihedrals,
                      calc, molh, conn_tol) -> float:
    """bouquet's OWN move applied to one embedding: dial the rotatable dihedrals to the
    CREST minimum's optimized values, constrained-relax (FixInternals) then RELEASE and
    relax -- exactly the search-evaluation + final-relaxation pipeline -- and return the
    full-molecule iRMSD to the CREST minimum.

    This is the only signal that separates an AXIAL vs EQUATORIAL (ring-flip) miss:
    ring-only iRMSD merges the two chairs, but bouquet can only turn dihedrals, never
    flip the ring, so from the wrong chair this move relaxes back to a different basin
    and stays far from CREST. Small => bouquet could reach CREST from this embedding;
    large => the ring/attachment is genuinely out of reach. inf on failure / bond change.
    """
    from bouquet.assess import evaluate_energy, relax_structure
    from bouquet.setup import connectivity_changed
    try:
        _e, a = evaluate_energy(crest_angles, embedding, dihedrals, calc, calc,
                                relax=True, steps=None)     # dihedrals -> CREST, hold + relax
        a = a.copy(); a.set_constraint()                    # release the constraints
        _e2, a2 = relax_structure(a, calc, calc, None)      # unconstrained final relax
    except Exception:
        return float("inf")
    if connectivity_changed(a2, molh, conn_tol):
        return float("inf")
    return full_irmsd(a2, crest_min)


# --------------------------------------------------------------------------- #
# Per-molecule diagnosis (one worker call)
# --------------------------------------------------------------------------- #
def _relax_pool(atoms_list, calc, molh, conn_tol):
    """Relax each geometry on GFN2; drop failures and connectivity-changers.
    Returns list of (relaxed_atoms, energy_eV)."""
    from bouquet.setup import connectivity_changed
    out = []
    for a in atoms_list:
        r = ref._optimize(a, calc)
        if r is None:
            continue
        if connectivity_changed(r[0], molh, conn_tol):
            continue
        out.append((r[0], r[1]))
    return out


def _read_reference_min(path: str):
    """Lowest-energy frame of a reference.py ``<mol_id>.xyz`` (deduped minima, comment
    ``E_rel_kcal=...``). For a CREST-wins molecule the global min IS the CREST minimum,
    so this is the CREST-winning geometry -- already GFN2-optimized, atom order = AddHs.
    Returns Atoms or None."""
    from ase.io import read
    frames = read(path, index=":")
    if not frames:
        return None
    if not isinstance(frames, list):
        frames = [frames]
    lines = Path(path).read_text().splitlines()
    energies, i = [], 0
    for _ in range(len(frames)):
        nat = int(lines[i])
        e = None
        for tok in lines[i + 1].split():
            if tok.startswith("E_rel_kcal="):
                try:
                    e = float(tok.split("=", 1)[1])
                except ValueError:
                    pass
        energies.append(e if e is not None else float("inf"))
        i += nat + 2
    if all(e == float("inf") for e in energies):
        return frames[0]  # written energy-ascending, so first is the min
    return frames[int(np.argmin(energies))]


def diagnose(task: dict) -> dict:
    from rdkit import Chem
    from bouquet.setup import (
        apply_charge_spin, default_multiplicity, detect_dihedrals,
        get_initial_candidates,
    )

    mid = task["mol_id"]
    out = {"mol_id": mid, "d": task.get("d"), "cls": task.get("cls"),
           "gap_kcal": task["gap_kcal"], "E_star_BO": task["E_star_BO"],
           "E_star_crest": task["E_star_crest"], "classification": None,
           "note": ""}
    try:
        mol = Chem.MolFromSmiles(task["smiles"])
        if mol is None:
            out["note"] = "bad SMILES"; return out
        molh = Chem.AddHs(mol)
        charge = Chem.GetFormalCharge(mol)
        calc = ref._gfnff_calc(task["method"], molh, charge)
        conn_tol = task["conn_tol"]
        ring_idx = flexible_ring_atoms(molh)
        out["n_flex_ring_atoms"] = len(ring_idx)

        def stamp(atoms_list):
            if not atoms_list:
                return atoms_list
            mult = default_multiplicity(atoms_list[0], charge)
            for a in atoms_list:
                apply_charge_spin(a, charge, mult)
            return atoms_list

        # -- CREST winner geometry ------------------------------------------------
        if task["crest_from_reference"]:
            # Global-min frame of reference/<mol_id>.xyz. For a CREST-wins molecule this
            # is the CREST minimum (already GFN2-optimized); use it when the raw CREST
            # cells are not on hand. Atom order matches AddHs, so indexing is consistent.
            crest_min = _read_reference_min(task["ref_xyz"])
            if crest_min is None:
                out["note"] = "no reference-min frame"; return out
            stamp([crest_min])
        else:
            crest_raw = []
            for cf in task["crest_files"][: task["crest_max"]]:
                crest_raw += ref._read_refined_xyz(cf)
            crest_opt = _relax_pool(stamp(crest_raw), calc, molh, conn_tol)
            if not crest_opt:
                out["note"] = "no CREST minimum survived"; return out
            crest_min, _crest_e = min(crest_opt, key=lambda t: t[1])

        # -- bouquet's best: relax the trail 'final' frame(s), take the min --------
        # A 'final' frame marks a COMPLETED run (bouquet's final unconstrained
        # relaxation). Right-censored runs (killed by the wall clock before finishing,
        # common at high d) have none; those are pre-filtered out in the driver so we
        # never compare against an unfinished search.
        finals = [a for tp in task["trail_files"]
                  for a, _e, k in ref._read_trail(tp) if k == "final"]
        bo_opt = _relax_pool(stamp(finals), calc, molh, conn_tol)
        if not bo_opt:
            out["note"] = "run incomplete (no final frame) / relaxation failed"; return out
        bo_best, _bo_e = min(bo_opt, key=lambda t: t[1])

        # -- rotatable-dihedral difference bouquet-best vs CREST-min ---------------
        dih = detect_dihedrals(molh)
        if dih:
            diffs = [circular_diff_deg(d.get_angle(bo_best), d.get_angle(crest_min))
                     for d in dih]
            out["dih_max_deg"] = round(max(diffs), 1)
            out["dih_mean_deg"] = round(float(np.mean(diffs)), 1)
        out["bo_best_ring_rmsd"] = (round(ring_rmsd(bo_best, crest_min, ring_idx), 4)
                                    if ring_idx else None)
        # Full-molecule iRMSD for context (how far bouquet's best sits from CREST overall).
        out["bo_best_full_irmsd"] = round(full_irmsd(bo_best, crest_min), 4)

        # -- ETKDG start (per run seed) + seed-pool + capability -------------------
        if ring_idx:
            run_seeds = task["run_seeds"]
            all_seeds = sorted(set(run_seeds) | set(range(1, task["cap_seeds"] + 1)))
            thr = task["ring_rmsd_thr"]

            start_atoms: dict[int, object] = {}   # run-seed -> relaxed start embedding
            start_rmsds: dict[int, float] = {}    # run-seed -> its ring rmsd
            pool: dict[int, list] = {}            # seed -> [(relaxed_atoms, ring_rmsd)]
            for s in all_seeds:
                cands, _m = get_initial_candidates(
                    task["smiles"], seed=s, max_confs=task["cap_confs"])
                stamp(cands)
                # bouquet seeds the SEARCH from the lowest single-point embedding.
                sp = []
                for a in cands:
                    aa = a.copy(); aa.calc = calc
                    try:
                        sp.append(float(aa.get_potential_energy()))
                    except Exception:
                        sp.append(float("inf"))
                start_i = int(np.argmin(sp)) if sp else 0
                relaxed = _relax_pool(cands, calc, molh, conn_tol)
                pool[s] = [(a, ring_rmsd(a, crest_min, ring_idx)) for a, _e in relaxed]
                rs = ref._optimize(cands[start_i], calc) if cands else None
                if rs is not None:
                    start_atoms[s] = rs[0]
                    start_rmsds[s] = ring_rmsd(rs[0], crest_min, ring_idx)
                else:
                    start_rmsds[s] = float("inf")

            start_min = min((start_rmsds[s] for s in run_seeds), default=float("inf"))
            seed_pool_min = min((min(r for _a, r in pool[s]) for s in run_seeds
                                 if pool[s]), default=float("inf"))
            capability_min = min((min(r for _a, r in pool[s]) for s in all_seeds
                                  if pool[s]), default=float("inf"))
            out["start_ring_rmsd_min"] = round(start_min, 4)
            out["seed_pool_ring_rmsd"] = round(seed_pool_min, 4)
            out["capability_ring_rmsd"] = round(capability_min, 4)
            out["start_ring_ok"] = bool(start_min < thr)
            out["seed_pool_hit"] = bool(seed_pool_min < thr)
            out["capability_hit"] = bool(capability_min < thr)

            # -- operational reachability (bouquet's own move; separates axial/eq) -----
            if task["operational"] and dih:
                op_tol, op_cap = task["op_rmsd"], task["op_max_confs"]
                crest_ang = [d.get_angle(crest_min) for d in dih]

                def op_reach(atoms):
                    return operational_irmsd(atoms, crest_min, crest_ang, dih,
                                             calc, molh, conn_tol)

                # Only the ring-CLOSEST op_cap embeddings per seed are worth testing:
                # the operational move needs a compatible ring, so farther puckers can
                # never reach CREST -- this bounds the extra relaxations.
                def pool_op_min(seeds):
                    best = float("inf")
                    for s in seeds:
                        for a, _r in sorted(pool[s], key=lambda t: t[1])[:op_cap]:
                            best = min(best, op_reach(a))
                    return best

                op_start_min = min((op_reach(start_atoms[s]) for s in run_seeds
                                    if s in start_atoms), default=float("inf"))
                op_seed_pool = pool_op_min(run_seeds)
                op_capability = pool_op_min(all_seeds)
                out["op_start_irmsd_min"] = round(op_start_min, 4)
                out["op_seed_pool_irmsd"] = round(op_seed_pool, 4)
                out["op_capability_irmsd"] = round(op_capability, 4)
                out["op_start_ok"] = bool(op_start_min < op_tol)
                out["op_seed_pool_hit"] = bool(op_seed_pool < op_tol)
                out["op_capability_hit"] = bool(op_capability < op_tol)

        # -- classification -------------------------------------------------------
        # Default (operational on): bouquet's own move decides reachability, and the
        # ring-shape signal splits out RING_FLIP_MISS -- the start had the right ring
        # SHAPE (ring iRMSD matched) but the move still cannot reach CREST, i.e. the
        # wrong flip / axial-vs-equatorial, which bouquet cannot fix by turning
        # dihedrals. Without operational (--no-operational) we fall back to the
        # ring-shape ladder alone and cannot distinguish a flip from a real search miss.
        if not ring_idx:
            out["classification"] = "DIHEDRAL_MISS"
            out["note"] = "no flexible ring; loss is dihedral/other"
        elif task["operational"] and dih:
            if out.get("op_start_ok"):
                out["classification"] = "DIHEDRAL_MISS"
            elif out["start_ring_ok"]:
                out["classification"] = "RING_FLIP_MISS"
            elif out.get("op_seed_pool_hit"):
                out["classification"] = "RING_MISS__IN_SEED_POOL"
            elif out.get("op_capability_hit"):
                out["classification"] = "RING_MISS__ETKDG_REACHABLE"
            else:
                out["classification"] = "RING_MISS__ETKDG_UNREACHABLE"
        else:  # ring-shape only (operational disabled): no RING_FLIP_MISS available
            if out["start_ring_ok"]:
                out["classification"] = "DIHEDRAL_MISS"
            elif out["seed_pool_hit"]:
                out["classification"] = "RING_MISS__IN_SEED_POOL"
            elif out["capability_hit"]:
                out["classification"] = "RING_MISS__ETKDG_REACHABLE"
            else:
                out["classification"] = "RING_MISS__ETKDG_UNREACHABLE"

        # -- optional geometry dump for eyeballing --------------------------------
        if task.get("dump_dir"):
            from ase.io import write
            dd = Path(task["dump_dir"]); dd.mkdir(parents=True, exist_ok=True)
            write(dd / f"{mid}_crest_min.xyz", crest_min)
            write(dd / f"{mid}_bo_best.xyz", bo_best)
    except Exception as e:  # keep the batch alive
        out["note"] = f"error: {e!r}"
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def load_manifest(path: Path) -> dict:
    import pandas as pd
    man = pd.read_csv(path)
    key = "mol_id"
    smi = "raw_smiles" if "raw_smiles" in man.columns else "smiles"
    return {r[key]: {"smiles": r[smi], "d": r.get("d"), "cls": r.get("cls")}
            for _, r in man.iterrows()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference-dir", type=Path, required=True,
                    help="dir of reference.py <mol_id>.json (has E_star_crest / E_star_BO)")
    ap.add_argument("--geom-dir", type=Path, required=True,
                    help="bouquet geometry trails (*_<mol_id>_seed*.xyz)")
    ap.add_argument("--crest-dir", type=Path, default=None,
                    help="ensemble_bench CREST cells (<mol_id>/<crest-glob>). Omit and pass "
                         "--crest-from-reference when the raw cells aren't available.")
    ap.add_argument("--crest-glob", default="crest_*/refined.xyz",
                    help="glob under <crest-dir>/<mol_id> (default: %(default)s)")
    ap.add_argument("--crest-from-reference", action="store_true",
                    help="take the CREST-winning geometry from the global-min frame of "
                         "reference/<mol_id>.xyz instead of the raw CREST cells (valid "
                         "because the global min of a CREST-wins molecule IS the CREST "
                         "minimum). Use when --crest-dir cells are not on hand.")
    ap.add_argument("--manifest", type=Path, required=True,
                    help="CSV with mol_id + raw_smiles (select_smiles / autosteps-calib)")
    ap.add_argument("--method", default="gfn2", help="relaxation surface (default gfn2)")
    ap.add_argument("--out", type=Path, required=True, help="output CSV")
    ap.add_argument("--dump-dir", type=Path, default=None,
                    help="if set, write crest-min and bouquet-best XYZ per molecule")
    ap.add_argument("--gap-kcal", type=float, default=0.5,
                    help="min kcal/mol CREST must beat bouquet by to be analysed")
    ap.add_argument("--ring-rmsd", type=float, default=0.125,
                    help="ring-atom RMSD (A) below which a pucker counts as reached")
    ap.add_argument("--run-seeds", type=int, nargs="+", default=[1, 2, 3],
                    help="seeds the benchmark actually ran (for the seed-pool probe)")
    ap.add_argument("--cap-seeds", type=int, default=5,
                    help="expanded-capability probe: ETKDG seeds 1..N (default 5)")
    ap.add_argument("--cap-confs", type=int, default=16,
                    help="max ETKDG embeddings per seed (default 16, = production cap)")
    ap.add_argument("--crest-max", type=int, default=60,
                    help="cap on CREST cell minima to re-optimize per molecule")
    ap.add_argument("--no-operational", dest="operational", action="store_false",
                    help="skip the operational reachability test (faster, ring-SHAPE only). "
                         "The operational test -- dial each embedding's rotatable dihedrals "
                         "to the CREST-min values, apply bouquet's own constrained-then-"
                         "unconstrained relaxation, and measure full iRMSD to CREST-min -- "
                         "is ON by default and drives the classification, including the "
                         "RING_FLIP_MISS (right ring shape, wrong flip / axial-vs-equatorial) "
                         "category. It costs ~2 extra GFN2 relaxations per tested embedding; "
                         "disabling it removes RING_FLIP_MISS (a flip then reads as "
                         "DIHEDRAL_MISS, since ring-shape iRMSD merges the two chairs).")
    ap.set_defaults(operational=True)
    ap.add_argument("--op-rmsd", type=float, default=0.125,
                    help="full-molecule iRMSD (A) below which the operational move counts "
                         "as reaching the CREST minimum (default 0.125)")
    ap.add_argument("--op-max-confs", type=int, default=8,
                    help="operational test runs on the ring-closest N embeddings per seed "
                         "(default 8; bounds the extra relaxations)")
    ap.add_argument("--conn-tol", type=float, default=1.3,
                    help="bond-perception tolerance for the connectivity guard")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mols", default=None, help="comma-separated mol_id subset")
    args = ap.parse_args()

    if not args.crest_from_reference and args.crest_dir is None:
        sys.exit("need --crest-dir OR --crest-from-reference to locate the CREST geometry")

    manifest = load_manifest(args.manifest)
    want = set(args.mols.split(",")) if args.mols else None

    def _run_completed(trail_paths) -> bool:
        """A completed bouquet run wrote a 'final' frame (its final unconstrained
        relaxation). Censored/killed runs (common at high d, not yet finished) have
        none -- exclude them so we never diagnose an unfinished search."""
        for tp in trail_paths:
            try:
                with open(tp) as fh:
                    if any("kind=final" in ln for ln in fh):
                        return True
            except OSError:
                pass
        return False

    # Select CREST-wins molecules from the reference JSONs.
    tasks = []
    skipped_no_crest = skipped_bo_wins = skipped_incomplete = 0
    for jf in sorted(glob.glob(str(args.reference_dir / "*.json"))):
        o = json.loads(Path(jf).read_text())
        mid = o.get("mol_id")
        if not o.get("ok") or mid is None:
            continue
        if want and mid not in want:
            continue
        ec, eb = o.get("E_star_crest"), o.get("E_star_BO")
        if ec is None:
            skipped_no_crest += 1; continue
        if eb is None:
            continue
        gap = (eb - ec) * KCAL  # >0 => CREST lower => CREST wins
        if gap < args.gap_kcal:
            skipped_bo_wins += 1; continue
        if mid not in manifest:
            continue
        m = manifest[mid]
        trails = sorted(glob.glob(str(args.geom_dir / f"*_{mid}_seed*.xyz")))
        if not trails:
            continue
        if not _run_completed(trails):
            skipped_incomplete += 1; continue   # censored / not-yet-finished run
        if args.crest_from_reference:
            ref_xyz = args.reference_dir / f"{mid}.xyz"
            if not ref_xyz.exists():
                continue
            crest_files = []
        else:
            crest_files = sorted(glob.glob(str(args.crest_dir / mid / args.crest_glob)))
            ref_xyz = None
            if not crest_files:
                continue
        tasks.append({
            "mol_id": mid, "smiles": m["smiles"], "d": m.get("d"), "cls": m.get("cls"),
            "gap_kcal": round(gap, 3), "E_star_BO": eb, "E_star_crest": ec,
            "method": args.method, "trail_files": trails, "crest_files": crest_files,
            "crest_from_reference": args.crest_from_reference,
            "ref_xyz": str(ref_xyz) if ref_xyz else None,
            "run_seeds": args.run_seeds, "cap_seeds": args.cap_seeds,
            "cap_confs": args.cap_confs, "crest_max": args.crest_max,
            "ring_rmsd_thr": args.ring_rmsd, "conn_tol": args.conn_tol,
            "operational": args.operational, "op_rmsd": args.op_rmsd,
            "op_max_confs": args.op_max_confs,
            "dump_dir": str(args.dump_dir) if args.dump_dir else None,
        })

    print(f"CREST-wins candidates: {len(tasks)}  (skipped: {skipped_no_crest} no CREST ref, "
          f"{skipped_bo_wins} bouquet>=CREST, {skipped_incomplete} incomplete/censored run)")
    if not tasks:
        print("Nothing to analyse."); return

    if not _HAVE_IRMSD:
        print("WARNING: iRMSD wheel not importable; ring metric falls back to Kabsch, "
              "which over-reports mirror-image puckers. Install the 'irmsd' extra "
              "(pixi -e irmsd / -e all) before trusting RING_MISS calls.")

    cols = ["mol_id", "d", "cls", "gap_kcal", "E_star_BO", "E_star_crest",
            "n_flex_ring_atoms", "classification",
            "start_ring_ok", "start_ring_rmsd_min", "bo_best_ring_rmsd",
            "bo_best_full_irmsd", "seed_pool_hit", "seed_pool_ring_rmsd",
            "capability_hit", "capability_ring_rmsd"]
    if args.operational:
        cols += ["op_start_ok", "op_start_irmsd_min", "op_seed_pool_hit",
                 "op_seed_pool_irmsd", "op_capability_hit", "op_capability_irmsd"]
    cols += ["dih_max_deg", "dih_mean_deg", "note"]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(diagnose, tasks):
            rows.append(r)
            print(f"  {r['mol_id']:28} {str(r['classification']):28} "
                  f"gap={r['gap_kcal']:+.2f}  {r.get('note','')}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # Summary tally.
    from collections import Counter
    tally = Counter(r["classification"] for r in rows if r["classification"])
    print("\n=== classification summary ===")
    for k in ("DIHEDRAL_MISS", "RING_FLIP_MISS", "RING_MISS__IN_SEED_POOL",
              "RING_MISS__ETKDG_REACHABLE", "RING_MISS__ETKDG_UNREACHABLE"):
        print(f"  {k:32} {tally.get(k, 0)}")
    errs = [r for r in rows if not r["classification"]]
    if errs:
        print(f"  {'(unanalysed)':32} {len(errs)}")
    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
