#!/usr/bin/env python
"""Benchmark ring-MTD seed recall against the CREST reference, across ring chemistries.

The question this answers is narrow and cheap on purpose: **does the RMSD-biased ring
metadynamics put the CREST global minimum's ring pucker into bouquet's seed set, where
ETKDG cannot?** It does NOT run the BO -- that is the follow-on end-to-end arm. Keeping
the capability question separate is what makes a broad sweep affordable, and it sidesteps
the fact that most d>=7 bouquet runs in stop_bench_gfn2 are right-censored (no `kind=final`
frame), which is why `crest_wins_diagnosis.py` left 55 of 82 CREST wins unclassified.

Per molecule, three arms scored with ONE metric -- iRMSD over the flexible (non-aromatic)
ring heavy atoms to the CREST global-min geometry, exactly as `crest_wins_diagnosis.py` and
`etkdg_ring_stress.py` define it:

  * ``etkdg``  -- bouquet's PRODUCTION embedding pool: ``get_initial_candidates`` over the
    run seeds x ``--cap-confs``. This is the incumbent, i.e. what bouquet gets today.
  * ``mtd``    -- ``run_ring_mtd`` seeds (the ring_seeds/1.0 set bouquet would consume).
  * ``mtd@k``  -- the seeds bouquet would actually start from: the first ``--seed-budget``
    under ``--select-rule``. Reported separately because the raw seed set can be large (57
    seeds on a 4-ring-system spiro, ~100 with the current pre-filter) and bouquet runs one
    BO per seed, so a hit that only appears at rank 40 is not a hit bouquet can afford.
    ``mtd_hit_rank`` stays the ENERGY rank regardless of the rule in force, which is what
    keeps every other rule scorable after the fact (`scripts/ring_seed_selection.py`).

Scoring protocol is identical across arms (apples-to-apples): screen all candidates by ring
iRMSD on their own geometry, GFN2-relax the ``--gfn2-cap`` closest, take the min. Hits are
reported at BOTH thresholds in play -- 0.125 A (the ensemble dedup tolerance the two existing
diagnostics use) and 0.25 A (RING_MTD_v2 Section 11.5's acceptance target) -- because the
proof-of-concept's best spiro result (0.157 A) falls between them, so the verdict depends on
which one is meant.

Molecules are stratified by ring class (SPIRO / BRIDGED_CAGE / MACROCYCLE / MEDIUM_RING /
FUSED / SIMPLE), since the proof-of-concept verdict was chemistry-gated. ``--all-flexible``
selects every molecule in the manifest that has a flexible ring AND a CREST reference
(166 of stopbench-500), not only the CREST-wins subset: molecules bouquet already wins are
the control arm for the Section 0.3 gating risk -- the MTD must not pay mini-CREST cost
where ETKDG already covers the rings.

Parallelism is per MOLECULE (``--workers``); the MTD runs its walkers sequentially inside
each task (``n_processes=1``) because a pool worker cannot spawn its own pool. Every worker
drives xtb, which is pinned to a single thread (see the env block below) -- so ``--workers``
is a real core count, not a nominal one. Keep it well under the machine's cores on a laptop;
the full sweep belongs on a node via ``scripts/ring_mtd_bench.slurm``.

Usage:
    pixi run python scripts/ring_mtd_bench.py \
        --reference-dir stop_bench_gfn2/reference_gfn2 \
        --manifest smiles/stopbench-500.csv \
        --all-flexible --workers 2 \
        --seeds-dir stop_bench_gfn2/analysis/ring_mtd_seeds \
        --out stop_bench_gfn2/analysis/ring_mtd_bench.csv
"""

from __future__ import annotations

import os

# Pin the numerical libraries to one thread BEFORE numpy/xtb load, matching
# sweep_common.single_threaded_env(). Every worker here runs its own xtb calls, and xtb's
# OpenMP otherwise grabs every core per worker -- --workers 8 then oversubscribes the machine
# by ~8x and thrashes it (this is not hypothetical; it took a laptop down). Set in the parent
# so pool children inherit it under both fork and spawn.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crest_wins_diagnosis as C  # noqa: E402  (ring_rmsd, flexible_ring_atoms, _read_reference_min)
import reference as ref           # noqa: E402  (_gfnff_calc, _optimize)
from bouquet.rings.select import RULES, select_seeds  # noqa: E402

# Both thresholds are reported; see the module docstring.
TOL_STRICT = 0.125
TOL_LOOSE = 0.25


def ring_class(mol) -> str:
    """Coarse ring-chemistry stratum of a molecule (its most demanding flexible system).

    Ordered most- to least-demanding so a molecule lands in the class that drives its
    difficulty: a spiro macrocycle is scored as a macrocycle. Aromatic-only molecules are
    NO_FLEX_RING and are excluded from the benchmark upstream.
    """
    ri = mol.GetRingInfo()
    rings = [frozenset(r) for r in ri.AtomRings()]
    flex = [r for r in rings if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    if not flex:
        return "NO_FLEX_RING"
    maxsize = max(len(r) for r in flex)
    pairs = [(a, b) for i, a in enumerate(rings) for b in rings[i + 1:]]
    if maxsize >= 10:
        return "MACROCYCLE"
    if any(len(a & b) > 2 for a, b in pairs):
        return "BRIDGED_CAGE"
    if any(len(a & b) == 1 for a, b in pairs):
        return "SPIRO"
    if maxsize >= 8:
        return "MEDIUM_RING"
    if any(len(a & b) == 2 for a, b in pairs):
        return "FUSED"
    return "SIMPLE"


def _score_pool(candidates, crest_min, ring_idx, calc, charge, gfn2_cap):
    """Screen candidates by ring iRMSD, GFN2-relax the closest, return the scoring summary.

    Returns ``(min_raw, min_gfn2, best_index)`` where ``best_index`` is the position in the
    INPUT list of the candidate whose relaxation gave ``min_gfn2`` (-1 if none relaxed).
    The screen-then-relax split is the same bounded-confirmation protocol
    ``etkdg_ring_stress.py`` uses: floppy ring systems are too expensive to relax wholesale,
    and a candidate far away before relaxation does not fall into the CREST basin after it.
    """
    from bouquet.setup import apply_charge_spin, default_multiplicity

    if not candidates:
        return float("inf"), float("inf"), -1
    screened = sorted(((C.ring_rmsd(c, crest_min, ring_idx), i) for i, c in enumerate(candidates)),
                      key=lambda t: t[0])
    mult = default_multiplicity(candidates[0], charge)
    best_gfn2, best_idx = float("inf"), -1
    for _r, i in screened[:gfn2_cap]:
        a = candidates[i].copy()
        apply_charge_spin(a, charge, mult)
        opt = ref._optimize(a, calc)
        if opt is None:
            continue
        r = C.ring_rmsd(opt[0], crest_min, ring_idx)
        if r < best_gfn2:
            best_gfn2, best_idx = r, i
    return screened[0][0], best_gfn2, best_idx


def _mtd_seeds(task):
    """Run (or load from ``--seeds-dir``) the ring MTD; return (atoms_list, meta).

    Seeds are cached on disk so re-scoring (a new threshold, a new seed budget) never pays
    the metadynamics again -- the sweep's expensive half.
    """
    from rdkit import Chem
    from bouquet.rings.mtd import RingMTDConfig, run_ring_mtd
    from bouquet.rings.seeds import read_seeds, write_seeds

    cache = Path(task["seeds_dir"]) / f"{task['mol_id']}.json" if task["seeds_dir"] else None
    if cache is not None and cache.exists():
        s = read_seeds(cache)
        d = s.data["diagnostics"]
        meta = {"n_seeds": len(s.seeds), "wall_s": s.data["method"].get("wall_seconds"),
                "select": lambda k, rule, _s=s: select_seeds(_s.seeds, k, rule, _s.ring_idx),
                "cached": True, "stereo_reject_rate": d.get("stereo_reject_rate"),
                "k_hill_final": d.get("k_hill_final"),
                "bias_rounds": len(d.get("bias_rounds") or []),
                "n_ring_systems": len(s.data["ring_systems"])}
        return s.conformers(), meta

    mol = Chem.MolFromSmiles(task["smiles"])
    cfg = RingMTDConfig(engine=task["mtd_engine"], t_run_ps=task["t_run_ps"],
                        n_walkers=task["walkers"], n_processes=1, seed=task["seed"])
    t0 = time.time()
    result = run_ring_mtd(mol, cfg)
    wall = time.time() - t0
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        write_seeds(cache, mol, result.systems, result, cfg, wall_seconds=wall)
    ring_idx = sorted({int(i) for sysm in result.systems for i in sysm.ring_idx})
    meta = {"n_seeds": len(result.seeds), "wall_s": round(wall, 1), "cached": False,
            "select": lambda k, rule, _r=result, _ri=ring_idx: select_seeds(
                [{"ring_state": s.ring_state, "rel_energy_kcal": s.rel_energy_kcal,
                  "cluster_size": s.cluster_size,
                  "coords": s.atoms.get_positions().tolist()} for s in _r.seeds],
                k, rule, _ri),
            "stereo_reject_rate": result.diagnostics.get("stereo_reject_rate"),
            "n_ring_systems": result.diagnostics.get("n_ring_systems"),
            "k_hill_final": result.diagnostics.get("k_hill_final"),
            "bias_rounds": len(result.diagnostics.get("bias_rounds") or []),
            "skipped": result.diagnostics.get("skipped")}
    # Seeds come out ranked by energy already (harvest.rank_and_window), so list position
    # IS the energy rank the --seed-budget truncation would apply.
    return [s.atoms for s in result.seeds], meta


def bench(task: dict) -> dict:
    # Re-assert the pin inside the worker: a spawned child re-imports this module, and any
    # environment the parent did not pass through would otherwise let xtb go multi-threaded.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"

    from rdkit import Chem
    from bouquet.setup import get_initial_candidates

    mid = task["mol_id"]
    row = {"mol_id": mid, "d": task["d"]}
    try:
        mol = Chem.MolFromSmiles(task["smiles"])
        molh = Chem.AddHs(mol)
        charge = Chem.GetFormalCharge(mol)
        ring_idx = C.flexible_ring_atoms(molh)
        crest_min = C._read_reference_min(task["ref_xyz"])
        row.update(ring_class=ring_class(mol), n_flex_ring_atoms=len(ring_idx),
                   crest_wins=task["crest_wins"], gap_kcal=task["gap_kcal"])
        if not ring_idx or crest_min is None:
            row["note"] = "no flexible ring / no reference min"
            return row

        calc = ref._gfnff_calc(task["method"], molh, charge)

        # -- arm 1: production ETKDG pool (what bouquet starts from today) ----------
        etkdg = []
        for s in range(1, task["run_seeds"] + 1):
            cands, _m = get_initial_candidates(task["smiles"], seed=s,
                                               max_confs=task["cap_confs"])
            etkdg += list(cands)
        e_raw, e_gfn2, _ = _score_pool(etkdg, crest_min, ring_idx, calc, charge,
                                       task["gfn2_cap"])
        row.update(etkdg_n=len(etkdg), etkdg_min_ring_irmsd_raw=round(e_raw, 4),
                   etkdg_min_ring_irmsd_gfn2=round(e_gfn2, 4),
                   etkdg_hit_125=bool(e_gfn2 < TOL_STRICT),
                   etkdg_hit_250=bool(e_gfn2 < TOL_LOOSE))

        # -- arm 2: ring-MTD seeds --------------------------------------------------
        seeds, meta = _mtd_seeds(task)
        row.update(mtd_n_seeds=meta["n_seeds"], mtd_wall_s=meta["wall_s"],
                   mtd_cached=meta["cached"], mtd_n_ring_systems=meta.get("n_ring_systems"),
                   mtd_stereo_reject_rate=meta.get("stereo_reject_rate"),
                   mtd_k_hill_final=meta.get("k_hill_final"),
                   mtd_bias_rounds=meta.get("bias_rounds"))
        if not seeds:
            row["note"] = f"no MTD seeds ({meta.get('skipped') or 'empty harvest'})"
            return row
        m_raw, m_gfn2, m_idx = _score_pool(seeds, crest_min, ring_idx, calc, charge,
                                           task["gfn2_cap"])
        row.update(mtd_min_ring_irmsd_raw=round(m_raw, 4),
                   mtd_min_ring_irmsd_gfn2=round(m_gfn2, 4),
                   mtd_hit_125=bool(m_gfn2 < TOL_STRICT),
                   mtd_hit_250=bool(m_gfn2 < TOL_LOOSE),
                   mtd_hit_rank=m_idx)

        # -- arm 3: the same seeds under a deliverable budget -----------------------
        # What bouquet would actually start from: the first k under the selection rule. A
        # hit outside it means the MTD FOUND the ring state but bouquet could not afford to
        # start from it. `mtd_hit_rank` is still the ENERGY rank, so every other rule stays
        # scorable offline from this CSV (scripts/ring_seed_selection.py).
        k, rule = task["seed_budget"], task["select_rule"]
        kept = set(meta["select"](k, rule)) if meta.get("select") else set(range(k))
        in_budget = m_idx in kept
        row.update(mtd_budget_k=k, mtd_select_rule=rule,
                   mtd_budget_hit_125=bool(in_budget and m_gfn2 < TOL_STRICT),
                   mtd_budget_hit_250=bool(in_budget and m_gfn2 < TOL_LOOSE))
    except Exception as e:  # one bad molecule must not sink the sweep
        row["note"] = f"error: {e!r}"
    return row


COLS = ["mol_id", "d", "ring_class", "n_flex_ring_atoms", "crest_wins", "gap_kcal",
        "etkdg_n", "etkdg_min_ring_irmsd_raw", "etkdg_min_ring_irmsd_gfn2",
        "etkdg_hit_125", "etkdg_hit_250",
        "mtd_n_seeds", "mtd_n_ring_systems", "mtd_wall_s", "mtd_cached",
        "mtd_stereo_reject_rate", "mtd_k_hill_final", "mtd_bias_rounds",
        "mtd_min_ring_irmsd_raw", "mtd_min_ring_irmsd_gfn2",
        "mtd_hit_125", "mtd_hit_250", "mtd_hit_rank",
        "mtd_budget_k", "mtd_select_rule", "mtd_budget_hit_125", "mtd_budget_hit_250",
        "note"]


def summarize(rows) -> None:
    """Per-class recall table: the MTD only earns its cost where it beats ETKDG."""
    import collections
    agg = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        cls = r.get("ring_class")
        if not cls or r.get("mtd_min_ring_irmsd_gfn2") is None:
            continue
        a = agg[cls]
        a["n"] += 1
        for key in ("etkdg_hit_125", "etkdg_hit_250", "mtd_hit_125", "mtd_hit_250",
                    "mtd_budget_hit_125", "mtd_budget_hit_250"):
            a[key] += bool(r.get(key))
        a["wall"] += r.get("mtd_wall_s") or 0
        a["seeds"] += r.get("mtd_n_seeds") or 0
    print(f"\n{'class':<14} {'n':>3} {'etkdg@.125':>11} {'mtd@.125':>9} {'mtd@.25':>8} "
          f"{'budget@.125':>12} {'med_seeds':>10} {'mean_wall_s':>12}")
    for cls, a in sorted(agg.items(), key=lambda kv: -kv[1]["n"]):
        n = a["n"]
        print(f"{cls:<14} {n:>3} {a['etkdg_hit_125']:>6}/{n:<4} {a['mtd_hit_125']:>4}/{n:<4} "
              f"{a['mtd_hit_250']:>3}/{n:<4} {a['mtd_budget_hit_125']:>7}/{n:<4} "
              f"{a['seeds']/n:>10.1f} {a['wall']/n:>12.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mols", help="comma-separated mol_id list")
    g.add_argument("--all-flexible", action="store_true",
                   help="every manifest molecule with a flexible ring and a CREST reference")
    ap.add_argument("--crest-wins", type=Path, default=None,
                    help="crest_wins.csv; annotates rows with gap_kcal / crest_wins")
    ap.add_argument("--wins-only", action="store_true",
                    help="restrict --all-flexible to the CREST-wins subset")
    ap.add_argument("--method", default="gfn2", help="scoring/relaxation surface (default gfn2)")
    ap.add_argument("--mtd-engine", default="gfnff", help="MTD walker engine (default gfnff)")
    ap.add_argument("--t-run-ps", type=float, default=20.0)
    ap.add_argument("--walkers", type=int, default=4)
    ap.add_argument("--seed-budget", type=int, default=8,
                    help="seeds bouquet would actually start from (default 8)")
    ap.add_argument("--select-rule", default="energy", choices=list(RULES),
                    help="rule choosing which seeds fall inside --seed-budget (default "
                         "hybrid). mtd_hit_rank stays the ENERGY rank, so the other rules "
                         "remain scorable offline via scripts/ring_seed_selection.py")
    ap.add_argument("--run-seeds", type=int, default=3, help="ETKDG arm: bouquet run seeds")
    ap.add_argument("--cap-confs", type=int, default=16,
                    help="ETKDG arm: embeddings per seed (default 16 = production cap)")
    ap.add_argument("--gfn2-cap", type=int, default=12,
                    help="GFN2-relax this many ring-closest candidates per arm")
    ap.add_argument("--seeds-dir", type=Path, default=None,
                    help="cache dir for ring-seed JSON (reused instead of re-running the MTD)")
    ap.add_argument("--seed", type=int, default=0xC0FFEE)
    ap.add_argument("--workers", type=int, default=2,
                    help="molecule-level parallelism. Each worker runs xtb pinned to one "
                         "thread; on a laptop stay well under the core count (default 2)")
    ap.add_argument("--shard", type=int, default=0,
                    help="run only molecules with (index %% --nshards) == --shard")
    ap.add_argument("--nshards", type=int, default=1,
                    help="split the selection into N interleaved shards (SLURM array axis)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not 0 <= args.shard < args.nshards:
        ap.error(f"--shard must be in [0, {args.nshards}), got {args.shard}")
    for name, val in (("--walkers", args.walkers), ("--gfn2-cap", args.gfn2_cap),
                      ("--workers", args.workers), ("--seed-budget", args.seed_budget)):
        if val <= 0:
            ap.error(f"{name} must be positive, got {val}")

    import pandas as pd
    from rdkit import Chem

    man = pd.read_csv(args.manifest).set_index("mol_id")
    smi_col = "raw_smiles" if "raw_smiles" in man.columns else "smiles"
    wins = {}
    if args.crest_wins and args.crest_wins.exists():
        wins = {r["mol_id"]: r for r in csv.DictReader(open(args.crest_wins))}

    ids = [m.strip() for m in args.mols.split(",")] if args.mols else list(man.index)
    tasks = []
    for mid in ids:
        if mid not in man.index:
            print(f"  {mid}: not in manifest, skipping"); continue
        ref_xyz = args.reference_dir / f"{mid}.xyz"
        if not ref_xyz.exists():
            continue  # no CREST reference -> nothing to score against
        smiles = man.loc[mid, smi_col]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or ring_class(mol) == "NO_FLEX_RING":
            continue
        if args.wins_only and mid not in wins:
            continue
        w = wins.get(mid)
        tasks.append({"mol_id": mid, "smiles": smiles,
                      "d": int(man.loc[mid, "d"]) if "d" in man.columns else None,
                      "ref_xyz": str(ref_xyz), "method": args.method,
                      "mtd_engine": args.mtd_engine, "t_run_ps": args.t_run_ps,
                      "walkers": args.walkers, "seed_budget": args.seed_budget,
                      "run_seeds": args.run_seeds, "cap_confs": args.cap_confs,
                      "gfn2_cap": args.gfn2_cap, "seed": args.seed,
                      "select_rule": args.select_rule,
                      "seeds_dir": str(args.seeds_dir) if args.seeds_dir else None,
                      "crest_wins": bool(w),
                      "gap_kcal": float(w["gap_kcal"]) if w else None})

    # Interleaved sharding (molecule i -> shard i mod N) keeps every shard's ring-class and
    # size mix representative, so no array task is systematically the expensive one.
    if args.nshards > 1:
        tasks = [t for i, t in enumerate(tasks) if i % args.nshards == args.shard]
        print(f"shard {args.shard}/{args.nshards}: {len(tasks)} molecule(s)")

    print(f"Ring-MTD bench on {len(tasks)} molecule(s): ETKDG({args.run_seeds}x{args.cap_confs}) "
          f"vs MTD({args.walkers} walkers x {args.t_run_ps} ps, {args.mtd_engine}), "
          f"scored on {args.method}")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(bench, tasks):
            rows.append(r)
            print(f"  {r['mol_id']:12} {r.get('ring_class','-'):<13} "
                  f"etkdg={r.get('etkdg_min_ring_irmsd_gfn2','-'):>7} "
                  f"mtd={r.get('mtd_min_ring_irmsd_gfn2','-'):>7} "
                  f"seeds={r.get('mtd_n_seeds','-'):>3} rank={r.get('mtd_hit_rank','-'):>3} "
                  f"{r.get('mtd_wall_s','-'):>7}s  {r.get('note','')}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")
    summarize(rows)


if __name__ == "__main__":
    main()
