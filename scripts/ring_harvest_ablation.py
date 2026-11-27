#!/usr/bin/env python
"""Paired ablation of a harvest parameter: one MD trajectory, several harvest settings.

Motivating question (RING_MTD_BENCH §9.9): the state-rich molecules regressed when
``max_coarse`` was introduced -- m047 0.280→0.444, m070 0.459→0.867, m122 0.480→0.689, with
seed counts falling -- but those runs were also different stochastic draws, and the §6
draw-to-draw spread is larger than the regressions. Independent arms cannot separate the two.

So this PAIRS them: run the walkers ONCE per (molecule, seed), keep the frames, and harvest
those SAME frames under each setting. Draw noise is then common to both arms and cancels
exactly -- any difference is the parameter. It also halves the cost, since the metadynamics
runs once instead of once per arm.

Scoring is the bench's: ring iRMSD to the CREST global-min geometry, screened on the seed
geometry then GFN2-relaxed (``--gfn2-cap`` closest), so numbers are comparable with
``ring_mtd_bench.py``.

Usage:
    pixi run python scripts/ring_harvest_ablation.py \
        --reference-dir stop_bench_gfn2/reference_gfn2 \
        --manifest smiles/stopbench-500.csv \
        --mols m047_d02,m070_d03,m121_d05,m122_d05 \
        --seeds 0,1,2 --max-coarse 60,180 \
        --workers 6 --out stop_bench_gfn2/analysis/ring_budget_ablation.csv
"""

from __future__ import annotations

import os

# Before numpy/xtb load; each worker drives its own xtb (see ring_mtd_bench.py).
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
import crest_wins_diagnosis as C  # noqa: E402
import reference as ref  # noqa: E402
from ring_mtd_bench import TOL_LOOSE, TOL_STRICT, _score_pool, ring_class  # noqa: E402


def run_one(task: dict) -> list[dict]:
    """One (molecule, MD seed): walk once, harvest under each --max-coarse, score each."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"

    from rdkit import Chem

    from bouquet.calculator import CalculatorFactory
    from bouquet.rings.harvest import harvest, quench, stereo_retained, stereo_signature
    from bouquet.rings.mtd import RingMTDConfig, _run_walker, seed_walkers
    from bouquet.rings.systems import perceive_ring_systems

    mid = task["mol_id"]
    base = {"mol_id": mid, "md_seed": task["seed"]}
    rows = []
    try:
        mol = Chem.MolFromSmiles(task["smiles"])
        molh = Chem.AddHs(mol)
        charge = Chem.GetFormalCharge(mol)
        systems = perceive_ring_systems(molh)
        ring_idx = C.flexible_ring_atoms(molh)
        crest_min = C._read_reference_min(task["ref_xyz"])
        base.update(ring_class=ring_class(mol), n_ring_systems=len(systems))
        if not systems or not ring_idx or crest_min is None:
            return [dict(base, note="no ring system / no reference min")]

        cfg = RingMTDConfig(n_walkers=task["walkers"], t_run_ps=task["t_run_ps"],
                            n_processes=1, seed=task["seed"], adaptive_bias=False)
        engine = CalculatorFactory.create(method=cfg.engine, mol=molh, charge=charge)

        # --- the shared half: walkers run ONCE ------------------------------------
        t0 = time.time()
        starts, mult = seed_walkers(molh, systems, cfg, charge)
        starts = [q[0] for q in (quench(s.copy(), engine, fmax=cfg.quench_fmax,
                                        steps=cfg.quench_maxsteps) for s in starts) if q]
        sig = stereo_signature(molh, starts[0])
        starts = [s for s in starts if stereo_retained(molh, s, sig)] or [starts[0]]
        frames = []
        for w, s in enumerate(starts):
            frames += _run_walker({
                "mol_binary": molh.ToBinary(), "numbers": s.get_atomic_numbers(),
                "positions": s.get_positions(), "charge": charge, "mult": mult,
                "cfg": cfg, "systems": systems, "vseed": cfg.seed * 1000 + w,
            })
        md_wall = time.time() - t0
        base.update(n_frames=len(frames), n_walkers=len(starts), md_wall_s=round(md_wall, 1))

        # --- the varied half: same frames, each budget -----------------------------
        scorer = CalculatorFactory.create(method=task["method"], mol=molh, charge=charge)
        for mc in task["max_coarse"]:
            t1 = time.time()
            res = harvest(frames, molh, systems, engine, start_signature=sig,
                          coarse_tol=cfg.coarse_dedup_A, fine_tol=cfg.fine_dedup_A,
                          max_coarse=mc, temperature=cfg.harvest_temperature_K,
                          energy_window_kcal=cfg.energy_window_kcal,
                          fmax=cfg.quench_fmax, steps=cfg.quench_maxsteps,
                          topology_check=cfg.topology_check)
            harvest_wall = time.time() - t1
            row = dict(base, max_coarse=mc, n_seeds=len(res.seeds),
                       n_after_coarse=res.diagnostics.get("n_after_coarse_dedup"),
                       n_clusters=res.diagnostics.get("n_clusters"),
                       harvest_wall_s=round(harvest_wall, 1))
            if not res.seeds:
                row["note"] = "no seeds"
                rows.append(row)
                continue
            seeds = [s.atoms for s in res.seeds]
            raw, gfn2, idx = _score_pool(seeds, crest_min, ring_idx, scorer, charge,
                                         task["gfn2_cap"])
            row.update(min_ring_irmsd_raw=round(raw, 4), min_ring_irmsd_gfn2=round(gfn2, 4),
                       hit_125=bool(gfn2 < TOL_STRICT), hit_250=bool(gfn2 < TOL_LOOSE),
                       hit_rank=idx)
            rows.append(row)
    except Exception as e:
        rows.append(dict(base, note=f"error: {e!r}"))
    return rows


COLS = ["mol_id", "md_seed", "ring_class", "n_ring_systems", "n_walkers", "n_frames",
        "md_wall_s", "max_coarse", "n_after_coarse", "n_clusters", "n_seeds",
        "harvest_wall_s", "min_ring_irmsd_raw", "min_ring_irmsd_gfn2", "hit_125",
        "hit_250", "hit_rank", "note"]


def summarize(rows, budgets) -> None:
    """Paired comparison: for each (molecule, seed), the same frames under each budget."""
    import statistics

    print(f"\n{'molecule':<12}{'seed':>5}{'frames':>8}" +
          "".join(f"{'mc=' + str(b):>22}" for b in budgets))
    print(f"{'':<12}{'':>5}{'':>8}" + "".join(f"{'seeds / ring-iRMSD':>22}" for b in budgets))
    print("-" * (25 + 22 * len(budgets)))
    paired = {}
    for key in sorted({(r["mol_id"], r["md_seed"]) for r in rows if "max_coarse" in r}):
        cells, vals = [], {}
        for b in budgets:
            m = next((r for r in rows if (r["mol_id"], r["md_seed"]) == key
                      and r.get("max_coarse") == b), None)
            if m and m.get("min_ring_irmsd_gfn2") is not None:
                cells.append(f"{m['n_seeds']:>6} / {m['min_ring_irmsd_gfn2']:<13.4f}")
                vals[b] = m["min_ring_irmsd_gfn2"]
            else:
                cells.append(f"{'-':>22}")
        f = next((r["n_frames"] for r in rows if (r["mol_id"], r["md_seed"]) == key), "-")
        print(f"{key[0]:<12}{key[1]:>5}{f:>8}" + "".join(cells))
        if len(vals) == len(budgets):
            paired[key] = vals

    if not paired:
        return
    lo, hi = budgets[0], budgets[-1]
    diffs = [v[hi] - v[lo] for v in paired.values()]
    better = sum(d < -1e-9 for d in diffs)
    worse = sum(d > 1e-9 for d in diffs)
    print(f"\nPaired over {len(diffs)} (molecule, seed) run(s): raising max_coarse "
          f"{lo} -> {hi} helped {better}, hurt {worse}, tied {len(diffs) - better - worse}.")
    print(f"  mean change in ring-iRMSD: {statistics.mean(diffs):+.4f} A "
          f"(negative = the larger budget is closer to CREST)")
    if len(diffs) > 1:
        print(f"  stdev of the paired change: {statistics.stdev(diffs):.4f} A")
    print("  Because the frames are shared, this difference is the BUDGET, not draw noise.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--mols", required=True)
    ap.add_argument("--seeds", default="0,1,2", help="MD seeds (the draw axis)")
    ap.add_argument("--max-coarse", default="60,180",
                    help="harvest budgets to compare on the SAME frames (0 = unlimited)")
    ap.add_argument("--method", default="gfn2", help="scoring surface")
    ap.add_argument("--t-run-ps", type=float, default=20.0)
    ap.add_argument("--walkers", type=int, default=4)
    ap.add_argument("--gfn2-cap", type=int, default=12)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import pandas as pd

    man = pd.read_csv(args.manifest).set_index("mol_id")
    smi_col = "raw_smiles" if "raw_smiles" in man.columns else "smiles"
    budgets = [int(b) for b in args.max_coarse.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    tasks = []
    for mid in [m.strip() for m in args.mols.split(",")]:
        if mid not in man.index:
            print(f"  {mid}: not in manifest, skipping")
            continue
        ref_xyz = args.reference_dir / f"{mid}.xyz"
        if not ref_xyz.exists():
            print(f"  {mid}: no reference xyz, skipping")
            continue
        for s in seeds:
            tasks.append({"mol_id": mid, "smiles": man.loc[mid, smi_col], "seed": s,
                          "ref_xyz": str(ref_xyz), "method": args.method,
                          "t_run_ps": args.t_run_ps, "walkers": args.walkers,
                          "gfn2_cap": args.gfn2_cap, "max_coarse": budgets})

    print(f"Paired harvest ablation: {len(tasks)} (molecule, seed) run(s) x "
          f"max_coarse {budgets}; the MD runs once per pair.")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(run_one, tasks):
            rows += res
            for r in res:
                print(f"  {r['mol_id']:12} seed={r['md_seed']} mc={r.get('max_coarse','-'):>4} "
                      f"seeds={r.get('n_seeds','-'):>3} "
                      f"irmsd={r.get('min_ring_irmsd_gfn2','-'):>7} "
                      f"coarse={r.get('n_after_coarse','-'):>4}  {r.get('note','')}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")
    summarize(rows, budgets)


if __name__ == "__main__":
    main()
