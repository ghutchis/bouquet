#!/usr/bin/env python
"""Can the ETKDG pool's OWN ring diversity predict where the ring-MTD adds value?

This is the last cheap test of RING_MTD_BENCH §9.14. Stage 1 showed the MTD's entire value
sits in a narrow band -- molecules whose ETKDG pool lands 0.15-0.30 A from the CREST ring --
but that band is defined against the CREST answer, so it is an oracle, not a gate. A usable
gate must be computable from the molecule alone.

The candidate signal: how much distinct ring geometry does ETKDG itself produce? If a pool
collapses to one or two ring states, that is evidence the embedder is not covering the ring
manifold, which is exactly when a metadynamics preprocessor might pay. No QM, no CREST
reference -- RDKit embedding plus the project's iRMSD, so it is affordable to run on
everything and, if it works, affordable to run as a gate in production.

Signals computed per molecule (all a priori):
  * ``n_states``      distinct ring states among the production pool, greedy-clustered at
                      ``--tol`` on flexible-ring HEAVY atoms (the metric the benchmark scores
                      with, so "state" means the same thing here as there);
  * ``states_per_atom`` ``n_states`` / flexible ring atoms -- size-normalised, since a big
                      floppy ring trivially shows more states than a cyclopentane;
  * ``spread``        max pairwise ring iRMSD in the pool (how far apart its extremes are);
  * ``mean_dist``     mean pairwise ring iRMSD (how tightly it clusters).

Scored against the Stage-1 CSV: does any of them separate the molecules where the MTD wins
alone from the 55% where ETKDG already succeeds and the MTD is pure cost?

Usage:
    pixi run python scripts/ring_etkdg_diversity.py \
        --bench stop_bench_gfn2/analysis/stage1_final.csv \
        --manifest smiles/stopbench-500.csv \
        --workers 6 --out stop_bench_gfn2/analysis/ring_etkdg_diversity.csv
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import itertools  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crest_wins_diagnosis as C  # noqa: E402


def diversity(task: dict) -> dict:
    """Ring-diversity descriptors of the production ETKDG pool. No QM, no reference."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"

    from rdkit import Chem

    from bouquet.setup import get_initial_candidates

    mid = task["mol_id"]
    row = {"mol_id": mid}
    try:
        molh = Chem.AddHs(Chem.MolFromSmiles(task["smiles"]))
        ring_idx = C.flexible_ring_atoms(molh)
        row["n_flex_ring_atoms"] = len(ring_idx)
        if not ring_idx:
            row["note"] = "no flexible ring"
            return row

        pool = []
        for s in range(1, task["run_seeds"] + 1):
            cands, _m = get_initial_candidates(task["smiles"], seed=s,
                                               max_confs=task["cap_confs"])
            pool += list(cands)
        row["n_pool"] = len(pool)
        if len(pool) < 2:
            row["note"] = "pool too small"
            return row

        # Greedy clustering at --tol, the same construction (and the same metric) the
        # benchmark uses to decide whether two ring geometries are the same state.
        reps = []
        for c in pool:
            if all(C.ring_rmsd(c, r, ring_idx) >= task["tol"] for r in reps):
                reps.append(c)
        d = [C.ring_rmsd(a, b, ring_idx) for a, b in itertools.combinations(pool, 2)]
        row.update(n_states=len(reps),
                   states_per_atom=round(len(reps) / len(ring_idx), 4),
                   spread=round(max(d), 4),
                   mean_dist=round(sum(d) / len(d), 4))
    except Exception as e:
        row["note"] = f"error: {e!r}"
    return row


def report(rows, bench_rows):
    """Does any descriptor separate MTD-only wins from 'ETKDG already succeeded'?"""
    b = {r["mol_id"]: r for r in bench_rows}
    def T(r, k): return r.get(k) == "True"
    joined = []
    for r in rows:
        m = b.get(r["mol_id"])
        if not m or r.get("n_states") is None:
            continue
        joined.append({**r,
                       "etkdg_hit": T(m, "etkdg_hit_125"), "mtd_hit": T(m, "mtd_hit_125"),
                       "mtd_only": T(m, "mtd_hit_125") and not T(m, "etkdg_hit_125"),
                       "wall": float(m["mtd_wall_s"]) if m.get("mtd_wall_s") else 0.0})
    if not joined:
        print("nothing to report")
        return
    print(f"\njoined {len(joined)} molecules with Stage-1 outcomes")

    for key in ("n_states", "states_per_atom", "spread", "mean_dist"):
        vals = sorted(j[key] for j in joined)
        qs = [vals[int(q * (len(vals) - 1))] for q in (0.25, 0.5, 0.75)]
        print(f"\n--- {key} (quartiles {qs[0]:.3g} / {qs[1]:.3g} / {qs[2]:.3g}) ---")
        edges = [-1e9, qs[0], qs[1], qs[2], 1e9]
        for lo, hi in zip(edges, edges[1:]):
            sub = [j for j in joined if lo < j[key] <= hi]
            if len(sub) < 5:
                continue
            n = len(sub)
            e = sum(j["etkdg_hit"] for j in sub); m = sum(j["mtd_hit"] for j in sub)
            u = sum(j["etkdg_hit"] or j["mtd_hit"] for j in sub)
            mo = sum(j["mtd_only"] for j in sub)
            print(f"  ({lo:>8.3g},{hi:>8.3g}]  n={n:<4} ETKDG {e/n:6.1%}  MTD {m/n:6.1%}  "
                  f"UNION {u/n:6.1%}  MTD-only {mo:<3} (gain {(u-e)/n:+5.1%})")

    # The decision-relevant view: gate on a threshold, pay the MTD only there.
    print("\n--- GATED DEPLOYMENT: run the MTD only where the signal says ETKDG is thin ---")
    base_e = sum(j["etkdg_hit"] for j in joined)
    n_all = len(joined)
    full_u = sum(j["etkdg_hit"] or j["mtd_hit"] for j in joined)
    full_cost = sum(j["wall"] for j in joined)
    print(f"  {'baseline ETKDG only':<34} {base_e}/{n_all} ({base_e/n_all:5.1%})  0.0 core-h")
    print(f"  {'ungated union (MTD everywhere)':<34} {full_u}/{n_all} ({full_u/n_all:5.1%})  "
          f"{full_cost/3600:.1f} core-h")
    best = None
    for key in ("n_states", "states_per_atom", "spread", "mean_dist"):
        vals = sorted({j[key] for j in joined})
        for thr in vals:
            gated = [j for j in joined if j[key] <= thr]      # low diversity => run the MTD
            if not 5 <= len(gated) <= 0.8 * n_all:
                continue
            hits = sum((j["etkdg_hit"] or j["mtd_hit"]) if j in gated else j["etkdg_hit"]
                       for j in joined)
            cost = sum(j["wall"] for j in gated)
            eff = (hits - base_e) / (cost / 3600 + 1e-9)
            if best is None or (hits, -cost) > (best[1], -best[2]):
                best = (f"{key} <= {thr:g}", hits, cost, len(gated), eff)
    if best:
        lbl, hits, cost, ngate, eff = best
        print(f"  {'best gate: ' + lbl:<34} {hits}/{n_all} ({hits/n_all:5.1%})  "
              f"{cost/3600:.1f} core-h on {ngate} molecules  "
              f"(+{hits - base_e} hits, {eff:.1f} hits/core-h)")
        print(f"\n  For reference the ORACLE gate (run the MTD only where it actually helps) "
              f"would give {full_u}/{n_all} ({full_u/n_all:5.1%}).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", type=Path, required=True, help="merged Stage-1 CSV")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=0.125,
                    help="ring-iRMSD below which two embeddings are the same state")
    ap.add_argument("--run-seeds", type=int, default=3)
    ap.add_argument("--cap-confs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import pandas as pd

    man = pd.read_csv(args.manifest).set_index("mol_id")
    smi_col = "raw_smiles" if "raw_smiles" in man.columns else "smiles"
    bench_rows = list(csv.DictReader(open(args.bench)))
    tasks = [{"mol_id": r["mol_id"], "smiles": man.loc[r["mol_id"], smi_col],
              "tol": args.tol, "run_seeds": args.run_seeds, "cap_confs": args.cap_confs}
             for r in bench_rows if r["mol_id"] in man.index]

    print(f"ETKDG ring-diversity on {len(tasks)} molecules "
          f"({args.run_seeds}x{args.cap_confs} embeddings, tol {args.tol} A, no QM)")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(diversity, tasks):
            rows.append(r)
            print(f"  {r['mol_id']:12} states={r.get('n_states','-'):>3}/{r.get('n_pool','-'):<3}"
                  f" spread={r.get('spread','-'):>7} {r.get('note','')}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mol_id", "n_flex_ring_atoms", "n_pool", "n_states",
                                          "states_per_atom", "spread", "mean_dist", "note"],
                           extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")
    report(rows, bench_rows)


if __name__ == "__main__":
    main()
