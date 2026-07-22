#!/usr/bin/env python
"""Stress-test ETKDG ring-pucker reachability for the RING_MISS__ETKDG_UNREACHABLE
molecules from crest_wins_diagnosis.py.

The main diagnostic's capability probe uses a PRODUCTION-scale embedding budget
(num_initial_embeddings = 1 + n_flexible_ring_atoms, capped, x a few seeds), so
"unreachable" there really means "the amount bouquet actually generates didn't hit
the CREST ring". This asks the stronger question: with a HUGE ETKDG budget
(``--n-confs``, default 1000), can distance geometry produce the CREST ring pucker
AT ALL -- and does ``useSmallRingTorsions`` change the answer?

Per molecule x {useSmallRingTorsions True, False}:
  1. EmbedMultipleConfs(numConfs=N) + MMFF -> N conformers.
  2. min ring-iRMSD (MMFF geometry) to the CREST ring over ALL N  [cheap screen; a
     large value here already means ETKDG never even gets close].
  3. GFN2-relax the --gfn2-cap MMFF-closest conformers (all that can plausibly reach
     CREST) and take the min ring-iRMSD to the CREST ring  [bounded confirmation on
     the GFN2 surface -- these ring systems are too floppy to relax every pucker].
Reachable iff the GFN2 min < --ring-rmsd (default 0.125 A). ``n_mmff_close`` (confs
within 2x tol at the MMFF level) reports how much of the huge budget got near at all.

CREST ring = the global-min frame of reference/<mol_id>.xyz (the CREST winner);
ring atoms = flexible (non-aromatic) ring heavy atoms; iRMSD via crest_wins_diagnosis
(rotation+permutation-invariant). Atom order is AddHs throughout, so indices align.

Usage:
    pixi run python scripts/etkdg_ring_stress.py \
        --reference-dir stop_bench_gfn2/reference_gfn2 \
        --manifest smiles/stopbench-500.csv \
        --mols m021_d01,m047_d02,... --n-confs 1000 --workers 8 \
        --out stop_bench_gfn2/analysis/etkdg_ring_stress.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crest_wins_diagnosis as C  # noqa: E402  (ring_rmsd, flexible_ring_atoms, _read_reference_min)
import reference as ref           # noqa: E402  (_gfnff_calc, _optimize)


def _embed(smiles, n, small_ring_torsions, seed):
    """N ETKDG conformers (MMFF-optimized) as ASE Atoms, AddHs order."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from bouquet.setup import mol_to_ase_atoms
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.useSmallRingTorsions = small_ring_torsions
    p.useMacrocycleTorsions = True
    p.randomSeed = seed
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n, params=p))
    if cids:
        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200)
    return [mol_to_ase_atoms(mol, conf_id=c) for c in cids]




def stress(task: dict) -> list[dict]:
    from rdkit import Chem
    from bouquet.setup import apply_charge_spin, default_multiplicity
    mid, smiles = task["mol_id"], task["smiles"]
    rows = []
    try:
        mol = Chem.MolFromSmiles(smiles)
        molh = Chem.AddHs(mol)
        charge = Chem.GetFormalCharge(mol)
        ring_idx = C.flexible_ring_atoms(molh)
        crest_min = C._read_reference_min(task["ref_xyz"])
        base = {"mol_id": mid, "d": task["d"], "gap_kcal": task["gap_kcal"],
                "n_flex_ring_atoms": len(ring_idx)}
        if not ring_idx or crest_min is None:
            base.update(note="no flexible ring / no reference min")
            return [base]
        calc = ref._gfnff_calc(task["method"], molh, charge)
        tol = task["ring_rmsd_thr"]
        for srt in (True, False):
            confs = _embed(smiles, task["n_confs"], srt, task["seed"])
            row = dict(base, use_small_ring_torsions=srt, n_embedded=len(confs))
            if not confs:
                row.update(note="embedding failed"); rows.append(row); continue
            # MMFF ring-iRMSD to the CREST ring for every conformer (cheap screen).
            mmff = sorted(((C.ring_rmsd(cf, crest_min, ring_idx), cf) for cf in confs),
                          key=lambda t: t[0])
            n_close = sum(1 for r, _ in mmff if r < 2 * tol)
            # GFN2-relax the MMFF-closest conformers and re-measure.
            mult = default_multiplicity(confs[0], charge)
            gfn2_rmsds = []
            for _r, cf in mmff[: task["gfn2_cap"]]:
                a = cf.copy(); apply_charge_spin(a, charge, mult)
                opt = ref._optimize(a, calc)
                if opt is not None:
                    gfn2_rmsds.append(C.ring_rmsd(opt[0], crest_min, ring_idx))
            min_gfn2 = min(gfn2_rmsds) if gfn2_rmsds else float("inf")
            row.update(n_mmff_close=n_close,
                       min_ring_irmsd_mmff=round(mmff[0][0], 4),
                       min_ring_irmsd_gfn2=round(min_gfn2, 4),
                       reachable=bool(min_gfn2 < tol))
            rows.append(row)
    except Exception as e:
        rows.append({"mol_id": mid, "note": f"error: {e!r}"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--mols", required=True, help="comma-separated mol_id list")
    ap.add_argument("--method", default="gfn2")
    ap.add_argument("--n-confs", type=int, default=1000, help="ETKDG conformers per setting")
    ap.add_argument("--gfn2-cap", type=int, default=30,
                    help="GFN2-relax this many MMFF-closest conformers per setting (default 30)")
    ap.add_argument("--ring-rmsd", type=float, default=0.125,
                    help="ring-iRMSD (A) below which the CREST ring counts as reached")
    ap.add_argument("--seed", type=int, default=0xC0FFEE)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import pandas as pd
    man = pd.read_csv(args.manifest).set_index("mol_id")
    smi_col = "raw_smiles" if "raw_smiles" in man.columns else "smiles"
    tasks = []
    for mid in args.mols.split(","):
        mid = mid.strip()
        if mid not in man.index:
            print(f"  {mid}: not in manifest, skipping"); continue
        ref_xyz = args.reference_dir / f"{mid}.xyz"
        if not ref_xyz.exists():
            print(f"  {mid}: no reference xyz, skipping"); continue
        tasks.append({"mol_id": mid, "smiles": man.loc[mid, smi_col],
                      "d": int(man.loc[mid, "d"]) if "d" in man.columns else None,
                      "gap_kcal": None, "ref_xyz": str(ref_xyz), "method": args.method,
                      "n_confs": args.n_confs, "gfn2_cap": args.gfn2_cap,
                      "ring_rmsd_thr": args.ring_rmsd, "seed": args.seed})
    print(f"Stress-testing {len(tasks)} molecule(s), {args.n_confs} ETKDG confs x "
          f"{{useSmallRingTorsions True, False}} each...")

    cols = ["mol_id", "d", "n_flex_ring_atoms", "use_small_ring_torsions", "n_embedded",
            "n_mmff_close", "min_ring_irmsd_mmff", "min_ring_irmsd_gfn2", "reachable", "note"]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(stress, tasks):
            for r in res:
                rows.append(r)
                srt = r.get("use_small_ring_torsions")
                print(f"  {r['mol_id']:12} srt={str(srt):5} "
                      f"mmff_close={r.get('n_mmff_close','-'):>4} "
                      f"mmff_min={r.get('min_ring_irmsd_mmff','-'):>7} "
                      f"gfn2_min={r.get('min_ring_irmsd_gfn2','-'):>7} "
                      f"reachable={r.get('reachable','-')}  {r.get('note','')}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
