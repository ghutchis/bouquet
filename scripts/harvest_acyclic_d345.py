#!/usr/bin/env python
"""Harvest ring-free (acyclic) molecules at d = 3, 4, 5 to rebalance the autosteps
calibration set, whose low-d buckets are aromatic-dominated (d3: 2 acyclic / 11 aromatic,
d4: 3/10, d5: 0/11). Acyclic molecules hit their global min fast, so adding them pulls
the n*_90(d) estimate off the ceiling that a few stubborn aromatics pin it to.

Definitions match select_smiles.py / the benchmark exactly:
  * d       = len(setup.detect_dihedrals(AddHs(mol)))   -- Bouquet's own dihedral count
  * acyclic = stop_rules.ring_class(smiles) == "acyclic" (RingInfo NumRings == 0)
  * inchikey= Chem.MolToInchiKey                         -- dedup + exclusion key

Filters (comparable, GFN2-friendly, unambiguous): neutral, single fragment, elements in
the benchmark set, heavy-atom cap. Excludes every inchikey already in stopbench-500,
stopbench-val-500, autosteps-calib, and the already-harvested stopbench-extras manifests
so the new set is disjoint from both the calibration and validation molecules.

Writes  smiles/<out-prefix>.{smi,csv}  (csv schema matches autosteps-calib.csv:
mol_id,raw_smiles,source,d,cls,inchikey; mol_id = e###_d0#).

Usage:
  python scripts/harvest_acyclic_d345.py --per-d 20 --workers 8
"""
import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

ALLOWED = set("H,B,C,N,O,F,Si,P,S,Cl,Se,Br,I".split(","))
# Reject energetic / exotic-bonding / reactive species that stress GFN2+CREST or aren't
# representative calibration molecules (a cluster of these turns up among small acyclic
# wiki entries: nitramines, nitrate esters, phosphaalkynes, acyl/phosphonyl/sulfonyl
# halides). SMARTS compiled lazily in-worker.
REJECT_SMARTS = [
    "[$([NX3][NX3]=O),$([NX3][N+](=O)[O-])]",   # nitramine / N-nitroso backbone
    "[OX2][N+](=O)[O-]",                          # nitrate ester
    "[#6]#[#15]",                                  # phosphaalkyne C#P
    "[CX3](=[OX1])[F,Cl,Br,I]",                   # acyl halide
    "[PX4](=O)[F,Cl,Br,I]",                        # phosphonyl/phosphoryl halide
    "[SX4](=O)(=O)[F,Cl,Br,I]",                    # sulfonyl halide
    "[NX2]=[NX2]",                                 # aliphatic azo
    "[#6]=[N+]=[N-]",                              # diazo
    "[N-]=[N+]=[N-,N]",                            # azide
    "[N+]#[C-]",                                   # isocyanide
    "[OX2][OX2]",                                   # peroxide / peroxyacid
    "[Cl,Br,I;X3,X4]",                             # hypervalent halogen (oxyacids)
    "[PX5,PX6]",                                    # hypervalent P (PF5/PF6, fluorophosphorane)
    "[#15][F]",                                     # P-F
]
NITRO = "[NX3+](=O)[O-]"                            # count these; >=2 = energetic, reject
_REJECT = None
# Source pools scanned in preference order. COD first: real, isolable, crystallographically
# characterised compounds (the stopbench/autosteps source), so the acyclic picks are stable
# and conventional rather than wiki's exotic small named reagents (isocyanides, diazo,
# peroxyacids, hypervalent-halogen oxyacids). wiki/ligand-expo are backfill.
POOLS = ["cod-59k", "platinum-diverse", "zinc-leads", "zinc_random",
         "ligand-expo", "wiki"]
EXCLUDE_MANIFESTS = ["stopbench-500.csv", "stopbench-val-500.csv",
                     "autosteps-calib.csv",
                     # earlier harvest batches -> keep successive pools disjoint
                     "stopbench-extras.csv"]


def _init():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


def descr(arg):
    """(smi, source) -> dict or None. Ring-free, neutral, single-fragment, allowed
    elements, with Bouquet's d and the ring class."""
    smi, source = arg
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        from bouquet.setup import detect_dihedrals
        from stop_rules import ring_class
        mol = Chem.MolFromSmiles(smi)
        if mol is None or Chem.GetFormalCharge(mol) != 0:
            return None
        if len(Chem.GetMolFrags(mol)) != 1:
            return None
        if any(a.GetSymbol() not in ALLOWED for a in mol.GetAtoms()):
            return None
        if mol.GetRingInfo().NumRings() != 0:          # acyclic only
            return None
        global _REJECT
        if _REJECT is None:
            _REJECT = ([Chem.MolFromSmarts(s) for s in REJECT_SMARTS], Chem.MolFromSmarts(NITRO))
        rej, nitro = _REJECT
        if any(mol.HasSubstructMatch(q) for q in rej if q is not None):
            return None
        if len(mol.GetSubstructMatches(nitro)) >= 2:      # polynitro = energetic
            return None
        n_heavy = sum(a.GetAtomicNum() > 1 for a in mol.GetAtoms())
        # heteroatom present -> genuine functional/conformational content (drops trivial
        # symmetric alkanes like neopentane whose d is only methyl-top rotations; pure
        # hydrocarbon chains are already covered by the oligo:* set).
        n_hetero = sum(a.GetAtomicNum() not in (1, 6) for a in mol.GetAtoms())
        d = len(detect_dihedrals(Chem.AddHs(mol)))
        return {"smiles": Chem.MolToSmiles(mol), "raw_smiles": smi, "source": source,
                "d": d, "cls": ring_class(smi), "n_heavy": n_heavy, "n_hetero": n_hetero,
                "inchikey": Chem.MolToInchiKey(mol)}
    except Exception:
        return None


def load_pool(name):
    p = ROOT / "smiles" / f"{name}.smi"
    out = []
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append((line.split()[0], name))     # SMILES is the first whitespace field
    return out


def load_exclusions():
    keys = set()
    for m in EXCLUDE_MANIFESTS:
        p = ROOT / "smiles" / m
        if not p.exists():
            continue
        with open(p) as fh:
            for r in csv.DictReader(fh):
                k = r.get("inchikey", "")
                if k:
                    keys.add(k)
    return keys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-d", type=int, default=20, help="target molecules per d bin")
    ap.add_argument("--dvals", default="3,4,5")
    ap.add_argument("--min-heavy", type=int, default=6)
    ap.add_argument("--max-heavy", type=int, default=40)
    ap.add_argument("--allow-hydrocarbon", action="store_true",
                    help="keep pure C/H molecules (default drops them: require a heteroatom)")
    ap.add_argument("--workers", "-w", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--id-prefix", default="e")
    ap.add_argument("--out-prefix", default="stopbench-extras-new",
                    help="writes smiles/<prefix>.{smi,csv}; merge into stopbench-extras "
                         "once the batch is accepted, and add it to EXCLUDE_MANIFESTS")
    ap.add_argument("--max-scan", type=int, default=0,
                    help="cap candidates scanned (0 = all); stops early once bins fill")
    args = ap.parse_args()
    import random
    rng = random.Random(args.seed)
    dvals = [int(x) for x in args.dvals.split(",")]
    target = {d: args.per_d for d in dvals}

    excl = load_exclusions()
    print(f"excluding {len(excl)} inchikeys already in stopbench/autosteps sets")

    seen = set(excl)
    bins = {d: [] for d in dvals}
    scanned = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as ex:
        for pool in POOLS:
            cands = load_pool(pool)
            if args.max_scan:
                cands = cands[:args.max_scan]
            print(f"scanning {pool}: {len(cands)} candidates ...", flush=True)
            for r in ex.map(descr, cands, chunksize=64):
                scanned += 1
                if not r or r["d"] not in target:
                    continue
                if not (args.min_heavy <= r["n_heavy"] <= args.max_heavy):
                    continue
                if not args.allow_hydrocarbon and r["n_hetero"] == 0:
                    continue
                if r["cls"] != "acyclic" or r["inchikey"] in seen:
                    continue
                seen.add(r["inchikey"])
                bins[r["d"]].append(r)
            have = {d: len(bins[d]) for d in dvals}
            print(f"  pool done. collected so far: {have}")
            if all(len(bins[d]) >= 3 * target[d] for d in dvals):
                print("  enough candidates for a diverse sample; stopping scan.")
                break

    rows = []
    for d in dvals:
        pool = bins[d]
        rng.shuffle(pool)
        take = pool[:target[d]]
        if len(take) < target[d]:
            print(f"WARNING: only {len(take)}/{target[d]} acyclic molecules found at d={d}")
        for i, r in enumerate(sorted(take, key=lambda x: x["inchikey"])):
            mid = f"{args.id_prefix}{len(rows):03d}_d{d:02d}"
            rows.append({"mol_id": mid, "raw_smiles": r["raw_smiles"],
                         "source": r["source"], "d": d, "cls": "acyclic",
                         "inchikey": r["inchikey"]})

    out_csv = ROOT / "smiles" / f"{args.out_prefix}.csv"
    out_smi = ROOT / "smiles" / f"{args.out_prefix}.smi"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["mol_id", "raw_smiles", "source", "d", "cls", "inchikey"])
        w.writeheader(); w.writerows(rows)
    with open(out_smi, "w") as fh:
        for r in rows:
            fh.write(f"{r['raw_smiles']}\t{r['mol_id']}\n")
    from collections import Counter
    print(f"\nscanned {scanned} molecules. wrote {len(rows)} -> {out_csv.name}, {out_smi.name}")
    print("per-d:", dict(sorted(Counter(r['d'] for r in rows).items())))


if __name__ == "__main__":
    main()
