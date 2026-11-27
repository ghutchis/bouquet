#!/usr/bin/env python
"""Build the combined stopbench + autosteps molecule set for the GFN-FF benchmark.

The two sets overlap (117 SMILES appear in more than one file, under different names --
e.g. stopbench's m086_d04 is autosteps' oli_thiophene_n5_d04), so a plain `cat` would run
those molecules twice and double-count them in every aggregate. Dedup is by canonical SMILES,
keeping the first name encountered in the file order given, so stopbench IDs win and existing
stopbench analyses keep their identifiers.

Writes both formats the pipeline needs:
  * ``<out>.smi``  SMILES<TAB>name   -- what stop_benchmark.py / the SLURM shards read
  * ``<out>.csv``  mol_id,raw_smiles,d,source -- what reference.py and the diagnostics read

    pixi run python scripts/make_combined_manifest.py \
        --inputs smiles/stopbench-500.smi smiles/autosteps-calib.smi \
                 smiles/stopbench-extras.smi \
        --out smiles/combined-gfnff

(stopbench-extras.smi is the merge of the former autosteps-d3d4d5 + autosteps-d6d7
acyclic harvest batches; autosteps-calib-val.smi was a subset of autosteps-calib.smi
and contributed nothing, so both are gone. The 761-row combined-gfnff.{smi,csv} in the
tree was built from those older inputs and is byte-for-byte reproducible from these --
do NOT regenerate it while runs are keyed to its line numbers as SLURM array indices.)
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="SMILES files (SMILES<TAB>name); earlier files win on a tie")
    ap.add_argument("--out", type=Path, required=True, help="output stem (.smi and .csv)")
    ap.add_argument("--max-d", type=int, default=0, help="drop molecules above this d (0 = keep all)")
    args = ap.parse_args()

    from rdkit import Chem

    seen: dict[str, tuple[str, str, str]] = {}     # canonical smiles -> (name, raw, source)
    n_dup = n_bad = 0
    for path in args.inputs:
        src = Path(path).stem
        for line in Path(path).read_text().splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            raw, name = parts[0], parts[1]
            mol = Chem.MolFromSmiles(raw)
            if mol is None:
                n_bad += 1
                continue
            # Canonical form is the dedup key: the same molecule carries different names in
            # the two sets, so name-based dedup would miss every cross-set duplicate.
            key = Chem.MolToSmiles(mol)
            if key in seen:
                n_dup += 1
                continue
            seen[key] = (name, raw, src)

    rows = []
    for key, (name, raw, src) in seen.items():
        m = re.search(r"_d(\d+)", name)
        d = int(m.group(1)) if m else None
        if args.max_d and d is not None and d > args.max_d:
            continue
        rows.append({"mol_id": name, "raw_smiles": raw, "d": d, "source": src})
    rows.sort(key=lambda r: (r["d"] if r["d"] is not None else 999, r["mol_id"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    smi = args.out.with_suffix(".smi")
    with open(smi, "w") as f:
        for r in rows:
            f.write(f"{r['raw_smiles']}\t{r['mol_id']}\n")
    csvp = args.out.with_suffix(".csv")
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mol_id", "raw_smiles", "d", "source"])
        w.writeheader(); w.writerows(rows)

    import collections
    byd = collections.Counter(r["d"] for r in rows)
    bysrc = collections.Counter(r["source"] for r in rows)
    print(f"{len(rows)} unique molecules ({n_dup} cross-file duplicates dropped, "
          f"{n_bad} unparseable)")
    print(f"  by source: {dict(bysrc)}")
    print(f"  by d: " + " ".join(f"d{k}={v}" for k, v in sorted(byd.items()) if k is not None))
    hi = sum(v for k, v in byd.items() if k is not None and k >= 7)
    print(f"  d>=7: {hi} molecules  (the band GFN2 could not reach)")
    print(f"\nwrote {smi} and {csvp}")


if __name__ == "__main__":
    main()
