#!/usr/bin/env python
"""Audit CREST coverage over a manifest and emit the work lists to close the gaps.

Replaces the hand-made nstar_merged/crest_todo/*.smi, which had gone stale: it listed
13 molecules when 20 actually lacked a usable CREST cell.

THE DISTINCTION THIS SCRIPT EXISTS TO MAKE. A reference with ``n_crest == 0`` does NOT
mean CREST never ran. There are two causes and they need opposite fixes:

  (a) NO CELL on disk -> CREST genuinely has not run. Fix: re-submit the CREST array
      (ensemble_bench/crest_fill.slurm for m###, scripts/autosteps_crest.slurm for the
      rest). Both are resumable, so a full re-submit only runs what is missing.

  (b) CELL EXISTS but every structure was REJECTED (connectivity guard ``n_rej_crest``
      or species guard ``n_species_rej_crest``). CREST ran fine; re-running it changes
      NOTHING. Fix: rebuild the reference WITH the orthogonal pool. These molecules were
      built under ``--no-orthogonal`` because a cell existed, so when the cell was
      rejected the pool collapsed to bouquet's own BO minima -- or to nothing at all
      (``n_orth=0, n_bo=0`` -> no valid conformer, E* is meaningless). A reference made
      only of bouquet's own minima is circular and scores trivially well, which is worse
      than having no reference, so this case must not be left silently in place.

Usage:
    pixi run python scripts/crest_gaps.py \
        --manifest smiles/combined-gfnff.csv \
        --crest-dir autosteps_ensemble ensemble_bench \
        --reference stop_bench_gfn2/reference_gfn2 autosteps_bouquet/reference \
        --out-dir nstar_merged/crest_todo
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path


def cells_on_disk(dirs, crest_glob: str) -> set[str]:
    """mol_ids with at least one NON-EMPTY refined cell in any --crest-dir."""
    found = set()
    for d in dirs:
        for p in glob.glob(str(Path(d) / "*" / crest_glob)):
            if os.path.getsize(p) > 0:
                # <crest-dir>/<mol_id>/crest_*/refined.xyz
                found.add(Path(p).parent.parent.name)
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True, help="CSV with a mol_id column")
    ap.add_argument("--crest-dir", type=Path, nargs="+", default=[],
                    help="dirs holding CREST cells (<dir>/<mol_id>/crest_*/refined.xyz)")
    ap.add_argument("--reference", type=Path, nargs="+", default=[],
                    help="reference.py output dirs holding <mol_id>.json")
    ap.add_argument("--crest-glob", default="crest_*/refined.xyz")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write missing_crest_<group>.smi + rebuild_orthogonal.txt here")
    ap.add_argument("--max-missing-frac", type=float, default=0.5,
                    help="refuse to write a group's list if more than this fraction of it "
                         "looks uncovered (default 0.5) -- see the guard note below")
    ap.add_argument("--force", action="store_true",
                    help="write the lists even if the guard trips")
    args = ap.parse_args()

    man = {r["mol_id"]: r["raw_smiles"] for r in csv.DictReader(open(args.manifest))}
    cells = cells_on_disk(args.crest_dir, args.crest_glob)
    ref = {}
    for d in args.reference:
        for f in glob.glob(str(d / "*.json")):
            try:
                ref[Path(f).stem] = json.load(open(f))
            except (json.JSONDecodeError, OSError):
                pass

    no_cell, rejected, ok, no_ref = [], [], [], []
    for mid in man:
        j = ref.get(mid)
        if mid not in cells:
            no_cell.append(mid)
        elif j is None:
            no_ref.append(mid)            # cell present, reference not built yet
        elif (j.get("n_crest") or 0) > 0:
            ok.append(mid)
        else:
            rejected.append(mid)

    print(f"manifest {args.manifest}: {len(man)} molecules")
    print(f"  CREST cell present and used  : {len(ok)}")
    print(f"  cell present, no ref yet     : {len(no_ref)}  (pending reference build)")
    print(f"  (a) NO cell -> run CREST     : {len(no_cell)}")
    print(f"  (b) cell REJECTED -> rebuild : {len(rejected)}")

    if rejected:
        print("\n  (b) detail -- re-running CREST will not help these:")
        for mid in sorted(rejected):
            j = ref[mid]
            empty = (j.get("n_orth") or 0) == 0 and (j.get("n_bo") or 0) == 0
            print(f"    {mid:<24} rej_conn={j.get('n_rej_crest')} "
                  f"rej_species={j.get('n_species_rej_crest')} "
                  f"n_orth={j.get('n_orth')} n_bo={j.get('n_bo')}"
                  f"{'   <- EMPTY POOL, E* invalid' if empty else ''}")

    if not args.out_dir:
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Split the CREST work by which array job owns it: m### -> crest_fill.slurm
    # (stopbench-500), everything else -> autosteps_crest.slurm.
    groups = {"stopbench": [m for m in no_cell if m.startswith("m")],
              "autosteps": [m for m in no_cell if not m.startswith("m")]}
    totals = {"stopbench": sum(1 for m in man if m.startswith("m")),
              "autosteps": sum(1 for m in man if not m.startswith("m"))}
    # GUARD. "No cell on disk" is only meaningful where the cells actually live. Point
    # --crest-dir at the wrong tree (or run this on a box that does not have it) and every
    # molecule in that group reports missing -- a plausible-looking list that would queue
    # hundreds of hours of redundant CREST and clobber a good todo file. A group that is
    # mostly "missing" is far more likely a bad path than a real gap, so refuse to write
    # it. This exact mistake produced a 500-molecule stopbench list from an empty dir.
    for name, mols in groups.items():
        tot = totals[name]
        if tot and len(mols) / tot > args.max_missing_frac and not args.force:
            print(f"\nREFUSING to write missing_crest_{name}.smi: {len(mols)}/{tot} "
                  f"({len(mols)/tot:.0%}) of the group has no cell -- that usually means "
                  f"--crest-dir is missing the tree holding this group's cells, not that "
                  f"the work is undone. Check the paths, or pass --force if it is real.")
            continue
        p = args.out_dir / f"missing_crest_{name}.smi"
        with open(p, "w") as fh:
            for mid in sorted(mols):
                fh.write(f"{man[mid]}\t{mid}\n")
        print(f"\nwrote {p} ({len(mols)} molecules)")
    p = args.out_dir / "rebuild_orthogonal.txt"
    p.write_text(",".join(sorted(rejected)) + "\n" if rejected else "")
    print(f"wrote {p} ({len(rejected)} molecules; feed to reference.py --mols, "
          f"WITHOUT --no-orthogonal)")


if __name__ == "__main__":
    main()
