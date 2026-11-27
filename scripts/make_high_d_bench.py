#!/usr/bin/env python
"""Build smiles/high-d-bench.{smi,csv} -- the consolidated high-d comparison set.

PURPOSE: the high-d cases where a *collective* move is the only way in -- oligomer
ladders with a folded/helical global minimum (oligoalanine's helix is the canonical
one), designed foldamers, and irregular high-d controls. This is the set for comparing
CREST and RDKit/ETKDG against bouquet's category moves, so it deliberately keeps the
homologous series intact: the interesting signal is where along a ladder each method
stops finding the fold.

Consolidates four overlapping manifests that had grown into strict-subset duplicates:

    high-d-phaseC (6) < high-d-ladder (8)          }
    high-d-phaseC (6) < high-d-lowmode-ext (103)   } 105 names, 101 unique molecules
    high-d-lowmode (29) < high-d-lowmode-ext (103) }

Two things are dropped or normalised on the way in:

  * The 32 ``m###_d##`` "diverse-*" rows are verbatim stopbench-500 members carrying
    stopbench IDs, already available as smiles/stopbench-d12-20.smi. They are drug-like
    one-offs with no repeating category structure -- the opposite of what this set is
    for -- so they stay on the stopbench side of the fence.
  * ``polyala_n{2,3,4,5}`` and ``polyalanine_n{2,3,4,5}`` are the same four molecules
    under two names. The systematic name wins; the old one is kept in ``aka`` so results
    from earlier high-d-ladder / high-d-phaseC runs can still be joined.

Columns beyond the original schema:
  * ``class``  ladder    -- homologous series, category/collective moves apply
                foldamer -- designed fold, one-off
                irregular-- high-d but no repeat structure (negative control)
  * ``tier``   core (d<=25) -- CREST reference is affordable, so all three methods
                              can be compared head to head
               ext  (d>25)  -- bouquet/RDKit only; CREST is not tractable here

Rows are sorted by (d, name), so truncating a SLURM array to 1-N gives you the N
cheapest cases rather than an arbitrary slice.

Usage:  pixi run python scripts/make_high_d_bench.py [--core-max-d 25]
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

SOURCES = ["high-d-ladder", "high-d-lowmode", "high-d-lowmode-ext", "high-d-phaseC"]

# Extra .smi sources that were never in the four CSVs. These carry no d column, so d is
# computed here with bouquet's own detect_dihedrals -- the same definition the CSVs use.
#   huc-foldamer: the Huc quinoline-amide ladder, n1-n6 at d 5/10/15/20/25/30 (exactly 5
#   rotors per repeat unit). Same chemistry as `napth`, which is a *different* molecule
#   (n=5 with nitro/OMe termini), so napth alone did not cover it. n1-n4 also sit in
#   stopbench-500 under m### IDs, which is how huc-n3 ended up inside high-d-lowmode-ext
#   mislabelled `m357_d15`/`diverse-weak` and got swept up by DROP_CHEMISTRY.
#   (stem, class, chemistry or None to use the molecule name, id_prefix)
# id_prefix names any row the source file leaves unnamed, as <id_prefix>_u<n> counting
# unnamed rows in file order -- unique, stable, and obviously synthetic, so it never
# reads as a recovered identity.
SMI_SOURCES = [
    ("huc-foldamer", "ladder", "huc-quinoline-amide", "huc"),
    ("foldamers", "foldamer", None, "fold"),
]
# Nothing to search: no rotatable dihedrals at all, so it cannot belong in a high-d set.
MIN_D = 1

# chemistry -> class. Anything not listed is a one-off foldamer.
LADDERS = {"polyethylene", "PEG", "polypropylene", "polyglycine", "polyalanine",
           "polyserine", "nylon6", "Aib", "b3homoAla"}
# high-d-ladder labelled polyalanine_n2..n5 "peptide" while high-d-lowmode labelled
# n6..n8 "polyalanine" -- one ladder under two labels, which split the very series this
# set exists to study (the oligoalanine helix). Relabel so n2..n8 read as one.
CHEMISTRY_ALIAS = {"peptide": "polyalanine"}
IRREGULAR = {"irregular"}
# Dropped: stopbench-500 members that came in with the lowmode-ext manifest.
DROP_CHEMISTRY = {"diverse-weak", "diverse-mod", "diverse-strong"}
# Duplicate names for the same molecule -- systematic name wins, other goes to `aka`.
ALIAS = {f"polyala_n{i}": f"polyalanine_n{i}" for i in (2, 3, 4, 5)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smiles-dir", type=Path, default=Path(__file__).resolve().parent.parent / "smiles")
    ap.add_argument("--core-max-d", type=int, default=25,
                    help="d at or below which a CREST reference is considered affordable")
    ap.add_argument("--out", default="high-d-bench", help="output stem under --smiles-dir")
    args = ap.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    rows: dict[str, dict] = {}
    aka: dict[str, str] = {}
    n_drop = 0
    for stem in SOURCES:
        path = args.smiles_dir / f"{stem}.csv"
        for r in csv.DictReader(open(path)):
            if r["chemistry"] in DROP_CHEMISTRY:
                n_drop += 1
                continue
            name = ALIAS.get(r["name"], r["name"])
            if name != r["name"]:
                aka[name] = r["name"]
            chem = CHEMISTRY_ALIAS.get(r["chemistry"], r["chemistry"])
            if chem != r["chemistry"]:
                assert name.startswith(f"{chem}_n"), (name, r["chemistry"], chem)
            prev = rows.get(name)
            if prev is not None:
                # Same name across manifests must mean the same molecule.
                assert Chem.CanonSmiles(prev["smiles"]) == Chem.CanonSmiles(r["smiles"]), name
                assert prev["d"] == int(r["d"]), (name, prev["d"], r["d"])
                prev["source"] = f"{prev['source']}+{stem}"
                continue
            rows[name] = {"name": name, "smiles": r["smiles"], "d": int(r["d"]),
                          "chemistry": chem, "source": stem}

    # --- extra .smi sources -------------------------------------------------------
    from bouquet.setup import detect_dihedrals

    by_canon = {Chem.CanonSmiles(r["smiles"]): n for n, r in rows.items()}
    skipped, autonamed = [], []
    for stem, cls, chem, prefix in SMI_SOURCES:
        n_unnamed = 0
        for line in (args.smiles_dir / f"{stem}.smi").read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            smi = parts[0]
            name = parts[1] if len(parts) > 1 else None
            if name is None:
                n_unnamed += 1
                name = f"{prefix}_u{n_unnamed}"
                assert name not in rows, f"auto-name {name} collides with a real name"
                autonamed.append((stem, name))
            key = Chem.CanonSmiles(smi)
            if key in by_canon:
                continue                       # already in via one of the CSVs
            d = len(detect_dihedrals(Chem.AddHs(Chem.MolFromSmiles(smi))))
            if d < MIN_D:
                skipped.append((stem, name, d))
                continue
            rows[name] = {"name": name, "smiles": smi, "d": d,
                          "chemistry": chem or name, "source": stem, "_class": cls}
            by_canon[key] = name

    # Names end up embedded in output filenames (grad_<name>_seed#.log and friends), so
    # anything needing shell quoting is a bug waiting to happen -- LQQ(4) was one.
    unsafe = [n for n in rows if not re.fullmatch(r"[A-Za-z0-9_.-]+", n)]
    assert not unsafe, f"names are used as filenames; these need renaming: {unsafe}"

    # Canonical SMILES must now be unique -- any collision left is an unhandled alias.
    seen: dict[str, str] = {}
    for name, r in rows.items():
        key = Chem.CanonSmiles(r["smiles"])
        assert key not in seen, f"duplicate molecule under two names: {seen[key]} / {name}"
        seen[key] = name

    # Many of these molecules were also sampled into stopbench-500 under m### IDs; record
    # the alias so existing stopbench results can be joined to a ladder position.
    sb = {}
    for line in (args.smiles_dir / "stopbench-500.smi").read_text().splitlines():
        p = line.split()
        if len(p) > 1:
            sb[Chem.CanonSmiles(p[0])] = p[1]

    out = []
    for name, r in sorted(rows.items(), key=lambda kv: (kv[1]["d"], kv[0])):
        cls = r.get("_class") or ("ladder" if r["chemistry"] in LADDERS else
                                  "irregular" if r["chemistry"] in IRREGULAR else "foldamer")
        out.append({"name": name, "smiles": r["smiles"], "d": r["d"],
                    "chemistry": r["chemistry"], "class": cls,
                    "tier": "core" if r["d"] <= args.core_max_d else "ext",
                    "aka": aka.get(name, ""),
                    "stopbench_id": sb.get(Chem.CanonSmiles(r["smiles"]), ""),
                    "source": r["source"]})

    stem = args.smiles_dir / args.out
    with open(stem.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "smiles", "d", "chemistry", "class",
                                          "tier", "aka", "stopbench_id", "source"])
        w.writeheader(); w.writerows(out)
    with open(stem.with_suffix(".smi"), "w") as f:
        for r in out:
            f.write(f"{r['smiles']}\t{r['name']}\n")

    import collections
    bycls = collections.Counter((r["class"], r["tier"]) for r in out)
    print(f"{len(out)} molecules ({n_drop} stopbench 'diverse-*' rows dropped, "
          f"{len(aka)} names normalised)")
    for k in sorted(bycls):
        print(f"  {k[0]:<10} {k[1]:<5} {bycls[k]}")
    print(f"  d range: {out[0]['d']}-{out[-1]['d']}")
    print(f"  core (d<={args.core_max_d}): {sum(1 for r in out if r['tier']=='core')}")
    print(f"  also in stopbench-500: {sum(1 for r in out if r['stopbench_id'])}")
    for src, name, d in skipped:
        print(f"  SKIPPED {src}:{name} d={d} -- no rotatable dihedrals")
    for src, name in autonamed:
        print(f"  AUTO-NAMED {name} -- unnamed row in {src}.smi")
    print(f"\nwrote {stem.with_suffix('.smi')} and {stem.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
