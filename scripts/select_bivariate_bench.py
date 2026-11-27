#!/usr/bin/env python
"""
Build the molecule set for the bivariate-prior benchmark (sweep_bivariate.py).

Selecting molecules that merely *match* a correlated pattern produces a
benchmark that cannot measure anything. ``DihedralPriorModule.forward`` averages
the log-prior over active dimensions, so on a molecule with 8 rotatable
dihedrals and one correlated pair the bivariate term contributes a quarter of a
signal that the PiBO exponent then scales again -- the arms end up nearly
identical by construction. A first attempt selected that way and the correlated
term reached ~30% of the dihedrals; nothing separated.

So rank by *covered fraction* -- the share of a molecule's dihedrals that the
matcher actually assigns to a correlated pair -- and keep the molecules where the
prior drives most of the search.

Selection uses bouquet's own matcher on an embedded conformer, not a bare SMARTS
test, because the terminal-atom offset that makes a fitted prior applicable is
measured off the geometry: a pattern can match a molecule and still yield no
usable pair. Murcko scaffolds are deduplicated so the set is not eight
stereoisomers of one scaffold.

    python scripts/select_bivariate_bench.py \
        --priors bivariate_priors.json \
        --smiles smiles/zinc.smi \
        --out bench_bivariate.csv --want 60 --min-covered 0.5

Writes a ``smiles,name`` CSV that ``sweep_bivariate.py run --input`` consumes,
and prints the covered-fraction distribution -- check it before running the
sweep, since it bounds what the benchmark can possibly detect.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bouquet.priors import (  # noqa: E402
    BivariateSMARTS,
    BivariateTopology,
    DihedralPriorMatcher,
)
from bouquet.setup import detect_dihedrals  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--priors", required=True, type=Path,
                   help="bivariate priors JSON (supplies SMARTS + topology)")
    p.add_argument("--smiles", required=True, type=Path,
                   help="SMILES library to draw from, one per line")
    p.add_argument("--out", required=True, type=Path, help="output smiles,name CSV")
    p.add_argument("--want", type=int, default=60)
    p.add_argument("--min-dihedrals", type=int, default=4)
    p.add_argument("--max-dihedrals", type=int, default=10)
    p.add_argument("--min-covered", type=float, default=0.40,
                   help="minimum share of dihedrals covered by a correlated pair")
    p.add_argument("--scan", type=int, default=0,
                   help="stop after this many parsed molecules (0 = whole file)")
    p.add_argument("--seed", type=int, default=0xF00D, help="embedding seed")
    a = p.parse_args()

    defs = json.loads(a.priors.read_text())
    patterns, queries = [], []
    for type_id, d in defs.items():
        smarts = d.get("smarts")
        if not smarts:
            continue
        q = Chem.MolFromSmarts(smarts)
        if q is None:
            continue
        patterns.append(BivariateSMARTS(
            smarts=smarts, prior_type=type_id,
            topology=BivariateTopology(d.get("topology", "adjacent")),
        ))
        queries.append(q)
    if not patterns:
        sys.exit(f"no usable SMARTS in {a.priors}")
    matcher = DihedralPriorMatcher([], patterns)
    print(f"{len(patterns)} correlated patterns loaded from {a.priors}")

    kept, scaffolds, scanned = [], set(), 0
    with a.smiles.open() as fh:
        for line in fh:
            if len(kept) >= a.want or (a.scan and scanned >= a.scan):
                break
            smi = line.split()[0] if line.strip() else ""
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                continue
            scanned += 1
            # Cheap reject first: embedding dominates the scan.
            if not any(mol.HasSubstructMatch(q) for q in queries):
                continue
            try:
                scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            except Exception:
                scaf = smi
            if scaf in scaffolds:
                continue
            mh = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mh, randomSeed=a.seed) != 0:
                continue
            dihedrals = [tuple(d.chain) for d in detect_dihedrals(mh)]
            nd = len(dihedrals)
            if not (a.min_dihedrals <= nd <= a.max_dihedrals):
                continue
            assigned = matcher.match_bivariate(mh, dihedrals)
            if not assigned:
                continue
            covered = len({d for pair in assigned for d in pair})
            frac = covered / nd
            if frac < a.min_covered:
                continue
            scaffolds.add(scaf)
            kept.append((smi, nd, len(assigned), covered, frac))

    if not kept:
        sys.exit("no molecule met the coverage threshold; lower --min-covered")

    kept.sort(key=lambda t: -t[4])
    with a.out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "name", "num_dihedrals", "n_pairs", "covered_frac"])
        for i, (smi, nd, npair, cov, frac) in enumerate(kept, 1):
            w.writerow([smi, f"biv{i:03d}", nd, npair, f"{frac:.3f}"])

    fr = sorted(t[4] for t in kept)
    print(f"scanned {scanned} molecules, selected {len(kept)} -> {a.out}")
    print(f"covered fraction: min {fr[0]:.2f}  median {fr[len(fr)//2]:.2f}  "
          f"max {fr[-1]:.2f}")
    print("If the median is well below ~0.5 the arms are nearly identical by "
          "construction; fit more patterns or raise --min-covered before running "
          "the sweep.")


if __name__ == "__main__":
    main()
