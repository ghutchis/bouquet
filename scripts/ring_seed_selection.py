#!/usr/bin/env python
"""Score ring-seed SELECTION rules offline against a finished ring_mtd_bench run.

The rules in :mod:`bouquet.rings.select` only reorder seeds a run already produced, so the
question "would a k=8 budget have kept the seed carrying the CREST ring conformer?" can be
answered from the cached artefacts alone -- no metadynamics, no quenching, no QM. That
matters because the answer decides whether the seed budget is costing bouquet real hits:
Stage 0 put two hits (m077, m124) at energy rank 9, one slot outside k=8.

Inputs are a bench CSV (for ``mtd_hit_rank`` and the hit flags) and its ``--seeds-dir`` (for
each seed's ring_state / energy / cluster_size). ``mtd_hit_rank`` is an index into the seed
list as written, which is energy order, so a rule "keeps the hit" iff that index appears in
its first k selections.

Usage:
    pixi run python scripts/ring_seed_selection.py \
        --bench stop_bench_gfn2/analysis/ring_mtd_bench_stage0_v2.csv \
        --seeds-dir stop_bench_gfn2/analysis/ring_mtd_seeds_v2 \
        --budgets 4,8,12
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from bouquet.rings.select import RULES, select_seeds


def load(bench: Path, seeds_dir: Path, tol_col: str):
    """Molecules that HIT, with their seed metadata and the hitting seed's index."""
    out = []
    for r in csv.DictReader(open(bench)):
        if r.get(tol_col) != "True":
            continue
        try:
            hit_rank = int(r["mtd_hit_rank"])
        except (KeyError, ValueError):
            continue
        f = seeds_dir / f"{r['mol_id']}.json"
        if not f.exists():
            print(f"  {r['mol_id']}: no seed file in {seeds_dir}, skipping")
            continue
        doc = json.loads(f.read_text())
        seeds = doc["seeds"]
        # Union of ring-system atoms, for the geometry rule.
        ring_idx = sorted({i for s in doc["ring_systems"] for i in s["ring_idx"]})
        if not 0 <= hit_rank < len(seeds):
            print(f"  {r['mol_id']}: hit rank {hit_rank} outside {len(seeds)} seeds, skipping")
            continue
        out.append({"mol_id": r["mol_id"], "ring_class": r.get("ring_class", ""),
                    "seeds": seeds, "hit": hit_rank, "ring_idx": ring_idx,
                    "n_systems": int(r.get("mtd_n_ring_systems") or 1)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", type=Path, required=True, action="append",
                    help="ring_mtd_bench CSV (repeatable; pair each with a --seeds-dir)")
    ap.add_argument("--seeds-dir", type=Path, required=True, action="append")
    ap.add_argument("--budgets", default="4,8,12")
    ap.add_argument("--tol", choices=["125", "250"], default="125",
                    help="score against hits at 0.125 A (default) or 0.25 A")
    args = ap.parse_args()
    if len(args.bench) != len(args.seeds_dir):
        ap.error("--bench and --seeds-dir must be given the same number of times")

    budgets = [int(b) for b in args.budgets.split(",")]
    tol_col = f"mtd_hit_{args.tol}"

    mols = []
    for b, s in zip(args.bench, args.seeds_dir):
        got = load(b, s, tol_col)
        print(f"{b.name}: {len(got)} molecule(s) with a hit at 0.{args.tol}")
        mols += got
    if not mols:
        raise SystemExit("no scorable molecules (no hits, or no seed files)")

    print(f"\nScoring {len(mols)} hit-carrying molecule(s). A rule 'keeps' a hit if the "
          f"hitting seed is inside its first k.\n")
    header = f"{'rule':<12}" + "".join(f"{'k=' + str(k):>10}" for k in budgets)
    print(header)
    print("-" * len(header))
    results = {}
    for rule in RULES:
        row = []
        for k in budgets:
            kept = sum(m["hit"] in select_seeds(m["seeds"], k, rule, m["ring_idx"])
                       for m in mols)
            row.append(kept)
        results[rule] = row
        print(f"{rule:<12}" + "".join(f"{v:>7}/{len(mols):<3}" for v in row))

    # Per-molecule detail at the production budget, so a win is attributable rather than
    # just a better aggregate.
    k = budgets[min(1, len(budgets) - 1)]
    print(f"\nPer molecule at k={k} (energy rank of the hitting seed -> kept by which rules):")
    for m in sorted(mols, key=lambda m: -m["hit"]):
        keep = [r for r in RULES
                if m["hit"] in select_seeds(m["seeds"], k, r, m["ring_idx"])]
        print(f"  {m['mol_id']:<12} {m['ring_class']:<13} rank {m['hit']:>3} of "
              f"{len(m['seeds']):<3} systems={m['n_systems']}  -> {', '.join(keep) or 'NONE'}")

    base = results["energy"]
    best = max(RULES, key=lambda r: (results[r][min(1, len(budgets) - 1)], -RULES.index(r)))
    i = min(1, len(budgets) - 1)
    print(f"\nAt k={budgets[i]}: energy keeps {base[i]}/{len(mols)}, "
          f"best rule '{best}' keeps {results[best][i]}/{len(mols)}.")
    if results[best][i] == base[i]:
        print("No rule beats energy ranking on this data.")


if __name__ == "__main__":
    main()
