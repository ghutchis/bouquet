#!/usr/bin/env python
"""Bouquet-vs-CREST winner analysis: which molecules did bouquet's torsion-space BO find
a LOWER global minimum than CREST's iMTD-GC (a genuine alternative basin CREST missed),
and which did CREST find lower than bouquet (a ring/biaryl-conformer bouquet could not
reach)? Both minima are unconstrained on the same GFN2 surface and the reference builder's
modal-fingerprint filter guarantees they are the SAME species, so a win/loss is a real
basin difference, not a rearrangement artifact.

Reads reference.py JSON outputs (E_star_BO, E_star_crest per molecule) from one or more
reference dirs, over molecules that have BOTH pools. gap = (E_star_BO - E_star_crest) in
kcal/mol:
    gap < -noise  -> bouquet lower  (bouquet WIN)
    gap > +noise  -> CREST lower    (CREST WIN, bouquet blind spot)
    |gap| <= noise-> tie

Writes, per direction, a CSV (name, d, cls, gap_kcal, smiles) and an annotated 2D
structure grid PDF, worst-first, so chemical motifs are eyeball-able. Prints the
res x ring-class crosstab.

Usage:
  python scripts/crest_vs_bouquet.py \
      --reference stop_bench_gfn2/reference_gfn2 autosteps_bouquet/reference \
      --out-dir nstar_merged/analysis --noise 0.5
"""
import argparse
import glob
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stop_rules import ring_class  # noqa: E402

KCAL = 23.060541945


def load_refs(dirs):
    rows = []
    for d in dirs:
        for f in glob.glob(str(Path(d) / "*.json")):
            o = json.loads(Path(f).read_text())
            if not o.get("ok"):
                continue
            if o.get("E_star_BO") is None or o.get("E_star_crest") is None:
                continue
            rows.append({"name": o["mol_id"], "d": int(o.get("d", 0)),
                         "smiles": o.get("smiles", ""),
                         "n_bo": o.get("n_bo", 0), "n_crest": o.get("n_crest", 0),
                         "gap_kcal": (o["E_star_BO"] - o["E_star_crest"]) * KCAL})
    return pd.DataFrame(rows)


def grid_pdf(df, out, title, ncol=5):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Draw import rdMolDraw2D
    RDLogger.DisableLog("rdApp.*")
    n = len(df)
    if n == 0:
        print(f"  (no molecules for {out.name}; skipping grid)")
        return
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.9, nrow * 3.1))
    axes = np.atleast_1d(axes).ravel()
    for i, (_, r) in enumerate(df.iterrows()):
        ax = axes[i]
        mol = Chem.MolFromSmiles(r.smiles)
        if mol is not None:
            dd = rdMolDraw2D.MolDraw2DCairo(340, 340)
            rdMolDraw2D.PrepareAndDrawMolecule(dd, mol)
            dd.FinishDrawing()
            ax.imshow(mpimg.imread(BytesIO(dd.GetDrawingText()), format="png"))
        ax.set_title(f"{r['name']} (d={int(r.d)})\ngap {r.gap_kcal:+.2f} kcal "
                     f"(bo{int(r.n_bo)}/crest{int(r.n_crest)})", fontsize=8)
        ax.axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=12, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}  ({n} molecules)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", nargs="+", required=True,
                   help="reference.py output dir(s) of <mol>.json")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--noise", type=float, default=0.5,
                   help="kcal/mol tie band; |gap|<=noise is a tie (default 0.5)")
    p.add_argument("--boreachable-dmax", type=int, default=9,
                   help="The BO-reachable CREST-wins subset (excl. flexible-ring) also "
                   "caps at this d: high-d losses are budget-limited (BO needs more steps), "
                   "so this isolates the cases with the right class AND enough budget that "
                   "bouquet still missed. Default 9 (d<10).")
    p.add_argument("--no-grids", action="store_true", help="CSVs only, skip the PDFs")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_refs(args.reference)
    if df.empty:
        raise SystemExit("no references with BOTH a BO and a CREST pool found")
    df["cls"] = df.smiles.map(ring_class)
    df["res"] = np.where(df.gap_kcal < -args.noise, "bouquet_win",
                np.where(df.gap_kcal > args.noise, "crest_win", "tie"))

    print(f"bouquet-vs-CREST over {len(df)} molecules with both pools "
          f"(noise={args.noise} kcal/mol):")
    print(f"  bouquet wins (BO lower) : {(df.res=='bouquet_win').sum()}")
    print(f"  ties                    : {(df.res=='tie').sum()}")
    print(f"  CREST wins (CREST lower): {(df.res=='crest_win').sum()}")
    print("\nres x ring class:")
    print(pd.crosstab(df.cls, df.res).to_string().replace("\n", "\n  "))

    crest = df[df.res == "crest_win"].sort_values("gap_kcal", ascending=False)   # biggest CREST advantage first
    boq = df[df.res == "bouquet_win"].sort_values("gap_kcal")                    # most-negative (biggest bouquet advantage) first
    cols = ["name", "d", "cls", "gap_kcal", "n_bo", "n_crest", "smiles"]
    # CREST wins WITHOUT flexible-ring (ring-pucker is a sampling gap, not a budget one)
    # AND capped at d<=boreachable_dmax (high-d losses are budget-limited). What remains is
    # the diagnostic subset: right class + enough budget, yet bouquet missed CREST's min.
    crest_br = crest[(crest.cls != "flexible-ring") & (crest.d <= args.boreachable_dmax)]
    crest[cols].to_csv(args.out_dir / "crest_wins.csv", index=False)
    crest_br[cols].to_csv(args.out_dir / "crest_wins_boreachable.csv", index=False)
    boq[cols].to_csv(args.out_dir / "bouquet_wins.csv", index=False)
    print(f"\nwrote {args.out_dir/'crest_wins.csv'} ({len(crest)}), "
          f"{args.out_dir/'crest_wins_boreachable.csv'} ({len(crest_br)}; excl flexible-ring, "
          f"d<={args.boreachable_dmax}), {args.out_dir/'bouquet_wins.csv'} ({len(boq)})")
    if not args.no_grids:
        grid_pdf(crest, args.out_dir / "crest_wins_structures.pdf",
                 f"CREST beats bouquet (bouquet blind spots, n={len(crest)}, >{args.noise} kcal)")
        grid_pdf(crest_br, args.out_dir / "crest_wins_boreachable_structures.pdf",
                 f"CREST beats bouquet, excl. flexible-ring, d<={args.boreachable_dmax} "
                 f"(acyclic+aromatic, n={len(crest_br)}, >{args.noise} kcal)")
        grid_pdf(boq, args.out_dir / "bouquet_wins_structures.pdf",
                 f"Bouquet beats CREST (basins CREST missed, n={len(boq)}, >{args.noise} kcal)")


if __name__ == "__main__":
    main()
