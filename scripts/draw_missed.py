#!/usr/bin/env python
"""2D structure grid of molecules that miss the search budget, annotated with d and
the closest approach to E* (best_gap, kcal/mol) -- the eyeball companion for spotting
*why* certain acyclic/aromatic molecules need many steps.

Input is hard_missed_boreachable.csv (from the miss analysis): columns
name,d,cls,n_seed,n_miss,best_gap,smiles. By default only ALL-SEED misses (n_miss ==
n_seed, i.e. consistently hard, not search variance) are drawn, sorted worst-first by
best_gap so the wrong-basin cases (large gap) lead.

Usage:
  python scripts/draw_missed.py --csv nstar_merged/analysis/hard_missed_boreachable.csv \
      --out nstar_merged/analysis/missed_structures.pdf
"""
import argparse
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


def render(smiles, px=380):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    d = rdMolDraw2D.MolDraw2DCairo(px, px)
    d.drawOptions().padding = 0.08
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    import matplotlib.image as mpimg
    return mpimg.imread(BytesIO(d.GetDrawingText()), format="png")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--all-misses", action="store_true",
                   help="Include partial (some-seed) misses too, not just all-seed.")
    p.add_argument("--ncol", type=int, default=4)
    p.add_argument("--near-thr", type=float, default=2.0,
                   help="best_gap (kcal) below which an UNCENSORED miss is 'near-miss'")
    args = p.parse_args()

    t = pd.read_csv(args.csv)
    t = t[t.n_miss > 0]
    if not args.all_misses:
        t = t[t.n_miss == t.n_seed]
    # genuine (uncensored) misses first -- they're the diagnosable ones -- then by gap.
    t = t.sort_values(["miss_censored_frac", "best_gap"],
                      ascending=[True, False]).reset_index(drop=True)
    n = len(t)
    if n == 0:
        raise SystemExit("no missed molecules to draw")

    ncol = args.ncol
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.1, nrow * 3.4))
    axes = axes.ravel() if n > 1 else [axes]

    # Honest axis is CENSORING, not gap: a censored miss was killed at the budget wall
    # (unfinished search -> its gap is a lower bound, not a wrong basin). Only an
    # uncensored miss (ran to completion, still short) reflects a genuine search/reference
    # failure, and only then does the gap distinguish near-miss vs stuck.
    def tag(row):
        if row.miss_censored_frac > 0.5:
            return "censored (budget)", "#8e44ad"
        if row.best_gap <= args.near_thr:
            return "genuine near-miss", "#27ae60"
        return "genuine stuck", "#c0392b"

    for i, row in t.iterrows():
        ax = axes[i]
        img = render(row.smiles)
        if img is not None:
            ax.imshow(img)
        lab, col = tag(row)
        miss = f"{int(row.n_miss)}/{int(row.n_seed)} seeds"
        ax.set_title(f"{row['name']}  (d={int(row.d)})\n"
                     f"gap {row.best_gap:.1f} kcal · {lab} · {miss}",
                     fontsize=8.5, color=col)
        ax.axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Acyclic/aromatic budget misses (n={n}; genuine/uncensored first, "
                 f"then wall-censored)", fontsize=12, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(args.out)
    print(f"wrote {args.out}  ({n} molecules)")


if __name__ == "__main__":
    main()
