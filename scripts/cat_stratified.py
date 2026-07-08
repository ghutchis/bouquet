#!/usr/bin/env python
"""Stratify category-move gain by repeat structure (max_spec) vs raw size (d).

Answers "when does the category move help?" -- is the win driven by molecule SIZE (d)
or by exploitable REPEAT STRUCTURE (max_spec = largest tied fitted-library category,
from scripts/cat_suitability.py)? The two are correlated in nature, so the tier-D
controls (large but irregular: d>=20, max_spec<=4) are the decisive negative arm: if
the win is repeat-driven, `cat` should be ~flat there while `lm_pca` may still gain.

Pairs every arm against the baseline per (molecule, seed) -- the same common-random-
number differencing the sweep analyze uses -- so `gain = E_base - E_arm` (>0 = arm
reached a lower/better energy). Then:
  1. gain stratified by max_spec bins and by d bins (median, win-rate);
  2. Spearman rho(gain, max_spec) vs rho(gain, d) -- which predicts the win better;
  3. the tier-D control arm (large + irregular) reported on its own;
  4. per-family gain-vs-d ladders (matched chemistry, varying d).

Usage:
  pixi run python scripts/cat_stratified.py "high_d_lowmode/*_traj.csv" \
      --manifest catsuitability.csv [--baseline base] [--arm cat --arm lm_pca] [--tol 1e-3]
"""
from __future__ import annotations
import argparse
import glob
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BROKEN_EV = -5.0  # final best below this = broken geometry (drop), matches sweep analyze


def load_finals(patterns: list[str]) -> pd.DataFrame:
    """Final best_so_far per (config, name, seed), with each molecule's d attached."""
    paths = [p for pat in patterns for p in glob.glob(pat)]
    if not paths:
        raise SystemExit(f"no trajectory CSVs matched {patterns}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    g = (
        df.groupby(["config", "name", "seed"])["best_so_far"].min()
        .reset_index().rename(columns={"best_so_far": "final"})
    )
    g["d"] = g["name"].map(df.groupby("name")["num_dihedrals"].median())
    return g


def pair_gains(g: pd.DataFrame, baseline: str, arm: str) -> pd.DataFrame:
    """Per-(name, seed) gain = E_baseline - E_arm (>0 = arm better). Drops the pair if
    either arm is missing or broken (< BROKEN_EV) for that seed."""
    wide = g.pivot_table(index=["name", "seed"], columns="config", values="final")
    if baseline not in wide or arm not in wide:
        return pd.DataFrame(columns=["name", "seed", "gain", "d"])
    sub = wide[[baseline, arm]].dropna()
    sub = sub[(sub[baseline] >= BROKEN_EV) & (sub[arm] >= BROKEN_EV)]
    out = (sub[baseline] - sub[arm]).rename("gain").reset_index()
    dd = g.drop_duplicates("name").set_index("name")["d"]
    out["d"] = out["name"].map(dd)
    return out


def stratify(gains: pd.DataFrame, col: str, edges: list[float], tol: float) -> pd.DataFrame:
    labels = [f"<= {edges[0]:g}"] + [
        f"{edges[i]:g}-{edges[i+1]:g}" for i in range(len(edges) - 1)
    ] + [f"> {edges[-1]:g}"]
    b = pd.cut(gains[col], [-np.inf] + edges + [np.inf], labels=labels)
    return gains.groupby(b, observed=True).agg(
        n=("gain", "size"),
        median_gain=("gain", "median"),
        mean_gain=("gain", "mean"),
        win_rate=("gain", lambda s: float((s > tol).mean())),
    ).round(3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj", nargs="+", help="trajectory CSV glob(s) (..._traj.csv)")
    ap.add_argument("--manifest", default="catsuitability.csv",
                    help="name,d,max_spec,tied_spec,tier (scripts/cat_suitability.py)")
    ap.add_argument("--baseline", default="base")
    ap.add_argument("--arm", action="append", dest="arms", default=None,
                    help="arm(s) to score vs baseline (repeatable; default cat + lm_pca)")
    ap.add_argument("--tol", type=float, default=1e-3, help="win threshold (eV)")
    ap.add_argument("--maxspec-edges", default="4,7",
                    help="upper edges for max_spec strata (default '4,7')")
    ap.add_argument("--d-edges", default="14,20,27,35",
                    help="upper edges for d strata (default '14,20,27,35')")
    args = ap.parse_args()
    arms = args.arms or ["cat", "lm_pca"]

    g = load_finals(args.traj)
    man = pd.read_csv(args.manifest).set_index("name")
    ms_edges = [float(x) for x in args.maxspec_edges.split(",")]
    d_edges = [float(x) for x in args.d_edges.split(",")]

    gains_by_arm = {}
    for arm in arms:
        gains = pair_gains(g, args.baseline, arm)
        gains_by_arm[arm] = gains
        if gains.empty:
            print(f"\n### {arm}: no paired data (missing arm?)")
            continue
        gains["max_spec"] = gains["name"].map(man["max_spec"])
        gains["tier"] = gains["name"].map(man.get("tier"))
        have_ms = gains.dropna(subset=["max_spec"])

        print(f"\n{'='*70}\n### {arm} vs {args.baseline}  "
              f"(gain = E_base - E_{arm} > 0 means {arm} better; "
              f"n={len(gains)} pairs, {gains['name'].nunique()} mols)\n{'='*70}")
        print(f"\n-- by max_spec (repeat structure; the mechanism's predicted axis) --")
        print(stratify(have_ms, "max_spec", ms_edges, args.tol).to_string())
        print(f"\n-- by d (raw dihedral count) --")
        print(stratify(gains, "d", d_edges, args.tol).to_string())

        # Which axis predicts the gain better? (points = (name, seed) pairs; note the
        # non-independence caveat -- seeds of one molecule share x. Descriptive.)
        if have_ms["max_spec"].nunique() >= 3:
            r_ms, p_ms = spearmanr(have_ms["max_spec"], have_ms["gain"])
            r_d, p_d = spearmanr(have_ms["d"], have_ms["gain"])
            print(f"\n-- Spearman rho(gain, .) : which predicts the win? --")
            print(f"   max_spec : rho={r_ms:+.3f}  p={p_ms:.3g}")
            print(f"   d        : rho={r_d:+.3f}  p={p_d:.3g}   (n={len(have_ms)})")

        # Control arm: large but irregular (tier D, or d>=20 & max_spec<=4).
        ctrl = have_ms[(have_ms["d"] >= 20) & (have_ms["max_spec"] <= 4)]
        if len(ctrl):
            print(f"\n-- CONTROL (d>=20 & max_spec<=4: large but no repeat) --")
            print(f"   n={len(ctrl)} pairs, {ctrl['name'].nunique()} mols  "
                  f"median_gain={ctrl['gain'].median():+.3f}  "
                  f"win_rate={float((ctrl['gain']>args.tol).mean()):.2f}")
            print("   (repeat-driven win => ~0 here; size-driven win => still positive)")

    # Per-family gain-vs-d ladder for the primary arm (matched chemistry, varying d).
    prim = arms[0]
    gains = gains_by_arm.get(prim)
    if gains is not None and not gains.empty:
        fam = lambda n: re.sub(r"_n?\d+$", "", str(n))
        gains["family"] = gains["name"].map(fam)
        permol = gains.groupby(["family", "name"]).agg(
            d=("d", "first"), median_gain=("gain", "median")).reset_index()
        big = permol.groupby("family").filter(lambda x: len(x) >= 3)
        if len(big):
            print(f"\n{'='*70}\n### {prim} gain-vs-d ladders (families with >=3 sizes)\n{'='*70}")
            for f, sub in big.sort_values("d").groupby("family"):
                cells = "  ".join(f"d{int(r.d)}:{r.median_gain:+.2f}"
                                  for r in sub.itertuples())
                print(f"  {f:16} {cells}")


if __name__ == "__main__":
    main()
