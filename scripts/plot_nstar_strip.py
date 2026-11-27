#!/usr/bin/env python
"""Stripplot of per-trial hitting time vs dihedral count, with n*(d) quantile curves
overlaid -- the eyeball companion to stop_rules.py's scaling fit.

Each point is one (molecule, seed) trial: x = dihedral count d (jittered), y = the
number of xTB calls at which the re-optimized running-min first reached the global
minimum E* (from events_nstar_reopt.csv). Trials that never reached E* within budget
are drawn as hollow up-triangles at their censoring horizon (ncalls_max) so misses are
visible, not hidden. Points are coloured by ring-flexibility class.

n_calls ~= BO step (ratio ~1.01, a small constant init offset), so the y-axis doubles
as a BO-step budget and compares directly to the AUTO_STEPS thresholds, drawn as a grey
reference staircase.

Overlaid lines: KM n*_90 / n*_95 / n*_99 from nstar.csv (gaps where censoring makes a
quantile unreachable). This makes fit artifacts obvious -- e.g. a through-origin power
law that reads ~150 calls at d=1 while the raw d=1 trials sit at a handful of calls.

Usage:
  python scripts/plot_nstar_strip.py \
      --events nstar_merged/analysis/events_nstar_reopt.csv \
      --nstar  nstar_merged/analysis/nstar.csv \
      --manifest nstar_merged/manifest.csv \
      --classes acyclic,aromatic \
      --out nstar_merged/analysis/nstar_strip.pdf
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stop_rules import ring_class, km_quantiles, fit_scaling, RHOS  # noqa: E402

# Legacy AUTO_STEPS staircase (the overlay below is currently disabled). The shipped
# budget is now the calibrated formula bouquet.config.auto_total_budget (n*_95:
# min(2000, max(7, round(1.95*d**3)))); update this list only if re-enabling the overlay.
AUTO_STEPS = [(3, 25), (5, 50), (7, 100), (12, 200)]  # (d<=key, steps)
CLASS_COLORS = {"acyclic": "#1b9e77", "aromatic": "#7570b3",
                "flexible-ring": "#d95f02", "?": "#999999"}
QUANTILE_STYLE = {"nstar_90": ("-", "n*_90"), "nstar_95": ("--", "n*_95"),
                  "nstar_99": (":", "n*_99")}


def auto_steps_curve(dmax):
    """The current AUTO_STEPS staircase as (d, steps) over 1..dmax."""
    xs, ys = [], []
    for d in range(1, dmax + 1):
        steps = 200
        for thr, s in AUTO_STEPS:
            if d <= thr:
                steps = s
                break
        xs.append(d)
        ys.append(steps)
    return xs, ys


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", type=Path, required=True,
                   help="events_nstar_reopt.csv (per-trial hitting times)")
    p.add_argument("--nstar", type=Path, default=None,
                   help="nstar.csv for the KM quantile overlay. If omitted, the overlay "
                   "is computed by KM directly from the plotted (class-filtered) events, "
                   "so it always matches the points shown.")
    p.add_argument("--manifest", type=Path, required=True,
                   help="manifest with mol_id,raw_smiles (for ring-class colouring)")
    p.add_argument("--classes", default=None,
                   help="Comma-separated ring classes to KEEP (default: all)")
    p.add_argument("--dmax", type=int, default=None,
                   help="Only fit/plot dihedral counts <= dmax (e.g. 7 while d>=8 is "
                   "still wall-censored).")
    p.add_argument("--exclude-ceiling", type=float, default=None, metavar="FRAC",
                   help="Drop molecules with ANY trial that misses the budget or hits "
                   "later than FRAC*C(d) (C(d)=per-bin run ceiling). These are the "
                   "ring/biaryl-conformer blind spots that pin n* to the ceiling; "
                   "excluding them gives the realistic n* over molecules bouquet solves. "
                   "Typical 0.5.")
    p.add_argument("--out", type=Path, required=True, help="output PDF")
    p.add_argument("--free-intercept", action="store_true",
                   help="Fit a + b*d^c (free intercept) instead of the default "
                   "through-origin b*d^c. Note a can go negative if high-d bins are "
                   "small-sample/steep; prefer capping --dmax to the well-sampled range.")
    p.add_argument("--floor-d1", action="store_true",
                   help="Hold d=1 OUT of the power-law fit and use its measured n* as a "
                   "floor: draw n*(d)=max(n1, b*d^c). d=1 carries a fixed init/search "
                   "overhead that a through-origin power law can't capture, so it needs "
                   "a separate value.")
    p.add_argument("--cap", type=float, default=None,
                   help="Cap the fitted n*(d) curve at this value: n*(d)=min(cap, ...). "
                   "High-d is where collective moves take over and flexible-ring sampling "
                   "dominates anyway, so the extrapolated power law is capped there.")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    ev = pd.read_csv(args.events)
    man = pd.read_csv(args.manifest)
    cls = {r.mol_id: ring_class(r.raw_smiles) for _, r in man.iterrows()}
    ev["cls"] = ev["name"].map(cls).fillna("?")
    keep = args.classes.split(",") if args.classes else sorted(ev["cls"].unique())
    ev = ev[ev["cls"].isin(keep)].copy()
    if args.dmax:
        ev = ev[ev["d"] <= args.dmax].copy()
    if args.exclude_ceiling is not None:
        # per-bin ceiling C(d) = max censoring horizon; a molecule is a blind spot if any
        # trial misses or converges only past FRAC*C(d).
        C = ev.groupby("d")["ncalls_max"].transform("max")
        bad = (~ev["hit"].astype(bool)) | (ev["ncalls_hit"] > args.exclude_ceiling * C)
        drop = set(ev.loc[bad, "name"])
        n0 = ev["name"].nunique()
        ev = ev[~ev["name"].isin(drop)].copy()
        print(f"excluded {len(drop)} ceiling-bound molecules "
              f"(>{args.exclude_ceiling:g}·C or miss); {ev['name'].nunique()}/{n0} kept")
    if ev.empty:
        sys.exit("no trials after filters")

    if args.nstar:
        ns = pd.read_csv(args.nstar).sort_values("d")
    else:
        # KM n*(d;rho) directly from the class-filtered events: hitting time = ncalls_hit
        # for hits, censored at ncalls_max otherwise -- identical machinery to stop_rules.
        rows = []
        for d, g in ev.groupby("d"):
            times = np.where(g["hit"].astype(bool), g["ncalls_hit"], g["ncalls_max"]).astype(float)
            q = km_quantiles(times, g["hit"].astype(int).to_numpy())
            rows.append({"d": int(d), **{f"nstar_{int(r*100)}": q[r] for r in RHOS}})
        ns = pd.DataFrame(rows).sort_values("d")
    dmax = int(ev["d"].max())

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    rng = np.random.default_rng(0)

    # --- per-trial strip ---
    for c in keep:
        sub = ev[ev["cls"] == c]
        hit = sub[sub["hit"].astype(bool)]
        miss = sub[~sub["hit"].astype(bool)]
        jx = hit["d"] + rng.uniform(-0.18, 0.18, len(hit))
        ax.scatter(jx, hit["ncalls_hit"].clip(lower=0.7), s=22, alpha=0.55,
                   color=CLASS_COLORS.get(c, "#999"), edgecolors="none",
                   label=f"{c} (hit, n={len(hit)})", zorder=3)
        if len(miss):
            jm = miss["d"] + rng.uniform(-0.18, 0.18, len(miss))
            ax.scatter(jm, miss["ncalls_max"].clip(lower=0.7), s=34, alpha=0.9,
                       facecolors="none", edgecolors=CLASS_COLORS.get(c, "#999"),
                       marker="^", linewidths=1.1,
                       label=f"{c} (missed budget, n={len(miss)})", zorder=4)

    # --- AUTO_STEPS staircase (current heuristic) ---
    # Leave this out for now to focus on the data
    # ax_x, ax_y = auto_steps_curve(dmax)
    # ax.step(ax_x, ax_y, where="mid", color="0.35", lw=1.4, ls="-.",
    #        label="current AUTO_STEPS", zorder=2)

    # --- n* overlays: SMOOTH scaling fit b*d^p (through origin), not the jumpy KM
    # points. The per-d KM estimates are shown as small unconnected markers so the
    # fit-vs-data spread is visible (e.g. the d3/d5/d6 KM values pinned to the run
    # ceiling C(d)) without implying they trend point-to-point. ---
    dfit = np.linspace(1, dmax, 200)
    for col, (style, lab) in QUANTILE_STYLE.items():
        if col not in ns:
            continue
        good = ns[np.isfinite(ns[col])]
        if len(good):
            ax.scatter(good["d"], good[col], marker="_", s=90, color="k",
                       linewidths=1.6, alpha=0.55, zorder=5)
        # d=1 carries a fixed overhead the power law can't reach, so optionally hold it
        # out of the fit and use its measured n* as a floor: n*(d)=max(n1, b*d^c).
        n1 = None
        fit_pts = good
        if args.floor_d1:
            d1row = good[good["d"] == 1]
            if len(d1row):
                n1 = float(d1row[col].iloc[0])
            fit_pts = good[good["d"] >= 2]
        fit = fit_scaling(fit_pts["d"].to_numpy(), fit_pts[col].to_numpy(),
                          through_origin=not args.free_intercept)
        if fit:
            a, b, pexp, rmse = fit
            curve = a + b * dfit ** pexp
            if n1 is not None:
                curve = np.maximum(n1, curve)
                core = f"max({n1:.0f}, {b:.1f}·d^{pexp:.2f})"
            elif args.free_intercept:
                core = f"{a:.0f}+{b:.1f}·d^{pexp:.2f}"
            else:
                core = f"{b:.1f}·d^{pexp:.2f}"
            if args.cap is not None:
                curve = np.minimum(args.cap, curve)
                core = f"min({args.cap:.0f}, {core})"
            ax.plot(dfit, curve, style, color="k", lw=2.0, zorder=6, label=f"{lab} ≈ {core}")
        elif len(good):
            ax.plot([], [], style, color="k", lw=2.0,
                    label=f"{lab} (only {len(good)} d-bins, no fit)")

    ax.set_yscale("log")
    ax.set_xlabel("dihedral count  d")
    ax.set_ylabel("xTB calls to reach global min  (≈ BO steps)")
    ax.set_xticks(range(1, dmax + 1))
    ax.set_xlim(0.5, dmax + 0.5)
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.set_title(args.title or f"Hitting time vs d  ({'+'.join(keep)})")
    ax.legend(fontsize=7, ncol=2, framealpha=0.9, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}  ({len(ev)} trials, d=1..{dmax})")


if __name__ == "__main__":
    main()
