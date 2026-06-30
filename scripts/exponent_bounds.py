#!/usr/bin/env python
"""Bounds on the basin-cap scaling exponent  n*(d) ~ A * d^p.

The single point fit in stop_rules.py (basin) reports one p, but it is sensitive
to (a) the fit form (through-origin power vs log-log slope), (b) the d-window
(high-d bins are truncated by the run ceiling -> plateau -> p biased low), and
(c) sampling noise. This script brackets p three ways:

  * log-log OLS slope of n*_q(d) vs d  (p = slope; analytic + bootstrap CI)
  * through-origin power-law grid fit   (matches stop_benchmark's C(d) form)
  * across several d-windows, to expose the high-d truncation bias

Input is the events_basin.csv written by `stop_rules.py basin`.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def nstar_by_d(ev: pd.DataFrame, q: float) -> pd.DataFrame:
    """Empirical n*_q per d-bin (basin reliability = 1, so KM = plain quantile)."""
    rows = []
    for d, g in ev.groupby("d"):
        rows.append({"d": int(d), "n": len(g),
                     "nstar": float(np.quantile(g["t_basin"], q))})
    return pd.DataFrame(rows).sort_values("d").reset_index(drop=True)


def fit_loglog(d, y):
    """OLS slope p of log(y) ~ log(d). Returns (p, intercept_logA)."""
    d, y = np.asarray(d, float), np.asarray(y, float)
    m = (d > 0) & (y > 0)
    x, ly = np.log(d[m]), np.log(y[m])
    p, b = np.polyfit(x, ly, 1)
    return float(p), float(b)


def fit_origin_power(d, y):
    """Through-origin y = A*d^p, grid over p (matches stop_rules.fit_scaling)."""
    d, y = np.asarray(d, float), np.asarray(y, float)
    m = (d > 0) & np.isfinite(y)
    d, y = d[m], y[m]
    best = None
    for p in np.linspace(0.5, 3.0, 251):
        A = float(np.linalg.lstsq((d**p)[:, None], y, rcond=None)[0][0])
        rmse = float(np.sqrt(np.mean((A * d**p - y) ** 2)))
        if best is None or rmse < best[-1]:
            best = (A, float(p), rmse)
    return best  # (A, p, rmse)


def bootstrap_exponent(ev: pd.DataFrame, q: float, dmin: int, dmax: int,
                       n_boot: int, rng, method: str) -> tuple:
    """Resample trials *within each d-bin*, recompute n*_q(d), refit p. CI on p."""
    sub = ev[(ev.d >= dmin) & (ev.d <= dmax)]
    groups = {d: g["t_basin"].to_numpy() for d, g in sub.groupby("d")}
    ds = sorted(groups)
    if len(ds) < 3:
        return (np.nan, np.nan, np.nan)
    ps = []
    for _ in range(n_boot):
        yy = []
        for d in ds:
            t = groups[d]
            bs = t[rng.integers(0, len(t), len(t))]
            yy.append(np.quantile(bs, q))
        if method == "loglog":
            p, _ = fit_loglog(ds, yy)
        else:
            _, p, _ = fit_origin_power(ds, yy)
        ps.append(p)
    ps = np.array(ps)
    return (float(np.median(ps)),
            float(np.percentile(ps, 2.5)), float(np.percentile(ps, 97.5)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("events", type=Path, help="events_basin.csv from stop_rules.py basin")
    ap.add_argument("--q", type=float, default=0.95, help="n* quantile (default 0.95)")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--windows", default="1-12,1-10,1-9,2-9,3-9",
                    help="comma list of dmin-dmax d-windows to fit")
    args = ap.parse_args()

    ev = pd.read_csv(args.events)
    rng = np.random.default_rng(args.seed)

    tab = nstar_by_d(ev, args.q)
    print(f"n*_{int(args.q*100)} per d-bin (basin discovery, identifiable):")
    print(tab.to_string(index=False))

    windows = []
    for w in args.windows.split(","):
        lo, hi = w.split("-")
        windows.append((int(lo), int(hi)))

    print(f"\nExponent p in  n*_{int(args.q*100)}(d) ~ A * d^p   "
          f"(95% bootstrap CI, {args.bootstrap} resamples)")
    print(f"{'window':>8} {'n_d':>4} | {'log-log slope':>22} | {'through-origin power':>22}")
    for lo, hi in windows:
        nd = tab[(tab.d >= lo) & (tab.d <= hi)].shape[0]
        pl, ll, lh = bootstrap_exponent(ev, args.q, lo, hi, args.bootstrap, rng, "loglog")
        po, ol, oh = bootstrap_exponent(ev, args.q, lo, hi, args.bootstrap, rng, "origin")
        print(f"{f'{lo}-{hi}':>8} {nd:>4} | "
              f"{pl:5.2f}  [{ll:4.2f}, {lh:4.2f}]   | "
              f"{po:5.2f}  [{ol:4.2f}, {oh:4.2f}]")

    # Point fits on the full and trimmed windows for reference.
    print("\npoint fits (no resampling):")
    for lo, hi in windows:
        t = tab[(tab.d >= lo) & (tab.d <= hi)]
        if len(t) < 3:
            continue
        p_ll, _ = fit_loglog(t.d, t.nstar)
        A, p_o, rmse = fit_origin_power(t.d, t.nstar)
        print(f"  {lo}-{hi}: log-log p={p_ll:.2f} | origin {A:.1f}*d^{p_o:.2f} (rmse {rmse:.0f})")


if __name__ == "__main__":
    main()
