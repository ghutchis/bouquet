#!/usr/bin/env python
"""
Offline stopping-rule replay + survival analysis for the conformer-search
stopping-rule benchmark (Phase C).

Consumes ONLY the per-step certificate CSV written by ``stop_benchmark.py run``
(``*_cert.csv``) -- every candidate stopping rule is a threshold on quantities
already logged during the single expensive run, evaluated here by replaying each
trajectory. Nothing re-runs xTB. Run it freely on partial data while the sweep is
still going.

What it computes:

  * E* per molecule -- the working global-best reference: the minimum ``e_best``
    over all seeds (the consensus pool minus CREST). Circular by construction (a
    basin no seed reached is invisible); ``reference.py`` will harden this with
    systematic/CREST references and the RMSD-identity criterion. Until then the
    success test is the energy criterion ``e_best - E* <= eps``.
  * An events table (per trial: d, hitting time/cost in xTB calls, censored flag).
  * Kaplan-Meier ``n*(d; rho)`` -- xTB-call budget to hit the global min with
    reliability rho in {0.90, 0.95, 0.99}, censoring-aware, with bootstrap CIs.
  * A scaling-law fit ``n*(d) ~ a + b*d^p``.
  * A Pareto front (realized reliability vs mean cost) over four stopping rules
    -- GP certificate, log-EI, stall, and the current auto table baseline.
  * A calibration curve for the certificate (realized reliability vs beta).

Usage:

  python scripts/stop_rules.py all runs/stopbench_cert.csv --out-dir runs/analysis
  # or individual: summary | nstar | pareto | calibrate
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bouquet.config import AUTO_STEPS_DEFAULT, AUTO_STEPS_THRESHOLDS  # noqa: E402

KCAL = 23.060541945  # eV -> kcal/mol
RHOS = (0.90, 0.95, 0.99)


# ---------------------------------------------------------------------------
# Load + reference
# ---------------------------------------------------------------------------


def load_cert(path: Path) -> pd.DataFrame:
    """Load the certificate CSV; energies -> kcal/mol; one tidy frame, sorted."""
    df = pd.read_csv(path)
    ecols = ["e_eval", "e_best", "mu_min", "alpha_max"] + [
        c for c in df.columns if c.startswith("lb_b")
    ]
    for c in ecols:
        df[c] = df[c] * KCAL
    df["d"] = pd.to_numeric(df["num_dihedrals"], errors="coerce").astype("Int64")
    if "censored" not in df.columns:
        df["censored"] = 0
    return df.sort_values(["name", "seed", "n_calls"]).reset_index(drop=True)


def load_reference(path: Path) -> dict:
    """Map mol_id -> E* (kcal/mol) from reference.py output (a reference_summary.csv
    or a directory of <mol_id>.json). E_star is stored relative-eV; convert to the
    kcal/mol scale used here. This is the non-circular reference -- prefer it over
    the within-data pool min, which makes reliability trivially 1.0."""
    path = Path(path)
    out = {}
    if path.is_dir():
        import json
        for f in path.glob("*.json"):
            o = json.loads(f.read_text())
            if o.get("ok") and o.get("E_star") is not None:
                out[o["mol_id"]] = o["E_star"] * KCAL
    else:
        ref = pd.read_csv(path)
        for _, r in ref.iterrows():
            if pd.notna(r.get("E_star")):
                out[r["mol_id"]] = float(r["E_star"]) * KCAL
    return out


def crest_coverage(path: Path) -> tuple[int, int]:
    """(# molecules whose reference E* is CREST-backed, # total references) from
    reference.py output. Surfaces partial-CREST coverage -- with the CREST fill still
    running on the high-d tail (d18-20), the reference E* there falls back to the
    orthogonal/BO pool, which is weaker; this makes that gap visible at a glance.
    Returns (0, n) for a pre-CREST reference (no E_star_crest column/field)."""
    path = Path(path)
    n_total = n_crest = 0
    if path.is_dir():
        import json
        for f in path.glob("*.json"):
            o = json.loads(f.read_text())
            if o.get("ok") and o.get("E_star") is not None:
                n_total += 1
                n_crest += int(o.get("E_star_crest") is not None)
    else:
        ref = pd.read_csv(path)
        has_col = "E_star_crest" in ref.columns
        for _, r in ref.iterrows():
            if pd.notna(r.get("E_star")):
                n_total += 1
                n_crest += int(has_col and pd.notna(r.get("E_star_crest")))
    return n_crest, n_total


def load_reference_abs(path: Path) -> dict:
    """Map mol_id -> E* in ABSOLUTE eV (E_star + E_start_eV) from reference.py output.

    The energy criterion compares bouquet's found minima to E*. Both the reference E*
    and (with --reopt) the re-optimized trajectory geometries are UNCONSTRAINED minima,
    but the cert's ``e_best`` is the CONSTRAINED search energy -- ~1 kcal/mol higher --
    so the naive ``e_best - E* <= eps`` undercounts basin hits ~2x. --reopt re-optimizes
    the trajectory geometries and compares on the absolute scale, where the per-molecule
    e_e0 anchor cancels; this returns that absolute E* (E_star is stored relative-eV,
    E_start_eV is the anchor, so their sum is the absolute minimum energy in eV)."""
    path = Path(path)
    out = {}
    if path.is_dir():
        import json
        for f in path.glob("*.json"):
            o = json.loads(f.read_text())
            if o.get("ok") and o.get("E_star") is not None and o.get("E_start_eV") is not None:
                out[o["mol_id"]] = float(o["E_star"]) + float(o["E_start_eV"])
    else:
        ref = pd.read_csv(path)
        if "E_start_eV" in ref.columns:
            for _, r in ref.iterrows():
                if pd.notna(r.get("E_star")) and pd.notna(r.get("E_start_eV")):
                    out[r["mol_id"]] = float(r["E_star"]) + float(r["E_start_eV"])
    return out


def _reopt_curve_worker(task: dict):
    """Re-optimize a trial's best-so-far trajectory geometries UNCONSTRAINED and return
    the anytime running-min ABSOLUTE energy curve: (name, seed, d, [(n_calls, e_abs_eV)]).

    Each improvement frame is a constrained best-so-far geometry; releasing it (tight
    unconstrained opt) gives the true basin energy the reference is built on. Frames whose
    unconstrained optimum changed connectivity (formed/broke a bond -> different species)
    are dropped, mirroring the reference build and the solver's final-relax guard."""
    import reference as ref  # same scripts/ dir; reuses the reference build machinery
    from rdkit import Chem
    from bouquet.setup import connectivity_changed

    name = task["name"]; seed = task["seed"]
    out = {"name": name, "seed": seed, "d": task["d"], "curve": []}
    frames = _trail_full(Path(task["geom_dir"]), task["config"], name, seed)
    if not frames:
        return out
    try:
        mol = Chem.MolFromSmiles(task["smiles"])
        if mol is None:
            return out
        charge = Chem.GetFormalCharge(mol)
        molh = Chem.AddHs(mol)
        calc = ref._gfnff_calc(task["method"], molh, charge)
        curve = []
        for n_calls, _e_e0, atoms in frames:
            r = ref._optimize(atoms, calc)
            if r is None:
                continue
            if connectivity_changed(r[0], molh, task["conn_tol"]):
                continue
            curve.append((n_calls, float(r[1])))  # (n_calls, absolute eV)
        out["curve"] = curve
    except Exception as e:  # keep the batch alive; a failed trial just has no curve
        out["error"] = repr(e)
    return out


def build_events_reopt(df: pd.DataFrame, args) -> pd.DataFrame:
    """Energy-criterion events using UNCONSTRAINED re-optimized trajectory geometries
    (the ``--reopt`` path). Same schema as :func:`build_events`, but the hitting time is
    the first n_calls at which the running-min *unconstrained* energy comes within eps of
    the absolute reference E*. Requires --manifest (SMILES) and --geom-dir."""
    from concurrent.futures import ProcessPoolExecutor

    cache = out_cache = getattr(args, "reopt_cache", None)
    if cache and Path(cache).exists() and not getattr(args, "reopt_refresh", False):
        print(f"Reusing cached reopt curves: {cache}")
        curves = pd.read_csv(cache)
    else:
        if not args.manifest:
            sys.exit("--reopt needs --manifest (SMILES) and --geom-dir")
        man = pd.read_csv(args.manifest).set_index("mol_id")
        ref_abs = load_reference_abs(args.reference)
        tasks = []
        for (name, seed), _g in df.groupby(["name", "seed"]):
            if name not in man.index or name not in ref_abs:
                continue
            tasks.append({
                "name": name, "seed": int(seed), "d": int(_g["d"].iloc[0]),
                "config": _g["config"].iloc[0], "smiles": man.loc[name, "raw_smiles"],
                "geom_dir": str(args.geom_dir), "method": args.reopt_method,
                "conn_tol": args.conn_tol,
            })
        print(f"Re-optimizing trajectory geometries for {len(tasks)} trials "
              f"({args.reopt_method}, {args.workers} workers)...")
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for o in ex.map(_reopt_curve_worker, tasks):
                for n_calls, e_abs in o["curve"]:
                    rows.append({"name": o["name"], "seed": o["seed"], "d": o["d"],
                                 "n_calls": n_calls, "e_abs": e_abs})
        curves = pd.DataFrame(rows)
        if out_cache:
            curves.to_csv(out_cache, index=False)
            print(f"wrote reopt curves -> {out_cache}")

    ref_abs = load_reference_abs(args.reference)
    eps_eV = args.eps / KCAL
    # censoring horizon per trial = last logged cert call
    horizon = df.groupby(["name", "seed"])["n_calls"].last()
    cens = df.groupby(["name", "seed"])["censored"].last()
    rows = []
    for (name, seed), g in curves.groupby(["name", "seed"]):
        estar_abs = ref_abs.get(name)
        if estar_abs is None:
            continue
        g = g.sort_values("n_calls")
        running = np.minimum.accumulate(g["e_abs"].to_numpy())
        hit_mask = (running - estar_abs) <= eps_eV
        hit = bool(hit_mask.any())
        ncalls_hit = int(g["n_calls"].to_numpy()[hit_mask][0]) if hit else np.nan
        key = (name, seed)
        rows.append({
            "name": name, "seed": seed, "d": int(g["d"].iloc[0]),
            "estar": estar_abs * KCAL, "hit": hit, "ncalls_hit": ncalls_hit,
            "ncalls_max": int(horizon.get(key, g["n_calls"].max())),
            "run_censored": int(cens.get(key, 0)),
        })
    return pd.DataFrame(rows)


def add_estar(df: pd.DataFrame, ref_map: dict | None = None) -> pd.DataFrame:
    """Add per-molecule E*. Default: min e_best over all seeds (within-data pool;
    circular -> reliability ~1). With ``ref_map`` (from reference.py), use the
    independent E* where available, falling back to the pool min otherwise."""
    df["estar"] = df.groupby("name")["e_best"].transform("min")
    if ref_map:
        ref = df["name"].map(ref_map)
        df["estar"] = ref.where(ref.notna(), df["estar"])
        df["has_reference"] = df["name"].isin(ref_map)
    else:
        df["has_reference"] = False
    return df


def beta_columns(df: pd.DataFrame):
    cols = sorted((c for c in df.columns if c.startswith("lb_b")),
                  key=lambda c: float(c[4:]))
    return cols, [float(c[4:]) for c in cols]


# ---------------------------------------------------------------------------
# Events table (rule-independent hitting times)
# ---------------------------------------------------------------------------


def build_events(df: pd.DataFrame, eps: float) -> pd.DataFrame:
    """Per-trial hitting time/cost for the energy criterion ``e_best - E* <= eps``.

    Returns one row per (name, seed): d, estar, hit(bool), ncalls_hit (xTB calls to
    first reach the criterion), ncalls_max (last logged call = censoring horizon),
    censored (trial was killed before finishing). A trial that never hits is
    right-censored at ncalls_max.
    """
    rows = []
    for (name, seed), g in df.groupby(["name", "seed"]):
        estar = g["estar"].iloc[0]
        hit_mask = g["e_best"] - estar <= eps
        hit = bool(hit_mask.any())
        ncalls_hit = int(g.loc[hit_mask, "n_calls"].iloc[0]) if hit else np.nan
        rows.append({
            "name": name, "seed": seed, "d": int(g["d"].iloc[0]),
            "estar": estar, "hit": hit,
            "ncalls_hit": ncalls_hit,
            "ncalls_max": int(g["n_calls"].iloc[-1]),
            "run_censored": int(g["censored"].iloc[-1]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stopping rules: trajectory -> stop row index (position in g)
# ---------------------------------------------------------------------------


def auto_total(d: int) -> int:
    """Current --auto total evaluation budget for dihedral count d (baseline)."""
    for thr, steps in sorted(AUTO_STEPS_THRESHOLDS.items()):
        if d <= thr:
            return steps
    return AUTO_STEPS_DEFAULT


def _first_true(mask: np.ndarray):
    nz = np.flatnonzero(mask)
    return int(nz[0]) if nz.size else None


def rule_table(g: pd.DataFrame, d: int):
    """Baseline: stop once cumulative xTB calls reach the auto-table budget."""
    return _first_true((g["n_calls"] >= auto_total(d)).to_numpy())


def rule_certificate(g: pd.DataFrame, eps: float, beta_col: str):
    """Stop when the certified gap e_best - lb(beta) closes below eps (kcal/mol)."""
    return _first_true(((g["e_best"] - g[beta_col]) < eps).to_numpy())


def rule_logei(g: pd.DataFrame, tau: float):
    """Stop when the max expected improvement falls below tau (kcal/mol)."""
    return _first_true((g["alpha_max"] < tau).to_numpy())


def rule_stall(g: pd.DataFrame, delta: float, k: int):
    """Stop when e_best improved by less than delta over the last k logged steps."""
    eb = g["e_best"].to_numpy()
    if len(eb) <= k:
        return None
    improved = eb[:-k] - eb[k:]            # improvement from i to i+k (>=0)
    pos = _first_true(improved < delta)    # first window with < delta improvement
    return None if pos is None else pos + k


def _stop_outcome(g: pd.DataFrame, idx, estar: float, success_eps: float):
    """(cost in xTB calls, success bool) for a stop at position idx (None = ran to
    the end of the logged trajectory = the ceiling/timeout)."""
    row = g.iloc[idx] if idx is not None else g.iloc[-1]
    return int(row["n_calls"]), bool(row["e_best"] - estar <= success_eps)


# ---------------------------------------------------------------------------
# Kaplan-Meier survival + n*(d; rho)
# ---------------------------------------------------------------------------


def km_quantiles(times, events, rhos=RHOS):
    """Kaplan-Meier (right-censored) hitting-time quantiles n*(rho).

    times: hitting time if event else censoring horizon. events: 1 if hit.
    Returns {rho: n* or nan} where n* is the smallest t with CDF F(t) >= rho.
    Returns nan when censoring prevents F from reaching rho (the budget is then a
    lower bound -- flagged by the caller).
    """
    times = np.asarray(times, float)
    events = np.asarray(events, int)
    order = np.argsort(times)
    times, events = times[order], events[order]
    n = len(times)
    surv, at_risk = 1.0, n
    out = {r: np.nan for r in rhos}
    remaining = set(rhos)
    for t in np.unique(times):
        d = int(events[times == t].sum())
        c = int((times == t).sum())
        if at_risk > 0 and d > 0:
            surv *= (1 - d / at_risk)
        at_risk -= c
        F = 1 - surv
        for r in sorted(remaining):
            if F >= r:
                out[r] = float(t)
                remaining.discard(r)
    return out


def bootstrap_nstar(ev: pd.DataFrame, rho: float, n_boot: int, rng) -> tuple:
    """Bootstrap CI (2.5/97.5%) for n*(rho) over trials within a d-bin."""
    times = np.where(ev.hit, ev.ncalls_hit, ev.ncalls_max).astype(float)
    events = ev.hit.astype(int).to_numpy()
    if events.sum() == 0:
        return (np.nan, np.nan)
    vals = []
    m = len(ev)
    for _ in range(n_boot):
        i = rng.integers(0, m, m)
        q = km_quantiles(times[i], events[i], (rho,))[rho]
        if not np.isnan(q):
            vals.append(q)
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def nstar_table(ev: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """n*(d; rho) per dihedral count with bootstrap CIs and censoring flag."""
    rng = np.random.default_rng(seed)
    rows = []
    for d, g in ev.groupby("d"):
        times = np.where(g.hit, g.ncalls_hit, g.ncalls_max).astype(float)
        q = km_quantiles(times, g.hit.astype(int).to_numpy())
        rec = {"d": int(d), "n_trials": len(g), "n_hit": int(g.hit.sum()),
               "reliability": g.hit.mean()}
        for r in RHOS:
            lo, hi = bootstrap_nstar(g, r, n_boot, rng)
            rec[f"nstar_{int(r*100)}"] = q[r]
            rec[f"nstar_{int(r*100)}_lo"] = lo
            rec[f"nstar_{int(r*100)}_hi"] = hi
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("d").reset_index(drop=True)


def fit_scaling(d, y, through_origin=True):
    """Fit n* ~ a + b*d^p (least squares over a p grid). Returns (a,b,p,rmse).

    ``through_origin`` (default) forces a=0: a molecule with no dihedrals needs no
    search, so a positive-or-zero intercept is the physical constraint, and a free
    fit can return an unphysical negative intercept (usually an artifact of an
    under-populated/outlier d-bin -- filter those out before fitting). Forcing the
    origin is typically within a few percent RMSE of the free fit on clean data.
    """
    d = np.asarray(d, float)
    y = np.asarray(y, float)
    m = np.isfinite(y) & (d > 0)
    d, y = d[m], y[m]
    if len(d) < 3:
        return None
    best = None
    for p in np.linspace(0.5, 3.0, 101):
        if through_origin:
            X = (d ** p)[:, None]
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            coef = (0.0, float(b[0]))
            pred = X @ b
        else:
            X = np.column_stack([np.ones_like(d), d ** p])
            c = np.linalg.lstsq(X, y, rcond=None)[0]
            coef = (float(c[0]), float(c[1]))
            pred = X @ c
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        if best is None or rmse < best[-1]:
            best = (coef[0], coef[1], float(p), rmse)
    return best


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_summary(df, args, out: Path):
    df = add_estar(df, getattr(args, 'ref_map', None))
    ev = build_events(df, args.eps)
    n_seeds = df.groupby("name").seed.nunique()
    print(f"molecules={df.name.nunique()}  trials={len(ev)}  "
          f"seeds/mol: min={n_seeds.min()} max={n_seeds.max()}")
    print(f"d range: {int(df.d.min())}-{int(df.d.max())}  "
          f"censored trials: {int(ev.run_censored.sum())}")
    print(f"\nenergy-criterion reliability (eps={args.eps} kcal/mol), overall: "
          f"{ev.hit.mean():.3f}")
    print("\nper-d: trials / hit / reliability / median calls-to-hit")
    for d, g in ev.groupby("d"):
        med = np.nanmedian(g.loc[g.hit, "ncalls_hit"]) if g.hit.any() else float("nan")
        print(f"  d={int(d):2d}: {len(g):3d} / {int(g.hit.sum()):3d} / "
              f"{g.hit.mean():.2f} / {med:.0f}")
    # Ring-flexibility stratification (Issue 2): BO-reachable (acyclic/aromatic) vs
    # flexible-ring (embedding/seed-gated, not budget-gated).
    classes = getattr(args, "classes", None)
    if classes:
        ev["ring_class"] = ev["name"].map(classes).fillna("?")
        print("\nreliability by ring class (acyclic/aromatic = BO-reachable; "
              "flexible-ring = embedding-gated):")
        for c, g in ev.groupby("ring_class"):
            print(f"  {c:14s} n={len(g):3d}  reliability={g.hit.mean():.3f}")
    ev.to_csv(out / "events.csv", index=False)
    print(f"\nwrote {out/'events.csv'}")
    return ev


def reliability_by_class(ev: pd.DataFrame, classes: dict | None, out: Path,
                         fname: str = "reliability_by_class.csv") -> None:
    """Print + write hit reliability stratified by ring-flexibility class
    (acyclic = straight-chain, flexible-ring = puckerable, aromatic = rigid rings).
    The stratification the ring-pucker story needs: flexible-ring misses are
    embedding/seed-gated (BO can't change pucker), acyclic/aromatic are BO-reachable
    so their misses are search/budget-limited."""
    if not classes:
        print("\n(reliability-by-class needs --manifest for ring classes)")
        return
    ev = ev.copy()
    ev["ring_class"] = ev["name"].map(classes).fillna("?")
    print("\nreliability by ring class (acyclic=straight-chain, "
          "flexible-ring=puckerable, aromatic=rigid):")
    overall = ev.groupby("ring_class").agg(
        n_trials=("hit", "size"), n_mols=("name", "nunique"),
        n_hit=("hit", "sum"), reliability=("hit", "mean"))
    with pd.option_context("display.width", 200):
        print(overall.round(3).to_string())
    # class x d grid of reliability (the ring-pucker signal is d-dependent)
    grid = ev.pivot_table(index="ring_class", columns="d", values="hit", aggfunc="mean")
    print("\nreliability by ring class x d:")
    with pd.option_context("display.width", 250):
        print(grid.round(2).to_string())
    overall.to_csv(out / fname)
    print(f"wrote {out / fname}")


def cmd_nstar(df, args, out: Path):
    if getattr(args, "reopt", False):
        # Unconstrained-reopt energy criterion (correct: matches the reference scale).
        ev = build_events_reopt(df, args)
        ev.to_csv(out / "events_nstar_reopt.csv", index=False)
    else:
        df = add_estar(df, getattr(args, 'ref_map', None))
        ev = build_events(df, args.eps)

    # AUTO_STEPS calibration discipline: the step budget is a SEARCH-DEPTH lever, but
    # flexible-ring misses are an EMBEDDING/pucker lever (BO never moves ring bonds), so
    # they plateau below reliability 1 no matter the budget -- including them inflates
    # the fitted step formula futilely. --fit-classes restricts the SCALING FIT (not the
    # reliability report) to the BO-reachable classes; default keeps all for compatibility.
    ev_fit = ev
    fit_classes = getattr(args, "fit_classes", None)
    classes = getattr(args, "classes", None)
    if fit_classes and classes:
        keep = set(fit_classes.split(","))
        cls = ev["name"].map(classes)
        ev_fit = ev[cls.isin(keep)]
        print(f"[fit restricted to ring classes {sorted(keep)}: "
              f"{len(ev_fit)}/{len(ev)} trials feed the n*(d) scaling]")

    tab = nstar_table(ev, args.bootstrap, args.seed)
    tab_fit = nstar_table(ev_fit, args.bootstrap, args.seed) if ev_fit is not ev else tab
    tab.to_csv(out / "nstar.csv", index=False)
    print("n*(d; rho) [xTB calls], censoring-aware (nan = censoring prevents rho):")
    with pd.option_context("display.width", 200):
        print(tab[["d", "n_trials", "n_hit", "reliability",
                   "nstar_90", "nstar_95", "nstar_99"]].to_string(index=False))
    # Fit only on adequately-populated d-bins; a single-molecule bin is an outlier
    # that distorts the scaling law (and drove the earlier negative intercept).
    ft = tab_fit[tab_fit.n_trials >= args.min_fit_trials]
    origin = not args.free_intercept
    fits = {}
    for r in RHOS:
        f = fit_scaling(ft.d, ft[f"nstar_{int(r*100)}"], through_origin=origin)
        if f:
            fits[r] = f
            print(f"  scaling n*_{int(r*100)} ~ {f[0]:.1f} + {f[1]:.2f}*d^{f[2]:.2f} "
                  f"(rmse {f[3]:.1f}{', origin' if origin else ''})")
    _plot_nstar(tab, fits, out / "nstar.pdf")
    (out / "fitted.json").write_text(json.dumps(
        {"eps_kcal": args.eps,
         "nstar": tab.to_dict(orient="records"),
         "scaling": {str(r): {"a": f[0], "b": f[1], "p": f[2], "rmse": f[3]}
                     for r, f in fits.items()}}, indent=2))
    print(f"wrote {out/'nstar.csv'}, {out/'nstar.pdf'}, {out/'fitted.json'}")
    reliability_by_class(ev, getattr(args, "classes", None), out)
    return tab


def cmd_pareto(df, args, out: Path):
    df = add_estar(df, getattr(args, 'ref_map', None))
    bcols, betas = beta_columns(df)
    trials = list(df.groupby(["name", "seed"]))

    def sweep(make_rule, grid):
        pts = []
        for thr in grid:
            costs, succ = [], []
            for (name, seed), g in trials:
                d = int(g["d"].iloc[0]); estar = g["estar"].iloc[0]
                idx = make_rule(thr)(g, d)
                c, s = _stop_outcome(g, idx, estar, args.eps)
                costs.append(c); succ.append(s)
            pts.append((thr, float(np.mean(costs)), float(np.mean(succ))))
        return pts

    results = {}
    # certificate: sweep eps at a fixed beta (default the middle of the grid)
    bcol = args.cert_beta_col if args.cert_beta_col in bcols else bcols[len(bcols)//2]
    results[f"certificate({bcol})"] = sweep(
        lambda e: (lambda g, d: rule_certificate(g, e, bcol)), args.cert_eps_grid)
    results["logei"] = sweep(
        lambda t: (lambda g, d: rule_logei(g, t)), args.logei_grid)
    results["stall(k=%d)" % args.stall_k] = sweep(
        lambda dl: (lambda g, d: rule_stall(g, dl, args.stall_k)), args.stall_grid)
    # baseline table = single point
    bcosts, bsucc = [], []
    for (name, seed), g in trials:
        d = int(g["d"].iloc[0]); estar = g["estar"].iloc[0]
        c, s = _stop_outcome(g, rule_table(g, d), estar, args.eps)
        bcosts.append(c); bsucc.append(s)
    table_pt = (float(np.mean(bcosts)), float(np.mean(bsucc)))

    print(f"Pareto (success eps={args.eps} kcal/mol). baseline auto-table: "
          f"cost={table_pt[0]:.0f} calls, reliability={table_pt[1]:.3f}")
    for rule, pts in results.items():
        print(f"\n{rule}:  threshold -> (mean calls, reliability)")
        for thr, c, r in pts:
            print(f"  {thr:8.3g}: {c:7.0f}  {r:.3f}")
    _plot_pareto(results, table_pt, out / "pareto.pdf")
    json.dump({"table": table_pt, "rules": results},
              open(out / "pareto.json", "w"), indent=2)
    print(f"\nwrote {out/'pareto.pdf'}, {out/'pareto.json'}")
    return results


def cmd_calibrate(df, args, out: Path):
    df = add_estar(df, getattr(args, 'ref_map', None))
    bcols, betas = beta_columns(df)
    trials = list(df.groupby(["name", "seed"]))
    rows = []
    for bcol, beta in zip(bcols, betas):
        costs, succ = [], []
        for (name, seed), g in trials:
            estar = g["estar"].iloc[0]
            idx = rule_certificate(g, args.cert_eps, bcol)
            c, s = _stop_outcome(g, idx, estar, args.eps)
            costs.append(c); succ.append(s)
        rows.append({"beta": beta, "mean_calls": float(np.mean(costs)),
                     "reliability": float(np.mean(succ))})
    tab = pd.DataFrame(rows)
    print(f"certificate calibration (stop when e_best-lb(beta) < {args.cert_eps} "
          f"kcal/mol; success eps={args.eps}):")
    print(tab.to_string(index=False))
    _plot_calibration(tab, out / "calibration.pdf")
    tab.to_csv(out / "calibration.csv", index=False)
    print(f"wrote {out/'calibration.pdf'}, {out/'calibration.csv'}")
    return tab


# ---------------------------------------------------------------------------
# RMSD-identity criterion (geometry, not energy)
# ---------------------------------------------------------------------------


def _load_ref_min(ref_dir: Path, mol_id: str):
    """The reference global-minimum geometry (frame 0 of <mol_id>.xyz, written by
    reference.py sorted by energy ascending), or None if absent."""
    from ase.io import read
    f = ref_dir / f"{mol_id}.xyz"
    if not f.exists():
        return None
    try:
        return read(f, index="0")
    except Exception:
        return None


def _safe_filename(name: str) -> str:
    """Sanitize a molecule name to a filename component -- must match the form
    sweep_common._safe_filename uses at write time, else the trail file isn't found."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "mol"


def _trail_frames(geom_dir: Path, config: str, mol_id: str, seed):
    """Geometry-trail frames for one trial as (n_calls, atoms), in n_calls order.
    Frames are the best-so-far improvements + final relaxed best (init_best too)."""
    from ase.io import read
    f = geom_dir / f"{config}_{_safe_filename(mol_id)}_seed{seed}.xyz"
    if not f.exists():
        return []
    frames = read(f, index=":")
    ncalls = []
    for ln in open(f):
        if "kind=" in ln:
            kv = dict(t.split("=", 1) for t in ln.split() if "=" in t)
            ncalls.append(int(kv.get("n_calls", "0")))
    out = list(zip(ncalls, frames))
    out.sort(key=lambda x: x[0])
    return out


def cmd_rmsd(df, args, out: Path):
    """RMSD-identity criterion: when did the run first reach the global-min BASIN
    (best-so-far geometry within --rmsd-identity of the reference global minimum),
    vs the energy criterion. Resolves the 'lower energy but same/wrong basin' issue:
    a run can be energy-close yet a distinct conformer, or in the right basin but
    not energy-converged. Reuses bouquet's symmetry-aware iRMSD (_rmsd)."""
    from bouquet.ensemble import _rmsd
    if not args.geom_dir or not args.reference:
        sys.exit("rmsd needs --geom-dir (trails) and --reference (reference dir)")
    ref_dir = Path(args.reference)
    if not ref_dir.is_dir():
        sys.exit("--reference must be the reference DIR (with <mol_id>.xyz) for rmsd")
    df = add_estar(df, getattr(args, "ref_map", None))
    thr = args.rmsd_identity
    rows = []
    for (name, seed), g in df.groupby(["name", "seed"]):
        gmin = _load_ref_min(ref_dir, name)
        frames = _trail_frames(Path(args.geom_dir), g["config"].iloc[0], name, seed)
        if gmin is None or not frames:
            continue
        estar = g["estar"].iloc[0]
        # energy hit: first call within eps of E*
        ehit = g.loc[g["e_best"] - estar <= args.eps, "n_calls"]
        t_energy = int(ehit.iloc[0]) if len(ehit) else np.nan
        # rmsd hit: first trail frame within thr of the global-min geometry
        t_rmsd = np.nan
        best_rmsd = np.inf
        for nc, atoms in frames:
            try:
                r = _rmsd(atoms, gmin)
            except Exception:
                continue
            best_rmsd = min(best_rmsd, r)
            if r < thr and np.isnan(t_rmsd):
                t_rmsd = nc
        rows.append({
            "name": name, "seed": seed, "d": int(g["d"].iloc[0]),
            "hit_energy": not np.isnan(t_energy), "t_energy": t_energy,
            "hit_rmsd": not np.isnan(t_rmsd), "t_rmsd": t_rmsd,
            "best_rmsd_to_gmin": round(float(best_rmsd), 3),
        })
    ev = pd.DataFrame(rows)
    if ev.empty:
        sys.exit("no trials had both a reference geometry and a trail")
    ev.to_csv(out / "events_rmsd.csv", index=False)
    print(f"RMSD-identity vs energy criterion (rmsd_identity={thr} A, eps={args.eps} "
          f"kcal/mol). {len(ev)} trials with geometry + reference.\n")
    print("per-d:  reliability(energy)  reliability(RMSD)  median t_energy  t_rmsd")
    for d, gg in ev.groupby("d"):
        te = np.nanmedian(gg.loc[gg.hit_energy, "t_energy"]) if gg.hit_energy.any() else float("nan")
        tr = np.nanmedian(gg.loc[gg.hit_rmsd, "t_rmsd"]) if gg.hit_rmsd.any() else float("nan")
        print(f"  d={int(d):2d}:  {gg.hit_energy.mean():.2f}             "
              f"{gg.hit_rmsd.mean():.2f}              {te:6.0f}        {tr:6.0f}")
    # disagreements: where the two criteria diverge
    e_not_r = ev[(ev.hit_energy) & (~ev.hit_rmsd)]
    r_not_e = ev[(~ev.hit_energy) & (ev.hit_rmsd)]
    print(f"\noverall: energy {ev.hit_energy.mean():.3f}  vs  RMSD {ev.hit_rmsd.mean():.3f}")
    print(f"  energy-hit but NOT RMSD-identical: {len(e_not_r)} "
          f"(energy-close but a distinct/near-degenerate conformer -- the case you flagged)")
    print(f"  RMSD-identical but NOT energy-hit: {len(r_not_e)} "
          f"(in the right basin but >eps from E*, e.g. not fully relaxed)")
    print(f"wrote {out/'events_rmsd.csv'}")
    return ev


# ---------------------------------------------------------------------------
# Ring-flexibility class (Issue 2 covariate) + self-referential basin (Issue 1)
# ---------------------------------------------------------------------------


def ring_class(smiles: str) -> str:
    """Classify a molecule by ring flexibility -- the key reliability covariate.

    'acyclic' and 'aromatic' (rings all aromatic) are BO-REACHABLE: every flexible
    DOF is a rotatable dihedral, so BO can in principle reach the global min and
    failures are search-limited. 'flexible-ring' has a non-aromatic ring with an
    sp3 atom (puckerable): BO cannot change ring pucker from a fixed embedding, so
    reliability is gated by embedding/seed diversity, not BO budget.
    """
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "?"
    ri = mol.GetRingInfo()
    if ri.NumRings() == 0:
        return "acyclic"
    for ring in ri.AtomRings():
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if all(a.GetIsAromatic() for a in atoms):
            continue
        # A single sp3 apex in an otherwise sp2/aromatic-fused ring (e.g. fluorene's
        # C9, an N-alkyl carbazole bridge) is RIGID -- it cannot pucker -- so it is a
        # BO-reachable aromatic-backbone case, not an embedding-gated one. Require >=2
        # sp3 ring atoms before calling a non-aromatic ring puckerable; saturated rings
        # (cyclohexane, piperidine, THF, ...) all have >=2 and stay flexible-ring.
        n_sp3 = sum(1 for a in atoms
                    if a.GetHybridization() == Chem.HybridizationType.SP3)
        if n_sp3 >= 2:
            return "flexible-ring"
    return "aromatic"


def load_classes(manifest: Path) -> dict:
    """mol_id -> ring_class from a select_smiles.py manifest CSV."""
    man = pd.read_csv(manifest)
    return {r.mol_id: ring_class(r.raw_smiles) for _, r in man.iterrows()}


def _trail_full(geom_dir: Path, config: str, mol_id: str, seed):
    """Trail frames as (n_calls, e_e0_eV, atoms) in n_calls order."""
    from ase.io import read
    f = geom_dir / f"{config}_{_safe_filename(mol_id)}_seed{seed}.xyz"
    if not f.exists():
        return []
    frames = read(f, index=":")
    out = []
    for a, ln in zip(frames, [c for c in open(f) if "kind=" in c]):
        kv = dict(t.split("=", 1) for t in ln.split() if "=" in t)
        out.append((int(kv.get("n_calls", "0")), float(kv.get("e_e0_eV", "nan")), a))
    out.sort(key=lambda x: x[0])
    return out


def cmd_basin(df, args, out: Path):
    """Issue 1 -- search efficiency: when does a run first ENTER the basin of its
    own best conformer? Self-referential (no external reference): the hitting time
    is the first trail frame within --rmsd-identity of the run's lowest-energy
    frame, so intra-basin energy refinement after that doesn't count. This is the
    budget the CAP should target -- it ignores the post-basin energy creep that
    inflates the 'step with best energy'. n*(d) here is identifiable (every run
    enters its own best basin -> reliability 1)."""
    from bouquet.ensemble import _rmsd
    if not args.geom_dir:
        sys.exit("basin needs --geom-dir (geometry trails)")
    thr = args.rmsd_identity
    classes = getattr(args, "classes", None)
    rows = []
    for (name, seed), g in df.groupby(["name", "seed"]):
        fr = _trail_full(Path(args.geom_dir), g["config"].iloc[0], name, seed)
        if len(fr) < 2:
            continue
        bi = min(range(len(fr)), key=lambda i: fr[i][1])   # lowest-energy frame
        best = fr[bi][2]
        t_basin = next((nc for nc, e, a in fr if _rmsd(a, best) < thr), fr[bi][0])
        rows.append({
            "name": name, "seed": seed, "d": int(g["d"].iloc[0]),
            "ring_class": classes.get(name, "?") if classes else "?",
            "t_basin": t_basin, "t_bestE": fr[bi][0],
            "wasted": fr[bi][0] - t_basin,
        })
    ev = pd.DataFrame(rows)
    if ev.empty:
        sys.exit("no trails found under --geom-dir")
    ev.to_csv(out / "events_basin.csv", index=False)
    # n*(d): KM on basin-discovery time (all hit -> identifiable quantiles = the cap)
    ev["hit"] = True; ev["ncalls_hit"] = ev["t_basin"]; ev["ncalls_max"] = ev["t_basin"]
    tab = nstar_table(ev[["d", "hit", "ncalls_hit", "ncalls_max"]], args.bootstrap, args.seed)
    print(f"Self-referential BASIN-discovery budget (Issue 1; rmsd_identity={thr} A). "
          f"{len(ev)} trials.")
    print(f"basin found earlier than best-energy step in {(ev.wasted>0).mean():.0%} of "
          f"runs; median wasted calls = {ev.loc[ev.wasted>0,'wasted'].median():.0f}\n")
    print("n*(d) to ENTER the best basin [xTB calls] vs the best-ENERGY step:")
    med_bestE = ev.groupby("d").t_bestE.median()
    with pd.option_context("display.width", 200):
        show = tab[["d", "nstar_90", "nstar_95", "nstar_99"]].copy()
        show["median_bestE_step"] = show.d.map(med_bestE).values
        print(show.to_string(index=False))
    origin = not args.free_intercept
    f = fit_scaling(tab[tab.n_trials >= args.min_fit_trials].d,
                    tab[tab.n_trials >= args.min_fit_trials]["nstar_95"],
                    through_origin=origin)
    if f:
        print(f"\nbasin n*_95 ~ {f[0]:.1f} + {f[1]:.2f}*d^{f[2]:.2f} "
              f"(rmse {f[3]:.1f}{', origin' if origin else ''})")
    if classes:
        print("\nbasin-discovery median calls by ring class:")
        for c, gg in ev.groupby("ring_class"):
            print(f"  {c:14s} n={len(gg):3d}  median t_basin={gg.t_basin.median():.0f}")
    tab.to_csv(out / "nstar_basin.csv", index=False)
    print(f"\nwrote {out/'events_basin.csv'}, {out/'nstar_basin.csv'}")
    return ev


# ---------------------------------------------------------------------------
# Ceiling-adequacy check (is the benchmark C(d) high enough?)
# ---------------------------------------------------------------------------


def benchmark_ceiling(d, anchor, p, margin, cap):
    """The benchmark's per-molecule run ceiling C(d) (matches stop_benchmark.py)."""
    return int(min(cap, math.ceil(margin * anchor * (d / 3.0) ** p)))


def cmd_ceiling_check(df, args, out: Path):
    """Was the benchmark ceiling C(d) high enough, or did it truncate the search?

    Within-trial test (no cross-seed energy issue): for every trial that ran PAST
    the original C(d) -- e.g. the extended 2x-ceiling seeds -- compare e_best at
    C(d) to e_best at the end. A meaningful drop AFTER C(d) means the original cap
    truncated real improvement (n* and E* are then lower bounds at that d). If drops
    are rare/tiny, C(d) was adequate and the basin cap stands.
    """
    rows = []
    for (name, seed), g in df.groupby(["name", "seed"]):
        g = g.sort_values("n_calls")
        d = int(g["d"].iloc[0])
        cd = benchmark_ceiling(d, args.ceiling_anchor, args.ceiling_p,
                               args.ceiling_margin, args.ceiling_cap)
        # C(d) is in BO steps; n_calls = start + init + BO, so the original ceiling
        # in n_calls is offset by the trial's pre-BO count (first row = BO step 0).
        boundary = int(g["n_calls"].iloc[0] - 1 + cd)
        if g["n_calls"].max() <= boundary:
            continue  # ran only to the original ceiling: no data beyond
        eb_at = g.loc[g["n_calls"] <= boundary, "e_best"].min()  # best by the ceiling
        eb_final = g["e_best"].min()                             # best with extension
        # n_calls where the post-ceiling best was reached (how far past the ceiling)
        reached = g.loc[g["e_best"] <= eb_final + 1e-9, "n_calls"].iloc[0]
        rows.append({"name": name, "seed": seed, "d": d, "cd": cd,
                     "boundary": boundary, "max_calls": int(g["n_calls"].max()),
                     "improve_beyond": float(eb_at - eb_final),  # kcal, >=0
                     "best_at_ncalls": int(reached)})
    ev = pd.DataFrame(rows)
    if ev.empty:
        sys.exit("No trials ran beyond C(d). Run extended-ceiling seeds first "
                 "(stop_benchmark.py ... --ceiling-margin 6).")
    ev.to_csv(out / "ceiling_check.csv", index=False)
    eps = args.improve_eps
    print(f"Ceiling-adequacy: {len(ev)} trials ran beyond the original C(d) "
          f"(anchor={args.ceiling_anchor}, p={args.ceiling_p}, margin={args.ceiling_margin}).")
    print(f"\n d : n / %improved>{eps}kcal beyond C(d) / median improve / max improve / "
          f"median(best_step/C(d))")
    for d, gg in ev.groupby("d"):
        frac = (gg.improve_beyond > eps).mean()
        ratio = (gg.best_at_ncalls / gg.boundary).median()
        print(f"{int(d):2d}: {len(gg):3d} / {frac:.0%} / "
              f"{gg.improve_beyond.median():.2f} / {gg.improve_beyond.max():.2f} / {ratio:.2f}")
    big = ev[ev.improve_beyond > eps]
    print(f"\nOVERALL: {(ev.improve_beyond>eps).mean():.0%} of extended runs improved "
          f">{eps} kcal beyond the original ceiling.")
    if len(big):
        print(f"  among those: median improvement {big.improve_beyond.median():.2f} kcal, "
              f"best reached at median {(big.best_at_ncalls/big.boundary).median():.2f}x the "
              f"original ceiling (in n_calls).")
    print("VERDICT:", "ceiling likely TOO LOW where %improved is high (revise cap up)"
          if (ev.improve_beyond > eps).mean() > 0.1 else
          "ceiling looks adequate (few/small improvements beyond C(d))")
    print(f"wrote {out/'ceiling_check.csv'}")
    return ev


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _ax():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_nstar(tab, fits, path):
    plt = _ax()
    fig, ax = plt.subplots(figsize=(8, 5))
    for r, c in zip(RHOS, ["tab:blue", "tab:orange", "tab:red"]):
        y = tab[f"nstar_{int(r*100)}"]
        lo = tab[f"nstar_{int(r*100)}_lo"]; hi = tab[f"nstar_{int(r*100)}_hi"]
        ax.errorbar(tab.d, y, yerr=[y - lo, hi - y], fmt="o", color=c,
                    capsize=2, label=f"n* (rho={r})")
        if r in fits:
            a, b, p, _ = fits[r]
            xs = np.linspace(tab.d.min(), tab.d.max(), 100)
            ax.plot(xs, a + b * xs ** p, "-", color=c, alpha=0.6,
                    label=f"  fit {a:.0f}+{b:.1f}d^{p:.2f}")
    ax.set_xlabel("d (rotatable dihedrals)")
    ax.set_ylabel("n* (xTB calls to global min)")
    ax.set_title("Required budget vs dihedral count (Kaplan-Meier)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130)


def _plot_pareto(results, table_pt, path):
    plt = _ax()
    fig, ax = plt.subplots(figsize=(8, 5))
    for rule, pts in results.items():
        pts = sorted(pts, key=lambda x: x[1])
        c = [p[1] for p in pts]; r = [p[2] for p in pts]
        ax.plot(c, r, "-o", ms=3, label=rule)
    ax.scatter([table_pt[0]], [table_pt[1]], marker="*", s=180, color="black",
               zorder=5, label="auto-table (baseline)")
    ax.set_xlabel("mean cost (xTB calls)")
    ax.set_ylabel("realized reliability")
    ax.set_title("Stopping-rule Pareto front")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130)


def _plot_calibration(tab, path):
    plt = _ax()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(tab.beta, tab.reliability, "-o", color="tab:purple")
    ax.set_xlabel("beta (certificate confidence multiplier)")
    ax.set_ylabel("realized reliability")
    ax.set_ylim(0, 1.02)
    ax.set_title("Certificate calibration")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130)


# ---------------------------------------------------------------------------


def _floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command",
                   choices=["summary", "nstar", "pareto", "calibrate", "rmsd",
                            "basin", "ceiling-check", "all"])
    p.add_argument("cert", type=Path, help="Certificate CSV (*_cert.csv)")
    p.add_argument("--out-dir", type=Path, default=Path("."))
    p.add_argument("--eps", type=float, default=0.5,
                   help="Energy success tolerance |e_best-E*| (kcal/mol, default 0.5)")
    p.add_argument("--bootstrap", type=int, default=500,
                   help="Bootstrap resamples for n* CIs (default 500)")
    p.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed")
    p.add_argument("--cert-eps-grid", type=_floats,
                   default=[0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    p.add_argument("--cert-beta-col", default="lb_b1",
                   help="Certificate beta column for the Pareto sweep (default lb_b1)")
    p.add_argument("--cert-eps", type=float, default=0.5,
                   help="Fixed certificate eps for the calibration-vs-beta curve")
    p.add_argument("--logei-grid", type=_floats, default=[0.01, 0.02, 0.05, 0.1, 0.2])
    p.add_argument("--stall-grid", type=_floats, default=[0.02, 0.05, 0.1, 0.2])
    p.add_argument("--stall-k", type=int, default=20)
    p.add_argument("--reference", type=Path, default=None,
                   help="reference.py output (reference_summary.csv or the reference "
                   "dir) to use as the non-circular E*; without it E* is the "
                   "within-data pool min (reliability ~1). The 'rmsd' command needs "
                   "the reference DIR (for <mol_id>.xyz global-min geometries).")
    p.add_argument("--geom-dir", type=Path, default=None,
                   help="Geometry-trail dir (*_geom) for the 'rmsd' command.")
    p.add_argument("--rmsd-identity", type=float, default=0.5,
                   help="Heavy-atom iRMSD (A) for 'found THE conformer' (default 0.5)")
    p.add_argument("--manifest", type=Path, default=None,
                   help="select_smiles.py manifest CSV; attaches a ring-flexibility "
                   "class (acyclic/aromatic/flexible-ring) for stratified reliability.")
    p.add_argument("--ceiling-anchor", type=float, default=25.0,
                   help="ceiling-check: original C(d) anchor (match the benchmark run)")
    p.add_argument("--ceiling-p", type=float, default=1.5,
                   help="ceiling-check: original C(d) exponent")
    p.add_argument("--ceiling-margin", type=float, default=3.0,
                   help="ceiling-check: ORIGINAL C(d) margin (the 1x run's margin, "
                   "default 3) -- the threshold past which improvement is measured")
    p.add_argument("--ceiling-cap", type=int, default=1500,
                   help="ceiling-check: original C(d) hard cap")
    p.add_argument("--improve-eps", type=float, default=0.5,
                   help="ceiling-check: kcal/mol drop beyond C(d) counted as a "
                   "meaningful improvement (default 0.5)")
    p.add_argument("--min-fit-trials", type=int, default=5,
                   help="Exclude d-bins with fewer trials from the scaling-law fit "
                   "(under-populated bins are outliers; default 5).")
    p.add_argument("--reopt", action="store_true",
                   help="Energy criterion: re-optimize each trajectory best-so-far "
                   "geometry UNCONSTRAINED (on --reopt-method) and score vs the absolute "
                   "reference E*, instead of the constrained cert e_best. Fixes the ~2x "
                   "undercount from the constrained/unconstrained scale mismatch. Needs "
                   "--manifest (SMILES), --geom-dir, and --reference with E_start_eV.")
    p.add_argument("--reopt-method", default="gfn2",
                   help="Calculator for --reopt (match the benchmark surface; default gfn2)")
    p.add_argument("--reopt-cache", type=Path, default=None,
                   help="CSV to cache/reuse the re-optimized (n_calls, e_abs) curves so "
                   "the expensive xTB replay runs once (default <out-dir>/reopt_curves.csv)")
    p.add_argument("--reopt-refresh", action="store_true",
                   help="Ignore any --reopt-cache and re-run the re-optimization")
    p.add_argument("--workers", "-w", type=int, default=8,
                   help="Parallel workers for the --reopt re-optimization (default 8)")
    p.add_argument("--conn-tol", type=float, default=1.3,
                   help="--reopt: bond-perception tolerance for the connectivity guard "
                   "that drops species-changed frames (default 1.3)")
    p.add_argument("--fit-classes", default=None,
                   help="Restrict the n*(d) SCALING FIT (the AUTO_STEPS calibration) to "
                   "these comma-separated ring classes, e.g. 'acyclic,aromatic' -- the "
                   "BO-reachable set. Flexible-ring reliability is embedding-gated (a "
                   "different lever) and plateaus below 1, so including it inflates the "
                   "step budget. Reliability is still reported for ALL classes. Needs "
                   "--manifest for the class map.")
    p.add_argument("--free-intercept", action="store_true",
                   help="Let the scaling fit have a free intercept; default forces "
                   "n*(d)=b*d^p through the origin (a molecule with no dihedrals "
                   "needs no search).")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if getattr(args, "reopt", False) and args.reopt_cache is None:
        args.reopt_cache = args.out_dir / "reopt_curves.csv"
    args.ref_map = load_reference(args.reference) if args.reference else None
    if args.ref_map:
        n_crest, n_total = crest_coverage(args.reference)
        print(f"Loaded reference E* for {len(args.ref_map)} molecules "
              f"(non-circular success criterion); {n_crest}/{n_total} CREST-backed.")
    args.classes = load_classes(args.manifest) if args.manifest else None
    if args.classes:
        print(f"Loaded ring-flexibility classes for {len(args.classes)} molecules.")
    df = load_cert(args.cert)
    runners = {"summary": cmd_summary, "nstar": cmd_nstar, "basin": cmd_basin,
               "pareto": cmd_pareto, "calibrate": cmd_calibrate, "rmsd": cmd_rmsd,
               "ceiling-check": cmd_ceiling_check}
    if args.command == "all":
        for name in ["summary", "nstar", "pareto", "calibrate"]:
            print(f"\n===== {name} =====")
            runners[name](df, args, args.out_dir)
    else:
        runners[args.command](df, args, args.out_dir)


if __name__ == "__main__":
    main()
