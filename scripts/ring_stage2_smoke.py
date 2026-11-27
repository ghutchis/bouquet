#!/usr/bin/env python
"""Stage 2 smoke: does STARTING the BO from an MTD ring state close the gap to CREST?

Stage 1 measured seed recall -- whether the CREST ring conformer is in the seed set. That is
necessary, not sufficient. This measures the thing that actually matters: the energy bouquet
reaches, against the CREST global minimum.

**Why `--file` and not `--conformer-file`.** The conformer-file path cannot answer this
question. `solver._evaluate_point` applies each supplied conformer's DIHEDRAL ANGLES to the
single `state.start_atoms` geometry, so a seed's ring pucker is discarded; the seeds reach
only `ensemble_seed_geometries`, which feeds the returned ensemble (and only under
`--ensemble`), never the search. RING_MTD_v2 SS9's actual design -- one torsional BO per ring
state -- is expressed today by making each ring state the START structure, which is what
`--file` does.

Matched design, equal BO budget per arm:
  * ``baseline``  3 runs, ``--smiles`` with seeds 1/2/3 (bouquet's normal ETKDG starts);
  * ``ringseed``  3 runs, ``--file`` on each of the 3 lowest-energy MTD ring states.
Each arm's score is its best run, so the comparison is "3 BO runs either way".

Energies are made comparable by recomputing BOTH sides here: every arm's final geometry and
the CREST reference minimum are GFN2-relaxed in this process, so the gap is a difference of
absolute energies and no E0 bookkeeping can drift between arms.

Usage:
    pixi run python scripts/ring_stage2_smoke.py \
        --reference-dir stop_bench_gfn2/reference_gfn2 \
        --manifest smiles/stopbench-500.csv \
        --seeds-dir ring_mtd_bench/seeds_s0 \
        --mols m009_d01,m025_d02,m075_d04,m160_d07 \
        --workers 6 --out stop_bench_gfn2/analysis/ring_stage2_smoke.csv
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference as ref  # noqa: E402

KCAL = 23.0605  # eV -> kcal/mol

# Measured GFN2 seconds per BO step, by dihedral count, from the stop_bench_gfn2 certificate
# logs (median over 8-25 trials per d). The cost is NOT flat in d: relaxation grows with
# molecule size and each BO step refits a GP over all observations so far, so d=10 is ~50x
# d=4 PER STEP on top of needing ~16x more steps. Extrapolating a low-d rate across the whole
# range is what made the first Stage-2 array unfinishable (an 8 h wall against a d=10 molecule
# needing ~383 h for its 6 runs), so the budget is now checked before a molecule is started.
SECONDS_PER_STEP = {1: 8.2, 2: 1.0, 3: 1.3, 4: 2.5, 5: 13.5, 6: 36.7,
                    7: 88.3, 8: 98.0, 9: 105.3, 10: 117.7, 11: 130.0, 12: 107.3}


def predicted_run_hours(d: int, steps: int) -> float:
    """Predicted wall hours for ONE BO run at this d and budget."""
    return steps * SECONDS_PER_STEP.get(min(d, 12), SECONDS_PER_STEP[12]) / 3600.0


def _write_xyz(path: Path, numbers, coords) -> None:
    from ase import Atoms
    from ase.io import write
    write(str(path), Atoms(numbers=numbers, positions=coords), format="xyz")


def _final_energy(geom_file: Path, molh, charge: int, method: str):
    """Absolute GFN2 energy of a run's final geometry (None if the run produced none)."""
    if not geom_file.exists():
        return None
    frames = [a for a, _e, k in ref._read_trail(str(geom_file)) if k == "final"]
    if not frames:
        frames = [a for a, _e, _k in ref._read_trail(str(geom_file))][-1:]
    if not frames:
        return None
    from bouquet.setup import apply_charge_spin, default_multiplicity
    calc = ref._gfnff_calc(method, molh, charge)
    mult = default_multiplicity(frames[0], charge)
    best = None
    for a in frames:
        aa = a.copy()
        apply_charge_spin(aa, charge, mult)
        opt = ref._optimize(aa, calc)
        if opt is not None and (best is None or opt[1] < best):
            best = opt[1]
    return best


def _run_bouquet(args_list, timeout: int, cwd) -> int:
    """Run the CLI in `cwd` -- bouquet creates its own output directory from --name in the
    working directory, so running from a scratch dir keeps the repo clean."""
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    try:
        p = subprocess.run([sys.executable, "-m", "bouquet.cli", *args_list],
                           capture_output=True, text=True, timeout=timeout, env=env,
                           cwd=str(cwd))
        return p.returncode
    except subprocess.TimeoutExpired:
        return 124


def smoke(task: dict) -> dict:
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"

    from rdkit import Chem

    mid, smiles = task["mol_id"], task["smiles"]
    row = {"mol_id": mid, "d": task["d"], "stratum": task.get("stratum", ""),
           "num_steps": task["num_steps"], "n_runs_per_arm": task["n_runs"]}
    try:
        mol = Chem.MolFromSmiles(smiles)
        molh = Chem.AddHs(mol)
        charge = Chem.GetFormalCharge(mol)

        # CREST global minimum, re-relaxed here so both sides share a footing.
        crest_min = ref._read_refined_xyz(task["ref_xyz"])[:1] or None
        if not crest_min:
            row["note"] = "no reference geometry"
            return row
        from bouquet.setup import apply_charge_spin, default_multiplicity
        calc = ref._gfnff_calc(task["method"], molh, charge)
        mult = default_multiplicity(crest_min[0], charge)
        a = crest_min[0].copy()
        apply_charge_spin(a, charge, mult)
        opt = ref._optimize(a, calc)
        if opt is None:
            row["note"] = "reference relaxation failed"
            return row
        e_crest = opt[1]

        seeds = json.loads(Path(task["seeds_file"]).read_text())
        numbers = seeds["molecule"]["atomic_numbers"]
        n = min(task["n_runs"], len(seeds["seeds"]))

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            results = {}
            n_censored = 0
            t0 = time.time()
            for arm in ("baseline", "ringseed"):
                energies = []
                for i in range(n):
                    geom = td / f"{arm}_{i}.xyz"
                    common = ["--num-steps", str(task["num_steps"]),
                              "--relax", "--energy", task["method"],
                              "--optimizer", task["method"],
                              "--geometry-log", str(geom),
                              "--name", f"{mid}_{arm}_{i}"]
                    if arm == "baseline":
                        cmd = ["--smiles", smiles, "--seed", str(i + 1)] + common
                    else:
                        start = td / f"seed{i}.xyz"
                        _write_xyz(start, numbers, seeds["seeds"][i]["coords"])
                        cmd = ["--file", str(start), "--seed", "1"] + common
                    rc = _run_bouquet(cmd, task["timeout"], td)
                    if rc == 124:
                        n_censored += 1
                    e = _final_energy(geom, molh, charge, task["method"]) if rc == 0 else None
                    if e is not None:
                        energies.append(e)
                results[arm] = min(energies) if energies else None
                row[f"{arm}_n_ok"] = len(energies)
            row["wall_s"] = round(time.time() - t0, 1)
            # A wall-capped run censors ONE arm's best; the arms have different per-step
            # costs (a GFN-FF-quenched ring start vs an MMFF one), so a cap can truncate
            # them unequally. Flag the whole pair rather than silently comparing a complete
            # arm against a truncated one.
            row["censored"] = n_censored > 0
            row["n_censored_runs"] = n_censored

        for arm in ("baseline", "ringseed"):
            e = results[arm]
            row[f"{arm}_gap_kcal"] = round((e - e_crest) * KCAL, 3) if e is not None else None
        if results["baseline"] is not None and results["ringseed"] is not None:
            row["improvement_kcal"] = round(
                (results["baseline"] - results["ringseed"]) * KCAL, 3)
    except Exception as e:
        row["note"] = f"error: {e!r}"
    return row


COLS = ["mol_id", "d", "stratum", "num_steps", "n_runs_per_arm", "baseline_n_ok",
        "ringseed_n_ok", "baseline_gap_kcal", "ringseed_gap_kcal", "improvement_kcal",
        "censored", "n_censored_runs", "wall_s", "note"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--seeds-dir", type=Path, required=True)
    ap.add_argument("--mols", default="", help="explicit mol_id list (or use --strata)")
    ap.add_argument("--method", default="gfn2")
    ap.add_argument("--n-runs", type=int, default=3, help="BO runs per arm (matched budget)")
    ap.add_argument("--budget", default="auto",
                    help="BO steps per run: an integer, or 'auto' to use bouquet's calibrated "
                         "auto_total_budget(d). Passed as --num-steps so BOTH arms get "
                         "exactly the same budget -- never --auto plus a wall cap, which "
                         "censors the arms asymmetrically.")
    ap.add_argument("--stage1", type=Path, default=None,
                    help="Stage-1 CSV; with --strata, selects molecules by outcome stratum")
    ap.add_argument("--strata", default=None,
                    help="comma-separated: mtd_only,etkdg_only,both,neither. mtd_only is the "
                         "claimed payoff; the others are the counterweights that keep this "
                         "from being a selected-for-success sample")
    ap.add_argument("--budget-cap", type=int, default=0,
                    help="cap the per-run step budget (0 = uncapped). The high-d tail costs "
                         "most of the sweep; capping keeps it affordable and is harmless here "
                         "because this is a PAIRED test -- both arms get the same budget, so "
                         "the comparison holds even where the budget is short of convergence")
    ap.add_argument("--control-cap", type=int, default=0,
                    help="subsample the 'both'/'neither' control strata to this many each "
                         "(0 = keep all). The decisive strata (mtd_only, etkdg_only) are "
                         "never subsampled")
    ap.add_argument("--crest-wins-only", action="store_true",
                    help="restrict to molecules where CREST currently beats bouquet")
    ap.add_argument("--max-run-hours", type=float, default=0.0,
                    help="skip molecules whose PREDICTED per-run cost exceeds this (0 = no "
                         "limit). Uses the measured SECONDS_PER_STEP model; set it to about "
                         "a quarter of the SLURM wall so all 6 runs of a molecule fit")
    ap.add_argument("--resume", action="store_true",
                    help="append to --out and skip molecules already present (results are "
                         "written per molecule, so a requeued job picks up where it stopped)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=7200, help="per-BO-run cap, seconds")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import pandas as pd

    man = pd.read_csv(args.manifest).set_index("mol_id")
    smi_col = "raw_smiles" if "raw_smiles" in man.columns else "smiles"
    # Population: an explicit list, or strata read off the Stage-1 outcomes.
    if args.strata:
        if not args.stage1:
            ap.error("--strata needs --stage1")
        want = {s.strip() for s in args.strata.split(",")}
        def stratum(r):
            e, m = r.get("etkdg_hit_125") == "True", r.get("mtd_hit_125") == "True"
            return ("mtd_only" if m and not e else "etkdg_only" if e and not m
                    else "both" if e and m else "neither")
        buckets, strat = {}, {}
        for r in csv.DictReader(open(args.stage1)):
            if args.crest_wins_only and r.get("crest_wins") != "True":
                continue
            s = stratum(r)
            if s in want:
                buckets.setdefault(s, []).append(r["mol_id"]); strat[r["mol_id"]] = s
        ids = []
        for s, members in sorted(buckets.items()):
            if args.control_cap and s in ("both", "neither") and len(members) > args.control_cap:
                # Deterministic subsample so every shard and every re-run agrees on the set.
                rng = random.Random(20260725)
                members = sorted(rng.sample(sorted(members), args.control_cap))
                print(f"  {s}: subsampled to {len(members)}")
            ids += members
    else:
        ids = [m.strip() for m in args.mols.split(",")]
        strat = {}

    tasks, skipped = [], []
    for mid in ids:
        sf = args.seeds_dir / f"{mid}.json"
        rx = args.reference_dir / f"{mid}.xyz"
        if mid not in man.index or not sf.exists() or not rx.exists():
            print(f"  {mid}: missing manifest/seeds/reference, skipping")
            continue
        d = int(man.loc[mid, "d"]) if "d" in man.columns else 1
        if args.budget == "auto":
            from bouquet.config import auto_total_budget
            steps = auto_total_budget(d)
        else:
            steps = int(args.budget)
        if args.budget_cap:
            steps = min(steps, args.budget_cap)
        if args.max_run_hours and predicted_run_hours(d, steps) > args.max_run_hours:
            print(f"  {mid}: predicted {predicted_run_hours(d, steps):.1f} h/run at d={d}, "
                  f"over --max-run-hours {args.max_run_hours} -- skipping")
            skipped.append({"mol_id": mid, "d": d, "stratum": strat.get(mid, ""),
                            "num_steps": steps,
                            "note": f"skipped: predicted {predicted_run_hours(d, steps):.1f} "
                                    f"h/run exceeds --max-run-hours"})
            continue
        tasks.append({"mol_id": mid, "smiles": man.loc[mid, smi_col], "d": d,
                      "ref_xyz": str(rx), "seeds_file": str(sf), "method": args.method,
                      "n_runs": args.n_runs, "timeout": args.timeout,
                      "num_steps": steps, "stratum": strat.get(mid, "")})

    if args.nshards > 1:
        # COST-BALANCED sharding, not index-interleaved: the budget is ~d^3, so a d=13
        # molecule costs ~300x a d=1 one and a naive split would leave one shard running
        # for hours after the rest have finished. Deal the most expensive molecules
        # round-robin (longest-processing-time-first), which balances the shards well.
        ordered = sorted(tasks, key=lambda t: -t["num_steps"])
        tasks = [t for i, t in enumerate(ordered) if i % args.nshards == args.shard]
        print(f"shard {args.shard}/{args.nshards}: {len(tasks)} molecule(s), "
              f"{sum(t['num_steps'] for t in tasks):,} steps/arm/run")

    print(f"Stage-2 smoke on {len(tasks)} molecule(s): {args.n_runs} BO runs per arm "
          f"(baseline ETKDG starts vs MTD ring-state starts), {args.method}")
    # Append each molecule as it finishes. A wall-clock kill then costs only the molecule in
    # flight instead of the whole shard -- the failure mode that lost the first array.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and args.out.exists():
        done = {r["mol_id"] for r in csv.DictReader(open(args.out))}
        tasks = [t for t in tasks if t["mol_id"] not in done]
        print(f"resuming: {len(done)} molecule(s) already done, {len(tasks)} to go")
    write_header = not (args.resume and args.out.exists())
    fh = open(args.out, "a" if (args.resume and args.out.exists()) else "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
    if write_header:
        writer.writeheader(); fh.flush()

    rows = list(skipped)
    for r in skipped:
        writer.writerow(r)
    fh.flush()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(smoke, tasks):
            rows.append(r)
            writer.writerow(r); fh.flush()
            print(f"  {r['mol_id']:12} baseline {str(r.get('baseline_gap_kcal')):>8} kcal   "
                  f"ringseed {str(r.get('ringseed_gap_kcal')):>8} kcal   "
                  f"improvement {str(r.get('improvement_kcal')):>8}  {r.get('note','')}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")
    summarize(rows)


def summarize(rows) -> None:
    """Per-stratum effect of ring seeding. Censored pairs are excluded, not averaged in."""
    import collections

    cens = [r for r in rows if r.get("censored")]
    good = [r for r in rows if r.get("improvement_kcal") is not None and not r.get("censored")]
    if cens:
        print(f"\nexcluded {len(cens)} censored molecule(s) (a wall-capped run truncates one "
              f"arm only -- see the note in smoke())")
    if not good:
        print("no complete pairs")
        return
    by = collections.defaultdict(list)
    for r in good:
        by[r.get("stratum") or "all"].append(r["improvement_kcal"])
    print(f"\ngap to CREST, baseline minus ring-seeded (+ = ring seeding WON), "
          f"{len(good)} complete pair(s)")
    print(f"  {'stratum':<12}{'n':>4}{'helped':>8}{'hurt':>6}{'tied':>6}{'mean':>9}{'median':>9}")
    for s in ("mtd_only", "etkdg_only", "both", "neither", "all"):
        v = by.get(s)
        if not v:
            continue
        v_sorted = sorted(v)
        print(f"  {s:<12}{len(v):>4}{sum(i > 0.05 for i in v):>8}{sum(i < -0.05 for i in v):>6}"
              f"{sum(abs(i) <= 0.05 for i in v):>6}{sum(v)/len(v):>+9.2f}"
              f"{v_sorted[len(v)//2]:>+9.2f}")
    allv = [r["improvement_kcal"] for r in good]
    print(f"  {'OVERALL':<12}{len(allv):>4}{sum(i > 0.05 for i in allv):>8}"
          f"{sum(i < -0.05 for i in allv):>6}{sum(abs(i) <= 0.05 for i in allv):>6}"
          f"{sum(allv)/len(allv):>+9.2f}{sorted(allv)[len(allv)//2]:>+9.2f}")


if __name__ == "__main__":
    main()
