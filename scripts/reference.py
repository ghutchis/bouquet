#!/usr/bin/env python
"""
Build ground-truth global-minimum references (E*) for the stopping-rule benchmark.

For a *global-best* benchmark the danger is circularity: if E* is built only from
the BO search, any basin the kernel never reaches is missing from BOTH the search
and the reference, so we'd report 100% reliability while systematically failing.
This module guards against that with an *orthogonal* sampler and an audit, plus an
optional CREST pool (``--crest-dir``) that ingests precomputed iMTD-GC minima where
they exist -- the expensive-but-strong ground truth we do NOT want to run on the
fly, but are happy to reuse when ensemble_bench already produced it.

Per molecule it pools, all optimized on the SAME Hamiltonian as the benchmark
(pass ``--method gfn2`` to match a GFN2 search, so we measure sampling error not
method error):

  * orthogonal -- RDKit ETKDG conformers (distance geometry, mechanistically
    orthogonal to torsion-space BO) plus uniform random-torsion restarts. Pool
    size scales with the dihedral count d. Captures ring pucker for free.
  * systematic (d <= --d-systematic, default 7) -- a staggered torsion grid; with
    ETKDG ring templates this is effectively exhaustive at low d (true ground
    truth for acyclic cases).
  * CREST (optional, ``--crest-dir``) -- ensemble_bench crest cells
    (``<crest-dir>/<mol_id>/crest_*/refined.xyz``), re-optimized on --method. The
    strong bouquet-independent reference; folded into E_star_nonBO alongside
    orthogonal, and reported via the CREST-vs-pool audit.
  * BO -- the minima bouquet actually visited, read from the geometry trails
    (``*_geom/grad_<mol_id>_seed*.xyz``), re-optimized unconstrained.

Outputs per molecule (``reference/<mol_id>.json`` + ``<mol_id>.xyz`` of the
deduplicated low-energy minima):

  * ``E_star``        -- min over the FULL pool (relative eV, the e_e0 convention),
  * ``E_star_crest``  -- min over the CREST pool only (None if no CREST cell),
  * ``E_star_BO``     -- min over the BO pool only,
  * ``E_star_nonBO``  -- min over orthogonal+systematic only (the leave-BO-out
                         reference). The BO-vs-nonBO gap is the audit AND a finding:
                         where BO beats nonBO by more than noise the orthogonal
                         reference was too weak there (flagged, low-confidence);
                         where they agree the reference is trustworthy.
  * reference-minimum geometries (for the RMSD-identity success criterion).

Energies are anchored to the benchmark's e_e0 scale via each trail's ``final``
frame: E_start = E_gfnff(final_geom) - e_e0(final). Molecules without a trail fall
back to the GFN-FF energy of bouquet's fixed initial structure.

Usage:
  python scripts/reference.py build --manifest smiles/stopbench-500.csv \
      --method gfn2 --geom-dir runs/stopbench_geom \
      --crest-dir /ihome/ghutchison/geoffh/ensemble_bench \
      --out-dir runs/reference_gfn2 --workers 8
  python scripts/reference.py report runs/reference_gfn2
"""

import argparse
import glob
import json
import logging
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

EV = 1.0
KCAL = 23.060541945  # eV -> kcal/mol


# ---------------------------------------------------------------------------
# Geometry helpers (worker-side; heavy imports are local so the pool stays light)
# ---------------------------------------------------------------------------


class _suppress_c_stdout:
    """Silence C-level stdout/stderr (GFN-FF prints a topology table per call)."""
    def __enter__(self):
        import os
        self._null = os.open(os.devnull, os.O_WRONLY)
        self._saved = (os.dup(1), os.dup(2))
        os.dup2(self._null, 1); os.dup2(self._null, 2)
        return self

    def __exit__(self, *exc):
        import os
        os.dup2(self._saved[0], 1); os.dup2(self._saved[1], 2)
        os.close(self._saved[0]); os.close(self._saved[1]); os.close(self._null)


def _conf_to_atoms(mol, conf_id):
    from ase import Atoms
    conf = mol.GetConformer(conf_id)
    pos = conf.GetPositions()
    sym = [a.GetSymbol() for a in mol.GetAtoms()]
    return Atoms(symbols=sym, positions=pos)


def _gfnff_calc(method, mol, charge):
    """The SAME calculator the benchmark uses (matched surface)."""
    from bouquet.calculator import CalculatorFactory
    return CalculatorFactory.create(method=method, mol=mol, charge=charge)


def _optimize(atoms, calc, fmax=0.05, steps=300):
    """Unconstrained local optimization; returns (atoms, E_eV) or None on failure."""
    from ase.optimize import LBFGS
    a = atoms.copy()
    a.calc = calc
    try:
        with _suppress_c_stdout():
            LBFGS(a, logfile=None).run(fmax=fmax, steps=steps)
            e = float(a.get_potential_energy())
        return a, e
    except Exception:
        return None


def _etkdg_confs(mol, n, seed):
    """RDKit ETKDG ensemble (MMFF pre-optimized); returns list of ase Atoms."""
    from rdkit.Chem import AllChem
    m = AllChem.AddHs(mol)
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    p.numThreads = 1
    cids = list(AllChem.EmbedMultipleConfs(m, numConfs=n, params=p))
    if cids:
        try:
            AllChem.MMFFOptimizeMoleculeConfs(m, numThreads=1, maxIters=200)
        except Exception:
            pass
    return [_conf_to_atoms(m, c) for c in cids]


def _random_torsion_confs(mol, dihedral_chains, n, seed):
    """Uniform random-torsion restarts: randomize each rotatable dihedral on an
    ETKDG template, MMFF-clean the clashes; returns list of ase Atoms."""
    import numpy as np
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdMolTransforms as rmt
    rng = np.random.default_rng(seed)
    m = AllChem.AddHs(mol)
    p = AllChem.ETKDGv3()
    p.randomSeed = seed + 1
    if AllChem.EmbedMolecule(m, p) != 0:
        return []
    out = []
    for _ in range(n):
        conf = m.GetConformer()
        for (i, j, k, l) in dihedral_chains:
            rmt.SetDihedralDeg(conf, i, j, k, l, float(rng.uniform(0, 360)))
        try:
            AllChem.MMFFOptimizeMolecule(m, maxIters=200)
        except Exception:
            pass
        out.append(_conf_to_atoms(m, conf.GetId()))
    return out


def _systematic_confs(mol, dihedral_chains, angles, cap):
    """Staggered torsion grid (cartesian product, capped); returns ase Atoms."""
    import itertools
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdMolTransforms as rmt
    m = AllChem.AddHs(mol)
    p = AllChem.ETKDGv3(); p.randomSeed = 1
    if AllChem.EmbedMolecule(m, p) != 0:
        return []
    combos = itertools.product(angles, repeat=len(dihedral_chains))
    out = []
    for combo in combos:
        if len(out) >= cap:
            break
        conf = m.GetConformer()
        for (i, j, k, l), ang in zip(dihedral_chains, combo):
            rmt.SetDihedralDeg(conf, i, j, k, l, float(ang))
        try:
            AllChem.MMFFOptimizeMolecule(m, maxIters=200)
        except Exception:
            pass
        out.append(_conf_to_atoms(m, conf.GetId()))
    return out


def _read_trail(path):
    """Read a geometry-trail XYZ; returns [(atoms, e_e0_eV, kind)] in file order."""
    from ase.io import read
    frames = read(path, index=":")
    out = []
    with open(path) as fh:
        comments = [ln.strip() for ln in fh if "kind=" in ln]
    # Every trail frame carries a "kind=" comment line, so the counts match by
    # construction; strict pairing surfaces a desync (e.g. a dropped frame) instead
    # of silently truncating trailing BO frames.
    for atoms, c in zip(frames, comments, strict=True):
        kv = dict(tok.split("=", 1) for tok in c.split() if "=" in tok)
        out.append((atoms, float(kv.get("e_e0_eV", "nan")), kv.get("kind", "")))
    return out


# Covalent-radius multiplier for distance-based bond perception in the species filter
# below. Matches ensemble_bench/bench.py's CONNECTIVITY_MULT: absolute perception is
# unreliable for large-radius atoms (P, S) -- it invents spurious long bonds -- so we never
# compare to an external graph, only structures of the same molecule to each other, where
# consistent perception quirks cancel out.
CONNECTIVITY_MULT = 1.15


def _bond_fingerprint(atoms, mult: float = CONNECTIVITY_MULT):
    """Perceived connectivity as a frozenset of (sorted element-pair, count).
    Ported from ensemble_bench/bench.py -- used only to compare structures of the SAME
    molecule to one another (see _drop_off_species)."""
    from ase.neighborlist import build_neighbor_list, natural_cutoffs

    nl = build_neighbor_list(atoms, natural_cutoffs(atoms, mult=mult),
                             self_interaction=False, bothways=True)
    m = nl.get_connectivity_matrix(sparse=False)
    syms = atoms.get_chemical_symbols()
    counts: dict = {}
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            if m[i, j]:
                pair = tuple(sorted((syms[i], syms[j])))
                counts[pair] = counts.get(pair, 0) + 1
    return frozenset(counts.items())


def _drop_off_species(pools: dict, mult: float = CONNECTIVITY_MULT):
    """Drop pool members that are not the same SPECIES as the pool majority.

    ``connectivity_changed`` only rejects *broken* bonds -- newly *formed* ones are
    ignored on purpose (strained/caged conformers are legitimate). But an unconstrained
    tight optimization can also FORM a bond (proton transfer, ring closure, H2 collapse),
    producing a different species that can sit thousands of kcal below the true conformers
    and hijack E* as a spurious global minimum. This is the same self-calibrating guard
    ensemble_bench/bench.py::_load_pool uses: keep only structures whose full perceived
    connectivity matches the pool's MODAL fingerprint (the correct connectivity is by far
    the most common; artifacts are the outliers).

    ``pools`` maps source -> [(atoms, E_eV)]. Returns (filtered_pools, n_rejected_by_src).
    """
    from collections import Counter

    recs = [(src, a, e) for src, res in pools.items() for a, e in res]
    n_rej = {src: 0 for src in pools}
    if not recs:
        return pools, n_rej
    fps = [_bond_fingerprint(a, mult) for _, a, _ in recs]
    modal = Counter(fps).most_common(1)[0][0]
    out: dict = {src: [] for src in pools}
    for (src, a, e), fp in zip(recs, fps):
        if fp == modal:
            out[src].append((a, e))
        else:
            n_rej[src] += 1
    return out, n_rej


def _read_refined_xyz(path):
    """Read an ensemble_bench ``refined.xyz`` cell (concatenated single-conformer
    XYZ frames, comment ``energy_kcal=...``) as a list of ase Atoms -- geometry only,
    since every candidate is re-optimized on the benchmark surface here anyway. A
    cell where zero structures survived refinement is written as an empty file; treat
    that (and any truncated/corrupt cell) as "no structures" rather than crash."""
    from ase.io import read
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    try:
        frames = read(p, index=":")
    except Exception:
        return []
    return frames if isinstance(frames, list) else [frames]


def _dedup(pairs, rmsd_thr, e_tol_eV):
    """Deduplicate (atoms, E_eV) by energy-AND-geometry, keeping lowest energy."""
    from bouquet.ensemble import _rmsd
    pairs = sorted(pairs, key=lambda p: p[1])
    uniq = []
    for atoms, e in pairs:
        if any(abs(e - ue) < e_tol_eV and _rmsd(ua, atoms) < rmsd_thr
               for ua, ue in uniq):
            continue
        uniq.append((atoms, e))
    return uniq


# ---------------------------------------------------------------------------
# Per-molecule reference (one worker call)
# ---------------------------------------------------------------------------


def _pool_size(d, base, per_d, cap):
    return int(min(cap, base + per_d * d))


def build_one(task):
    """Build the reference for one molecule. ``task`` is a plain dict so it pickles
    cleanly into the process pool."""
    logging.getLogger("bouquet").setLevel(logging.ERROR)
    mol_id = task["mol_id"]
    out = {"mol_id": mol_id, "smiles": task["smiles"], "d": task["d"], "ok": False}
    try:
        from rdkit import Chem
        from bouquet.setup import detect_dihedrals

        mol = Chem.MolFromSmiles(task["smiles"])
        if mol is None:
            out["error"] = "unparseable SMILES"
            return out
        charge = Chem.GetFormalCharge(mol)
        molh = Chem.AddHs(mol)
        chains = [tuple(di.chain) for di in detect_dihedrals(molh)]
        calc = _gfnff_calc(task["method"], molh, charge)

        # --- candidate geometries by source ---
        cand = {"orth": [], "crest": [], "bo": []}
        # The ETKDG/random/systematic orthogonal pool is the independent acyclic/ring
        # reference. Skip it (--no-orthogonal) for large foldamers: distance-geometry
        # embeds never find the fold, so it is pure GFN2 cost -- there CREST IS the
        # reference and only the bouquet-vs-CREST comparison matters.
        if not task.get("skip_orth"):
            n_orth = _pool_size(task["d"], task["base"], task["per_d"], task["cap"])
            cand["orth"] += _etkdg_confs(mol, n_orth, task["seed"])
            cand["orth"] += _random_torsion_confs(mol, chains, max(1, n_orth // 2),
                                                  task["seed"])
            if task["d"] <= task["d_systematic"] and chains:
                cand["orth"] += _systematic_confs(mol, chains, task["angles"],
                                                  task["sys_cap"])

        # CREST pool: ensemble_bench crest cells (GFN2 iMTD-GC minima). Re-optimized
        # on the matched surface below like every other source, so the fact that they
        # were refined at GFN2 kcal is irrelevant -- only the geometry is used. This is
        # the strong, bouquet-independent reference; a molecule whose CREST cell is
        # still filling (d18-20) simply contributes no member here.
        for cf in task.get("crest_files", []):
            cand["crest"] += _read_refined_xyz(cf)

        # BO pool + e_e0 anchor from a `final` frame
        e_start = None
        for tf in task["trail_files"]:
            for atoms, e_e0, kind in _read_trail(tf):
                cand["bo"].append(atoms)
                if kind == "final" and not math.isnan(e_e0) and e_start is None:
                    sp = atoms.copy(); sp.calc = calc
                    try:
                        with _suppress_c_stdout():
                            e_start = float(sp.get_potential_energy()) - e_e0
                    except Exception:
                        pass

        # Fallback anchor: GF N-FF energy of bouquet's fixed initial structure.
        if e_start is None:
            from bouquet.setup import get_initial_structure
            try:
                init_atoms, _ = get_initial_structure(task["smiles"])
                opt = _optimize(init_atoms, calc)  # constrained-free start proxy
                e_start = opt[1] if opt else 0.0
            except Exception:
                e_start = 0.0

        # --- optimize all candidates on the matched surface; reject any whose
        # optimized geometry changed connectivity (GFN-FF can rearrange/dissociate
        # odd species into a spuriously low, non-conformer structure -- e.g. the
        # 101 kcal m017 artifact). The check shares atom ordering with molh. ---
        from bouquet.setup import connectivity_changed
        raw = {}
        rejected = {}
        for src, atoms_list in cand.items():
            res = []
            n_rej = 0
            for a in atoms_list:
                r = _optimize(a, calc)
                if r is None:
                    continue
                if connectivity_changed(r[0], molh, task["conn_tol"]):
                    n_rej += 1
                    continue
                res.append(r)
            raw[src] = res
            rejected[src] = n_rej

        # Second guard, across the WHOLE pool: drop bond-FORMED species that the
        # broken-bond check above ignores by design (see _drop_off_species). Without
        # this a single proton-transferred/ring-closed structure can sit thousands of
        # kcal below the real conformers and become a spurious E*.
        raw, species_rej = _drop_off_species(raw)
        opt = {src: _dedup(res, task["rmsd_thr"], task["e_tol"] / KCAL)
               for src, res in raw.items()}

        # The leave-BO-out (independent) reference is orthogonal + CREST; CREST is
        # the strong member where present. allpool adds bouquet's own BO minima on top.
        nonbo = _dedup(opt["orth"] + opt["crest"], task["rmsd_thr"],
                       task["e_tol"] / KCAL)
        allpool = _dedup(opt["orth"] + opt["crest"] + opt["bo"], task["rmsd_thr"],
                         task["e_tol"] / KCAL)

        def rel_min(pairs):
            return (min(e for _, e in pairs) - e_start) if pairs else None

        out.update({
            "ok": True,
            "n_orth": len(opt["orth"]), "n_crest": len(opt["crest"]),
            "n_bo": len(opt["bo"]),
            "n_rej_orth": rejected.get("orth", 0),
            "n_rej_crest": rejected.get("crest", 0),
            "n_rej_bo": rejected.get("bo", 0),
            # bond-FORMED species dropped by the modal-fingerprint guard, per source
            "n_species_rej_orth": species_rej.get("orth", 0),
            "n_species_rej_crest": species_rej.get("crest", 0),
            "n_species_rej_bo": species_rej.get("bo", 0),
            "E_start_eV": e_start,
            "E_star": rel_min(allpool),
            "E_star_BO": rel_min(opt["bo"]),
            "E_star_crest": rel_min(opt["crest"]),
            "E_star_nonBO": rel_min(nonbo),
        })
        # store the deduped low-energy minima geometries (within a window) for RMSD
        if allpool:
            emin = min(e for _, e in allpool)
            window = task["geom_window"] / KCAL
            keep = [(a, e) for a, e in allpool if e - emin <= window]
            xyz = task["out_dir"] / f"{mol_id}.xyz"
            from ase.io import write
            for a, e in sorted(keep, key=lambda p: p[1]):
                a.info["e_rel_kcal"] = (e - e_start) * KCAL
                write(xyz, a, append=(xyz.exists()), comment=f"E_rel_kcal={(e-e_start)*KCAL:.4f}")
            out["n_minima"] = len(keep)
    except Exception as e:  # keep the batch alive
        out["error"] = repr(e)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def cmd_build(args):
    import pandas as pd
    man = pd.read_csv(args.manifest)
    if "mol_id" not in man.columns:
        sys.exit("manifest needs a mol_id column (use a select_smiles.py manifest CSV)")
    if args.mols:
        want = set(args.mols.split(","))
        man = man[man.mol_id.isin(want)]
    if args.limit:
        man = man.head(args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    geom_dir = args.geom_dir
    angles = [float(a) for a in args.angles.split(",")]

    crest_dir = args.crest_dir
    tasks = []
    for _, r in man.iterrows():
        mid = r.mol_id
        trails = sorted(glob.glob(str(geom_dir / f"*_{mid}_seed*.xyz"))) if geom_dir else []
        # ensemble_bench layout: <crest-dir>/<mol_id>/<crest-glob> (crest_*/refined.xyz).
        crest_files = (sorted(glob.glob(str(crest_dir / mid / args.crest_glob)))
                       if crest_dir else [])
        # skip molecules already done (resume)
        if args.resume and (args.out_dir / f"{mid}.json").exists():
            continue
        tasks.append({
            "mol_id": mid, "smiles": r.raw_smiles, "d": int(r.d),
            "method": args.method, "seed": args.seed,
            "base": args.base, "per_d": args.per_d, "cap": args.cap,
            "d_systematic": args.d_systematic, "angles": angles,
            "sys_cap": args.sys_cap, "rmsd_thr": args.rmsd, "e_tol": args.e_tol,
            "geom_window": args.geom_window, "trail_files": trails,
            "crest_files": crest_files, "skip_orth": args.no_orthogonal,
            "out_dir": args.out_dir, "conn_tol": args.conn_tol,
        })
    print(f"Building references for {len(tasks)} molecules "
          f"({args.method}, {args.workers} workers)...")

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for out in ex.map(build_one, tasks):
            (args.out_dir / f"{out['mol_id']}.json").write_text(json.dumps(out, indent=2))
            done += 1
            tag = "ok" if out.get("ok") else f"FAIL {out.get('error','')[:50]}"
            if out.get("ok") and out.get("E_star") is None:
                # connectivity guard rejected every candidate -> no valid conformer
                tag = (f"E*=None EXCLUDED (rejected "
                       f"{out.get('n_rej_orth',0)}+{out.get('n_rej_crest',0)}"
                       f"+{out.get('n_rej_bo',0)} conn, "
                       f"{out.get('n_species_rej_orth',0)}"
                       f"+{out.get('n_species_rej_crest',0)}"
                       f"+{out.get('n_species_rej_bo',0)} species)")
            elif out.get("ok"):
                gap = (out["E_star_BO"] - out["E_star_nonBO"]) * KCAL \
                    if out["E_star_BO"] is not None and out["E_star_nonBO"] is not None else float("nan")
                tag = (f"E*={out['E_star']*KCAL:7.2f}  BO-nonBO={gap:+6.2f} kcal  "
                       f"(orth {out['n_orth']}, crest {out.get('n_crest', 0)}, "
                       f"bo {out['n_bo']})")
            print(f"[{done}/{len(tasks)}] {out['mol_id']}: {tag}")
    print(f"\nWrote {done} references to {args.out_dir}")


def cmd_report(args):
    import numpy as np
    rows = []
    for f in sorted(glob.glob(str(args.ref_dir / "*.json"))):
        o = json.loads(Path(f).read_text())
        if o.get("ok"):
            rows.append(o)
    if not rows:
        sys.exit("no successful references found")
    import pandas as pd
    df = pd.DataFrame(rows)
    df["bo_minus_nonbo_kcal"] = (df.E_star_BO - df.E_star_nonBO) * KCAL
    df["d"] = df.d.astype(int)
    has_crest = df["E_star_crest"].notna() if "E_star_crest" in df.columns else pd.Series(False, index=df.index)
    print(f"references: {len(df)}  (with BO pool: {df.E_star_BO.notna().sum()}, "
          f"with CREST pool: {int(has_crest.sum())})")

    # CREST-vs-pool audit: where the FULL pool (incl. bouquet BO) beats the CREST
    # minimum by more than noise, bouquet reached a lower basin than CREST -> a real
    # win, and a heads-up that CREST alone would have under-counted reliability there.
    if has_crest.any():
        dfc = df[has_crest].copy()
        dfc["crest_minus_pool_kcal"] = (dfc.E_star_crest - dfc.E_star) * KCAL
        beat = dfc[dfc.crest_minus_pool_kcal > args.noise]
        print(f"\nCREST-vs-pool audit (noise={args.noise} kcal/mol):")
        print(f"  full pool found a LOWER minimum than CREST on {len(beat)}/{len(dfc)} "
              f"molecules (bouquet/orth beat CREST there).")
        print(f"  CREST matched the pool minimum on {len(dfc)-len(beat)}/{len(dfc)}.")

        # bouquet-vs-CREST audit: E_star_BO vs E_star_crest directly -- the "does the BO
        # search find a lower conformer than CREST's metadynamics" question (both are
        # unconstrained minima on the same surface; the modal-fingerprint filter already
        # guarantees they are the SAME species, so a win is a genuine alternative basin,
        # not a rearrangement artifact). Winners are written to bouquet_beats_crest.csv
        # (with the gap + geometry path) for follow-up on e.g. larger foldamers.
        dbo = dfc[dfc.E_star_BO.notna()].copy()
        if len(dbo):
            dbo["bo_minus_crest_kcal"] = (dbo.E_star_BO - dbo.E_star_crest) * KCAL
            win = dbo[dbo.bo_minus_crest_kcal < -args.noise]
            tie = dbo[dbo.bo_minus_crest_kcal.abs() <= args.noise]
            loss = dbo[dbo.bo_minus_crest_kcal > args.noise]
            print(f"\nbouquet-vs-CREST audit (E_star_BO vs E_star_crest, "
                  f"noise={args.noise} kcal/mol; {len(dbo)} mols with both pools):")
            print(f"  bouquet LOWER than CREST : {len(win):3d}  "
                  f"(median gap {win.bo_minus_crest_kcal.median():+.2f} kcal)"
                  if len(win) else f"  bouquet LOWER than CREST : {len(win):3d}")
            print(f"  tie (within noise)       : {len(tie):3d}")
            print(f"  CREST LOWER than bouquet : {len(loss):3d}")
            if "cls" not in dbo.columns:
                # ring-class stratification (reuses the benchmark's ring_class defn)
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                try:
                    from stop_rules import ring_class
                    dbo["cls"] = dbo.smiles.map(ring_class)
                    res = dbo.assign(res=np.where(dbo.bo_minus_crest_kcal < -args.noise, "win",
                                     np.where(dbo.bo_minus_crest_kcal > args.noise, "loss", "tie")))
                    print("  by ring class:")
                    print(pd.crosstab(res.cls, res.res).to_string().replace("\n", "\n    "))
                except Exception:
                    pass
            if len(win):
                w = win.sort_values("bo_minus_crest_kcal")[
                    ["mol_id", "d", "bo_minus_crest_kcal"]].copy()
                w["xyz"] = [str(args.ref_dir / f"{m}.xyz") for m in w.mol_id]
                w.to_csv(args.ref_dir / "bouquet_beats_crest.csv", index=False)
                print(f"  wrote {args.ref_dir / 'bouquet_beats_crest.csv'} "
                      f"(winners + gap + geometry path)")
    # Leave-BO-out audit: BO beats the orthogonal reference by > noise => orthogonal
    # ref too weak there (low-confidence); BO worse/equal => reference trustworthy.
    noise = args.noise
    weak = df[df.bo_minus_nonbo_kcal < -noise]
    print(f"\nLeave-BO-out audit (noise={noise} kcal/mol):")
    print(f"  BO found a LOWER minimum than the orthogonal reference on "
          f"{len(weak)}/{len(df)} molecules -> those references are low-confidence.")
    print(f"  orthogonal matched/beat BO on {len(df)-len(weak)}/{len(df)}.")
    print("\nBO - nonBO gap (kcal/mol) by d (negative = BO found lower = ref weak):")
    g = df.groupby("d").bo_minus_nonbo_kcal
    for d, s in g:
        print(f"  d={d:2d}: n={len(s):3d}  median={s.median():+.3f}  "
              f"min={s.min():+.3f}  weak={int((s < -noise).sum())}")
    df.sort_values("d").to_csv(args.ref_dir / "reference_summary.csv", index=False)
    print(f"\nwrote {args.ref_dir/'reference_summary.csv'}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="Build per-molecule references")
    b.add_argument("--manifest", type=Path, required=True,
                   help="select_smiles.py manifest CSV (mol_id, raw_smiles, d)")
    b.add_argument("--geom-dir", type=Path, default=None,
                   help="Geometry-trail dir (*_geom) for the BO pool; optional")
    b.add_argument("--crest-dir", type=Path, default=None,
                   help="ensemble_bench workdir holding CREST cells "
                   "(<crest-dir>/<mol_id>/crest_*/refined.xyz); adds them to the "
                   "reference pool as the strong bouquet-independent source. Optional.")
    b.add_argument("--crest-glob", default="crest_*/refined.xyz",
                   help="Glob under <crest-dir>/<mol_id>/ for CREST refined cells "
                   "(default crest_*/refined.xyz).")
    b.add_argument("--no-orthogonal", action="store_true",
                   help="Skip the ETKDG/random/systematic orthogonal pool (pure GFN2 "
                   "cost that never finds a fold) and build E* from CREST + bouquet BO "
                   "trails only. Use for large foldamers where CREST is the reference "
                   "and only the bouquet-vs-CREST comparison matters.")
    b.add_argument("--out-dir", type=Path, default=Path("reference"))
    b.add_argument("--method", default="gfnff", help="Hamiltonian (match the run)")
    b.add_argument("--workers", "-w", type=int, default=8)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--base", type=int, default=20, help="ETKDG base pool size")
    b.add_argument("--per-d", type=int, default=15, help="ETKDG pool growth per d")
    b.add_argument("--cap", type=int, default=300, help="ETKDG pool cap")
    b.add_argument("--d-systematic", type=int, default=7,
                   help="Add the systematic torsion grid for d <= this (default 7)")
    b.add_argument("--angles", default="60,180,300",
                   help="Staggered grid angles per torsion (default 60,180,300)")
    b.add_argument("--sys-cap", type=int, default=729,
                   help="Cap on the systematic grid product (default 729=3^6)")
    b.add_argument("--rmsd", type=float, default=0.25, help="Dedup RMSD (A)")
    b.add_argument("--e-tol", type=float, default=0.1,
                   help="Dedup energy tolerance (kcal/mol)")
    b.add_argument("--geom-window", type=float, default=5.0,
                   help="Store reference minima within this window of E* (kcal/mol)")
    b.add_argument("--conn-tol", type=float, default=1.3,
                   help="Bond-perception tolerance for the connectivity guard "
                   "(reject optimized conformers whose bond graph changed; default 1.3)")
    b.add_argument("--mols", default=None, help="Comma-separated mol_id subset")
    b.add_argument("--limit", type=int, default=None, help="First N molecules only")
    b.add_argument("--resume", action="store_true",
                   help="Skip molecules whose <mol_id>.json already exists")
    b.set_defaults(func=cmd_build)

    r = sub.add_parser("report", help="Summarize references + leave-BO-out audit")
    r.add_argument("ref_dir", type=Path)
    r.add_argument("--noise", type=float, default=0.2,
                   help="kcal/mol below which a BO-nonBO gap is noise (default 0.2)")
    r.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
