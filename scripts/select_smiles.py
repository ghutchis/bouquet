#!/usr/bin/env python
"""
Build a stratified molecule manifest for the stopping-rule benchmark.

The ``smiles/`` tree pools many sources (Platinum, ZINC, COD, Ligand-Expo, Wiki,
oligomer/foldamer series, ...) with heavy subset overlap and a sharply peaked
dihedral-count (``d``) histogram. Pooling them raw gives a confident-looking
``n*(d)`` that is actually one chemotype per bin. This script fixes that at the
manifest stage: ingest everything, deduplicate, compute per-molecule descriptors,
show the per-source ``d`` histogram, then **stratified-sample by d-bin** so every
bin gets a comparable, chemically diverse sample.

``d`` is Bouquet's OWN rotatable-dihedral count (``setup.detect_dihedrals`` on the
H-added molecule), so it matches what ``bouquet --auto`` and the benchmark see --
not RDKit's ``CalcNumRotatableBonds`` (logged too, for comparison). It is a pure
graph property, so no 3D embedding is needed and ~300k molecules ingest in
minutes.

Two phases (the first is the expensive, cacheable one):

  # 1. Ingest: parse all sources, dedup raw SMILES (first-seen source wins),
  #    compute descriptors in parallel -> descriptors.csv (resumable).
  python scripts/select_smiles.py ingest --output descriptors.csv -w 8

  # 2. Select: dedup by InChIKey, filter, plot the d-histogram by source, then
  #    stratified-sample K molecules per d-bin -> manifest.smi (+ manifest.csv).
  python scripts/select_smiles.py select descriptors.csv \
      --out-prefix manifest --per-bin 30 --d-max 20

``manifest.smi`` is ``smiles<TAB>name`` -- feed it straight to
``stop_benchmark.py run --input manifest.smi``. High-d bins are usually
underpopulated by real molecules; top them up with the oligomer/foldamer series
(``scripts/generate_oligomers.py``, ``smiles/oligo-scan.smi``).
"""

import argparse
import csv
import glob
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Default source priority: curated/diverse sets first, so when the same molecule
# appears in several files the *kept* source is the most informative one. Unlisted
# stems sort after these (alphabetically). Order matters only for the recorded
# `source`; dedup is by structure regardless.
SOURCE_PRIORITY = [
    "platinum-diverse", "platinum-500", "platinum-50", "platinum-10",
    "zinc-leads", "zinc_random", "zinc-50", "zinc",
    "cod-50", "cod-59k", "cod",
    "ligand-expo", "wiki",
    "foldamers", "huc-foldamer", "oligo-scan", "polyyne",
]

# Elements GFN2-xTB / GFN-FF handle well and that we want in the benchmark; a
# molecule with anything else (metals, etc.) is dropped at select time.
DEFAULT_ELEMENTS = "H,B,C,N,O,F,Si,P,S,Cl,Se,Br,I"

DESCRIPTOR_FIELDS = [
    "raw_smiles", "name", "source", "inchikey", "parse_ok",
    "d", "d_rdkit", "n_heavy", "n_atoms", "mw", "charge",
    "n_hbd", "n_hba", "frac_aromatic", "n_rings", "max_ring_size",
    "n_frags", "n_sym_classes", "elements",
]


# ---------------------------------------------------------------------------
# Parsing the smiles/ tree
# ---------------------------------------------------------------------------


def _priority_key(path: Path):
    stem = path.stem
    idx = SOURCE_PRIORITY.index(stem) if stem in SOURCE_PRIORITY else len(SOURCE_PRIORITY)
    return (idx, stem)


def parse_sources(inputs):
    """Yield (raw_smiles, name, source), dedup'd by raw SMILES string with the
    first-seen (highest-priority source) kept. Files are read in priority order;
    each line is ``SMILES [name...]`` with any whitespace delimiter (the first
    token is the SMILES, the rest -- if any -- the name)."""
    seen = {}
    for path in sorted((Path(p) for p in inputs), key=_priority_key):
        source = path.stem
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                smi = parts[0]
                name = " ".join(parts[1:]).strip() or f"{source}_{lineno}"
                if smi not in seen:
                    seen[smi] = (name, source)
    for smi, (name, source) in seen.items():
        yield smi, name, source


# ---------------------------------------------------------------------------
# Descriptors (one worker call per unique raw SMILES)
# ---------------------------------------------------------------------------


def _init_worker():
    """Silence bouquet's per-call dihedral logging in each worker process."""
    logging.getLogger("bouquet").setLevel(logging.ERROR)


def compute_descriptors(arg):
    """(raw_smiles, name, source) -> descriptor dict. ``parse_ok=0`` on any
    failure (kept in the CSV so the loss is auditable). ``d`` is Bouquet's own
    rotatable-dihedral count on the H-added molecule (matches the CLI)."""
    smi, name, source = arg
    row = {f: "" for f in DESCRIPTOR_FIELDS}
    row.update({"raw_smiles": smi, "name": name, "source": source, "parse_ok": 0})
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors

        from bouquet.setup import detect_dihedrals

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return row
        molh = Chem.AddHs(mol)
        heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
        n_heavy = len(heavy)
        rings = mol.GetRingInfo().AtomRings()
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
        elements = sorted({a.GetSymbol() for a in mol.GetAtoms()})
        try:
            inchikey = Chem.MolToInchiKey(mol)
        except Exception:
            inchikey = ""
        row.update({
            "inchikey": inchikey,
            "parse_ok": 1,
            "d": len(detect_dihedrals(molh)),          # Bouquet's definition
            "d_rdkit": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "n_heavy": n_heavy,
            "n_atoms": molh.GetNumAtoms(),
            "mw": round(Descriptors.MolWt(mol), 2),
            "charge": Chem.GetFormalCharge(mol),
            "n_hbd": rdMolDescriptors.CalcNumHBD(mol),
            "n_hba": rdMolDescriptors.CalcNumHBA(mol),
            "frac_aromatic": round(
                sum(a.GetIsAromatic() for a in heavy) / n_heavy, 3
            ) if n_heavy else 0.0,
            "n_rings": len(rings),
            "max_ring_size": max((len(r) for r in rings), default=0),
            "n_frags": len(Chem.GetMolFrags(mol)),
            "n_sym_classes": len(set(ranks)),
            "elements": "/".join(elements),
        })
    except Exception:
        return row
    return row


def ingest(args):
    """Phase 1: parse, dedup raw SMILES, compute descriptors -> CSV (resumable)."""
    inputs = args.inputs or sorted(glob.glob(str(Path("smiles") / "*.smi")))
    if not inputs:
        sys.exit("No input .smi files found (pass --inputs or run from the repo root).")

    pairs = list(parse_sources(inputs))
    print(f"Parsed {len(pairs)} unique raw SMILES from {len(inputs)} file(s).")

    done = set()
    if args.resume and Path(args.output).exists():
        with open(args.output, newline="") as f:
            done = {r["raw_smiles"] for r in csv.DictReader(f)}
        pairs = [p for p in pairs if p[0] not in done]
        print(f"--resume: {len(done)} already done; {len(pairs)} to compute.")

    write_header = not (args.resume and Path(args.output).exists())
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if write_header else "a"
    n = 0
    with open(args.output, mode, newline="") as out:
        w = csv.DictWriter(out, fieldnames=DESCRIPTOR_FIELDS)
        if write_header:
            w.writeheader()
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker
        ) as ex:
            for row in ex.map(compute_descriptors, pairs, chunksize=200):
                w.writerow(row)
                n += 1
                if n % 5000 == 0:
                    out.flush()
                    print(f"  ...{n}/{len(pairs)}")
    print(f"Wrote {n} descriptor rows to {args.output}")


# ---------------------------------------------------------------------------
# Selection (dedup by structure, filter, stratify)
# ---------------------------------------------------------------------------


def select(args):
    """Phase 2: dedup by InChIKey, filter, plot d-histogram, stratified-sample."""
    import numpy as np
    import pandas as pd

    df = pd.read_csv(args.descriptors)
    n0 = len(df)
    df = df[df.parse_ok == 1].copy()
    for c in ["d", "d_rdkit", "n_heavy", "n_atoms", "charge", "n_rings",
              "max_ring_size", "n_frags", "n_sym_classes"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Dedup by structure (InChIKey); rows are already in source-priority order from
    # ingest, so keep="first" retains the highest-priority source. Rows missing an
    # InChIKey fall back to raw SMILES so they aren't all collapsed together.
    # A missing InChIKey is NaN (truthy under astype(bool)) or empty; treat both as
    # absent and fall back to raw_smiles, else all keyless rows collapse to one key.
    _has_key = df.inchikey.notna() & df.inchikey.astype(str).str.strip().astype(bool)
    df["dedup_key"] = df.inchikey.where(_has_key, df.raw_smiles)
    n_before = len(df)
    df["n_sources"] = df.groupby("dedup_key")["source"].transform("nunique")
    df = df.drop_duplicates("dedup_key", keep="first")
    print(f"Loaded {n0} rows; {n_before} parsed; {len(df)} unique structures "
          f"({(df.n_sources > 1).sum()} appear in >1 source).")

    # Exclude structures already chosen for a prior manifest (e.g. the calibration
    # set), so a held-out validation set is strictly disjoint by structure. Match on
    # dedup_key, falling back to inchikey/raw_smiles if an older manifest lacks it.
    excluded = set()
    for ex in (args.exclude or []):
        edf = pd.read_csv(ex)
        if "dedup_key" in edf.columns:
            keys = edf.dedup_key.astype(str)
        elif "inchikey" in edf.columns:
            _ek = edf.inchikey.notna() & edf.inchikey.astype(str).str.strip().astype(bool)
            keys = edf.inchikey.where(_ek, edf.raw_smiles).astype(str)
        else:
            keys = edf.raw_smiles.astype(str)
        excluded.update(keys)
    if excluded:
        n = len(df)
        df = df[~df.dedup_key.isin(excluded)]
        print(f"Excluded {n - len(df)} structures present in "
              f"{len(args.exclude)} prior manifest(s) ({len(excluded)} keys).")

    allowed = set(args.allowed_elements.split(","))

    def elements_ok(s):
        return all(e in allowed for e in str(s).split("/") if e)

    # Filters (each reported so the funnel is auditable).
    funnel = [("unique", len(df))]
    df = df[df.n_frags <= 1]; funnel.append(("single-fragment", len(df)))
    df = df[df.elements.map(elements_ok)]; funnel.append(("allowed-elements", len(df)))
    df = df[df.n_heavy <= args.max_heavy]; funnel.append((f"<= {args.max_heavy} heavy", len(df)))
    df = df[(df.d >= args.d_min) & (df.d <= args.d_max)]
    funnel.append((f"d in [{args.d_min},{args.d_max}]", len(df)))
    print("\nselection funnel:")
    for label, n in funnel:
        print(f"  {label:24s} {n}")

    # d-bin (right-closed width-`bin_width` bins on d).
    bw = args.bin_width
    df["d_bin"] = ((df.d - args.d_min) // bw) * bw + args.d_min
    df["d_bin_label"] = df.d_bin.map(lambda b: f"{int(b)}-{int(b + bw - 1)}"
                                     if bw > 1 else f"{int(b)}")

    if args.hist:
        _plot_d_hist_by_source(df, args.out_prefix)

    # Stratified sample: up to per_bin per d-bin, diversified within bin by
    # spreading across sources (round-robin) then chemotype. Seeded for repeatability.
    rng = np.random.default_rng(args.seed)
    picked = []
    print(f"\nstratified sample ({args.per_bin}/bin, width {bw}):")
    for b, g in df.sort_values("d").groupby("d_bin"):
        take = _sample_bin(g, args.per_bin, rng)
        picked.append(take)
        flag = "" if len(take) == args.per_bin else "  (under-target)"
        print(f"  d {g.d_bin_label.iloc[0]:>7s}: {len(take):3d} / {len(g):<5d} available{flag}")
    manifest = pd.concat(picked).sort_values(["d", "source", "name"]).reset_index(drop=True)

    # Clean, unique trial id. The original source names are unusable as keys --
    # some have spaces (split by the .smi parser), duplicates, or a leading '-'
    # (argparse would read it as a flag after --name). mol_id encodes d for
    # readability; the original name + InChIKey stay in the CSV for provenance.
    manifest = manifest.rename(columns={"name": "orig_name"})
    manifest.insert(0, "mol_id", [
        f"{args.id_prefix}{i:03d}_d{int(d):02d}" for i, d in enumerate(manifest.d)
    ])

    smi_path = Path(f"{args.out_prefix}.smi")
    csv_path = Path(f"{args.out_prefix}.csv")
    manifest.to_csv(csv_path, index=False)
    with open(smi_path, "w") as f:
        for _, r in manifest.iterrows():
            f.write(f"{r.raw_smiles}\t{r.mol_id}\n")
    print(f"\nWrote {len(manifest)} molecules -> {smi_path} and {csv_path}")
    under = [b for b, g in manifest.groupby("d_bin") if len(g) < args.per_bin]
    if under:
        print(f"Under-target d-bins (top up with oligomers/foldamers): "
              f"{', '.join(str(int(b)) for b in under)}")


def _sample_bin(g, k, rng):
    """Pick up to k rows from a d-bin, spreading across sources round-robin so a
    single dominant source can't fill the bin."""
    if len(g) <= k:
        return g
    by_source = {s: sub.sample(frac=1, random_state=int(rng.integers(1 << 31)))
                 for s, sub in g.groupby("source")}
    order = sorted(by_source, key=lambda s: -len(by_source[s]))
    chosen, i = [], 0
    while len(chosen) < k and any(len(by_source[s]) > 0 for s in order):
        s = order[i % len(order)]
        if len(by_source[s]):
            chosen.append(by_source[s].iloc[0])
            by_source[s] = by_source[s].iloc[1:]
        i += 1
    import pandas as pd
    return pd.DataFrame(chosen)


def _plot_d_hist_by_source(df, out_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sources = sorted(df.source.unique(), key=lambda s: -len(df[df.source == s]))
    dmax = int(df.d.max())
    bins = np.arange(0, dmax + 2) - 0.5
    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(bins) - 1)
    for s in sources:
        counts, _ = np.histogram(df[df.source == s].d, bins=bins)
        ax.bar(np.arange(0, dmax + 1), counts, bottom=bottom, width=0.9,
               label=f"{s} ({len(df[df.source == s])})")
        bottom += counts
    ax.set_xlabel("d (Bouquet rotatable dihedrals)")
    ax.set_ylabel("unique molecules")
    ax.set_title("Dihedral-count distribution by source (after dedup + filters)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = f"{out_prefix}_d_hist_by_source.pdf"
    fig.savefig(out, dpi=130)
    print(f"d-histogram -> {out}")


# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    i = sub.add_parser("ingest", help="Parse sources + compute descriptors -> CSV")
    i.add_argument("--inputs", nargs="*", default=None,
                   help="Input .smi files (default: smiles/*.smi from the repo root)")
    i.add_argument("--output", default="descriptors.csv", help="Descriptor CSV path")
    i.add_argument("--workers", "-w", type=int, default=8)
    i.add_argument("--resume", action="store_true",
                   help="Skip raw SMILES already present in --output")
    i.set_defaults(func=ingest)

    s = sub.add_parser("select", help="Dedup + filter + stratified sample -> manifest")
    s.add_argument("descriptors", help="Descriptor CSV from 'ingest'")
    s.add_argument("--out-prefix", default="manifest",
                   help="Output prefix: <prefix>.smi, <prefix>.csv, histogram PDF")
    s.add_argument("--per-bin", type=int, default=30, help="Target molecules per d-bin")
    s.add_argument("--bin-width", type=int, default=1, help="d-bin width (default 1)")
    s.add_argument("--d-min", type=int, default=1, help="Minimum d to include (default 1)")
    s.add_argument("--d-max", type=int, default=20, help="Maximum d to include (default 20)")
    s.add_argument("--max-heavy", type=int, default=80,
                   help="Drop molecules with more heavy atoms (xTB cost; default 80)")
    s.add_argument("--allowed-elements", default=DEFAULT_ELEMENTS,
                   help="Comma-separated element whitelist")
    s.add_argument("--seed", type=int, default=42, help="Sampling RNG seed")
    s.add_argument("--id-prefix", default="m",
                   help="Prefix for generated mol_id (use a distinct one per set, "
                   "e.g. 'v' for the validation set, so ids never collide if merged).")
    s.add_argument("--exclude", nargs="*", default=None,
                   help="Prior manifest CSV(s) whose molecules to exclude, so a "
                   "held-out set is disjoint by structure (e.g. the calibration set).")
    s.add_argument("--no-hist", dest="hist", action="store_false",
                   help="Skip the per-source d-histogram PDF")
    s.set_defaults(func=select)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
