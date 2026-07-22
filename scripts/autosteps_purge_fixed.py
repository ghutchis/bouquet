#!/usr/bin/env python
"""Purge stale results for the connectivity-fixed autosteps oligomers so a
``--resume`` rerun recomputes them.

Background: fixing the fluorene/carbazole backbone coupling in
``generate_oligomers.py`` changed the *chemistry* of a few calibration molecules
WITHOUT changing their ``mol_id`` (the dihedral count -- hence the ``_d##`` label
-- was unchanged). The sweep keys resume off ``(config, name, seed)`` and only
``drop_failed`` clears rows, and only for ``success != 1``. The fixed molecules
were recorded ``success=1`` on the *wrong* structure, so a plain ``--resume``
would silently keep the stale results. This script removes their rows (summary,
trajectory, certificate CSVs) and per-trial geometry / certificate files for the
requested seeds, keyed by name, so the next ``--resume`` re-runs exactly them.

The target names come from the subset manifest ``smiles/autosteps-calib-fixed.smi``
(kept in sync there); the d-bin for each is read from its ``_d##`` suffix, which
determines the ``as_s{seed}_d{dd}`` file family written by autosteps_bouquet.slurm.

Usage:
    python scripts/autosteps_purge_fixed.py --results <RESULTS_DIR>
    python scripts/autosteps_purge_fixed.py --results <RESULTS_DIR> --dry-run
    python scripts/autosteps_purge_fixed.py --results <RESULTS_DIR> --seeds 1 2 3
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

# Default subset manifest listing the fixed molecules (SMILES<TAB>name).
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "smiles" / "autosteps-calib-fixed.smi"


def load_targets(manifest: Path) -> list[str]:
    """Molecule names (mol_ids) to purge, from the subset .smi (2nd column)."""
    names = []
    for line in manifest.read_text().splitlines():
        if line.strip():
            names.append(line.split("\t")[1])
    return names


def dbin_of(name: str) -> str:
    """Two-digit d-bin from a ``..._d##`` name, matching the slurm file naming."""
    m = re.search(r"_d(\d+)$", name)
    if not m:
        raise ValueError(f"name {name!r} has no _d## suffix")
    return f"{int(m.group(1)):02d}"


def purge_csv(path: Path, targets: set[str], seed: str, config: str, dry: bool) -> int:
    """Drop rows whose (config, name, seed) matches a target. Preserves the file's
    own header (certificate columns are dynamic). Returns the number removed."""
    if not path.exists():
        return 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or "name" not in fields:
        return 0
    kept, removed = [], 0
    for r in rows:
        hit = (
            r.get("name") in targets
            and str(r.get("seed")) == seed
            and (r.get("config") == config if "config" in fields else True)
        )
        if hit:
            removed += 1
        else:
            kept.append(r)
    if removed and not dry:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(kept)
    return removed


def safe_name(name: str) -> str:
    """Mirror sweep_common._safe_filename for per-trial file names."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "mol"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True,
                    help="RESULTS dir holding as_s{seed}_d{dd}.csv (+ _traj/_cert, geom/)")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                    help="subset .smi listing the fixed molecules (default: %(default)s)")
    ap.add_argument("--seeds", nargs="+", default=["1", "2", "3"],
                    help="seeds to purge (default: 1 2 3)")
    ap.add_argument("--config", default="grad",
                    help="sweep config/arm name (default: grad, per stop_benchmark.py)")
    ap.add_argument("--geom-subdir", default="geom",
                    help="geometry subdir under --results (default: geom)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be removed; change nothing")
    args = ap.parse_args()

    names = load_targets(args.manifest)
    targets = set(names)
    by_bin: dict[str, set[str]] = {}
    for n in names:
        by_bin.setdefault(dbin_of(n), set()).add(n)

    tag = "[dry-run] " if args.dry_run else ""
    print(f"{tag}Purging {len(names)} molecule(s) across seeds {args.seeds} in {args.results}")
    for n in names:
        print(f"  - {n} (d-bin {dbin_of(n)})")

    total_rows = total_files = 0
    geom_dir = args.results / args.geom_subdir
    for seed in args.seeds:
        for dd, bin_names in by_bin.items():
            stem = args.results / f"as_s{seed}_d{dd}"
            for suffix in ("", "_traj", "_cert"):
                p = stem.with_name(stem.name + suffix + ".csv")
                total_rows += purge_csv(p, bin_names, seed, args.config, args.dry_run)
            certfiles = stem.with_name(stem.name + "_certfiles")
            for n in bin_names:
                base = f"{args.config}_{safe_name(n)}_seed{seed}"
                for fp in (geom_dir / f"{base}.xyz", certfiles / f"{base}.csv"):
                    if fp.exists():
                        total_files += 1
                        if not args.dry_run:
                            fp.unlink()

    verb = "would remove" if args.dry_run else "removed"
    print(f"{tag}{verb} {total_rows} CSV row(s) and {total_files} per-trial file(s).")
    if not args.dry_run:
        print("Now re-submit autosteps_bouquet.slurm (with --resume) for the affected "
              "d-bins/seeds; it will recompute exactly these trials.")


if __name__ == "__main__":
    main()
