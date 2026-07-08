#!/usr/bin/env python
"""Merge per-seed (or per-chunk) sweep CSVs from a distributed SLURM run.

When a sweep is split across nodes -- one task per seed writing its own
``--output`` -- this concatenates the per-task summary CSVs and their
``_traj`` siblings into a single pair of files for ``analyze``/``traj``.

It is deliberately a collect-then-merge step rather than a shared append:
multiple nodes appending to one CSV on a network drive race on the header and
interleave rows (NFS append is not atomic, and the trajectory writer emits many
rows per trial), so each task writes its own files and they are joined here.

The merge is header-deduplicating row concatenation, with three safety checks:
  - every input's columns must match sweep_common.FIELDNAMES (resp.
    TRAJ_FIELDNAMES) exactly, so a stale-schema file can't corrupt the merge;
  - duplicate (config, name, seed) keys -- two tasks that ran the same cell --
    are reported (fatal unless --allow-overlap, which keeps the first);
  - each summary file's ``<stem>_traj.csv`` sibling is required unless
    --no-traj, so the two merged files stay row-consistent for the paired
    analysis.

Usage:
    # Explicit files (shell-expanded glob):
    python scripts/merge_sweep.py results/init_seed*.csv --output results/init_all.csv

    # Or point at a directory; *_seed*.csv (minus _traj/_cert) are discovered:
    python scripts/merge_sweep.py results/ --output results/init_all.csv

    # Then analyze the merged pair as usual:
    python scripts/sweep_init.py analyze results/init_all.csv
    python scripts/sweep_init.py traj    results/init_all_traj.csv --plot-dir results/plots
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

# Reuse the canonical column definitions (same directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_common import FIELDNAMES, TRAJ_FIELDNAMES  # noqa: E402


def traj_sibling(summary_path: Path) -> Path:
    """The ``<stem>_traj<suffix>`` path for a summary CSV (matches run_sweep's
    default trajectory naming in sweep_common.run_sweep)."""
    return summary_path.with_name(
        f"{summary_path.stem}_traj{summary_path.suffix or '.csv'}"
    )


def discover_inputs(args_inputs: List[str]) -> List[Path]:
    """Expand the positional inputs into a sorted list of summary CSV paths.

    A directory is scanned for ``*.csv`` that are neither ``_traj`` nor ``_cert``
    files (those are siblings/aggregates, not summaries); a file is taken as-is.
    Sorting keeps the merge order deterministic, so the surviving row on an
    --allow-overlap duplicate is stable run-to-run.
    """
    def is_sibling(path: Path) -> bool:
        # _traj/_cert files are siblings/aggregates, not summaries. The summary
        # merge re-derives each summary's _traj sibling itself, so these must be
        # dropped however they arrive -- including when a shell glob like
        # ``init_seed*.csv`` expands to include ``init_seed1_traj.csv``.
        return path.stem.endswith("_traj") or path.stem.endswith("_cert")

    paths: List[Path] = []
    for raw in args_inputs:
        p = Path(raw)
        if p.is_dir():
            paths.extend(c for c in sorted(p.glob("*.csv")) if not is_sibling(c))
        elif p.exists():
            if not is_sibling(p):
                paths.append(p)
        else:
            sys.exit(f"Input not found: {raw}")
    # De-duplicate while preserving order (a dir + an explicit file can overlap).
    seen, unique = set(), []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def read_checked(path: Path, expected_fields: List[str]) -> List[dict]:
    """Read a CSV, asserting its header is exactly ``expected_fields``."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != expected_fields:
            sys.exit(
                f"Schema mismatch in {path}:\n"
                f"  expected: {expected_fields}\n"
                f"  found:    {reader.fieldnames}\n"
                "Refusing to merge mismatched files (stale script version?)."
            )
        return list(reader)


def merge_summary(
    summary_paths: List[Path], out_path: Path, allow_overlap: bool
) -> Tuple[int, dict]:
    """Concatenate summary CSVs into ``out_path``; return (n_rows, key_map).

    ``key_map`` is (config, name, seed) -> the path that won that cell, so
    ``merge_traj`` can keep only trajectory rows from the same file that
    survived the summary merge (relevant when ``--allow-overlap`` drops a
    duplicate: its traj sibling must be dropped too, not just its summary row).

    Keys are (config, name, seed). A key appearing in two files means two tasks
    ran the same cell -- fatal unless ``allow_overlap`` (then the first wins).
    """
    seen_keys: dict = {}  # key -> source path of the first occurrence
    merged: List[dict] = []
    dupes: List[Tuple[tuple, Path, Path]] = []
    for path in summary_paths:
        for row in read_checked(path, FIELDNAMES):
            key = (row["config"], row["name"], str(row["seed"]))
            if key in seen_keys:
                dupes.append((key, seen_keys[key], path))
                continue  # keep the first occurrence
            seen_keys[key] = path
            merged.append(row)

    if dupes:
        print(f"WARNING: {len(dupes)} duplicate (config, name, seed) cell(s) across files:")
        for key, first, second in dupes[:10]:
            print(f"  {key}  first={first.name}  dropped={second.name}")
        if len(dupes) > 10:
            print(f"  ... and {len(dupes) - 10} more")
        if not allow_overlap:
            sys.exit(
                "Refusing to merge overlapping cells (the tasks were not cleanly "
                "partitioned). Re-check the seed/chunk split, or pass --allow-overlap "
                "to keep the first occurrence of each key."
            )

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(merged)
    return len(merged), seen_keys


def merge_traj(
    summary_paths: List[Path], out_path: Path, keep_keys: dict
) -> int:
    """Concatenate the ``_traj`` siblings of each summary file into ``out_path``.

    Only rows whose (config, name, seed) survived the summary merge, AND whose
    file is the one that won that key, are kept -- so an --allow-overlap drop
    can't leak a duplicate's trajectory rows in. A missing sibling is fatal
    (the pair must stay consistent for paired analysis).
    """
    merged: List[dict] = []
    for path in summary_paths:
        sib = traj_sibling(path)
        if not sib.exists():
            sys.exit(
                f"Missing trajectory sibling for {path.name}: expected {sib.name}. "
                "Pass --no-traj to merge summaries only."
            )
        for row in read_checked(sib, TRAJ_FIELDNAMES):
            key = (row["config"], row["name"], str(row["seed"]))
            if keep_keys.get(key) == path:
                merged.append(row)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRAJ_FIELDNAMES)
        w.writeheader()
        w.writerows(merged)
    return len(merged)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help="Per-task summary CSVs (shell glob) or a directory to scan for them.",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Merged summary CSV; the trajectory goes to <stem>_traj<suffix>.",
    )
    p.add_argument(
        "--no-traj",
        action="store_true",
        help="Merge only the summary CSVs (skip the _traj siblings).",
    )
    p.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Permit duplicate (config, name, seed) cells across files, keeping "
        "the first occurrence instead of aborting.",
    )
    args = p.parse_args()

    summary_paths = discover_inputs(args.inputs)
    # Never ingest our own output (or its traj sibling) -- merging into the same
    # directory you scan would otherwise slurp a previous run's merged file and
    # flag every cell as a duplicate.
    exclude = {args.output.resolve(), traj_sibling(args.output).resolve()}
    summary_paths = [p for p in summary_paths if p.resolve() not in exclude]
    if not summary_paths:
        sys.exit("No input summary CSVs found.")
    print(f"Merging {len(summary_paths)} summary file(s):")
    for sp in summary_paths:
        print(f"  {sp}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_sum, keep_keys = merge_summary(summary_paths, args.output, args.allow_overlap)
    print(f"\nWrote {n_sum} summary rows -> {args.output}")

    if not args.no_traj:
        traj_out = traj_sibling(args.output)
        n_traj = merge_traj(summary_paths, traj_out, keep_keys)
        print(f"Wrote {n_traj} trajectory rows -> {traj_out}")

    # Quick coverage report: trials per (config, seed) so a missing task is obvious.
    from collections import Counter

    by_cfg_seed = Counter((k[0], k[2]) for k in keep_keys)
    print("\nCoverage (trials per config x seed):")
    for (cfg, seed), n in sorted(by_cfg_seed.items()):
        print(f"  {cfg:<16} seed={seed:<8} {n}")


if __name__ == "__main__":
    main()
