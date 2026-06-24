#!/usr/bin/env python
"""Surgically drop specific trials from a sweep's summary + trajectory CSVs.

``--retry-failed`` only removes rows recorded ``success != 1``. When a trial
completes but its data is tainted (e.g. the wrong git branch was checked out
mid-run), it is recorded ``success=1`` and the retry path leaves it in place.
This tool removes rows matching a (name[, config][, seed]) selector from BOTH
CSVs -- keeping them consistent the way ``sweep_common.drop_failed`` does -- so a
subsequent ``run ... --resume`` re-runs exactly the removed (config, name, seed)
keys.

Example -- drop the tainted gradfreeze runs of one molecule, then re-run them:

    python scripts/drop_sweep_entry.py grad.csv --name T74_3EQR_A --config gradfreeze
    python scripts/sweep_gradient.py run --input mols.csv --output grad.csv --resume ...

Omit --config and/or --seed to widen the selector (e.g. drop every config/seed
for a molecule). Use --dry-run first to see what would be removed.
"""

import argparse
import csv
import sys
from pathlib import Path

# Reuse the sweep's column schemas so the rewritten files stay canonical.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_common as sc  # noqa: E402


def _rewrite(path: Path, fieldnames: list, match) -> tuple[int, int]:
    """Drop rows where ``match(row)`` is true; return (removed, total)."""
    if not path.exists():
        return (0, 0)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    kept = [r for r in rows if not match(r)]
    removed = len(rows) - len(kept)
    if removed:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(kept)
    return (removed, len(rows))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("output", type=Path, help="Summary CSV from 'sweep ... run'")
    p.add_argument("--traj-output", type=Path, default=None,
                   help="Trajectory CSV (default: <output>_traj<suffix>)")
    p.add_argument("--name", required=True, help="Molecule name to drop")
    p.add_argument("--config", default=None,
                   help="Restrict to this arm (default: all configs)")
    p.add_argument("--seed", default=None,
                   help="Restrict to this seed (default: all seeds)")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be removed without rewriting files")
    args = p.parse_args()

    traj_path = args.traj_output or args.output.with_name(
        f"{args.output.stem}_traj{args.output.suffix or '.csv'}"
    )

    def match(r) -> bool:
        return (
            r["name"] == args.name
            and (args.config is None or r["config"] == args.config)
            and (args.seed is None or str(r["seed"]) == str(args.seed))
        )

    sel = f"name={args.name}"
    if args.config is not None:
        sel += f", config={args.config}"
    if args.seed is not None:
        sel += f", seed={args.seed}"

    if args.dry_run:
        for path, fields in ((args.output, sc.FIELDNAMES),
                             (traj_path, sc.TRAJ_FIELDNAMES)):
            if not path.exists():
                print(f"{path}: (missing)")
                continue
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            hit = [r for r in rows if match(r)]
            keys = sorted({(r["config"], r["name"], str(r["seed"])) for r in hit})
            print(f"{path}: would remove {len(hit)}/{len(rows)} rows; keys={keys}")
        print(f"\nDry run only ({sel}); nothing written.")
        return

    n_sum, t_sum = _rewrite(args.output, sc.FIELDNAMES, match)
    n_traj, t_traj = _rewrite(traj_path, sc.TRAJ_FIELDNAMES, match)
    print(f"Removed {n_sum}/{t_sum} summary rows from {args.output}")
    print(f"Removed {n_traj}/{t_traj} trajectory rows from {traj_path}")
    if n_sum or n_traj:
        print(f"Re-run with: sweep_gradient.py run ... --output {args.output} --resume")
    else:
        print(f"No rows matched ({sel}); nothing changed.")


if __name__ == "__main__":
    main()
