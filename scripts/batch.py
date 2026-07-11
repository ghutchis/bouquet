#!/usr/bin/env python
"""
Batch processing script for bouquet conformer optimization.

Runs multiple seeds for each molecule in a CSV file and collects results.
"""

import argparse
import csv
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bouquet.calculator import CalculatorFactory

# Method choices from the calculator registry (installed subset) -- tracks new
# methods and never offers an unavailable one.
_METHOD_CHOICES = list(CalculatorFactory.available_methods())


def parse_log_output(log_text: str) -> dict:
    """
    Parse the log output to extract optimization metrics.

    Expected log lines:
        - Best energy found on step X
        - Found low energy on step X
        - Found first good energy on step X
        - Performed final relaxation with dihedral constraints. E: X. E-E0: X
        - Performed final relaxation without dihedral constraints. E: X. E-E0: X
    """
    results = {
        "best_step": None,
        "low_step": None,
        "good_step": None,
        "e_e0_constrained": None,
        "e_e0_unconstrained": None,
        "num_dihedrals": None,
    }

    # Parse number of dihedral angles
    match = re.search(r"Detected (\d+) dihedral angles", log_text)
    if match:
        results["num_dihedrals"] = int(match.group(1))

    # Parse best energy step
    match = re.search(r"Best energy found on step (\d+)", log_text)
    if match:
        results["best_step"] = int(match.group(1))

    # Parse low energy step
    match = re.search(r"Found low energy on step (\d+)", log_text)
    if match:
        results["low_step"] = int(match.group(1))

    # Parse first good energy step
    match = re.search(r"Found first good energy on step (\d+)", log_text)
    if match:
        results["good_step"] = int(match.group(1))

    # Parse E-E0 with dihedral constraints
    match = re.search(
        r"with dihedral constraints\. E: [^\s]+\. E-E0: ([-\d.eE]+)", log_text
    )
    if match:
        results["e_e0_constrained"] = float(match.group(1))

    # Parse E-E0 without dihedral constraints
    match = re.search(
        r"without dihedral constraints\. E: [^\s]+\. E-E0: ([-\d.eE]+)", log_text
    )
    if match:
        results["e_e0_unconstrained"] = float(match.group(1))

    return results


def run_single(
    smiles: str,
    name: str,
    seed: int,
    energy_method: str | None = None,
    optimizer_method: str | None = None,
) -> tuple[dict, bool, str]:
    """
    Run the CLI for a single molecule with a single seed.

    Returns:
        tuple of (results dict, success bool, log text)
    """
    # can also use --file f"platinum/{name}.mol",
    cmd = [
        sys.executable,
        "-m",
        "bouquet.cli",
        "--smiles",
        smiles,
        "--name",
        name,
        "--seed",
        str(seed),
        "--auto",
        "--relax",
        "--priors",
        "bouquet/data/gfn2_priors.json"
    ]

    if energy_method:
        cmd.extend(["--energy", energy_method])
    if optimizer_method:
        cmd.extend(["--optimizer", optimizer_method])

    result = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)

    # Combine stdout and stderr since logging might go to either
    log_text = result.stdout + "\n" + result.stderr

    return parse_log_output(log_text), result.returncode == 0, log_text


def build_header(seeds: list[int]) -> list[str]:
    """Build the CSV header row."""
    header = ["smiles", "name", "num_dihedrals"]
    for seed in seeds:
        header.extend(
            [
                f"seed{seed}_best_step",
                f"seed{seed}_low_step",
                f"seed{seed}_good_step",
                f"seed{seed}_e_e0_constrained",
                f"seed{seed}_e_e0_unconstrained",
            ]
        )
    return header


def process_molecule(
    smiles: str,
    name: str,
    seeds: list[int],
    energy_method: str | None,
    optimizer_method: str | None,
    verbose: bool = False,
    workers: int = 1,
) -> list:
    """Process a single molecule with all seeds."""
    row = [smiles, name]
    num_dihedrals = None
    seed_results = {seed: None for seed in seeds}

    if workers == 1:
        # Sequential execution
        for seed in seeds:
            print(f"  Running seed {seed}...", end=" ", flush=True)
            try:
                results, success, log_text = run_single(
                    smiles, name, seed, energy_method, optimizer_method
                )
                if success and results["best_step"] is not None:
                    if num_dihedrals is None and results["num_dihedrals"] is not None:
                        num_dihedrals = results["num_dihedrals"]
                    seed_results[seed] = [
                        results["best_step"],
                        results["low_step"],
                        results["good_step"],
                        results["e_e0_constrained"],
                        results["e_e0_unconstrained"],
                    ]
                    e_e0 = results["e_e0_unconstrained"]
                    e_e0_str = f"{e_e0:.4f}" if e_e0 is not None else "N/A"
                    print(
                        f"done (best step: {results['best_step']}, "
                        f"E-E0: {e_e0_str})"
                    )
                else:
                    seed_results[seed] = ["", "", "", "", ""]
                    print("FAILED")
                    if verbose:
                        print(f"    Log output:\n{log_text}")
            except Exception as e:
                seed_results[seed] = ["", "", "", "", ""]
                print(f"ERROR: {e}")
    else:
        # Parallel execution
        # print(f"  Running {len(seeds)} seeds in parallel (workers={workers})...", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_seed = {
                executor.submit(
                    run_single, smiles, name, seed, energy_method, optimizer_method
                ): seed
                for seed in seeds
            }

            for future in as_completed(future_to_seed):
                seed = future_to_seed[future]
                try:
                    results, success, log_text = future.result()
                    if success and results["best_step"] is not None:
                        if num_dihedrals is None and results["num_dihedrals"] is not None:
                            num_dihedrals = results["num_dihedrals"]
                        seed_results[seed] = [
                            results["best_step"],
                            results["low_step"],
                            results["good_step"],
                            results["e_e0_constrained"],
                            results["e_e0_unconstrained"],
                        ]
                        e_e0 = results["e_e0_unconstrained"]
                        e_e0_str = f"{e_e0:.4f}" if e_e0 is not None else "N/A"
                        print(
                            f"    Seed {seed}: done (best step: {results['best_step']}, "
                            f"E-E0: {e_e0_str})"
                        )
                    else:
                        seed_results[seed] = ["", "", "", "", ""]
                        print(f"    Seed {seed}: FAILED")
                        if verbose:
                            print(f"      Log output:\n{log_text}")
                except Exception as e:
                    seed_results[seed] = ["", "", "", "", ""]
                    print(f"    Seed {seed}: ERROR: {e}")

    # Build final row: smiles, name, num_dihedrals, then seed results in order
    row.append(num_dihedrals if num_dihedrals is not None else "")
    for seed in seeds:
        row.extend(seed_results[seed] if seed_results[seed] else ["", "", "", "", ""])

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Batch process molecules with multiple seeds"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input CSV file with smiles and name columns",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV file for results",
    )
    parser.add_argument(
        "--seeds",
        default="1234,12345,3141,314159,42",
        help="Comma-separated list of seeds (e.g., '1234,12345,3141,314159,42')",
    )
    parser.add_argument(
        "--energy",
        default=None,
        choices=_METHOD_CHOICES,
        help="Energy method (uses default if not specified)",
    )
    parser.add_argument(
        "--optimizer",
        default=None,
        choices=_METHOD_CHOICES,
        help="Optimizer method (uses default if not specified)",
    )
    parser.add_argument(
        "--smiles-column",
        default="smiles",
        help="Name of the SMILES column in input CSV (default: 'smiles')",
    )
    parser.add_argument(
        "--name-column",
        default="name",
        help="Name of the name column in input CSV (default: 'name')",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file (skip already processed molecules)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print full log output on failures",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Number of parallel workers for running seeds (default: 1, sequential)",
    )

    args = parser.parse_args()

    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    print(f"Using seeds: {seeds}")
    if args.workers > 1:
        print(f"Running with {args.workers} parallel workers")

    # Read input CSV
    # Detect delimiter and whether there's a header row
    with open(args.input, newline="") as f:
        # Read all rows first
        sample = f.read()
        f.seek(0)

        # Detect delimiter (tab or comma)
        dialect = csv.Sniffer().sniff(sample, delimiters="\t, ")

        reader = csv.reader(f, dialect)
        rows = list(reader)

    if not rows:
        print(f"Error: Empty input file {args.input}")
        sys.exit(1)

    # Check if first row looks like a header. Require BOTH configured columns so
    # downstream mol[smiles_column]/mol[name_column] lookups can't KeyError; if
    # only one (or neither) is present, fall back to positional header below.
    first_row = rows[0]
    has_header = args.smiles_column in first_row and args.name_column in first_row

    # Convert rows to list of dicts
    if has_header:
        # Use the header row
        header = first_row
        data_rows = rows[1:]
    else:
        # No header - assume column 0 is SMILES, column 1 is name
        header = [args.smiles_column, args.name_column] + [f"col{i}" for i in range(2, len(first_row))]
        data_rows = rows
        print(f"No header row detected. Assuming column 0 is SMILES, column 1 is name.")

    molecules = []
    for row in data_rows:
        if len(row) >= 2:  # Need at least SMILES and name
            mol_dict = {header[i]: row[i] for i in range(min(len(header), len(row)))}
            molecules.append(mol_dict)

    print(f"Loaded {len(molecules)} molecules from {args.input}")

    # Handle resume mode
    processed_names = set()
    if args.resume and args.output.exists():
        with open(args.output, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_names.add(row["name"])
        print(f"Resuming: {len(processed_names)} molecules already processed")
    else:
        # Write header to output
        header = build_header(seeds)
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # Track summary statistics
    total_trials = 0
    improved_trials = 0

    # Process each molecule
    for i, mol in enumerate(molecules):
        smiles = mol[args.smiles_column]
        name = mol[args.name_column]

        if name in processed_names:
            print(f"[{i+1}/{len(molecules)}] Skipping {name} (already processed)")
            continue

        print(f"[{i+1}/{len(molecules)}] Processing {name}...")

        start_time = time.time()
        row = process_molecule(
            smiles,
            name,
            seeds,
            args.energy,
            args.optimizer,
            args.verbose,
            args.workers,
        )
        elapsed = time.time() - start_time
        print(f"  took {elapsed:.1f} seconds")

        # Append row incrementally
        with open(args.output, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # Update summary statistics
        # best_step values are at indices 3, 8, 13, ... (3 + 5*i for each seed)
        # (after smiles, name, num_dihedrals)
        for j, seed in enumerate(seeds):
            best_step_idx = 3 + 5 * j
            best_step = row[best_step_idx]
            if best_step != "":  # successful trial
                total_trials += 1
                if best_step > 1:
                    improved_trials += 1

    # Print summary
    print(f"\nBatch complete. Results saved to {args.output}")
    if total_trials > 0:
        pct_improved = 100 * improved_trials / total_trials
        print(
            f"Summary: {improved_trials}/{total_trials} trials ({pct_improved:.1f}%) "
            f"found a better conformer than the first step"
        )


if __name__ == "__main__":
    main()
