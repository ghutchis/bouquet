#!/usr/bin/env python3
"""
Bouquet Analysis Script

Processes multiple CSV files (e.g., platinum-ei.csv, platinum-lei.csv, etc.)
and extracts the minimum e_e0_unconstrained value across all seeds for each molecule.

Output: A combined CSV with columns:
  - name
  - num_dihedrals
  - min_unconstrained_{suffix} for each input file
"""

import pandas as pd
import argparse
from pathlib import Path


def get_unconstrained_columns(df: pd.DataFrame) -> list[str]:
    """Find all e_e0_unconstrained columns in the dataframe."""
    return [col for col in df.columns if col.endswith('_e_e0_unconstrained')]


def compute_min_unconstrained(df: pd.DataFrame) -> pd.Series:
    """Compute the minimum e_e0_unconstrained value across all seeds for each row."""
    unconstrained_cols = get_unconstrained_columns(df)
    # Use min with skipna=True to handle missing values
    return df[unconstrained_cols].min(axis=1, skipna=True)


def extract_suffix_from_filename(filepath: Path) -> str:
    """
    Extract the suffix from a filename like 'platinum-ei.csv' -> 'ei'
    Assumes format: {prefix}-{suffix}.csv
    """
    stem = filepath.stem  # e.g., 'platinum-ei'
    if '-' in stem:
        return stem.split('-')[-1]
    else:
        # Fallback: use the whole stem if no hyphen
        return stem


def process_files(input_files: list[Path], output_file: Path) -> pd.DataFrame:
    """
    Process multiple CSV files and combine results into a single output CSV.

    Args:
        input_files: List of paths to input CSV files
        output_file: Path for the output CSV

    Returns:
        The combined DataFrame
    """
    result_df = None

    for filepath in input_files:
        print(f"Processing: {filepath}")

        # Read the CSV
        df = pd.read_csv(filepath)

        # Extract suffix for column naming
        suffix = extract_suffix_from_filename(filepath)

        # Compute minimum unconstrained energy
        min_col_name = f"min_unconstrained_{suffix}"
        min_values = compute_min_unconstrained(df)

        if result_df is None:
            # Initialize result with name and num_dihedrals from first file
            result_df = df[['name', 'num_dihedrals']].copy()
            result_df[min_col_name] = min_values
        else:
            # Verify that names match across files
            if not df['name'].equals(result_df['name']):
                print(f"  Warning: Names in {filepath} don't match the first file!")
                print("  Attempting to merge by name...")
                temp_df = df[['name']].copy()
                temp_df[min_col_name] = min_values
                result_df = result_df.merge(temp_df, on='name', how='outer')
            else:
                result_df[min_col_name] = min_values

        # Report statistics
        n_valid = min_values.notna().sum()
        print(f"  Found {len(get_unconstrained_columns(df))} seed columns")
        print(f"  Computed min for {n_valid}/{len(df)} molecules")

    # Save to output file
    result_df.to_csv(output_file, index=False)
    print(f"\nSaved results to: {output_file}")
    print(f"Output shape: {result_df.shape}")

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze bouquet CSV files and extract minimum unconstrained energies."
    )
    parser.add_argument(
        'input_files',
        nargs='+',
        type=Path,
        help="Input CSV files (e.g., platinum-ei.csv platinum-lei.csv)"
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('bouquet_analysis.csv'),
        help="Output CSV file (default: bouquet_analysis.csv)"
    )

    args = parser.parse_args()

    # Validate input files exist
    for filepath in args.input_files:
        if not filepath.exists():
            print(f"Error: File not found: {filepath}")
            return 1

    # Process files
    process_files(args.input_files, args.output)

    return 0


if __name__ == '__main__':
    exit(main())
