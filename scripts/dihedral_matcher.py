#!/usr/bin/env python3
"""
Dihedral Angle Matcher

Matches dihedral angles in a molecule against a list of SMARTS patterns.
When multiple patterns match the same dihedral, uses the most specific
(later in the list) pattern.

Usage:
    python dihedral_matcher.py molecule.mol patterns.txt -o results.csv
"""

import argparse
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
import math


def load_smarts_patterns(filepath: str) -> list[str]:
    """Load SMARTS patterns from a file (one per line)."""
    patterns = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                patterns.append(line)
    return patterns


def get_dihedral_angle(mol, atom_indices: tuple[int, int, int, int]) -> float:
    """
    Calculate the dihedral angle in degrees for four atom indices.

    Args:
        mol: RDKit molecule with 3D coordinates
        atom_indices: Tuple of 4 atom indices (i, j, k, l)

    Returns:
        Dihedral angle in degrees (-180 to 180)
    """
    conf = mol.GetConformer()
    return rdMolTransforms.GetDihedralDeg(conf, *atom_indices)


def normalize_dihedral(indices: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """
    Normalize dihedral indices so the smaller central bond atom comes first.
    This ensures dihedrals (a,b,c,d) and (d,c,b,a) are treated as the same.
    """
    i, j, k, l = indices
    # Compare central atoms; if reversed order has smaller first central atom, flip
    if j > k or (j == k and i > l):
        return (l, k, j, i)
    return indices


def match_dihedrals_to_patterns(mol, smarts_patterns: list[str]) -> dict:
    """
    Match all dihedrals in a molecule to SMARTS patterns.

    For each unique dihedral, finds all matching patterns and keeps
    the most specific one (highest index in the pattern list).

    Args:
        mol: RDKit molecule with 3D coordinates
        smarts_patterns: List of SMARTS patterns (general to specific)

    Returns:
        Dictionary mapping normalized dihedral indices to
        (pattern_index, angle_degrees, smarts_pattern)
    """
    # Dictionary to store best match for each unique dihedral
    # Key: normalized (i, j, k, l) tuple
    # Value: (pattern_index, angle, smarts_pattern)
    dihedral_matches = {}

    for pattern_idx, smarts in enumerate(smarts_patterns):
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            print(f"Warning: Invalid SMARTS pattern at index {pattern_idx}: {smarts}")
            continue

        # Find all matches for this pattern
        matches = mol.GetSubstructMatches(pattern)

        for match in matches:
            if len(match) != 4:
                continue  # Dihedral patterns should match exactly 4 atoms

            # Normalize the dihedral to avoid duplicates
            norm_indices = normalize_dihedral(match)

            # Always update with later (more specific) pattern
            # Or add if this is a new dihedral
            if norm_indices not in dihedral_matches or pattern_idx > dihedral_matches[norm_indices][0]:
                angle = get_dihedral_angle(mol, match)
                dihedral_matches[norm_indices] = (pattern_idx, angle, smarts)

    return dihedral_matches


def main():
    parser = argparse.ArgumentParser(
        description='Match dihedral angles to SMARTS patterns'
    )
    parser.add_argument('molfile', help='Input molecule file (.mol, .mol2, .sdf)')
    parser.add_argument('patterns', help='File containing SMARTS patterns (one per line)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output file (CSV format). If not specified, prints to stdout')
    parser.add_argument('--add-hydrogens', action='store_true',
                        help='Add hydrogens to the molecule before matching')
    args = parser.parse_args()

    # Load molecule
    mol_path = Path(args.molfile)
    if mol_path.suffix.lower() in ['.mol', '.sdf']:
        mol = Chem.MolFromMolFile(str(mol_path), removeHs=False)
    elif mol_path.suffix.lower() == '.mol2':
        mol = Chem.MolFromMol2File(str(mol_path), removeHs=False)
    else:
        raise ValueError(f"Unsupported file format: {mol_path.suffix}")

    if mol is None:
        raise ValueError(f"Could not parse molecule from {args.molfile}")

    # Optionally add hydrogens
    if args.add_hydrogens:
        mol = Chem.AddHs(mol, addCoords=True)

    # Ensure we have 3D coordinates
    if mol.GetNumConformers() == 0:
        print("Warning: No 3D coordinates found. Generating conformer...")
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)

    # Load SMARTS patterns
    smarts_patterns = load_smarts_patterns(args.patterns)
    print(f"Loaded {len(smarts_patterns)} SMARTS patterns")

    # Match dihedrals
    matches = match_dihedrals_to_patterns(mol, smarts_patterns)
    print(f"Found {len(matches)} unique dihedrals")

    # Output results
    lines = ["atom_i,atom_j,atom_k,atom_l,pattern_index,angle_degrees,smarts"]
    for (i, j, k, l), (pattern_idx, angle, smarts) in sorted(matches.items()):
        # Escape any commas in SMARTS for CSV
        smarts_escaped = f'"{smarts}"' if ',' in smarts else smarts
        lines.append(f"{i},{j},{k},{l},{pattern_idx},{angle:.2f},{smarts_escaped}")

    output_text = '\n'.join(lines)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_text)
        print(f"Results written to {args.output}")
    else:
        print("\nResults:")
        print(output_text)


if __name__ == '__main__':
    main()
