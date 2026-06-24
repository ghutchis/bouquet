"""Generate oligomer SMILES of varying length for dihedral-scaling benchmarks.

Emits a `.smi` file (SMILES<TAB>name) covering several oligomer families across
a range of lengths, so the benchmark sweeps bouquet's searchable dihedral count
(as reported by `bouquet.setup.detect_dihedrals`, which runs after AddHs).

Usage:
    pixi run python scripts/generate_oligomers.py --nmin 1 --nmax 8 \
        --out smiles/oligo-scan.smi
"""

from __future__ import annotations

import argparse

from rdkit import Chem
from rdkit.Chem import AllChem

from bouquet.setup import detect_dihedrals

# name -> (prefix cap, repeat unit, suffix cap)
# Peptides use neutral Ac- / -NHMe caps; nylon-6 / PEG use H/OH caps.
FAMILIES: dict[str, tuple[str, str, str]] = {
    "PPE": ("", "C#Cc1ccc(cc1)", ""),  # rigid-rod control: 0 dihedrals
    "thiophene": ("", "c1ccc(s1)", ""),
    "polyethylene": ("", "CC", "C"),
    "PEG": ("", "OCC", "O"),
    "polypropylene": ("", "CC(C)", "C"),  # isotactic
    "polyglycine": ("CC(=O)", "NCC(=O)", "NC"),
    "polyalanine": ("CC(=O)", "N[C@@H](C)C(=O)", "NC"),
    "polyserine": ("CC(=O)", "N[C@@H](CO)C(=O)", "NC"),
    "nylon6": ("", "NCCCCCC(=O)", "O"),
    # Foldamers: gas-phase helices driven by intramolecular H-bonds / sterics.
    # Aib (alpha-aminoisobutyric acid) -> 3_10/alpha helix (gem-dimethyl funnel).
    # beta3-homoalanine -> 14-helix. Both verified to fold under MMFF (n>=4).
    "Aib": ("CC(=O)", "NC(C)(C)C(=O)", "NC"),
    "b3homoAla": ("CC(=O)", "N[C@@H](C)CC(=O)", "NC"),
}


def build_smiles(pre: str, unit: str, post: str, n: int) -> str:
    """Concatenate caps and n repeat units into an oligomer SMILES."""
    return pre + unit * n + post


def count_dihedrals(smiles: str) -> int | None:
    """Number of rotatable dihedrals bouquet would search, or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=0xF00D) != 0:
        # embedding failure doesn't change topology-based detection, but
        # detect_dihedrals expects a conformer-bearing mol downstream
        AllChem.EmbedMolecule(mol, AllChem.EmbedParameters())
    return len(detect_dihedrals(mol))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nmin", type=int, default=1, help="minimum oligomer length")
    ap.add_argument("--nmax", type=int, default=8, help="maximum oligomer length")
    ap.add_argument("--out", default="smiles/oligo-scan.smi", help="output .smi path")
    args = ap.parse_args()

    rows: list[tuple[str, str]] = []
    print(f"{'name':18} {'smiles':50} dihedrals")
    for fam, (pre, unit, post) in FAMILIES.items():
        for n in range(args.nmin, args.nmax + 1):
            smi = build_smiles(pre, unit, post, n)
            ndih = count_dihedrals(smi)
            if ndih is None:
                raise ValueError(f"Invalid SMILES for {fam} n={n}: {smi}")
            name = f"{fam}_n{n}"
            rows.append((smi, name))
            print(f"{name:18} {smi:50} {ndih}")

    with open(args.out, "w") as fh:
        for smi, name in rows:
            fh.write(f"{smi}\t{name}\n")
    print(f"\nWrote {len(rows)} oligomers to {args.out}")


if __name__ == "__main__":
    main()
