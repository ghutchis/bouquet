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
    # Conjugated side-chain polymers (organic electronics): a rigid aromatic backbone
    # (planar-preferring 2-fold inter-ring/aryl-vinyl dihedrals) carrying FLEXIBLE alkyl/
    # alkoxy side chains -- a distinct torsional regime (chain torsions dominate the DOF
    # count) and real chemistry. They gap-fill d-values the plain backbones miss: PPV hits
    # ODD d (1,3,5,7,9,11); the side chains give large per-unit increments for high-d
    # anchors (P3HT +7/unit, 3-(2-ethylhexyl) +10, MEH-PPV +13). All 'aromatic' class.
    "P3HT": ("", "c1cc(CCCCCC)c(s1)", ""),               # poly(3-hexylthiophene)
    "P3EHT": ("", "c1cc(CCC(CC)CCCC)c(s1)", ""),         # 3-(2-ethylhexyl)thiophene (branched)
    "PPV": ("", "C=Cc1ccc(cc1)", ""),                    # poly(phenylene-vinylene)
    "MEH_PPV": ("", "C=Cc1cc(OC)c(OCC(CC)CCCC)cc1", ""),  # 2-methoxy-5-(2-ethylhexyloxy)-PPV
    "polyaniline": ("", "Nc1ccc(cc1)", "N"),             # leucoemeraldine (aryl-NH-); EVEN d 2..12
    # Polystyrene + derivatives: flexible sp3 -CH2-CH(Ar)- backbone with PENDANT aromatic
    # rings (aromatic class, but the difficulty is backbone + phenyl-rotation torsions --
    # a distinct regime from the rigid conjugated backbones above).
    "polystyrene": ("", "CC(c1ccccc1)", "C"),            # atactic, d 3,6,9,12
    "PS_4Me": ("", "CC(c1ccc(C)cc1)", "C"),              # poly(4-methylstyrene)
    "PS_4OMe": ("", "CC(c1ccc(OC)cc1)", "C"),            # poly(4-methoxystyrene)
    # Rigid engineering/optoelectronic backbones. Fluorene/carbazole have an sp3/N bridge
    # in a FUSED 5-ring that is rigid (not puckerable) -> 'aromatic' class via the ring_class
    # single-sp3-apex rule. High-d anchors with real chemistry.
    "polycarbonate": ("", "Oc1ccc(cc1)C(C)(C)c1ccc(cc1)OC(=O)", "O"),  # bisphenol-A PC
    "polyfluorene": ("", "CC1(C)c2cc(ccc2-c2ccc(cc21))", ""),          # poly(9,9-dimethylfluorene), 2,7
    "PF_hexyl": ("", "C(CCCCC)C1(CCCCCC)c2cc(ccc2-c2ccc(cc21))", ""),  # 9,9-dihexylfluorene (alkyl)
    "polycarbazole": ("", "Cn1c2cc(ccc2c2ccc(cc21))", ""),            # poly(N-methylcarbazole), 3,6
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
    """Number of rotatable dihedrals bouquet would search, or None if invalid.

    ``detect_dihedrals`` is a pure graph property (no conformer needed), so we skip
    3D embedding -- embedding large fused oligomers (polyfluorene, MEH-PPV) is slow
    enough to stall a full --nmax sweep."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return len(detect_dihedrals(Chem.AddHs(mol)))


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
