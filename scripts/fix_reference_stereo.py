#!/usr/bin/env python
"""Audit and repair CREST reference minima that are the wrong STEREOISOMER of the input.

Why this exists: `reference.py`'s species guards compare bond CONNECTIVITY, and a double-bond
isomer has an identical bond graph. CREST's iMTD-GC runs hot with an RMSD bias and will cross
a conjugated C=N / C=C isomerisation barrier, so an isomerised structure passes every check
and becomes the reference global minimum. Bouquet only turns single bonds, so it is then
scored against a compound it cannot reach -- and the loss reads as a search failure.
Measured on stopbench-500: 5 of 30 molecules with a stereo-specified double bond, 3 of them
counted among the 82 CREST wins.

This is a CONFORMER benchmark, so the reference must be the same stereoisomer as the input.
(A structure-search benchmark would want the opposite policy plus a bouquet move that can flip
a double bond; neither exists today.)

Needs no bouquet run -- it compares the input SMILES against the reference geometry, so it
covers every molecule including the censored high-d ones, and it reuses the per-frame
`E_rel_kcal` already written by reference.py rather than re-relaxing anything.

    pixi run python scripts/fix_reference_stereo.py \
        --manifest smiles/stopbench-500.csv \
        --reference-dir stop_bench_gfn2/reference_gfn2 --report
    ... add --write to rewrite the affected <mol_id>.json in place (a .bak is kept).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

KCAL = 23.060541945


def cip_from_graph(molh):
    """Double-bond CIP labels required by the INPUT (never taken from a pool member, so a
    fully-isomerised pool is reported as having no valid reference rather than re-baselined
    onto the wrong isomer)."""
    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler

    m = Chem.Mol(molh)
    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    rdCIPLabeler.AssignCIPLabels(m)
    return {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): b.GetProp("_CIPCode")
            for b in m.GetBonds() if b.HasProp("_CIPCode")}


def cip_from_3d(molh, atoms):
    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler
    from rdkit.Geometry import Point3D

    m = Chem.Mol(molh)
    conf = Chem.Conformer(m.GetNumAtoms())
    for i, p in enumerate(atoms.get_positions()):
        conf.SetAtomPosition(i, Point3D(*map(float, p)))
    m.RemoveAllConformers()
    m.AddConformer(conf, assignId=True)
    Chem.AssignStereochemistryFrom3D(m)
    rdCIPLabeler.AssignCIPLabels(m)
    return {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): b.GetProp("_CIPCode")
            for b in m.GetBonds() if b.HasProp("_CIPCode")}


def frames_with_energy(path: Path):
    """Reference frames paired with the E_rel_kcal reference.py wrote in each comment line."""
    from ase.io import read

    fr = read(str(path), index=":")
    if not isinstance(fr, list):
        fr = [fr]
    es = [float(m.group(1))
          for ln in path.read_text().splitlines()
          if (m := re.search(r"E_rel_kcal=([-\d.eE+]+)", ln))]
    return list(zip(fr, es)) if len(es) == len(fr) else [(f, None) for f in fr]


def audit_one(mid, smiles, refdir: Path):
    from rdkit import Chem

    xyz = refdir / f"{mid}.xyz"
    if not xyz.exists():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    molh = Chem.AddHs(mol)
    want = cip_from_graph(molh)
    if not want:
        return None                                  # nothing stereo-specified to enforce
    frames = frames_with_energy(xyz)
    if not frames or len(frames[0][0]) != molh.GetNumAtoms():
        return None
    good = []
    for a, e in frames:
        got = cip_from_3d(molh, a)
        if all(got.get(k) is None or got[k] == v for k, v in want.items()):
            good.append(e)
    good = [e for e in good if e is not None]
    gmin = frames[0][1]
    if gmin is None:
        return None
    corrected = min(good) if good else None
    # The stored minima are energy-sorted, so "the global min is fine" means frame 0 survived.
    flipped = corrected is None or corrected > gmin + 1e-6
    return {"mol_id": mid, "n_frames": len(frames), "gmin_kcal": gmin,
            "corrected_kcal": corrected, "flipped": flipped,
            "n_valid": len(good), "shift_kcal": None if corrected is None else corrected - gmin}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--reference-dir", type=Path, required=True)
    ap.add_argument("--report", action="store_true", help="print the audit (default)")
    ap.add_argument("--write", action="store_true",
                    help="rewrite affected <mol_id>.json: E_star_crest/E_star/E_star_nonBO "
                         "re-derived from the correct isomer, or the molecule marked "
                         "no_valid_reference when the whole pool is isomerised")
    ap.add_argument("--out", type=Path, default=None, help="write the audit as CSV")
    args = ap.parse_args()

    import pandas as pd

    man = pd.read_csv(args.manifest).set_index("mol_id")
    col = "raw_smiles" if "raw_smiles" in man.columns else "smiles"

    rows = []
    for mid in man.index:
        r = audit_one(mid, man.loc[mid, col], args.reference_dir)
        if r:
            rows.append(r)
    flipped = [r for r in rows if r["flipped"]]
    unfixable = [r for r in flipped if r["corrected_kcal"] is None]

    print(f"{len(rows)} molecules with a stereo-specified double bond and a reference")
    print(f"  reference global minimum is the WRONG isomer: {len(flipped)}")
    print(f"  ... repairable from frames already on disk:   {len(flipped)-len(unfixable)}")
    print(f"  ... NO valid-isomer structure anywhere:       {len(unfixable)}  "
          f"(no usable reference; drop from the benchmark)")
    if flipped:
        print(f"\n  {'mol':<12}{'bad gmin':>10}{'corrected':>11}{'shift':>8}{'valid/frames':>14}")
        for r in flipped:
            c = f"{r['corrected_kcal']:.2f}" if r["corrected_kcal"] is not None else "--"
            s = f"{r['shift_kcal']:+.2f}" if r["shift_kcal"] is not None else "n/a"
            print(f"  {r['mol_id']:<12}{r['gmin_kcal']:>10.2f}{c:>11}{s:>8}"
                  f"{str(r['n_valid'])+'/'+str(r['n_frames']):>14}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote audit to {args.out}")

    if not args.write:
        print("\n(dry run -- pass --write to repair the JSONs)")
        return

    for r in flipped:
        jf = args.reference_dir / f"{r['mol_id']}.json"
        if not jf.exists():
            continue
        shutil.copy(jf, jf.with_suffix(".json.bak"))
        d = json.loads(jf.read_text())
        d["stereo_audit"] = {"reference_gmin_was_wrong_isomer": True,
                             "shift_kcal": r["shift_kcal"], "n_valid": r["n_valid"],
                             "n_frames": r["n_frames"]}
        if r["corrected_kcal"] is None:
            # Every pooled structure is the wrong isomer: there is no reference to correct to.
            d["ok"] = False
            d["no_valid_reference"] = "all pooled minima are a different stereoisomer"
        else:
            shift_eV = r["shift_kcal"] / KCAL
            for key in ("E_star", "E_star_crest", "E_star_nonBO"):
                if d.get(key) is not None:
                    d[key] = d[key] + shift_eV
        jf.write_text(json.dumps(d, indent=2))
        print(f"  repaired {jf.name}")


if __name__ == "__main__":
    main()
