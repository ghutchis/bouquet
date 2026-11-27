"""Build a real-molecule 1D/2D dihedral slice for the illustrative BO figures.

Evaluates bouquet's own objective -- :func:`bouquet.assess.evaluate_energy_with_gradient`
with constrained relaxation (the scanned AND the remaining dihedrals held fixed,
everything else relaxed for ``DEFAULT_RELAXATION_STEPS`` FIRE steps, exactly as in
the search) -- on a regular periodic grid over one or two rotors. The remaining
rotors are anchored at the reference conformer (the GFN2 CREST global minimum for
the stopbench molecules in ``data/bo_slices/``).

The output ``.npz`` is read by ``scripts/plot_bo_1d.py`` / ``plot_bo_2d.py``
via ``--slice``:

  grid_deg     (M,)            grid angles per axis, degrees, [0, 360)
  energy_eV    (M,) / (M, M)   energy relative to the slice minimum, eV
  grad_eV_rad  (..., k)        dE/dtheta for the k scanned rotors, eV/rad
  anchor_deg   (k,)            the reference conformer's scanned angles
  chains       (k, 4)          the scanned dihedrals' atom chains
  smiles, mol_id, method, relax_steps, e_min_eV (absolute slice minimum)

Run (2D, rotors 1 and 2 of m139_d06; ~20 min on 8 cores):
    PYTHONPATH=. pixi run python scripts/build_bo_slice.py \
        data/bo_slices/m139_d06.xyz --rotors 1 2 --n-grid 72
"""

from __future__ import annotations

import os

# One thread per worker: unpinned xtb OpenMP x N pool workers oversubscribes the
# machine badly. Must be set before numpy / xtb load.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import itertools
import json
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

EV2KCAL = 23.060541945

_STATE: dict = {}


def _load(xyz: str, smiles: str, method: str):
    """Per-process structure, rotors, calculator and anchor angles (cached)."""
    if _STATE:
        return _STATE["v"]
    import ase.io
    from rdkit import Chem
    from bouquet.calculator import CalculatorFactory
    from bouquet.setup import (
        apply_charge_spin,
        default_multiplicity,
        detect_dihedrals,
        get_initial_structure_from_file,
    )

    # Rewrite through ASE so the comment line (which carries numbers in the
    # reference files) can't trip RDKit's XYZ reader.
    clean = Path(tempfile.mkdtemp()) / Path(xyz).name
    ase.io.write(clean, ase.io.read(xyz, index=0), format="xyz", comment=Path(xyz).stem)
    atoms, mol = get_initial_structure_from_file(str(clean))
    charge = Chem.GetFormalCharge(Chem.MolFromSmiles(smiles)) if smiles else 0
    apply_charge_spin(atoms, charge, default_multiplicity(atoms, charge))
    dihedrals = detect_dihedrals(mol)
    calc = CalculatorFactory.create(method, mol)
    anchor = np.array([d.get_angle(atoms) for d in dihedrals])
    _STATE["v"] = (atoms, dihedrals, calc, anchor)
    return _STATE["v"]


def _evaluate(task):
    from bouquet.assess import evaluate_energy_with_gradient

    xyz, smiles, method, steps, rotors, values = task
    atoms, dihedrals, calc, anchor = _load(xyz, smiles, method)
    angles = anchor.copy()
    angles[list(rotors)] = values
    energy, _, grad = evaluate_energy_with_gradient(
        angles, atoms, dihedrals, calc, calc, relax=True, steps=steps
    )
    return energy, grad[list(rotors)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xyz", help="reference conformer (.xyz); a sibling .json supplies smiles/mol_id")
    ap.add_argument("--rotors", type=int, nargs="+", required=True,
                    help="1 or 2 rotor indices (detect_dihedrals order)")
    ap.add_argument("--n-grid", type=int, default=72, help="grid points per axis")
    ap.add_argument("--method", default="gfn2")
    ap.add_argument("--steps", type=int, default=None,
                    help="relaxation steps (default: bouquet's DEFAULT_RELAXATION_STEPS)")
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--out", default=None, help="output .npz (default: next to the xyz)")
    args = ap.parse_args()

    if len(args.rotors) not in (1, 2):
        ap.error("--rotors takes one or two indices")

    from bouquet.config import DEFAULT_RELAXATION_STEPS, RELAX_FAILURE_ENERGY_EV

    steps = DEFAULT_RELAXATION_STEPS if args.steps is None else args.steps
    xyz = Path(args.xyz)
    meta_path = xyz.with_suffix(".json")
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    smiles = meta.get("smiles", "")
    mol_id = meta.get("mol_id", xyz.stem)
    rotors = tuple(args.rotors)

    _, dihedrals, _, anchor = _load(str(xyz), smiles, args.method)
    if max(rotors) >= len(dihedrals):
        ap.error(f"{mol_id} has {len(dihedrals)} rotors; got {rotors}")

    grid = np.linspace(0.0, 360.0, args.n_grid, endpoint=False)
    points = list(itertools.product(grid, repeat=len(rotors)))
    tasks = [(str(xyz), smiles, args.method, steps, rotors, np.array(p)) for p in points]

    t0 = time.time()
    with Pool(args.nproc) as pool:
        res = pool.map(_evaluate, tasks, chunksize=4)
    shape = (args.n_grid,) * len(rotors)
    energy = np.array([r[0] for r in res]).reshape(shape)
    grad = np.array([r[1] for r in res]).reshape(shape + (len(rotors),))

    failed = energy >= RELAX_FAILURE_ENERGY_EV
    if failed.any():
        print(f"WARNING: {failed.sum()} failed evaluations (set to the slice max)")
        energy[failed] = energy[~failed].max()
        grad[failed] = 0.0
    e_min = float(energy.min())

    out = Path(args.out) if args.out else xyz.with_name(
        f"{mol_id}_slice_{'_'.join(map(str, rotors))}.npz")
    np.savez(
        out,
        grid_deg=grid,
        energy_eV=energy - e_min,
        grad_eV_rad=grad,
        anchor_deg=anchor[list(rotors)] % 360.0,
        chains=np.array([dihedrals[r].chain for r in rotors]),
        failed=failed,
        smiles=smiles,
        mol_id=mol_id,
        method=args.method,
        relax_steps=steps,
        e_min_eV=e_min,
    )
    rel = (energy - e_min) * EV2KCAL
    imin = np.unravel_index(int(energy.argmin()), shape)
    print(f"{mol_id} rotors {rotors}: {len(points)} points in {time.time() - t0:.0f} s; "
          f"slice min at {[float(grid[i]) for i in imin]} deg, "
          f"anchor at {np.round(anchor[list(rotors)] % 360.0, 1).tolist()} deg, "
          f"max {rel.max():.1f} kcal/mol -> {out}")


if __name__ == "__main__":
    main()
