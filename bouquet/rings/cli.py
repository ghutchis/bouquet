"""``bouquet-ring-mtd`` -- discover ring-conformer seeds for a molecule.

    bouquet-ring-mtd --smiles "C1CC2(C1)NCC[C@@H]2N[C@@H]1CCC2(C1)OCCO2" --out seeds.json

Runs RMSD-biased ring metadynamics (:func:`bouquet.rings.run_ring_mtd`) and writes the
``ring_seeds/1.0`` JSON that bouquet consumes as ``initial_conformers``. An acyclic or
purely-aromatic molecule writes an empty seed set (the gate declined to run).
"""

from __future__ import annotations

import argparse
import sys
import time

from bouquet.rings.mtd import RingMTDConfig, run_ring_mtd
from bouquet.rings.seeds import write_seeds


def build_config(args) -> RingMTDConfig:
    """Map parsed CLI args onto a :class:`RingMTDConfig` (unset flags keep the defaults)."""
    d = RingMTDConfig()
    overrides = {
        "engine": args.engine,
        "temperature_K": args.temperature,
        "t_run_ps": args.t_run_ps,
        "t_equil_ps": args.t_equil_ps,
        "n_walkers": args.walkers,
        "n_processes": args.processes,
        "k_hill_kcal": args.k_hill,
        "energy_window_kcal": args.energy_window,
        "topology_check": args.topology_check,
        "seed": args.seed,
    }
    return RingMTDConfig(**{**d.__dict__, **{k: v for k, v in overrides.items() if v is not None}})


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bouquet-ring-mtd", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smiles", required=True, help="input SMILES (stereochemistry honoured)")
    p.add_argument("--out", required=True, help="output ring-seed JSON path")
    p.add_argument("--engine", default=None, help="calculator method (default gfnff)")
    p.add_argument("--temperature", type=float, default=None, help="MD temperature K (default 500)")
    p.add_argument("--t-run-ps", type=float, default=None, help="biased run per walker (default 20)")
    p.add_argument("--t-equil-ps", type=float, default=None, help="unbiased alpha-calibration run (default 2)")
    p.add_argument("--walkers", type=int, default=None, help="number of walkers (default 4)")
    p.add_argument("--processes", type=int, default=None, help="parallel walker processes (default 1)")
    p.add_argument("--k-hill", type=float, default=None, help="hill height kcal/mol (default 0.5)")
    p.add_argument("--energy-window", type=float, default=None, help="seed energy window kcal/mol (default 12)")
    p.add_argument("--topology-check", action="store_true", default=None,
                   help="reject bond-graph changes (needed for gfn2/aimnet2)")
    p.add_argument("--seed", type=int, default=None, help="random seed (default 0)")
    p.add_argument("--quiet", action="store_true", help="suppress the summary line")
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    cfg = build_config(args)

    from rdkit import Chem
    mol = Chem.MolFromSmiles(args.smiles)
    if mol is None:
        print(f"error: could not parse SMILES {args.smiles!r}", file=sys.stderr)
        return 2

    t0 = time.time()
    result = run_ring_mtd(mol, cfg)
    wall = time.time() - t0

    from bouquet.rings.systems import perceive_ring_systems
    systems = perceive_ring_systems(Chem.AddHs(mol))
    write_seeds(args.out, mol, systems, result, cfg, wall_seconds=wall)

    if not args.quiet:
        d = result.diagnostics
        if d.get("skipped"):
            print(f"{args.smiles}: gate declined ({d['skipped']}); wrote empty seed set -> {args.out}")
        else:
            alarm = " [STEREO-REJECT ALARM: k_hill too high]" if d.get("stereo_reject_alarm") else ""
            print(f"{args.smiles}: {len(result.seeds)} ring-state seed(s) from "
                  f"{d.get('n_walkers')} walker(s), {d.get('n_clusters')} clusters, "
                  f"{wall:.1f}s -> {args.out}{alarm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
