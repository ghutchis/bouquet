"""Quick before/after check for the periodic-kernel period fix.

Background: the value-only BO surrogate normalizes dihedral inputs by /360 (full
turn = 1.0), but historically set the periodic kernel period to 360, which makes
the covariance near-degenerate on [0, 1] inputs (k(0deg, 180deg) ~= 1). The fix
sets the period to 1.0 (config.GP_PERIOD_LENGTH_MEAN).

This script runs the *value-only* optimizer on a few small molecules with the
old (360) and new (1.0) period, fixed seeds, fixed evaluation budget, and
compares the best relative energy found (lower is better). It reuses the solver's
internal pipeline so we can read the full energy trace.

Run: KMP_DUPLICATE_LIB_OK=TRUE pixi run python scripts/period_fix_benchmark.py
"""

from __future__ import annotations

import numpy as np

from bouquet import solver
from bouquet.calc_rdkit import RDKitMMFFCalculator
from bouquet.setup import detect_dihedrals, get_initial_structure

# Flexible H-bonding chains whose *folded* (gauche) conformer is well below the
# extended start geometry -- so finding the minimum genuinely exercises the
# surrogate's point selection (rather than the start already being optimal).
MOLECULES = {
    "butanediol": "OCCCCO",
    "pentanediol": "OCCCCCO",
    "diethyleneglycol": "OCCOCCO",
    "aminobutanol": "NCCCCO",
}
SEEDS = [0, 1, 2, 3]
INIT_STEPS = 6
N_STEPS = 30
RELAX = True


def run_once(smiles: str, seed: int, period_mean: float) -> np.ndarray:
    """Run the value-only optimizer; return the best-so-far energy trace (eV)."""
    solver.GP_PERIOD_LENGTH_MEAN = period_mean  # patched module global
    atoms, mol = get_initial_structure(smiles)
    dihedrals = detect_dihedrals(mol)
    calc = RDKitMMFFCalculator(mol)

    state = solver._setup_initial_state(atoms, dihedrals, calc, calc, RELAX, None)
    state.prior_module = None
    solver._evaluate_initial_guesses(
        state, dihedrals, calc, calc, RELAX, INIT_STEPS, seed, None, None
    )
    solver._run_optimization_loop(state, N_STEPS, dihedrals, calc, calc, RELAX, None)
    energies = state.observed_energies.detach().cpu().numpy()
    return np.minimum.accumulate(energies)


def main():
    print(f"init_steps={INIT_STEPS}  n_steps={N_STEPS}  seeds={SEEDS}")
    print(f"{'molecule':<17}{'d':>3}  {'best old(360)':>14}{'best new(1.0)':>14}"
          f"{'improvement':>13}{'new wins':>10}")
    grand_old, grand_new = [], []
    wins = losses = ties = 0
    for name, smiles in MOLECULES.items():
        _, mol = get_initial_structure(smiles)
        d = len(detect_dihedrals(mol))
        best_old, best_new = [], []
        nwin = 0
        for seed in SEEDS:
            old = run_once(smiles, seed, 360.0)[-1]
            new = run_once(smiles, seed, 1.0)[-1]
            best_old.append(old)
            best_new.append(new)
            if new < old - 1e-4:
                nwin += 1
                wins += 1
            elif new > old + 1e-4:
                losses += 1
            else:
                ties += 1
        mo, mn = float(np.mean(best_old)), float(np.mean(best_new))
        grand_old.append(mo)
        grand_new.append(mn)
        print(f"{name:<17}{d:>3}  {mo:>14.4f}{mn:>14.4f}{mo - mn:>13.4f}"
              f"{f'{nwin}/{len(SEEDS)}':>10}")
    print("-" * 71)
    print(f"{'mean':<20}  {np.mean(grand_old):>14.4f}{np.mean(grand_new):>14.4f}"
          f"{np.mean(grand_old) - np.mean(grand_new):>13.4f}")
    print(f"\nper-(molecule,seed) record  new better: {wins}   worse: {losses}   tie: {ties}")
    print("(energies relative to start geometry, eV; lower = better;"
          " positive improvement = new period found a lower minimum)")


if __name__ == "__main__":
    main()
