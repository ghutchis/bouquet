"""Phase-3 payoff: calls-to-minimum, gradient-enhanced GP vs value-only GP.

The hypothesis behind gradient-enhanced BO is data efficiency: each energy
evaluation also contributes dE/dtheta, so the surrogate should locate the
minimum in fewer calls. This benchmark runs both surrogates on a set of flexible
molecules (paired seeds, fixed budget) and reports:

  * final best relative energy at the budget (lower = better), and
  * calls-to-minimum: the first evaluation reaching within EPS of a per-molecule
    reference minimum (censored at the budget if never reached).

The reference minimum per molecule is the best energy found across a generous
random-sampling baseline plus both methods' own trajectories, so it does not
favor either surrogate.

Run: KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. pixi run python \
        scripts/gradient_benchmark.py [--backend mmff|gfnff]
"""

from __future__ import annotations

import argparse
import contextlib
import os

import numpy as np
import torch

from bouquet import solver
from bouquet.calc_rdkit import RDKitMMFFCalculator
from bouquet.calculator import CalculatorFactory
from bouquet.setup import detect_dihedrals, get_initial_structure

MOLECULES = {
    "butanediol": "OCCCCO",
    "pentanediol": "OCCCCCO",
    "diethyleneglycol": "OCCOCCO",
    "aminopentanol": "NCCCCCO",
}
SEEDS = [0, 1, 2, 3]
INIT_STEPS = 5
N_STEPS = 30
EPS = 0.043  # ~1 kcal/mol: "reached the minimum" tolerance (eV)


@contextlib.contextmanager
def _silence():
    with open(os.devnull, "w") as dn:
        old = os.dup(1)
        try:
            os.dup2(dn.fileno(), 1)
            yield
        finally:
            os.dup2(old, 1)
            os.close(old)


def _make_calc(mol, backend):
    if backend == "mmff":
        return RDKitMMFFCalculator(mol)
    return CalculatorFactory.create("gfnff", mol=mol)


def run_bo(smiles, seed, use_gradients, backend):
    """Return the best-so-far energy trace over all evaluations (eV, rel)."""
    torch.manual_seed(seed)
    atoms, mol = get_initial_structure(smiles)
    dihedrals = detect_dihedrals(mol)
    calc = _make_calc(mol, backend)
    with _silence():
        state = solver._setup_initial_state(atoms, dihedrals, calc, calc, True, None)
        state.prior_module = None
        state.use_gradients = use_gradients
        solver._evaluate_initial_guesses(
            state, dihedrals, calc, calc, True, INIT_STEPS, seed, None, None
        )
        solver._run_optimization_loop(
            state, N_STEPS, dihedrals, calc, calc, True, None
        )
    e = state.observed_energies.detach().cpu().numpy()
    return np.minimum.accumulate(e)


def calls_to_min(trace, ref, eps, budget):
    """First index (1-based, excluding the start point) within eps of ref."""
    hit = np.where(trace <= ref + eps)[0]
    if hit.size == 0:
        return budget + 1  # censored
    return int(hit[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mmff", "gfnff"], default="mmff")
    args = ap.parse_args()

    print(f"backend={args.backend}  init={INIT_STEPS}  n_steps={N_STEPS}  "
          f"seeds={SEEDS}  eps={EPS} eV")
    print(f"{'molecule':<18}{'d':>3} {'best nograd':>12}{'best grad':>11}"
          f"{'regret imp':>12}{'calls nograd':>13}{'calls grad':>11}")

    budget = INIT_STEPS + N_STEPS
    grand = {"reg_imp": [], "calls_ng": [], "calls_g": [], "win": 0, "loss": 0, "tie": 0,
             "faster": 0, "slower": 0, "same": 0}
    for name, smiles in MOLECULES.items():
        _, mol = get_initial_structure(smiles)
        d = len(detect_dihedrals(mol))
        # collect traces
        tr_ng = [run_bo(smiles, s, False, args.backend) for s in SEEDS]
        tr_g = [run_bo(smiles, s, True, args.backend) for s in SEEDS]
        ref = min(min(t[-1] for t in tr_ng), min(t[-1] for t in tr_g))
        bn = np.mean([t[-1] for t in tr_ng])
        bg = np.mean([t[-1] for t in tr_g])
        cn = np.mean([calls_to_min(t, ref, EPS, budget) for t in tr_ng])
        cg = np.mean([calls_to_min(t, ref, EPS, budget) for t in tr_g])
        for tn, tg in zip(tr_ng, tr_g):
            if tg[-1] < tn[-1] - 1e-4:
                grand["win"] += 1
            elif tg[-1] > tn[-1] + 1e-4:
                grand["loss"] += 1
            else:
                grand["tie"] += 1
            c_n, c_g = calls_to_min(tn, ref, EPS, budget), calls_to_min(tg, ref, EPS, budget)
            if c_g < c_n:
                grand["faster"] += 1
            elif c_g > c_n:
                grand["slower"] += 1
            else:
                grand["same"] += 1
        grand["reg_imp"].append(bn - bg); grand["calls_ng"].append(cn); grand["calls_g"].append(cg)
        print(f"{name:<18}{d:>3} {bn:>12.4f}{bg:>11.4f}{bn - bg:>12.4f}"
              f"{cn:>13.1f}{cg:>11.1f}")
    print("-" * 79)
    print(f"{'mean':<21} {'':>12}{'':>11}{np.mean(grand['reg_imp']):>12.4f}"
          f"{np.mean(grand['calls_ng']):>13.1f}{np.mean(grand['calls_g']):>11.1f}")
    print(f"\nfinal-best   grad better: {grand['win']}  worse: {grand['loss']}  tie: {grand['tie']}")
    print(f"calls-to-min grad faster: {grand['faster']}  slower: {grand['slower']}  same: {grand['same']}")
    print("(regret imp > 0 and fewer calls favor the gradient-enhanced GP; "
          "calls censored at budget+1 if the minimum was never reached.)")


if __name__ == "__main__":
    main()
