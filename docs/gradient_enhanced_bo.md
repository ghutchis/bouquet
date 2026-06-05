# Gradient-enhanced Bayesian optimization — design & handoff

Branch: `claude/gradient-boosted-bayesian-opt-h7l7H`

This document is the working plan for adding **gradient-enhanced Bayesian
optimization** to bouquet: feeding analytic `dE/dθ` (energy gradient w.r.t. each
torsion) into the GP surrogate so each expensive energy evaluation contributes
`1 + d` numbers instead of one.

## Decisions (already made)

- **Approach:** gradient-enhanced GP-BO — derivative observations in the GP,
  not gradient-boosted trees and not (yet) a local-polish hybrid.
- **Gradient source:** relaxed-torque via the **envelope theorem**. The objective
  the BO fits is the *relaxed* surface `E*(θ) = min_{other DOF} E(x; θ)`; at the
  constrained minimum the only residual generalized force is along the torsion,
  so projecting the Cartesian forces at the relaxed geometry gives `dE*/dθ`
  directly. Works for the default `gfn2` (xTB returns analytic forces) — autodiff
  via ANI is just one of several gradient sources, not required.
- **First milestone:** prototype/validate on a cheap backend (RDKit MMFF) before
  touching the default `gfn2` path.

## Key insight that simplifies the math

bouquet sets a dihedral by **rigidly rotating a defined atom group** about the
central bond (`atoms.set_dihedral(*di.chain, angle, indices=di.group)`). So the
position Jacobian is trivial — no general Wilson B-matrix needed. For atom `k`
in `DihedralInfo.group`, with `û` the unit bond axis (between `chain[1]` and
`chain[2]`) and `p` any point on it:

```
dx_k/dθ = û × (x_k − p)
dE/dθ   = − Σ_k F_k · (û × (x_k − p))      # = −torque about the bond axis; θ in radians
```

On-axis atoms contribute zero, so they're harmless to include. Validated against
central finite differences to **< 0.1 %**.

---

## Phase 0 — Gradient extraction + finite-difference validation ✅ DONE

Implemented, tested, committed on this branch.

- **`bouquet/gradients.py`**
  - `project_torsion_gradient(positions, dihedrals, forces, per_degree=False)` —
    pure function: Cartesian forces → `dE/dθ` (eV/rad or eV/deg).
  - `compute_torsion_gradient(atoms, dihedrals, calc, per_degree=False)` —
    fetches **unconstrained** forces from `calc` (drops any `FixInternals` on a
    copy so the torsion component survives) and projects.
  - Units default to **eV/radian**.
- **`bouquet/assess.py`**
  - `evaluate_energy_with_gradient(...)` → `(energy, atoms, gradient)`; returns a
    NaN gradient when the energy evaluation fails (so callers can drop the point).
- **`bouquet/config.py`**
  - `RELAX_FAILURE_ENERGY_EV = 1000.0` — names the relaxation-failure sentinel
    that was previously a magic `1000.0` in `assess.py`.
- **`tests/test_gradients.py`** (8 tests, all passing)
  - Projection vs. finite differences on the **rigid scan**.
  - Envelope-theorem check: `dE*/dθ` vs. finite differences of the **relaxed**
    surface (other DOF re-relaxed at each angle).
  - Properties: on-axis atoms contribute zero, degenerate axis raises, input
    `Atoms` not mutated, NaN gradient on failed evaluation.

**Status:** `pytest` → new tests 8 passed; full suite 73 passed, 24 skipped
(skips are pre-existing xtb/slow and optional-dep tests — no regressions).

---

## Phase 1 — Gradient-enhanced GP surrogate (NEXT)

Build a GP that trains on stacked `[E, dE/dθ]` observations. Two candidate
routes; **pick by validating posterior mean AND gradient against a known
analytic periodic function + finite differences** (mirror the Phase 0 gate):

- **(a) sin/cos input embedding + `gpytorch.kernels.RBFKernelGrad`.** Map each
  angle `θ → (cos θ, sin θ)`; an RBF on the embedding is periodic, and
  `RBFKernelGrad` supplies derivative observations. Reuses existing machinery;
  the wrinkle is the chain rule mapping `dE/dθ` to the embedding's tangent
  derivative (1 physical dim ↔ 2 embedding dims — provide the directional
  derivative along the circle tangent).
- **(b) custom periodic gradient kernel.** Hand-derive the first/cross
  derivatives of `PeriodicKernel` (gpytorch ships `RBFKernelGrad` but **no**
  `PeriodicKernelGrad`). Cleanest semantics, more effort.

Then wrap as a BoTorch-compatible model exposing the **function-value** posterior
so `LogExpectedImprovement` and the PiBO `PriorGuidedAcquisitionFunction` keep
working unchanged. **This value-posterior wrapper is the main integration risk.**

Reference: gpytorch "GP Regression with derivatives" example (uses
`RBFKernelGrad` + a `(n, d+1)` target stack).

## Phase 2 — Wire into the solver loop

- `OptimizationState` (`solver.py:71`) gains an index-aligned
  `observed_gradients` tensor; extend `append_observation`.
- `_select_next_points_botorch` uses the gradient-enhanced GP when enabled, else
  falls back to the current `_periodic_covar_module` GP (`solver.py:112`). Keep
  the PiBO path intact.
- **Filtering** (important): drop gradients from failed evaluations (NaN /
  `RELAX_FAILURE_ENERGY_EV`) and from non-converged FIRE2 relaxations. The
  relaxed surface can **jump basins** as θ sweeps; near those boundaries the
  gradient is locally valid but globally misleading, so it can corrupt the GP.

## Phase 3 — CLI, config, benchmark

- Add `--use-gradients` flag + `Configuration` field; log it into
  `run_params.json`.
- Benchmark **calls-to-minimum** with vs. without gradients on a small molecule
  set — MMFF first, then the default `gfn2`. This is the payoff check.

## Phase 4 — Stretch

- Hybrid local polish (gradient descent / trust-region on BO proposals).
- Confirm transparent extension to `gfn2`/Psi4 analytic forces (the projection
  is backend-agnostic); add the ANI autograd path as an alternative source.

---

## Resuming locally

```bash
git fetch origin
git checkout claude/gradient-boosted-bayesian-opt-h7l7H
# use the project env (environment.yml / pixi) for the full toolchain incl. xtb
pytest tests/test_gradients.py -v
```

Note: the web container had `torch`, `botorch`, `ase`, `rdkit`, `pytest`
pip-installed only to run the Phase 0 validation — these are **not** committed;
use the repo's normal environment locally.

## Risks / open items

- Value-posterior wrapper so BoTorch acquisition sees only `E` (Phase 1).
- Basin-jump discontinuities in `E*(θ)` corrupting gradients near boundaries
  (mitigated by the convergence/failure filtering in Phase 2).
- Gradient-enhanced GP cost scales as `O((n(d+1))³)` — fine at small `d` (typical
  conformer searches), watch it as `d` grows.
- Gradient unit convention (eV/rad vs eV/deg) must stay consistent with
  `observed_coords`, which are stored in **degrees** (`solver.py`).
