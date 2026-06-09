"""End-to-end wiring tests for gradient-enhanced BO in the solver.

These exercise the Phase-2 integration: recording dE/dtheta per evaluation,
keeping the gradient tensor index-aligned, and routing the acquisition step
through the gradient-enhanced GP. They use the fast RDKit MMFF backend.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")

from bouquet import solver  # noqa: E402
from bouquet.calc_rdkit import RDKitMMFFCalculator  # noqa: E402
from bouquet.setup import detect_dihedrals, get_initial_structure  # noqa: E402


@pytest.fixture
def butane_calc():
    atoms, mol = get_initial_structure("CCCC")
    dihedrals = detect_dihedrals(mol)
    return atoms, mol, dihedrals, RDKitMMFFCalculator(mol)


def test_select_next_points_gradient_path(butane_calc):
    """The gradient branch returns a valid in-range candidate (degrees)."""
    torch.manual_seed(0)
    n, d = 6, 2
    X = torch.rand(n, d, dtype=torch.float64) * 360.0
    y = torch.rand(n, dtype=torch.float64)  # relative energies (eV)
    g = 0.01 * torch.randn(n, d, dtype=torch.float64)

    out = solver._select_next_points_botorch(
        X, y, observed_gradients=g, use_gradients=True
    )
    assert out.shape == (d,)
    assert np.all(out >= 0) and np.all(out <= 360)


def test_gradient_path_handles_nan_rows(butane_calc):
    """NaN gradient rows (failed evals) are dropped, not propagated."""
    torch.manual_seed(1)
    n, d = 5, 2
    X = torch.rand(n, d, dtype=torch.float64) * 360.0
    y = torch.rand(n, dtype=torch.float64)
    g = 0.01 * torch.randn(n, d, dtype=torch.float64)
    g[0] = float("nan")  # e.g. the start point / a failed evaluation

    out = solver._select_next_points_botorch(
        X, y, observed_gradients=g, use_gradients=True
    )
    assert out.shape == (d,)
    assert np.all(np.isfinite(out))


def test_run_optimization_records_aligned_gradients(butane_calc):
    """Full pipeline with use_gradients keeps gradients index-aligned & runs."""
    atoms, _, dihedrals, calc = butane_calc
    state = solver._setup_initial_state(
        atoms, dihedrals, calc, calc, False, None, use_gradients=True
    )
    solver._evaluate_initial_guesses(
        state, dihedrals, calc, calc, False, 4, 0, None, None
    )
    solver._run_optimization_loop(state, 4, dihedrals, calc, calc, False, None)

    n = state.observed_energies.shape[0]
    assert state.observed_gradients.shape == (n, len(dihedrals))
    # The start point now contributes a gradient too (not NaN).
    assert not torch.isnan(state.observed_gradients[0]).any()
    finite = (~torch.isnan(state.observed_gradients).any(dim=1)).sum().item()
    assert finite >= n - 1


def test_gradient_steps_switches_to_value_only(butane_calc, monkeypatch):
    """gradient_steps caps the gradient-enhanced GP to the first N BO steps; the
    rest fall back to the value-only GP."""
    atoms, _, dihedrals, calc = butane_calc
    state = solver._setup_initial_state(
        atoms, dihedrals, calc, calc, False, None, use_gradients=True
    )
    state.gradient_steps = 2
    solver._evaluate_initial_guesses(
        state, dihedrals, calc, calc, False, 3, 0, None, None
    )

    # Spy on the per-step use_gradients flag without paying for the real GP fit.
    seen = []

    def spy(*args, **kwargs):
        seen.append(kwargs.get("use_gradients"))
        return np.zeros(len(dihedrals))  # valid degrees; stays at a fixed point

    monkeypatch.setattr(solver, "_select_next_points_botorch", spy)
    solver._run_optimization_loop(state, 5, dihedrals, calc, calc, False, None)

    assert seen == [True, True, False, False, False]


def test_gradient_steps_zero_keeps_gradients_whole_run(butane_calc, monkeypatch):
    """gradient_steps <= 0 (default) keeps the gradient GP for every BO step."""
    atoms, _, dihedrals, calc = butane_calc
    state = solver._setup_initial_state(
        atoms, dihedrals, calc, calc, False, None, use_gradients=True
    )
    assert state.gradient_steps == 0  # dataclass default
    solver._evaluate_initial_guesses(
        state, dihedrals, calc, calc, False, 3, 0, None, None
    )

    seen = []

    def spy(*args, **kwargs):
        seen.append(kwargs.get("use_gradients"))
        return np.zeros(len(dihedrals))

    monkeypatch.setattr(solver, "_select_next_points_botorch", spy)
    solver._run_optimization_loop(state, 4, dihedrals, calc, calc, False, None)

    assert seen == [True, True, True, True]


def _record_frozen(monkeypatch, dihedrals):
    """Spy on _select_next_points_botorch, recording whether each step is a
    condition-only update (gp_frozen_hypers set) vs. a cold fit (None). Emulates the
    GP returning fitted hypers when the loop asks (cold-fit steps)."""
    seen = []

    def spy(*args, **kwargs):
        seen.append(kwargs.get("gp_frozen_hypers") is not None)
        out = kwargs.get("gp_hyper_out")
        if out is not None:
            out["hypers"] = {"dummy": torch.zeros(())}
        return np.zeros(len(dihedrals))

    monkeypatch.setattr(solver, "_select_next_points_botorch", spy)
    return seen  # True = condition-only (frozen), False = cold fit


def test_grad_refit_freeze_schedule(butane_calc, monkeypatch):
    """Cold full fits during the dense phase, then condition-only updates on the
    frozen hyperparameters. Cold fits never load hypers (warm-starting drifts them)."""
    atoms, _, dihedrals, calc = butane_calc
    state = solver._setup_initial_state(
        atoms, dihedrals, calc, calc, False, None, use_gradients=True
    )
    state.grad_refit_dense_until = 3
    state.grad_refit_every = 0  # freeze after the dense phase
    solver._evaluate_initial_guesses(
        state, dihedrals, calc, calc, False, 3, 0, None, None
    )

    frozen = _record_frozen(monkeypatch, dihedrals)
    solver._run_optimization_loop(state, 6, dihedrals, calc, calc, False, None)

    # steps 0,1,2: cold fits; steps 3-5: condition-only on the frozen hypers.
    assert frozen == [False, False, False, True, True, True]
    assert state.grad_gp_hypers is not None  # frozen after the dense phase


def test_grad_refit_periodic_cold_refresh(butane_calc, monkeypatch):
    """grad_refit_every > 0 cold-refreshes the frozen hypers on a stride, with
    condition-only updates between."""
    atoms, _, dihedrals, calc = butane_calc
    state = solver._setup_initial_state(
        atoms, dihedrals, calc, calc, False, None, use_gradients=True
    )
    state.grad_refit_dense_until = 3
    state.grad_refit_every = 2
    solver._evaluate_initial_guesses(
        state, dihedrals, calc, calc, False, 3, 0, None, None
    )

    frozen = _record_frozen(monkeypatch, dihedrals)
    solver._run_optimization_loop(state, 8, dihedrals, calc, calc, False, None)

    # dense 0,1,2 cold; then cold refit when (step-3)%2==0 -> steps 3,5,7 cold,
    # steps 4,6 condition-only on frozen hypers.
    assert frozen == [False, False, False, False, True, False, True, False]


def test_grad_refit_dense_zero_fits_every_step(butane_calc, monkeypatch):
    """dense_until=0 (the explicit opt-out from the default freeze schedule) cold-fits
    every step -- the slow full-gradient reference, never conditioning on frozen
    hypers."""
    atoms, _, dihedrals, calc = butane_calc
    state = solver._setup_initial_state(
        atoms, dihedrals, calc, calc, False, None, use_gradients=True
    )
    state.grad_refit_dense_until = 0  # opt out of the default (20-step) freeze
    solver._evaluate_initial_guesses(
        state, dihedrals, calc, calc, False, 3, 0, None, None
    )

    frozen = _record_frozen(monkeypatch, dihedrals)
    solver._run_optimization_loop(state, 4, dihedrals, calc, calc, False, None)

    assert frozen == [False, False, False, False]  # all cold fits


def test_value_only_path_leaves_gradients_nan(butane_calc):
    """With use_gradients off, no gradients are computed (all NaN)."""
    atoms, _, dihedrals, calc = butane_calc
    state = solver._setup_initial_state(atoms, dihedrals, calc, calc, False, None)
    state.use_gradients = False
    solver._evaluate_initial_guesses(
        state, dihedrals, calc, calc, False, 3, 0, None, None
    )
    assert torch.isnan(state.observed_gradients).all()
