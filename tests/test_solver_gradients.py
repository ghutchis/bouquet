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


def test_value_only_path_leaves_gradients_nan(butane_calc):
    """With use_gradients off, no gradients are computed (all NaN)."""
    atoms, _, dihedrals, calc = butane_calc
    state = solver._setup_initial_state(atoms, dihedrals, calc, calc, False, None)
    state.use_gradients = False
    solver._evaluate_initial_guesses(
        state, dihedrals, calc, calc, False, 3, 0, None, None
    )
    assert torch.isnan(state.observed_gradients).all()
