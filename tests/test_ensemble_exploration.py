"""Tests for active level-set ensemble exploration.

The pure-function tests run without a calculator; the integration test is marked
``slow`` and exercises a tiny end-to-end run with active exploration enabled.
"""

import numpy as np
import pytest
import torch
from ase import Atoms

from bouquet.config import KCAL_TO_EV, RunOptions
from bouquet.ensemble import (
    _LevelSetAcquisition,
    _periodic_min_dist,
    _resolve_ensemble_budget,
)
from bouquet.solver import (
    OptimizationState,
    _build_selection_gp,
    _fit_selection_gp_valid,
    _initial_basins,
    _max_posterior_sigma,
)


def _h2(length: float) -> Atoms:
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, length]])


def _make_state(energies, coords, atoms, start_energy=-100.0):
    return OptimizationState(
        start_atoms=atoms[0],
        start_coords=np.zeros(coords.shape[1]),
        start_energy=start_energy,
        initial_coords=coords,
        initial_energies=energies,
        observed_atoms=atoms,
    )


def _toy_selection_gp(seed=0, n=15):
    """A fitted 2-D selection GP with a clear low-energy basin near (60, 60)."""
    torch.manual_seed(seed)
    X = torch.rand(n, 2, dtype=torch.float64) * 360.0
    E = 0.5 + 0.4 * torch.rand(n, dtype=torch.float64)
    E[0] = 0.0
    X[0] = torch.tensor([60.0, 60.0], dtype=torch.float64)
    gp = _build_selection_gp(X, E)
    gp.eval()
    return gp, float(E.min())


class TestPeriodicMinDist:
    def test_wraps_across_360(self):
        # 350 deg and 10 deg are 20 deg apart, not 340.
        d = _periodic_min_dist(torch.tensor([350.0]), torch.tensor([[10.0]]))
        assert float(d) == pytest.approx(20.0)

    def test_wraps_negative(self):
        # 350 deg and -20 deg are 10 deg apart.
        d = _periodic_min_dist(torch.tensor([350.0]), torch.tensor([[-20.0]]))
        assert float(d) == pytest.approx(10.0)

    def test_wraps_large_difference(self):
        # 700 deg (which is 340 deg) and -10 deg (which is 350 deg) are 10 deg apart.
        d = _periodic_min_dist(torch.tensor([700.0]), torch.tensor([[-10.0]]))
        assert float(d) == pytest.approx(10.0)

    def test_min_over_basins(self):
        # Nearest of two basins wins.
        x = torch.tensor([[0.0, 0.0]])
        m = torch.tensor([[180.0, 180.0], [5.0, 5.0]])
        assert float(_periodic_min_dist(x, m)) == pytest.approx(5.0)

    def test_rms_over_dihedrals(self):
        # sqrt(mean(30^2, 40^2)) = sqrt((900+1600)/2) = sqrt(1250).
        x = torch.tensor([[0.0, 0.0]])
        m = torch.tensor([[30.0, 40.0]])
        assert float(_periodic_min_dist(x, m)) == pytest.approx(np.sqrt(1250.0))

    def test_leading_batch_dims_preserved(self):
        x = torch.zeros(4, 1, 2)  # (b, q, d)
        m = torch.tensor([[10.0, 10.0]])
        assert _periodic_min_dist(x, m).shape == (4, 1)


class TestResolveEnsembleBudget:
    def test_zero_disables(self):
        assert _resolve_ensemble_budget(0, 8) == 0

    def test_positive_is_verbatim(self):
        assert _resolve_ensemble_budget(30, 8) == 30

    def test_auto_scales_and_is_bounded(self):
        assert _resolve_ensemble_budget(-1, 1) == 25  # floor
        assert _resolve_ensemble_budget(-1, 4) == 60  # 15 * 4
        assert _resolve_ensemble_budget(-1, 100) == 250  # cap


class TestLevelSetAcquisition:
    def test_shapes_and_nonnegative(self):
        gp, e_min = _toy_selection_gp()
        acqf = _LevelSetAcquisition(gp, e_min + 10.0 * KCAL_TO_EV)
        X = torch.rand(5, 1, 2, dtype=torch.float64)  # (b, q=1, d)
        with torch.no_grad():
            vals = acqf(X)
        assert vals.shape == (5,)
        assert torch.all(vals >= 0)

    def test_diversity_suppresses_at_seed_only(self):
        # Multiplicative local penalization: acq at a known basin is scaled by
        # (1 - lambda); regions far from every basin are left untouched.
        gp, e_min = _toy_selection_gp()
        seed = torch.tensor([[60.0, 60.0]], dtype=torch.float64)
        near = torch.tensor([[60.0, 60.0]], dtype=torch.float64) / 360.0
        far = torch.tensor([[250.0, 250.0]], dtype=torch.float64) / 360.0
        Xq = torch.stack([near, far])  # (2, 1, 2)

        base = _LevelSetAcquisition(gp, e_min + 10.0 * KCAL_TO_EV)
        half = _LevelSetAcquisition(
            gp, e_min + 10.0 * KCAL_TO_EV, minima_deg=seed, diversity_lambda=0.5
        )
        full = _LevelSetAcquisition(
            gp, e_min + 10.0 * KCAL_TO_EV, minima_deg=seed, diversity_lambda=1.0
        )
        with torch.no_grad():
            b, h, f = base(Xq), half(Xq), full(Xq)

        # At the seed: halved, then fully suppressed.
        assert float(h[0]) == pytest.approx(0.5 * float(b[0]), rel=1e-4)
        assert float(f[0]) == pytest.approx(0.0, abs=1e-9)
        # Far from the seed: essentially untouched.
        assert float(h[1]) == pytest.approx(float(b[1]), rel=1e-3)
        assert float(f[1]) == pytest.approx(float(b[1]), rel=1e-3)

    def test_optimize_acqf_runs(self):
        from botorch.optim import optimize_acqf

        gp, e_min = _toy_selection_gp()
        acqf = _LevelSetAcquisition(gp, e_min + 10.0 * KCAL_TO_EV)
        bounds = torch.zeros(2, 2, dtype=torch.float64)
        bounds[1, :] = 1.0
        cand, _ = optimize_acqf(
            acqf, bounds=bounds, q=1, num_restarts=4, raw_samples=32
        )
        assert cand.shape == (1, 2)
        assert torch.all((cand >= 0) & (cand <= 1))


class TestFitSelectionGpValid:
    def test_none_with_too_few_valid(self):
        energies = torch.tensor([0.0, 1000.0], dtype=torch.float64)
        coords = torch.zeros(2, 2, dtype=torch.float64)
        state = _make_state(energies, coords, [_h2(1.0), _h2(1.01)])
        gp, e_min = _fit_selection_gp_valid(state)
        assert gp is None and e_min is None

    def test_fits_and_reports_min_dropping_sentinels(self):
        energies = torch.tensor(
            [0.2, 0.0, 1000.0, 0.1, 0.3], dtype=torch.float64
        )
        rng = np.random.RandomState(1)
        coords = torch.tensor(rng.uniform(0, 360, (5, 2)), dtype=torch.float64)
        state = _make_state(energies, coords, [_h2(1.0 + 0.01 * i) for i in range(5)])
        gp, e_min = _fit_selection_gp_valid(state)
        assert gp is not None
        # e_min ignores the 1000 eV failure sentinel.
        assert e_min == pytest.approx(0.0)

    def test_max_posterior_sigma_positive(self):
        gp, _ = _toy_selection_gp()
        s = _max_posterior_sigma(gp, 2, torch.float64, torch.device("cpu"))
        assert s > 0


class TestInitialBasins:
    def _state(self, energies, coords):
        e = torch.tensor(energies, dtype=torch.float64)
        c = torch.tensor(coords, dtype=torch.float64)
        atoms = [_h2(1.0 + 0.01 * i) for i in range(len(energies))]
        return _make_state(e, c, atoms)

    def test_dedup_and_window(self):
        delta = 10.0 * KCAL_TO_EV
        # Two near-identical low points (basin A), one distinct (basin B), one
        # far outside the window that must be excluded even though geometrically
        # distinct.
        state = self._state(
            energies=[0.0, 0.01 * KCAL_TO_EV, 2.0 * KCAL_TO_EV, 50.0 * KCAL_TO_EV],
            coords=[[0.0, 0.0], [10.0, 10.0], [200.0, 200.0], [100.0, 100.0]],
        )
        basins = _initial_basins(state, delta_eV=delta, basin_deg=40.0)
        # Basin A collapses to one seed; basin B adds a second; the out-of-window
        # point is dropped.
        assert basins.shape == (2, 2)

    def test_empty_when_all_failed(self):
        state = self._state(energies=[1000.0, 1001.0], coords=[[0.0, 0.0], [1.0, 1.0]])
        basins = _initial_basins(state, delta_eV=10.0 * KCAL_TO_EV, basin_deg=40.0)
        assert basins.shape == (0, 2)


class TestRunOptionsValidation:
    def test_rejects_below_auto_sentinel(self):
        with pytest.raises(ValueError):
            RunOptions(ensemble_steps=-2)

    def test_rejects_negative_diversity(self):
        with pytest.raises(ValueError):
            RunOptions(ensemble_diversity=-0.1)

    def test_accepts_sentinels(self):
        assert RunOptions(ensemble_steps=-1).ensemble_steps == -1
        assert RunOptions(ensemble_steps=0).ensemble_steps == 0


@pytest.mark.slow
class TestExplorationIntegration:
    """End-to-end: active exploration appends points and yields a valid ensemble."""

    @pytest.fixture
    def gfnff_calc(self):
        pytest.importorskip("xtb")
        from bouquet.calculator import CalculatorFactory

        return CalculatorFactory.create("gfnff")

    def test_active_exploration_enriches_ensemble(self, gfnff_calc, temp_dir, monkeypatch):
        monkeypatch.chdir(temp_dir)
        from bouquet.setup import detect_dihedrals, get_initial_structure
        from bouquet.solver import run_optimization

        atoms, mol = get_initial_structure("CCCC")
        dihedrals = detect_dihedrals(mol)

        n_steps, init_steps, explore = 4, 3, 6
        final_atoms, ensemble = run_optimization(
            atoms,
            dihedrals,
            n_steps=n_steps,
            calc=gfnff_calc,
            relaxCalc=gfnff_calc,
            init_steps=init_steps,
            out_dir=None,
            relax=True,
            seed=0,
            return_ensemble=True,
            opts=RunOptions(ensemble_steps=explore),
        )

        # A valid, sorted, normalized ensemble comes back.
        assert len(ensemble) >= 1
        energies = [e for _, e, _ in ensemble]
        assert energies == sorted(energies)
        assert np.isclose(sum(w for _, _, w in ensemble), 1.0)
        assert final_atoms is ensemble[0][0]
        # Every reported member is within the 6 kcal/mol reporting window.
        e_min = energies[0]
        assert all((e - e_min) <= 6.0 * KCAL_TO_EV + 1e-9 for e in energies)
