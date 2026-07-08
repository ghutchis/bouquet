"""Tests for probabilistic Boltzmann ensemble selection.

The pure-function tests run without a calculator; the integration test is marked
``slow`` and exercises a tiny end-to-end optimization with ``return_ensemble``.
"""

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import molecule

from bouquet.config import KCAL_TO_EV
from bouquet.io import save_ensemble
from bouquet.solver import (
    _HAVE_IRMSD,
    OptimizationState,
    _boltzmann_weights,
    _dedup,
    _rmsd,
    _select_ensemble_candidates,
)


def _h2(length: float) -> Atoms:
    """A simple two-atom molecule with a tunable bond length.

    Used only as a lightweight placeholder where the RMSD backend is never
    invoked (selection and file-writing tests).
    """
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, length]])


def _mol() -> Atoms:
    """A real multi-atom molecule for RMSD tests.

    The iRMSD backend canonicalizes/aligns whole molecules and is undefined for
    trivial 2-atom inputs, so RMSD/dedup tests use ethanol (9 atoms), which
    works under both the iRMSD and Kabsch-fallback code paths.
    """
    return molecule("CH3CH2OH")


class TestBoltzmannWeights:
    def test_weights_normalize(self):
        w = _boltzmann_weights(np.array([0.0, 0.02, 0.05]), 298.15)
        assert np.isclose(w.sum(), 1.0)

    def test_temperature_flattens_distribution(self):
        e = np.array([0.0, 0.02, 0.05])  # eV
        w_low = _boltzmann_weights(e, 100.0)
        w_high = _boltzmann_weights(e, 1000.0)
        # Higher temperature => more uniform => smaller max population.
        assert w_high.max() < w_low.max()
        # Lowest-energy member always has the largest population.
        assert w_low.argmax() == 0 and w_high.argmax() == 0


class TestRmsdAndDedup:
    def test_rmsd_invariant_to_rotation_and_translation(self):
        a = _mol()
        b = a.copy()
        b.rotate(57.0, "y")
        b.translate([5.0, -2.0, 1.0])
        # Same conformer up to rigid-body motion: RMSD ~ 0 under both backends.
        assert _rmsd(a, b) < 1e-3

    def test_rmsd_distinct_geometry_is_large(self):
        a = _mol()
        c = a.copy()
        c.rattle(0.5, seed=42)  # large random displacement
        assert _rmsd(a, c) > 0.125

    def test_dedup_merges_near_identical(self):
        e_tol = 0.1 * KCAL_TO_EV
        a = _mol()
        b = a.copy()  # geometrically identical, energy-close
        b.rotate(30.0, "z")
        b.translate([1.0, 0.0, -2.0])
        unique = _dedup([(a, 0.0), (b, 0.0005)], rmsd_thr=0.125, e_tol_eV=e_tol)
        assert len(unique) == 1

    def test_dedup_keeps_distinct_geometry(self):
        e_tol = 0.1 * KCAL_TO_EV
        a = _mol()
        c = a.copy()
        c.rattle(0.5, seed=7)  # distinct geometry, but energy-close
        unique = _dedup([(a, 0.0), (c, 0.0005)], rmsd_thr=0.125, e_tol_eV=e_tol)
        assert len(unique) == 2

    def test_dedup_keeps_energy_separated(self):
        e_tol = 0.1 * KCAL_TO_EV
        a = _mol()
        b = a.copy()  # identical geometry, but far in energy
        unique = _dedup([(a, 0.0), (b, 5.0 * KCAL_TO_EV)], rmsd_thr=0.125, e_tol_eV=e_tol)
        assert len(unique) == 2

    @pytest.mark.skipif(
        not _HAVE_IRMSD, reason="permutation invariance requires the iRMSD backend"
    )
    def test_rmsd_is_permutation_invariant(self):
        # Swapping two symmetry-equivalent hydrogens leaves the molecule
        # physically unchanged. iRMSD canonicalizes atom ordering and reports
        # ~0; the Kabsch fallback (ordering-dependent) would report ~1.1 A.
        m = molecule("CH4")
        swapped = m.copy()
        pos = swapped.get_positions()
        pos[[1, 2]] = pos[[2, 1]]
        swapped.set_positions(pos)

        assert _rmsd(m, swapped) < 1e-3


def _make_state(energies, coords, atoms, start_energy=-100.0):
    return OptimizationState(
        start_atoms=atoms[0],
        start_coords=np.zeros(coords.shape[1]),
        start_energy=start_energy,
        initial_coords=coords,
        initial_energies=energies,
        observed_atoms=atoms,
    )


class TestSelectEnsembleCandidates:
    def _coords(self, n, dim=2, seed=0):
        rng = np.random.RandomState(seed)
        return torch.tensor(rng.uniform(0, 360, size=(n, dim)), dtype=torch.float64)

    def test_failure_sentinel_excluded(self):
        # One observation is a failed-eval sentinel (~1000 eV relative).
        energies = torch.tensor(
            [0.0, 0.01, 0.02, 1000.0, 0.03, 0.05], dtype=torch.float64
        )
        coords = self._coords(6)
        atoms = [_h2(1.0 + 0.01 * i) for i in range(6)]
        state = _make_state(energies, coords, atoms)

        cands = _select_ensemble_candidates(
            state,
            window_eV=6.0 * KCAL_TO_EV,
            p_threshold=0.01,
            sigma_floor_eV=0.1 * KCAL_TO_EV,
            failure_energy_eV=100.0,
        )
        # Five valid observations survive failure filtering.
        assert len(cands) == 5
        # The sentinel's coordinates never appear among the candidates.
        assert all(not np.allclose(c[0], coords[3].numpy()) for c in cands)

    def test_alignment_assertion(self):
        energies = torch.tensor([0.0, 0.01, 0.02], dtype=torch.float64)
        coords = self._coords(3)
        atoms = [_h2(1.0), _h2(1.01)]  # deliberately one short
        state = _make_state(energies, coords, atoms)
        with pytest.raises(AssertionError):
            _select_ensemble_candidates(
                state,
                window_eV=6.0 * KCAL_TO_EV,
                p_threshold=0.01,
                sigma_floor_eV=0.1 * KCAL_TO_EV,
                failure_energy_eV=100.0,
            )

    def test_few_points_fallback(self):
        # < 3 valid points: flat-window fallback, no GP fit, must not raise.
        energies = torch.tensor([0.0, 0.02], dtype=torch.float64)
        coords = self._coords(2)
        atoms = [_h2(1.0), _h2(1.01)]
        state = _make_state(energies, coords, atoms)
        cands = _select_ensemble_candidates(
            state,
            window_eV=6.0 * KCAL_TO_EV,
            p_threshold=0.01,
            sigma_floor_eV=0.1 * KCAL_TO_EV,
            failure_energy_eV=100.0,
        )
        assert len(cands) == 2

    def test_no_valid_observations(self):
        energies = torch.tensor([1000.0, 1001.0], dtype=torch.float64)
        coords = self._coords(2)
        atoms = [_h2(1.0), _h2(1.01)]
        state = _make_state(energies, coords, atoms)
        cands = _select_ensemble_candidates(
            state,
            window_eV=6.0 * KCAL_TO_EV,
            p_threshold=0.01,
            sigma_floor_eV=0.1 * KCAL_TO_EV,
            failure_energy_eV=100.0,
        )
        assert cands == []


class TestSaveEnsemble:
    def test_writes_xyz_and_csv(self, temp_dir):
        ensemble = [
            (_h2(1.0), 0.0, 0.6),
            (_h2(1.05), 0.01, 0.4),  # 0.01 eV above the minimum
        ]
        save_ensemble(temp_dir, ensemble)

        xyz = temp_dir / "ensemble_final.xyz"
        csv = temp_dir / "ensemble.csv"
        assert xyz.exists() and csv.exists()

        # Two frames in the XYZ.
        assert xyz.read_text().count("pop=") == 2

        lines = csv.read_text().strip().splitlines()
        assert lines[0] == "index,rel_energy_kcal,weight"
        assert len(lines) == 3  # header + 2 rows
        # Relative energy reported in kcal/mol from the minimum member.
        rel_kcal = float(lines[2].split(",")[1])
        assert np.isclose(rel_kcal, 0.01 / KCAL_TO_EV, atol=1e-4)

    def test_empty_ensemble_writes_nothing(self, temp_dir):
        save_ensemble(temp_dir, [])
        assert not (temp_dir / "ensemble_final.xyz").exists()
        assert not (temp_dir / "ensemble.csv").exists()


@pytest.mark.slow
class TestEnsembleIntegration:
    """End-to-end check that observed_atoms stays aligned and an ensemble is built."""

    @pytest.fixture
    def gfnff_calc(self):
        pytest.importorskip("xtb")
        from bouquet.calculator import CalculatorFactory

        return CalculatorFactory.create("gfnff")

    def test_butane_ensemble_alignment_and_populations(self, gfnff_calc, temp_dir, monkeypatch):
        monkeypatch.chdir(temp_dir)
        from bouquet.setup import detect_dihedrals, get_initial_structure
        from bouquet.solver import run_optimization

        atoms, mol = get_initial_structure("CCCC")
        dihedrals = detect_dihedrals(mol)

        final_atoms, ensemble = run_optimization(
            atoms,
            dihedrals,
            n_steps=4,
            calc=gfnff_calc,
            relaxCalc=gfnff_calc,
            init_steps=3,
            out_dir=None,
            relax=True,
            seed=0,
            return_ensemble=True,
        )

        # Ensemble is non-empty, sorted by energy, populations normalized.
        assert len(ensemble) >= 1
        energies = [e for _, e, _ in ensemble]
        assert energies == sorted(energies)
        weights = np.array([w for _, _, w in ensemble])
        assert np.isclose(weights.sum(), 1.0)
        # The returned best structure is the lowest-energy ensemble member.
        assert final_atoms is ensemble[0][0]
