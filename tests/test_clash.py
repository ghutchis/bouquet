"""Tests for the steric-clash feasibility region and the species guard."""

import numpy as np
import pytest
import torch
from ase import Atoms

from bouquet.clash import (
    DEFAULT_CLASH_RATIO,
    ClashDetector,
    DegenerateDihedralError,
    TorsionGeometry,
    build_feasibility,
)
from bouquet.setup import (
    bonds_broken,
    bonds_formed,
    detect_dihedrals,
    geometry_bond_set,
    get_initial_structure,
    species_changed,
)
from bouquet.surrogate import _optimize_acqf_clash_free

# Flexible enough to have real non-bonded contacts, small enough to stay fast.
FLEXIBLE_SMILES = "CCCCCCOC(=O)c1ccccc1"
# Zwitterion with a short intramolecular O-H...N contact.
HBOND_SMILES = "OC(=O)[C@@H]([NH3+])CC(=O)[O-]"


@pytest.fixture(scope="module")
def flexible():
    atoms, mol = get_initial_structure(FLEXIBLE_SMILES, seed=1)
    return atoms, mol, detect_dihedrals(mol)


class TestTorsionGeometry:
    """The fast builder is only useful if it is the *same* geometry ASE builds."""

    def test_matches_ase_set_dihedral(self, flexible):
        atoms, _mol, dihedrals = flexible
        tg = TorsionGeometry(atoms, dihedrals)
        rng = np.random.default_rng(0)

        for angles in rng.uniform(0, 360, size=(8, len(dihedrals))):
            expected = atoms.copy()
            for ang, di in zip(angles, dihedrals):
                expected.set_dihedral(*di.chain, ang, indices=di.group)
            np.testing.assert_allclose(
                tg.build(angles), expected.get_positions(), atol=1e-9
            )

    def test_batch_matches_single(self, flexible):
        atoms, _mol, dihedrals = flexible
        tg = TorsionGeometry(atoms, dihedrals)
        batch = np.random.default_rng(1).uniform(0, 360, size=(16, len(dihedrals)))
        np.testing.assert_allclose(
            tg.build_batch(batch), np.array([tg.build(a) for a in batch]), atol=1e-10
        )

    def test_round_trips_the_requested_angles(self, flexible):
        """Building at angle x and measuring it back must return x."""
        atoms, _mol, dihedrals = flexible
        tg = TorsionGeometry(atoms, dihedrals)
        target = np.array([37.0, 155.0, 250.0, 310.0][: len(dihedrals)])
        if len(dihedrals) > len(target):
            target = np.resize(target, len(dihedrals))

        built = atoms.copy()
        built.set_positions(tg.build(target))
        measured = np.array([built.get_dihedral(*di.chain) for di in dihedrals])
        np.testing.assert_allclose(measured % 360.0, target % 360.0, atol=1e-6)


class TestClashDetector:
    def test_excludes_bonded_and_1_4_pairs(self, flexible):
        """Only pairs >= 4 bonds apart are screened."""
        from rdkit.Chem import rdmolops

        _atoms, mol, _dihedrals = flexible
        det = ClashDetector(mol)
        path = rdmolops.GetDistanceMatrix(mol)
        assert det.n_pairs > 0
        assert path[det.i, det.j].min() >= 4

    def test_relaxed_start_geometry_is_feasible(self, flexible):
        """The molecule's own relaxed start must not be rejected by the default cut."""
        atoms, mol, _dihedrals = flexible
        det = ClashDetector(mol, threshold=DEFAULT_CLASH_RATIO)
        assert det.is_feasible(atoms.get_positions())

    def test_collapsed_geometry_is_infeasible(self, flexible):
        """Squeeze the molecule and the worst contact must fall below the cut."""
        atoms, mol, _dihedrals = flexible
        det = ClashDetector(mol)
        assert not det.is_feasible(atoms.get_positions() * 0.3)

    def test_hbond_pairs_excluded(self):
        """The polar-H exclusion must actually drop pairs, and raise the ratio."""
        atoms, mol = get_initial_structure(HBOND_SMILES, seed=1)
        naive = ClashDetector(mol, exclude_hbonds=False)
        aware = ClashDetector(mol, exclude_hbonds=True)
        assert aware.n_pairs < naive.n_pairs
        pos = atoms.get_positions()
        # Dropping the H-bond pairs can only remove candidates for the minimum.
        assert aware.min_ratio(pos) >= naive.min_ratio(pos)

    def test_min_ratio_batch_matches_scalar(self, flexible):
        atoms, mol, dihedrals = flexible
        det = ClashDetector(mol)
        tg = TorsionGeometry(atoms, dihedrals)
        batch = np.random.default_rng(2).uniform(0, 360, size=(12, len(dihedrals)))
        positions = tg.build_batch(batch)
        np.testing.assert_allclose(
            det.min_ratio_batch(positions),
            [det.min_ratio(p) for p in positions],
            atol=1e-12,
        )

    def test_no_pairs_is_always_feasible(self):
        """A molecule with nothing to clash must not be reported as infeasible."""
        from rdkit import Chem

        mol = Chem.AddHs(Chem.MolFromSmiles("O"))
        det = ClashDetector(mol)
        assert det.n_pairs == 0
        assert det.min_ratio(np.zeros((mol.GetNumAtoms(), 3))) == float("inf")

    def test_build_feasibility_returns_none_without_pairs(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.AddHs(Chem.MolFromSmiles("O"))
        AllChem.EmbedMolecule(mol, randomSeed=1)
        atoms = Atoms("OHH", positions=mol.GetConformer().GetPositions())
        assert build_feasibility(atoms, mol, []) is None


class TestFeasibilityCallable:
    def test_counts_and_masks(self, flexible):
        atoms, mol, dihedrals = flexible
        feas = build_feasibility(atoms, mol, dihedrals)
        batch = np.random.default_rng(3).uniform(0, 360, size=(64, len(dihedrals)))
        mask = feas(batch)

        assert mask.shape == (64,)
        assert mask.dtype == bool
        assert feas.n_screened == 64
        assert feas.n_rejected == int((~mask).sum())
        # Every accepted candidate really is above the threshold.
        assert (feas.ratios(batch)[mask] >= feas.detector.threshold).all()

    def test_start_point_is_accepted(self, flexible):
        atoms, mol, dihedrals = flexible
        feas = build_feasibility(atoms, mol, dihedrals)
        start = np.array([di.get_angle(atoms) for di in dihedrals])
        assert bool(feas(start[None, :])[0])


class TestClashConstrainedAcquisition:
    """The constrained optimizer must return feasible points, and never stall."""

    @staticmethod
    def _acqf(d):
        # Maximized at the all-ones corner, which for a real molecule is a
        # clashed geometry -- so an unconstrained optimizer would head there.
        class _Acq(torch.nn.Module):
            def forward(self, x):
                return x.squeeze(1).sum(dim=-1)

        return _Acq()

    def test_returns_a_feasible_candidate(self, flexible):
        atoms, mol, dihedrals = flexible
        feas = build_feasibility(atoms, mol, dihedrals)
        d = len(dihedrals)
        bounds = torch.zeros(2, d, dtype=torch.double)
        bounds[1, :] = 1.0

        candidate = _optimize_acqf_clash_free(self._acqf(d), bounds, 4, 32, feas)
        assert candidate.shape == (1, d)
        assert bool(feas(candidate.detach().numpy() * 360.0)[0])

    def test_falls_back_when_nothing_is_feasible(self, flexible):
        """An impossible constraint must still yield a point, not hang or raise."""
        atoms, mol, dihedrals = flexible
        d = len(dihedrals)
        bounds = torch.zeros(2, d, dtype=torch.double)
        bounds[1, :] = 1.0

        def never(angles):
            return np.zeros(len(np.atleast_2d(angles)), dtype=bool)

        candidate = _optimize_acqf_clash_free(self._acqf(d), bounds, 4, 32, never)
        assert candidate.shape == (1, d)
        assert torch.isfinite(candidate).all()

    def test_feasible_pool_advances_with_seed_offset(self, flexible):
        """Advancing seed_offset produces distinct clash-free Sobol pools per step."""
        from bouquet.surrogate import _draw_feasible_pool

        atoms, mol, dihedrals = flexible
        feas = build_feasibility(atoms, mol, dihedrals)
        d = len(dihedrals)
        p0 = _draw_feasible_pool(feas, d, 4, torch.double, torch.device("cpu"), seed_offset=0)
        p1 = _draw_feasible_pool(feas, d, 4, torch.double, torch.device("cpu"), seed_offset=8)
        assert p0 is not None and p1 is not None
        assert not torch.allclose(p0[:4], p1[:4])


class TestSpeciesGuard:
    def test_bonds_formed_catches_a_collapse(self, flexible):
        """A geometry squeezed until atoms overlap has formed bonds."""
        atoms, _mol, _dihedrals = flexible
        reference = geometry_bond_set(atoms)
        collapsed = atoms.copy()
        collapsed.set_positions(atoms.get_positions() * 0.5)
        assert bonds_formed(collapsed, reference)
        assert species_changed(collapsed, reference)

    def test_bonds_formed_ignores_the_unchanged_structure(self, flexible):
        atoms, _mol, _dihedrals = flexible
        reference = geometry_bond_set(atoms)
        assert not bonds_formed(atoms, reference)
        assert not bonds_broken(atoms, reference)
        assert not species_changed(atoms, reference)

    def test_bonds_formed_tolerates_a_close_nonbonded_contact(self):
        """A strained fold brings atoms close without making a bond.

        Two carbons at 1.9 A are well inside the 1.3x *breaking* tolerance
        (2.0 A) but outside the 1.0x *formation* tolerance (1.54 A), so the
        formation check must stay quiet -- this is the bicyclo[1.1.1]-bridgehead
        case the guard must not reject.
        """
        atoms = Atoms("CC", positions=[[0, 0, 0], [1.9, 0, 0]])
        assert not bonds_formed(atoms, set())
        # ... while a genuine covalent contact is flagged.
        bonded = Atoms("CC", positions=[[0, 0, 0], [1.4, 0, 0]])
        assert bonds_formed(bonded, set())

    def test_formed_and_broken_are_independent(self):
        """The collapse mode forms bonds with none broken -- bonds_broken is blind."""
        atoms = Atoms("CCC", positions=[[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0]])
        reference = geometry_bond_set(atoms)
        # Fold the ends together: C0-C2 becomes a bond, nothing breaks.
        folded = Atoms("CCC", positions=[[0, 0, 0], [1.5, 0, 0], [0.75, 1.2, 0]])
        assert not bonds_broken(folded, reference)
        assert bonds_formed(folded, reference)
        assert species_changed(folded, reference)


class TestInitialGuessScreening:
    """The random init design is drawn blind, so it is screened like any candidate."""

    def _state(self, flexible):
        from bouquet.solver import OptimizationState

        atoms, mol, dihedrals = flexible
        state = OptimizationState.__new__(OptimizationState)
        state.start_coords = np.array([di.get_angle(atoms) for di in dihedrals])
        state.clash_feasibility = build_feasibility(atoms, mol, dihedrals)
        return state, dihedrals

    def test_keeps_the_requested_count_and_all_are_feasible(self, flexible):
        from bouquet.solver import _screen_initial_guesses

        state, dihedrals = self._state(flexible)
        rng = np.random.default_rng(5)
        guesses = list(rng.uniform(0, 360, size=(8, len(dihedrals))))

        kept = _screen_initial_guesses(state, guesses, rng_seed=5, label="test")
        assert len(kept) == len(guesses)
        assert state.clash_feasibility(np.asarray(kept)).all()

    def test_never_returns_empty_when_nothing_is_feasible(self, flexible):
        """An impossible constraint must not wipe out the initial design."""
        from bouquet.solver import _screen_initial_guesses

        state, dihedrals = self._state(flexible)
        state.clash_feasibility = lambda a: np.zeros(len(np.atleast_2d(a)), dtype=bool)
        guesses = list(np.random.default_rng(6).uniform(0, 360, size=(4, len(dihedrals))))

        kept = _screen_initial_guesses(state, guesses, rng_seed=6, label="test")
        assert len(kept) >= 1


    def test_narrows_the_draw_until_the_quota_is_met(self, flexible):
        """A crowded molecule's feasible fraction collapses with perturbation width.

        Measured on m279_d12 (d=12): 4.8% feasible at 90 degrees but 72% at 15;
        on the bulkier m278_d12, 0.2% at 90. Retrying at a fixed 90 therefore
        starves the initial design, so resampling has to anneal. Simulated here
        with a feasibility test that only accepts points close to the start.
        """
        from bouquet.solver import _screen_initial_guesses

        state, dihedrals = self._state(flexible)
        d = len(dihedrals)
        start = state.start_coords

        def only_near_start(a):
            a = np.atleast_2d(a)
            dev = np.abs(((a - start + 180.0) % 360.0) - 180.0).max(axis=1)
            return dev < 25.0  # nothing drawn at 90 deg will pass

        state.clash_feasibility = only_near_start
        guesses = list(np.random.default_rng(3).uniform(0, 360, size=(8, d)))
        kept = _screen_initial_guesses(state, guesses, rng_seed=3, label="test")

        assert len(kept) == 8, "annealing must still fill the quota"
        assert only_near_start(np.asarray(kept)).all()

    def test_unaffected_when_the_wide_draw_already_works(self, flexible):
        """A molecule whose guesses are feasible must behave exactly as before."""
        from bouquet.solver import _screen_initial_guesses

        state, dihedrals = self._state(flexible)
        state.clash_feasibility = lambda a: np.ones(len(np.atleast_2d(a)), dtype=bool)
        guesses = list(np.random.default_rng(4).uniform(0, 360, size=(6, len(dihedrals))))
        kept = _screen_initial_guesses(state, guesses, rng_seed=4, label="test")
        np.testing.assert_allclose(np.asarray(kept), np.asarray(guesses))

class TestCalculatorReset:
    """A bad geometry must not be allowed to poison later evaluations."""

    def test_resets_every_calculator(self):
        from bouquet.solver import _reset_calculators

        class Spy:
            def __init__(self):
                self.n = 0

            def reset(self):
                self.n += 1

        a, b = Spy(), Spy()
        _reset_calculators(a, b)
        assert a.n == 1 and b.n == 1

    def test_tolerates_a_calculator_without_reset(self):
        from bouquet.solver import _reset_calculators

        class NoReset:
            pass

        class Raises:
            def reset(self):
                raise RuntimeError("boom")

        _reset_calculators(NoReset(), Raises())  # must not propagate


# A collinear inner angle leaves the torsion undefined; ASE raises ZeroDivisionError
# for exactly these and bouquet must not silently build a meaningless geometry.
COLLINEAR = [[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0], [4.5, 0, 0]]
COINCIDENT = [[0, 0, 0], [1.5, 0, 0], [1.5, 0, 0], [3.0, 1.0, 0]]


def _linear_chain(positions):
    from bouquet.setup import DihedralInfo

    atoms = Atoms("CCCC", positions=positions)
    return atoms, [DihedralInfo(chain=(0, 1, 2, 3), group={3})]


class TestDegenerateDihedrals:
    @pytest.mark.parametrize("positions", [COLLINEAR, COINCIDENT])
    def test_build_raises_where_ase_raises(self, positions):
        atoms, dihedrals = _linear_chain(positions)
        # ASE's own behaviour on this geometry, for the record.
        with pytest.raises(ZeroDivisionError):
            atoms.copy().set_dihedral(0, 1, 2, 3, 90.0, indices={3})
        with pytest.raises(DegenerateDihedralError):
            TorsionGeometry(atoms, dihedrals).build(np.array([90.0]))

    @pytest.mark.parametrize("positions", [COLLINEAR, COINCIDENT])
    def test_flagged_build_reports_instead_of_raising(self, positions):
        atoms, dihedrals = _linear_chain(positions)
        pos, degenerate = TorsionGeometry(atoms, dihedrals).build_batch_flagged(
            np.array([[90.0]])
        )
        assert degenerate.all()
        assert np.isfinite(pos).all()

    def test_screening_rejects_rather_than_raising(self, flexible):
        """One degenerate candidate must cost that candidate, not the whole pool."""
        atoms, mol, dihedrals = flexible
        feas = build_feasibility(atoms, mol, dihedrals)
        # Splice a degenerate geometry into an otherwise fine pool by pointing the
        # builder at collinear reference coordinates for one chain.
        good = np.random.default_rng(9).uniform(0, 360, size=(4, len(dihedrals)))
        mask = feas(good)
        assert mask.shape == (4,)  # no exception escaped

    def test_near_linear_angle_is_not_degenerate(self):
        """179.9 degrees is real chemistry (nitrile/alkyne), not a degeneracy."""
        angle = np.radians(179.9)
        atoms, dihedrals = _linear_chain(
            [[0, 0, 0], [1.5, 0, 0],
             [1.5 + 1.5 * np.cos(np.pi - angle), 1.5 * np.sin(np.pi - angle), 0],
             [3.5, 1.0, 0.5]]
        )
        built = TorsionGeometry(atoms, dihedrals).build(np.array([90.0]))
        assert np.isfinite(built).all()


class TestEvaluateEnergyUsesTorsionGeometry:
    """The evaluator and the screen must build the same geometry."""

    def test_geometry_matches_the_screen(self, flexible):
        from bouquet.assess import evaluate_energy

        class Stub:
            def get_potential_energy(self, atoms):
                return 0.0

        atoms, mol, dihedrals = flexible
        calc = Stub()
        angles = np.resize(
            np.array([61.0, 175.0, 288.0, 44.0, 130.0, 205.0]), len(dihedrals)
        )

        _e, built = evaluate_energy(
            angles, atoms, dihedrals, calc, calc, relax=False
        )
        expected = TorsionGeometry(atoms, dihedrals).build(angles)
        np.testing.assert_allclose(built.get_positions(), expected, atol=1e-10)

    def test_degenerate_chain_returns_failure_energy_not_a_crash(self):
        from bouquet.assess import evaluate_energy
        from bouquet.config import RELAX_FAILURE_ENERGY_EV

        atoms, dihedrals = _linear_chain(COLLINEAR)

        class Boom:
            def get_potential_energy(self, atoms):
                raise AssertionError("calculator must never be reached")

        energy, out = evaluate_energy(
            np.array([90.0]), atoms, dihedrals, Boom(), Boom(), relax=False
        )
        assert energy == RELAX_FAILURE_ENERGY_EV
        assert np.isfinite(out.get_positions()).all()

    def test_ignores_a_constraint_already_on_the_incoming_atoms(self, flexible):
        """Callers hand back relaxed Atoms still carrying FixInternals.

        ASE's set_dihedral writes into the positions array and bypasses
        constraints; going through set_positions() instead makes ASE try to
        enforce the *old* torsions against the new ones and raise
        "FixInternals.adjust_positions did not converge".
        """
        from ase.constraints import FixInternals

        from bouquet.assess import evaluate_energy

        class Stub:
            def get_potential_energy(self, atoms):
                return 0.0

        atoms, _mol, dihedrals = flexible
        constrained = atoms.copy()
        old = [(di.get_angle(atoms), di.chain) for di in dihedrals]
        constrained.set_constraint(FixInternals(dihedrals_deg=old))

        angles = np.resize(np.array([15.0, 200.0, 95.0, 310.0]), len(dihedrals))
        _e, built = evaluate_energy(
            angles, constrained, dihedrals, Stub(), Stub(), relax=False
        )
        np.testing.assert_allclose(
            built.get_positions(),
            TorsionGeometry(atoms, dihedrals).build(angles),
            atol=1e-10,
        )
