"""The collective-move path must carry the same guards as an ordinary BO step.

This is a regression suite for a real gap: the clash feasibility region, the
bond-formation species guard and the calculator reset were wired into
``solver._evaluate_point`` (BO steps and initial guesses) but NOT into the
low-mode / category moves. Those moves are a committed geometry change followed
by an UNCONSTRAINED relaxation -- the step with the most freedom to collapse --
and low-mode auto-enables at high d, so the hole opened exactly where it did the
most damage: two d=12 molecules in the kernel A/B collapsed mid-run and stayed
collapsed in every arm, reporting energies near -6000 kcal/mol.
"""

import numpy as np
import pytest
from ase import Atoms

from bouquet.config import RELAX_FAILURE_ENERGY_EV
from bouquet.setup import DihedralInfo
from bouquet.solver_moves import _collective_clash_reject, _collective_result


class _State:
    """Minimal stand-in for OptimizationState's guard-related fields."""

    def __init__(self, **kw):
        self.clash_feasibility = None
        self.required_bonds = None
        self.check_bond_forms = True
        self.check_bond_breaks = False
        self.n_bond_forms = 0
        self.n_bond_breaks = 0
        self.n_collective_clash_rejects = 0
        self.__dict__.update(kw)


class _SpyCalc:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


@pytest.fixture
def chain():
    """A short carbon chain with one dihedral, and its reference bond set."""
    from bouquet.setup import geometry_bond_set

    atoms = Atoms("CCCC", positions=[[0, 0, 0], [1.5, 0, 0], [2.2, 1.3, 0], [3.7, 1.3, 0]])
    dihedrals = [DihedralInfo(chain=(0, 1, 2, 3), group={3})]
    return atoms, dihedrals, geometry_bond_set(atoms)


class TestClashPreScreen:
    def test_rejects_an_infeasible_proposal(self):
        state = _State(clash_feasibility=lambda a: np.zeros(len(np.atleast_2d(a)), bool))
        assert _collective_clash_reject(state, np.array([10.0, 20.0]))
        assert state.n_collective_clash_rejects == 1

    def test_accepts_a_feasible_proposal(self):
        state = _State(clash_feasibility=lambda a: np.ones(len(np.atleast_2d(a)), bool))
        assert not _collective_clash_reject(state, np.array([10.0, 20.0]))
        assert state.n_collective_clash_rejects == 0

    def test_no_constraint_configured_is_a_no_op(self):
        state = _State(clash_feasibility=None)
        assert not _collective_clash_reject(state, np.array([10.0, 20.0]))

    def test_screens_the_proposal_as_a_single_batch_row(self):
        """The feasibility callable takes (n_candidates, n_dihedrals)."""
        seen = {}

        def feas(a):
            seen["shape"] = np.atleast_2d(a).shape
            return np.ones(len(np.atleast_2d(a)), bool)

        _collective_clash_reject(_State(clash_feasibility=feas), np.array([1.0, 2.0, 3.0]))
        assert seen["shape"] == (1, 3)


class TestSpeciesGuardOnCollectiveResult:
    def test_rejects_a_collapsed_geometry_by_default(self, chain):
        _atoms, dihedrals, reference = chain
        collapsed = Atoms("CCCC", positions=[[0, 0, 0], [1.5, 0, 0], [0.8, 0.7, 0],
                                             [1.4, 1.0, 0]])
        state = _State(required_bonds=reference)
        assert _collective_result(state, collapsed, -5.0, dihedrals) is None
        assert state.n_bond_forms == 1

    def test_accepts_an_intact_geometry(self, chain):
        atoms, dihedrals, reference = chain
        state = _State(required_bonds=reference)
        out = _collective_result(state, atoms, -5.0, dihedrals)
        assert out is not None
        coords, got, energy = out
        assert energy == -5.0 and len(coords) == 1
        assert state.n_bond_forms == 0

    def test_bond_breaking_stays_opt_in(self, chain):
        """Matches the BO path: formation is default-on, breaking needs --retain-bonds."""
        atoms, dihedrals, reference = chain
        broken = Atoms("CCCC", positions=[[0, 0, 0], [1.5, 0, 0], [2.2, 1.3, 0],
                                          [9.0, 9.0, 9.0]])
        assert _collective_result(_State(required_bonds=reference), broken, -5.0,
                                  dihedrals) is not None
        state = _State(required_bonds=reference, check_bond_breaks=True)
        assert _collective_result(state, broken, -5.0, dihedrals) is None
        assert state.n_bond_breaks == 1


class TestCalculatorResetOnCollectiveRejection:
    def test_resets_when_the_geometry_is_rejected(self, chain):
        _atoms, dihedrals, reference = chain
        collapsed = Atoms("CCCC", positions=[[0, 0, 0], [1.5, 0, 0], [0.8, 0.7, 0],
                                             [1.4, 1.0, 0]])
        calc, relax = _SpyCalc(), _SpyCalc()
        _collective_result(_State(required_bonds=reference), collapsed, -5.0,
                           dihedrals, calc, relax)
        assert calc.resets == 1 and relax.resets == 1

    def test_resets_on_a_failed_evaluation(self, chain):
        atoms, dihedrals, reference = chain
        calc = _SpyCalc()
        _collective_result(_State(required_bonds=reference), atoms,
                           RELAX_FAILURE_ENERGY_EV, dihedrals, calc)
        assert calc.resets == 1

    def test_does_not_reset_on_a_good_move(self, chain):
        atoms, dihedrals, reference = chain
        calc = _SpyCalc()
        _collective_result(_State(required_bonds=reference), atoms, -5.0, dihedrals, calc)
        assert calc.resets == 0, "resetting on the good path throws away the SCF restart"


class TestEveryCollectiveMoveIsGuarded:
    def test_no_move_reaches_the_calculator_without_the_screen(self):
        """A new collective move must not be able to skip the guards.

        Both move functions have to route through _collective_clash_reject before
        _collective_kick_relax, and through _collective_result afterwards. Checked
        on the source so an added move that calls _collective_kick_relax directly
        fails here rather than silently reopening the hole.
        """
        import inspect

        from bouquet import solver_moves

        for name in ("_low_mode_move", "_category_move"):
            src = inspect.getsource(getattr(solver_moves, name))
            assert "_collective_kick_relax" in src, f"{name} changed shape"
            assert "_collective_clash_reject" in src, f"{name} skips the clash screen"
            assert "_collective_result" in src, f"{name} skips the species guard"

    def test_category_move_does_not_record_collapsed_energy_in_reduced_gp(
        self, chain, monkeypatch
    ):
        """A species-guard-rejected collapse must record the failure sentinel in _cat_Y,
        not its bogus huge negative energy."""
        import torch
        from bouquet import solver_moves

        atoms, dihedrals, reference = chain
        collapsed = Atoms(
            "CCCC",
            positions=[[0, 0, 0], [1.5, 0, 0], [0.8, 0.7, 0], [1.4, 1.0, 0]],
        )
        state = _State(
            required_bonds=reference,
            category_groups=[[0]],
            category_min_moves=6,
            prior_module=None,
            observed_atoms=[atoms],
            observed_energies=torch.tensor([0.0], dtype=torch.float64),
            observed_coords=torch.tensor([[180.0]], dtype=torch.float64),
            start_energy=-10.0,
            _cat_Z=[],
            _cat_Y=[],
        )
        monkeypatch.setattr(
            solver_moves, "_collective_kick_relax", lambda *a, **kw: (-5000.0, collapsed)
        )
        out = solver_moves._category_move(
            state, dihedrals, _SpyCalc(), _SpyCalc(), np.random.default_rng(0)
        )
        assert out is None
        assert len(state._cat_Y) == 1
        assert state._cat_Y[0] == pytest.approx(RELAX_FAILURE_ENERGY_EV - state.start_energy)
