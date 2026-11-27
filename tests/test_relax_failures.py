# Copyright (c) 2026 Geoffrey Hutchison
# SPDX-License-Identifier: MIT
"""relax_structure must swallow calculator failures during the optimizer loop.

A collective category / low-mode move can drive the geometry into a clashing
region where the calculator fails (xtb raises CalculationFailed, an Exception
that is not a ValueError) mid-relaxation. That must be caught and reported as
the sentinel failure energy, not propagated -- an uncaught exception there
crashes the whole run.
"""

import numpy as np
import pytest

from ase import Atoms
from ase.calculators.calculator import CalculationFailed, Calculator, all_changes

from bouquet.assess import RELAX_FAILURE_ENERGY_EV, relax_structure


class _FailingCalculator(Calculator):
    """Raises CalculationFailed on any evaluation, mimicking xtb SCF failure."""

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        raise CalculationFailed("Self consistent charge iterator did not converge")


def test_relax_structure_swallows_calculator_failure():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]])
    calc = _FailingCalculator()

    # A positive step limit selects the FIRE2 search relaxation, whose first
    # get_forces() raises. The failure must be caught and reported, not raised.
    energy, out = relax_structure(atoms, calc, calc, steps=10)

    assert energy == RELAX_FAILURE_ENERGY_EV
    assert isinstance(out, Atoms)


def test_relax_structure_tight_path_swallows_failure():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]])
    calc = _FailingCalculator()

    # steps=None selects the tight LBFGS final relaxation; same guarantee.
    energy, _ = relax_structure(atoms, calc, calc, steps=None)
    assert energy == RELAX_FAILURE_ENERGY_EV


class _PoisoningCalculator(Calculator):
    """Fails once, then returns a catastrophically wrong energy until reset.

    This is xtb's failure mode in miniature: the SCF that did not converge is kept
    as the restart guess, so the NEXT call succeeds and quietly returns a bogus
    energy. ``reset()`` drops the guess and restores sane numbers.
    """

    implemented_properties = ["energy", "forces"]
    POISONED_ENERGY = -2745.0
    GOOD_ENERGY = -1.0

    def __init__(self):
        super().__init__()
        self.poisoned = False
        self.n_resets = 0
        self.n_failures = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        # Fail exactly once, on the first call ever -- like a single SCF that does
        # not converge. Everything after that succeeds; whether it returns sense
        # depends only on whether the stale restart guess was dropped.
        if self.n_failures == 0:
            self.n_failures += 1
            self.poisoned = True
            raise CalculationFailed("Self consistent charge iterator did not converge")
        energy = self.POISONED_ENERGY if self.poisoned else self.GOOD_ENERGY
        self.results = {"energy": energy, "forces": np.zeros((len(atoms), 3))}

    def reset(self):
        super().reset()
        self.poisoned = False
        self.n_resets += 1


def test_relax_structure_resets_after_calculator_failure():
    """A thrown evaluation must reset the calculator, not just be swallowed.

    Regression for a009_d02 in the d1-8 stopbench: a failed evaluation left xtb's
    restart wavefunction corrupted, the guards never fired (nothing was REJECTED --
    every geometry was intact), and the trial reported -65,366 kcal/mol from a
    perfectly valid structure. Swallowing the exception is not enough; the poisoned
    state has to be dropped or every later step inherits it.
    """
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]])
    calc = _PoisoningCalculator()

    energy, _ = relax_structure(atoms, calc, calc, steps=10)

    assert calc.n_failures == 1, "the calculator should have failed once"
    assert calc.n_resets >= 1, "relax_structure must reset a calculator that threw"
    assert not calc.poisoned, "the poisoned restart guess must be cleared"
    # The energy must not be the poisoned value silently passed off as a minimum.
    assert energy != _PoisoningCalculator.POISONED_ENERGY


def test_evaluate_energy_resets_after_calculator_failure():
    """Same guarantee on the non-relaxing path in evaluate_energy."""
    from bouquet.assess import evaluate_energy
    from bouquet.setup import DihedralInfo

    atoms = Atoms("H4", positions=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1.0]])
    calc = _PoisoningCalculator()
    dihedrals = [DihedralInfo(chain=(0, 1, 2, 3), group=[3], type="rotatable")]

    energy, _ = evaluate_energy([60.0], atoms, dihedrals, calc, calc, relax=False)

    assert calc.n_failures == 1
    assert calc.n_resets >= 1, "evaluate_energy must reset a calculator that threw"
    assert energy == RELAX_FAILURE_ENERGY_EV


def test_reference_optimize_resets_after_failure():
    """scripts/reference.py::_optimize must reset too -- it shares one calc per molecule.

    The reference builder creates a single calculator per molecule and reuses it for the
    start proxy and every pool member in turn, so a failed optimization poisons the rest
    of that molecule's pool, not just its own structure. That is how a009_d02 produced
    E_star = -97283 kcal/mol. Regression for the same bug class as the assess.py
    handlers above, in the scoring path rather than the search path.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import reference

    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]])
    calc = _PoisoningCalculator()

    assert reference._optimize(atoms, calc) is None, "a thrown optimization returns None"
    assert calc.n_failures == 1
    assert calc.n_resets >= 1, "_optimize must reset a calculator that threw"
    assert not calc.poisoned, "the next pool member must not inherit the bad SCF guess"
