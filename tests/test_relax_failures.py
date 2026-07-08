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
