"""Calculator factory for the ring-MTD, delegating to bouquet's own factory.

Kept as a one-function seam so the MTD driver never imports xtb/tblite/psi4 directly and
so the ring subpackage picks up any engine bouquet already supports (``gfnff``, ``gfn2``,
...). This does NOT re-implement a registry -- it wraps
:meth:`bouquet.calculator.CalculatorFactory.create`.
"""

from __future__ import annotations


def make_engine(name: str, mol=None, charge: int = 0, multiplicity: int = 1,
                num_threads: int = 1):
    """Build an ASE calculator for the ring-MTD.

    Args:
        name: method name, one of :func:`available_engines` (e.g. ``"gfnff"``, ``"gfn2"``).
        mol: RDKit molecule (xtb-family calculators read elements/topology from it).
        charge: total molecular charge.
        multiplicity: spin multiplicity (2S+1); 1 for a closed shell.
        num_threads: calculator threads.
    """
    from bouquet.calculator import CalculatorFactory
    return CalculatorFactory.create(method=name, mol=mol, charge=charge,
                                    multiplicity=multiplicity, num_threads=num_threads)


def available_engines() -> tuple[str, ...]:
    """Method names available in this environment (depends on installed backends)."""
    from bouquet.calculator import CalculatorFactory
    return CalculatorFactory.available_methods()
