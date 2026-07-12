"""Utilities for conformer optimization"""

import importlib
from typing import TYPE_CHECKING

# Defer the heavy scientific stack (torch, botorch, gpytorch, rdkit, ase,
# networkx) pulled in transitively by these submodules: `import bouquet` stays
# cheap and each submodule loads only when first accessed.
#
# On Python 3.15+ the interpreter honors ``__lazy_modules__`` natively for the
# ``from . import ...`` statement. On earlier runtimes (3.7+) the PEP 562
# ``__getattr__`` below provides the same lazy behavior, so importing bouquet
# does not eagerly drag in torch/botorch/rdkit/ase.
__lazy_modules__ = [
    "bouquet.assess",
    "bouquet.calculator",
    "bouquet.config",
    "bouquet.ensemble",
    "bouquet.gradients",
    "bouquet.io",
    "bouquet.priors",
    "bouquet.setup",
    "bouquet.solver",
]

_SUBMODULES = frozenset(name.split(".", 1)[1] for name in __lazy_modules__)

if TYPE_CHECKING:  # keep static analyzers / IDEs aware of the attributes
    from . import (
        assess,
        calculator,
        config,
        ensemble,
        gradients,
        io,
        priors,
        setup,
        solver,
    )


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module  # cache so subsequent access skips __getattr__
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _SUBMODULES)
