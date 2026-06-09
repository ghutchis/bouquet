"""Utilities for conformer optimization"""

# Defer the heavy scientific stack (torch, botorch, gpytorch, rdkit, ase,
# networkx) pulled in transitively by these submodules: `import bouquet` stays
# cheap and each submodule loads only when first accessed (Python 3.15+).
__lazy_modules__ = [
    "bouquet.assess",
    "bouquet.calculator",
    "bouquet.config",
    "bouquet.gradients",
    "bouquet.io",
    "bouquet.priors",
    "bouquet.setup",
    "bouquet.solver",
]

from . import assess, calculator, config, gradients, io, priors, setup, solver
