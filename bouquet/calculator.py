"""Factory for creating ASE calculators with capability detection.

   If you want to add a new calculator, see
   for example _ani() or _psi4_calculator().
   And add it to _build_full_registry().
"""

import importlib.util
import shutil
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from bouquet.config import Configuration
    from rdkit import Chem

_DEFAULT_NUM_THREADS = 4


# ---- Metadata model ---------------------------------------------------------


@dataclass(frozen=True)
class MethodSpec:
    builder: Callable[[Optional["Chem.Mol"], int, int, int], "Calculator"]
    category: str
    requires: Tuple[str, ...] = ()       # importable Python modules
    executables: Tuple[str, ...] = ()    # CLI tools that must be on PATH (e.g. gcp)
    description: str = ""


# ---- Dependency checking ----------------------------------------------------


def _module_available(module: str) -> bool:
    # Already-imported (or test-injected) modules count as available without
    # re-importing -- mirrors __import__'s sys.modules-first lookup and lets tests
    # inject mocks via patch.dict("sys.modules", ...).
    if module in sys.modules:
        return True
    # Otherwise check importability without executing the module, so building the
    # registry never eagerly imports heavy backends. For dotted names, find_spec
    # imports the parent packages and raises ModuleNotFoundError if one is missing.
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def _check_requirements(spec: MethodSpec) -> bool:
    return all(_module_available(m) for m in spec.requires) and all(
        shutil.which(exe) is not None for exe in spec.executables
    )


# ---- Helpers ----------------------------------------------------------------


def _psi4_calculator(
    method: str,
    *,
    basis: Optional[str],
    num_threads: int,
    charge: int,
    multiplicity: int,
) -> "Calculator":
    try:
        from ase.calculators.psi4 import Psi4
    except ImportError as e:
        raise RuntimeError("Psi4 is required for this method") from e

    kwargs = dict(
        method=method,
        num_threads=num_threads,
        charge=charge,
        multiplicity=multiplicity,
    )

    if basis is not None:
        kwargs["basis"] = basis

    return Psi4(**kwargs)


# ---- Builders ---------------------------------------------------------------


def _ani(*_):
    import torchani
    return torchani.models.ANI2x().ase()


def _aimnet2(mol, num_threads, charge, multiplicity):
    # AIMNet2's ASE calculator takes the total charge in its constructor (it is a
    # charge-aware potential). Spin/multiplicity is not part of the documented
    # AIMNet2ASE API, so only charge is passed; the charge stamped on the Atoms by
    # setup.apply_charge_spin is redundant here but harmless.
    from aimnet.calculators import AIMNet2ASE
    return AIMNet2ASE("aimnet2", charge=charge)


def _xtb(method: str):
    # xtb's ASE calculator takes neither charge nor spin in its constructor -- it
    # reads them from each Atoms it evaluates (sum of initial charges / magnetic
    # moments). So the builder ignores the charge/multiplicity args; those are
    # stamped on the Atoms instead (see setup.apply_charge_spin).
    def builder(*_):
        from xtb.ase.calculator import XTB
        return XTB(method=method)
    return builder


def _rdkit_mmff(mol, *_):
    if mol is None:
        raise ValueError("MMFF requires RDKit molecule")
    from bouquet.calc_rdkit import RDKitMMFFCalculator
    return RDKitMMFFCalculator(mol)


def _rdkit_uff(mol, *_):
    if mol is None:
        raise ValueError("UFF requires RDKit molecule")
    from bouquet.calc_rdkit import RDKitUFFCalculator
    return RDKitUFFCalculator(mol)


# ---- Full registry ----------------------------------------------------------


def _build_full_registry() -> Dict[str, MethodSpec]:
    return {
        # ---- ML --------------------------------------------------------------
        "ani": MethodSpec(
            builder=_ani,
            category="ml",
            requires=("torchani",),
            description="ANI-2x neural network potential",
        ),
        "aimnet2": MethodSpec(
            builder=_aimnet2,
            category="ml",
            requires=("aimnet.calculators",),
            description="AIMNet2 neural network potential",
        ),

        # ---- xTB -------------------------------------------------------------
        "gfn2": MethodSpec(
            builder=_xtb("GFN2xTB"),
            category="semiempirical",
            requires=("xtb.ase.calculator",),
            description="GFN2-xTB tight-binding method",
        ),
        "gfn0": MethodSpec(
            builder=_xtb("GFN0xTB"),
            category="semiempirical",
            requires=("xtb.ase.calculator",),
            description="GFN0-xTB very fast tight-binding",
        ),
        "gfnff": MethodSpec(
            builder=_xtb("gfnff"),
            category="forcefield",
            requires=("xtb.ase.calculator",),
            description="GFN-FF force field",
        ),

        # ---- DFT -------------------------------------------------------------
        "b3lyp": MethodSpec(
            builder=lambda m, nt, q, mult: _psi4_calculator(
                "b3lyp-d4", basis="def2-svp",
                num_threads=nt, charge=q, multiplicity=mult
            ),
            category="dft",
            requires=("ase.calculators.psi4",),
            executables=("dftd4"),
            description="B3LYP with D4 dispersion",
        ),
        "wb97x": MethodSpec(
            builder=lambda m, nt, q, mult: _psi4_calculator(
                "wb97x-d3bj", basis="def2-svp",
                num_threads=nt, charge=q, multiplicity=mult
            ),
            category="dft",
            requires=("ase.calculators.psi4",),
            executables=("s-dftd3"),
            description="ωB97X with D3(BJ) dispersion",
        ),

        # ---- Wavefunction ----------------------------------------------------
        "mp2": MethodSpec(
            builder=lambda m, nt, q, mult: _psi4_calculator(
                "df-mp2", basis="def2-tzvp",
                num_threads=nt, charge=q, multiplicity=mult
            ),
            category="wavefunction",
            requires=("ase.calculators.psi4",),
            description="Density-fitted MP2",
        ),

        # ---- 3c methods ------------------------------------------------------
        # The -3c composites need psi4 plus the gCP and dispersion CLI tools
        # (conda-forge: gcp-correction -> `gcp`; s-dftd3 -> `s-dftd3`; dftd4 ->
        # `dftd4`). These are command-line programs, not importable modules, so
        # they are checked via `executables` (shutil.which), not `requires`.
        "hf3c": MethodSpec(
            builder=lambda m, nt, q, mult: _psi4_calculator(
                "hf3c", basis=None,
                num_threads=nt, charge=q, multiplicity=mult
            ),
            category="qm-fast",
            requires=("ase.calculators.psi4",),
            executables=("gcp", "s-dftd3"),
            description="HF-3c composite method",
        ),
        "b973c": MethodSpec(
            builder=lambda m, nt, q, mult: _psi4_calculator(
                "b973c", basis=None,
                num_threads=nt, charge=q, multiplicity=mult
            ),
            category="qm-fast",
            requires=("ase.calculators.psi4",),
            executables=("gcp", "s-dftd3"),
            description="B97-3c composite DFT",
        ),
        "r2scan3c": MethodSpec(
            builder=lambda m, nt, q, mult: _psi4_calculator(
                "r2scan3c", basis=None,
                num_threads=nt, charge=q, multiplicity=mult
            ),
            category="qm-fast",
            requires=("ase.calculators.psi4",),
            executables=("gcp", "dftd4"),
            description="r2SCAN-3c composite DFT",
        ),

        # ---- RDKit ----------------------------------------------------------
        "mmff": MethodSpec(
            builder=_rdkit_mmff,
            category="forcefield",
            requires=("rdkit",),
            description="MMFF94 force field",
        ),
        "uff": MethodSpec(
            builder=_rdkit_uff,
            category="forcefield",
            requires=("rdkit",),
            description="Universal force field",
        ),
    }


_FULL_REGISTRY = _build_full_registry()


def _build_available_registry() -> Dict[str, MethodSpec]:
    # Resolved at call time (not import time) so that requirement checks reflect
    # the modules/executables present when a calculator is actually requested.
    # This also lets tests inject optional dependencies via
    # `patch.dict("sys.modules", ...)` before the registry is consulted.
    return {
        name: spec
        for name, spec in _FULL_REGISTRY.items()
        if _check_requirements(spec)
    }


# Methods are registry-driven (not a fixed Literal), so the config field/type-hint
# alias is just a str. The selectable set is CalculatorFactory.available_methods()
# -- the installed subset of the registry.
MethodType = str


# ---- Factory ----------------------------------------------------------------


class CalculatorFactory:
    """Factory for creating ASE calculators with automatic capability detection."""

    @staticmethod
    def create(
        method: str,
        mol: Optional["Chem.Mol"] = None,
        num_threads: int = _DEFAULT_NUM_THREADS,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> "Calculator":
        registry = _build_available_registry()
        try:
            spec = registry[method]
        except KeyError:
            available = ", ".join(sorted(registry))
            raise ValueError(
                f"Method '{method}' is not available. Available methods: {available}"
            ) from None

        return spec.builder(mol, num_threads, charge, multiplicity)

    @classmethod
    def available_methods(cls) -> Tuple[str, ...]:
        return tuple(sorted(_build_available_registry()))

    @classmethod
    def describe_methods(cls) -> Dict[str, str]:
        return {
            name: spec.description
            for name, spec in _build_available_registry().items()
        }

    @classmethod
    def methods_by_category(cls) -> Dict[str, Tuple[str, ...]]:
        out: Dict[str, list[str]] = {}
        for name, spec in _build_available_registry().items():
            out.setdefault(spec.category, []).append(name)
        return {k: tuple(sorted(v)) for k, v in out.items()}

    @classmethod
    def from_config(
        cls,
        config: "Configuration",
        for_optimizer: bool = False,
        mol: Optional["Chem.Mol"] = None,
    ) -> "Calculator":
        method = config.optimizer_method if for_optimizer else config.energy_method
        return cls.create(
            method=method,
            mol=mol,
            num_threads=config.num_threads,
            charge=config.charge,
            multiplicity=config.multiplicity,
        )
