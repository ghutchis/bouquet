"""Dihedral angle priors for PiBO-guided conformer optimization.

This module provides:
- Prior distributions for dihedral angles (univariate and bivariate)
- SMARTS-based automatic prior type assignment
- JSON loading for custom fitted priors
"""

import json
import logging
import math
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical, MixtureSameFamily, VonMises

if TYPE_CHECKING:
    from rdkit import Chem

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

logger = logging.getLogger(__name__)

PriorTypeId = Union[str, int]

# Default cap on von Mises concentration (kappa) when a fitted prior is used as a
# PiBO search prior. kappa=50 corresponds to a ~8 deg (1-sigma) basin, broad enough
# for the acquisition optimizer to climb while still expressing a clear preference.
# Raw histogram fits can reach kappa ~ 1e4 (~0.5 deg), which is unusable for search.
DEFAULT_MAX_CONCENTRATION: float = 50.0

# Default weight of a uniform background component mixed into each univariate prior.
# 0.0 disables it (pure fitted mixture). A small weight (e.g. 0.05-0.2) bounds the
# prior's dynamic range so no single mode can dominate the acquisition, and replaces
# the hard log-prob clamp with a smooth, principled floor.
DEFAULT_BACKGROUND_WEIGHT: float = 0.0


class BivariateTopology(Enum):
    """Defines how matched atoms map to two dihedrals in a bivariate pattern.

    Each topology specifies:
    - The minimum number of atoms in the SMARTS match
    - Which atom indices form the first dihedral (4 atoms)
    - Which atom indices form the second dihedral (4 atoms)

    Topologies:
        ADJACENT: 5 atoms, dihedrals share 3 atoms (1-2-3-4 and 2-3-4-5)
        SKIP_ONE: 6 atoms, dihedrals share central bond only (1-2-3-4 and 3-4-5-6)
        W_SHAPE: 6 atoms, branched/W pattern (1-2-3-4 and 4-3-5-6)
    """

    ADJACENT = "adjacent"
    SKIP_ONE = "skip_one"
    W_SHAPE = "w_shape"


# Mapping from topology to (min_atoms, dihedral1_indices, dihedral2_indices)
TOPOLOGY_DEFINITIONS: Dict[BivariateTopology, Tuple[int, Tuple[int, ...], Tuple[int, ...]]] = {
    BivariateTopology.ADJACENT: (5, (0, 1, 2, 3), (1, 2, 3, 4)),
    BivariateTopology.SKIP_ONE: (6, (0, 1, 2, 3), (2, 3, 4, 5)),
    BivariateTopology.W_SHAPE: (6, (0, 1, 2, 3), (3, 2, 4, 5)),
}


def _try_int(value: Any) -> PriorTypeId:
    """Convert to int if possible, otherwise return as-is."""
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def _matches_dihedral(match: Tuple, dihedral: Tuple) -> bool:
    """Check if match equals dihedral in either direction."""
    return match == dihedral or match[::-1] == dihedral


# ============================================================================
# Built-in Prior Definitions (Generic Fallbacks)
# ============================================================================

BUILTIN_UNIVARIATE_PRIORS: Dict[PriorTypeId, Dict[str, Any]] = {
    # Generic patterns - these serve as fallbacks
    "sp3_sp3": {
        "description": "Generic sp3-sp3 single bond (gauche/anti)",
        "means_deg": [60.0, 180.0, 300.0],
        "concentrations": [5.0, 6.0, 5.0],
        "weights": [0.28, 0.44, 0.28],
    },
    "sp3_sp2": {
        "description": "sp3-sp2 bond (eclipsed/anti)",
        "means_deg": [0.0, 180.0],
        "concentrations": [4.0, 4.0],
        "weights": [0.5, 0.5],
    },
    "uniform": {
        "description": "No angular preference (uniform)",
        "means_deg": [],
        "concentrations": [],
        "weights": [],
    },
}


# ============================================================================
# Built-in SMARTS Patterns (Generic)
# ============================================================================


@dataclass
class UnivariateSMARTS:
    """SMARTS pattern for assigning 1D prior types."""

    smarts: str
    prior_type: PriorTypeId
    description: str = ""
    priority: int = 0


@dataclass
class BivariateSMARTS:
    """SMARTS pattern for identifying correlated dihedral pairs.

    The topology field determines how the matched atoms map to two dihedrals:
    - ADJACENT (default): 5 atoms, dihedrals 0-1-2-3 and 1-2-3-4
    - SKIP_ONE: 6 atoms, dihedrals 0-1-2-3 and 2-3-4-5
    - W_SHAPE: 6 atoms, dihedrals 0-1-2-3 and 3-2-4-5 (branched)
    """

    smarts: str
    prior_type: PriorTypeId
    topology: BivariateTopology = BivariateTopology.ADJACENT
    description: str = ""
    priority: int = 0


# Generic 1D patterns (low priority fallbacks)
BUILTIN_UNIVARIATE_SMARTS: List[UnivariateSMARTS] = [
    UnivariateSMARTS(
        smarts="[CX4:1]-[CX4:2]-[CX4:3]-[CX4:4]",
        prior_type="sp3_sp3",
        description="sp3-sp3 carbon chain",
        priority=10,
    ),
    UnivariateSMARTS(
        smarts="[*:1]-[CX4:2]-[CX3:3]=[*:4]",
        prior_type="sp3_sp2",
        description="sp3-sp2 rotation",
        priority=15,
    ),
]


# ============================================================================
# Bivariate Von Mises Mixture
# ============================================================================


class BivariateVonMisesMixture(nn.Module):
    """Mixture of bivariate von Mises distributions for correlated angles."""

    def __init__(
        self,
        components: List[Dict[str, float]],
        weights: List[float],
        max_concentration: Optional[float] = None,
    ):
        super().__init__()

        weights_t = torch.tensor(weights, dtype=torch.float64)
        if weights_t.sum() == 0:
            raise ValueError(f"Weights must sum to a non-zero value")
        weights_t = weights_t / weights_t.sum()
        self.register_buffer("weights", weights_t)
        self.register_buffer("log_weights", torch.log(weights_t))

        mu1 = [math.radians(c["mu1_deg"]) for c in components]
        mu2 = [math.radians(c["mu2_deg"]) for c in components]
        k1 = torch.tensor([c["kappa1"] for c in components], dtype=torch.float64)
        k2 = torch.tensor([c["kappa2"] for c in components], dtype=torch.float64)
        corr = [c.get("correlation", 0.0) for c in components]

        # Cap concentrations so a near-delta histogram fit becomes a usable, smooth
        # *search* prior. See DihedralPriorModule.max_concentration for rationale.
        if max_concentration is not None:
            k1 = k1.clamp(max=max_concentration)
            k2 = k2.clamp(max=max_concentration)

        self.register_buffer("mu1", torch.tensor(mu1, dtype=torch.float64))
        self.register_buffer("mu2", torch.tensor(mu2, dtype=torch.float64))
        self.register_buffer("kappa1", k1)
        self.register_buffer("kappa2", k2)
        self.register_buffer("correlation", torch.tensor(corr, dtype=torch.float64))

    def log_prob(self, phi: Tensor, psi: Tensor) -> Tensor:
        """Compute log probability (unnormalized)."""
        phi_exp = phi.unsqueeze(-1)
        psi_exp = psi.unsqueeze(-1)

        d_phi = phi_exp - self.mu1
        d_psi = psi_exp - self.mu2

        # use the cosine form, e.g.
        # BOKEI - https://doi.org/10.1039/C9CP06688H
        component_log_probs = (
            self.kappa1 * torch.cos(d_phi)
            + self.kappa2 * torch.cos(d_psi)
            + self.correlation * torch.cos(d_phi - d_psi)
        )

        return torch.logsumexp(component_log_probs + self.log_weights, dim=-1)


# ============================================================================
# Prior Registry
# ============================================================================


class DihedralPriorModule(nn.Module):
    """
    Combined prior module for PiBO optimization.

    Supports both independent (1D) and correlated (2D) dihedral priors.

    Usage with PiBO (forward returns a LOG probability, so pair it with a
    log-scale acquisition and log=True):
        prior_module = DihedralPriorModule(...)
        pibo_acqf = PriorGuidedAcquisitionFunction(
            acq_function=LogExpectedImprovement(...),
            prior_module=prior_module,
            log=True,
            prior_exponent=2.0,
        )
    """

    def __init__(
        self,
        dim: int,
        univariate_assignments: Dict[int, PriorTypeId],
        bivariate_assignments: Dict[Tuple[int, int], PriorTypeId],
        univariate_priors: Dict[PriorTypeId, Dict],
        bivariate_priors: Dict[PriorTypeId, Dict],
        fallback_type: PriorTypeId = "sp3_sp3",
        input_in_degrees: bool = True,
        max_concentration: Optional[float] = DEFAULT_MAX_CONCENTRATION,
        background_weight: float = DEFAULT_BACKGROUND_WEIGHT,
    ):
        """
        Args:
            dim: Number of dihedral dimensions
            univariate_assignments: {dim_index: prior_type_id}
            bivariate_assignments: {(dim_i, dim_j): prior_type_id}
            univariate_priors: Prior type definitions for 1D
            bivariate_priors: Prior type definitions for 2D
            fallback_type: Default type for unassigned dimensions
            input_in_degrees: If True, input is in degrees [0, 360); else [0, 1]
            max_concentration: Upper bound applied to every von Mises concentration
                (kappa). Histogram fits can produce kappa ~ 1e4 (a ~0.5 deg-wide
                spike) for near-rigid bonds; as a PiBO *search* prior that collapses
                to a near-delta that is flat (and gradient-free) across virtually the
                entire domain, so the acquisition optimizer cannot follow it. Capping
                kappa turns each mode into a finite-width basin the optimizer can
                actually climb. ``None`` disables the cap (use the raw fitted values).
            background_weight: Weight ``w`` in [0, 1) of a uniform background mixed
                into each univariate factor: ``(1-w) * vonMises_mixture + w * U``,
                where ``U = 1/(2*pi)`` is the uniform density on the circle. This
                bounds the log-prob range a single mode can contribute (so the prior
                informs without dominating/trapping the search) and gives a smooth
                lower floor in place of the hard log-prob clamp. 0.0 disables it.
        """
        super().__init__()

        if not 0.0 <= background_weight < 1.0:
            raise ValueError("background_weight must be in [0, 1)")

        self.dim = dim
        self.fallback_type = fallback_type
        self.input_in_degrees = input_in_degrees
        self.max_concentration = max_concentration
        self.background_weight = background_weight
        self.univariate_assignments = univariate_assignments
        self.bivariate_assignments = bivariate_assignments

        # Merge built-in with custom priors
        self.univariate_priors = {**BUILTIN_UNIVARIATE_PRIORS, **univariate_priors}

        # Build distributions
        self._build_univariate()
        self._build_bivariate()

    @cached_property
    def bivariate_dims(self) -> Set[int]:
        """Dimensions handled by bivariate priors."""
        return {d for pair in self.bivariate_assignments for d in pair}

    def _build_univariate(self):
        """Build 1D von Mises mixture distributions."""
        self.univariate_dists: Dict[int, Optional[MixtureSameFamily]] = {
            d: self._create_univariate_dist(d)
            for d in range(self.dim)
            if d not in self.bivariate_dims
        }

    def _create_univariate_dist(self, d: int) -> Optional[MixtureSameFamily]:
        """Create a univariate von Mises distribution for a dimension."""
        type_id = self.univariate_assignments.get(d, self.fallback_type)
        type_def = self.univariate_priors.get(type_id)

        if type_def is None:
            logger.warning(f"Unknown prior type '{type_id}' for dim {d}, using fallback")
            type_def = self.univariate_priors[self.fallback_type]

        if not type_def.get("means_deg"):
            return None  # Uniform

        means = torch.tensor(
            [math.radians(m) for m in type_def["means_deg"]], dtype=torch.float64
        )
        concs = torch.tensor(type_def["concentrations"], dtype=torch.float64)
        if self.max_concentration is not None:
            concs = concs.clamp(max=self.max_concentration)
        weights = torch.tensor(type_def["weights"], dtype=torch.float64)
        if weights.sum() == 0:
            raise ValueError(f"Weights must sum to a non-zero value for dim {d}")
        weights = weights / weights.sum()

        # Always return MixtureSameFamily for consistency
        return MixtureSameFamily(Categorical(weights), VonMises(means, concs))

    def _build_bivariate(self):
        """Build 2D bivariate von Mises distributions."""
        self.bivariate_dists: Dict[Tuple[int, int], BivariateVonMisesMixture] = {}

        for (d1, d2), type_id in self.bivariate_assignments.items():
            type_def = self.bivariate_priors.get(type_id)

            if type_def is None:
                logger.warning(f"Unknown bivariate type '{type_id}' for dims ({d1}, {d2})")
                continue

            self.bivariate_dists[(d1, d2)] = BivariateVonMisesMixture(
                components=type_def["components"],
                weights=type_def["weights"],
                max_concentration=self.max_concentration,
            )

    def _to_radians(self, x: Tensor) -> Tensor:
        """Convert input to radians in [-π, π]."""
        if self.input_in_degrees:
            # Input is [0, 360) degrees
            return (x / 180.0 - 1.0) * math.pi
        else:
            # Input is [0, 1) normalized
            return (x * 2.0 - 1.0) * math.pi

    def forward(self, X: Tensor) -> Tensor:
        """
        Compute the LOG probability of X under the prior.

        This is paired with ``PriorGuidedAcquisitionFunction(..., log=True)`` and a
        log-scale base acquisition (``LogExpectedImprovement``). In that mode
        botorch forms the PiBO objective additively:

            weighted_af = logEI(X) + prior_exponent * log_prior(X)

        which is the correct PiBO formula (Hvarfner 2022). Returning a *probability*
        here instead (and using ``log=False``) would multiply a log-scale, almost
        always negative ``logEI`` by ``prior**exponent``, which inverts the prior's
        influence and steers the search away from the prior's own modes.

        Args:
            X: Shape (..., q, dim) - dihedral values

        Returns:
            Log probability (unnormalized), shape (..., q). With
            ``background_weight == 0`` the far field is clamped at -20; with a
            background the uniform component provides the (smooth) lower floor.
        """
        log_prob = torch.zeros(X.shape[:-1], dtype=X.dtype, device=X.device)

        # Each univariate factor is a normalized circular density, so we can mix in
        # a uniform background per-dimension: log[(1-w)*vM(theta) + w/(2*pi)].
        w_bg = self.background_weight
        if w_bg > 0.0:
            log_uniform = -math.log(2.0 * math.pi)  # uniform density on the circle
            log_one_minus_w = math.log(1.0 - w_bg)
            log_w = math.log(w_bg)

        # Univariate contributions
        for d, dist in self.univariate_dists.items():
            if dist is None:
                continue
            angle = self._to_radians(X[..., d])
            vm_lp = dist.log_prob(angle)
            if w_bg > 0.0:
                bg_lp = torch.full_like(vm_lp, log_w + log_uniform)
                vm_lp = torch.logaddexp(log_one_minus_w + vm_lp, bg_lp)
            log_prob = log_prob + vm_lp

        # Bivariate contributions. These factors are *unnormalized*, so the uniform
        # background (which needs a normalized density to mix against) is not applied
        # here; they keep the hard clamp below as their floor.
        for (d1, d2), dist in self.bivariate_dists.items():
            angle1 = self._to_radians(X[..., d1])
            angle2 = self._to_radians(X[..., d2])
            log_prob = log_prob + dist.log_prob(angle1, angle2)

        if w_bg > 0.0:
            # The per-dimension uniform background already bounds log_prob from
            # below (>= dim * log(w/(2*pi))), so no hard clamp is needed -- clamping
            # here would flatten the smooth background floor back to a dead region.
            return log_prob
        # No background: clamp the floor so a candidate far from every prior mode
        # incurs a bounded penalty (still carrying a gradient) instead of -inf.
        return torch.clamp(log_prob, min=-20.0)

    def describe(self) -> str:
        """Human-readable description of assignments."""
        lines = [f"DihedralPriorModule (dim={self.dim})"]

        if self.bivariate_assignments:
            lines.append("\nCorrelated pairs (2D):")
            for (d1, d2), type_id in self.bivariate_assignments.items():
                desc = self.bivariate_priors.get(type_id, {}).get("description", "")
                lines.append(f"  ({d1}, {d2}): {type_id} - {desc}")

        lines.append("\nIndependent dihedrals (1D):")
        for d in range(self.dim):
            if d in self.bivariate_dims:
                continue
            type_id = self.univariate_assignments.get(d, self.fallback_type)
            is_fb = d not in self.univariate_assignments
            desc = self.univariate_priors.get(type_id, {}).get("description", "")
            fb_str = " (fallback)" if is_fb else ""
            lines.append(f"  {d}: {type_id}{fb_str} - {desc}")

        return "\n".join(lines)


# ============================================================================
# SMARTS Matching
# ============================================================================


class DihedralPriorMatcher:
    """Assigns prior types to dihedrals using SMARTS patterns."""

    def __init__(
        self,
        univariate_smarts: List[UnivariateSMARTS],
        bivariate_smarts: List[BivariateSMARTS],
    ):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit required for SMARTS matching")

        # Sort by priority (highest first)
        self.univariate_patterns = sorted(univariate_smarts, key=lambda p: p.priority, reverse=True)
        self.bivariate_patterns = sorted(bivariate_smarts, key=lambda p: p.priority, reverse=True)

        # Compile patterns
        self._compiled_uni = self._compile_patterns(self.univariate_patterns)
        self._compiled_bi = self._compile_patterns(self.bivariate_patterns)

    @staticmethod
    def _compile_patterns(patterns: List) -> Dict[str, Chem.Mol]:
        """Compile SMARTS patterns into RDKit molecule objects."""
        compiled = {}
        for p in patterns:
            mol = Chem.MolFromSmarts(p.smarts)
            if mol:
                compiled[p.smarts] = mol
        return compiled

    def match_univariate(
        self,
        mol: Chem.Mol,
        dihedral: Tuple[int, int, int, int],
    ) -> Optional[PriorTypeId]:
        """Find matching 1D prior type for a dihedral."""
        for pattern in self.univariate_patterns:
            smarts_mol = self._compiled_uni.get(pattern.smarts)
            if smarts_mol is None:
                continue

            for match in mol.GetSubstructMatches(smarts_mol):
                # these SMARTS have atom maps, so convert them
                # http://www.rdkit.org/docs/GettingStartedInPython.html#atom-map-indices-in-smarts
                index_map = {}
                for atom in smarts_mol.GetAtoms():
                    map_num = atom.GetAtomMapNum()
                    if map_num:
                        index_map[map_num-1] = atom.GetIdx()
                map_list = [index_map[x] for x in sorted(index_map)]
                mapped = tuple(match[x] for x in map_list)

                if len(mapped) >= 4 and _matches_dihedral(mapped, dihedral):
                    return pattern.prior_type

        return None

    def match_bivariate(
        self,
        mol: Chem.Mol,
        dihedrals: List[Tuple[int, int, int, int]],
    ) -> Dict[Tuple[int, int], PriorTypeId]:
        """Find correlated pairs among dihedrals."""
        results = {}

        for pattern in self.bivariate_patterns:
            smarts_mol = self._compiled_bi.get(pattern.smarts)
            if smarts_mol is None:
                continue

            # Get topology definition for this pattern
            min_atoms, dih1_indices, dih2_indices = TOPOLOGY_DEFINITIONS[pattern.topology]

            for match in mol.GetSubstructMatches(smarts_mol):
                if len(match) < min_atoms:
                    continue

                # Extract dihedrals based on topology
                dih1 = tuple(match[i] for i in dih1_indices)
                dih2 = tuple(match[i] for i in dih2_indices)

                # Find indices in our dihedral list
                idx1 = idx2 = None
                for i, dih in enumerate(dihedrals):
                    if _matches_dihedral(dih, dih1):
                        idx1 = i
                    if _matches_dihedral(dih, dih2):
                        idx2 = i

                if idx1 is not None and idx2 is not None:
                    pair = (min(idx1, idx2), max(idx1, idx2))
                    if pair not in results:
                        results[pair] = pattern.prior_type

        return results

    def assign_all(
        self,
        mol: Chem.Mol,
        dihedrals: List[Tuple[int, int, int, int]],
    ) -> Tuple[Dict[int, PriorTypeId], Dict[Tuple[int, int], PriorTypeId]]:
        """
        Assign both univariate and bivariate types.

        Returns:
            (univariate_assignments, bivariate_assignments)
        """
        # First find bivariate pairs
        bivariate = self.match_bivariate(mol, dihedrals)
        bivariate_dims = set()
        for d1, d2 in bivariate.keys():
            bivariate_dims.add(d1)
            bivariate_dims.add(d2)

        # Then assign univariate to remaining
        univariate = {}
        for i, dih in enumerate(dihedrals):
            if i in bivariate_dims:
                continue
            type_id = self.match_univariate(mol, dih)
            if type_id is not None:
                univariate[i] = type_id

        return univariate, bivariate


# ============================================================================
# JSON Loading
# ============================================================================


def load_univariate_priors_from_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load univariate prior definitions from a JSON file.

    Expected format:
    {
        "type_id": {
            "smarts": "[*:1]~[CX4:2]!@[n:3]~[*:4]",
            "description": "Fit from N obs.",
            "means_deg": [...],
            "concentrations": [...],
            "weights": [...]
        },
        ...
    }

    Each entry contains both the SMARTS pattern and the fitted parameters.

    Args:
        filepath: Path to the JSON file

    Returns:
        Dict with keys:
            - "priors": {type_id: prior_definition} - the fit parameters
            - "smarts": List[UnivariateSMARTS] - SMARTS patterns for matching
    """
    filepath = Path(filepath)

    with filepath.open("r") as f:
        data = json.load(f)

    result = {
        "priors": {},
        "smarts": [],
    }

    for type_id, defn in data.items():
        type_key = _try_int(type_id)

        # Extract SMARTS pattern from the entry
        smarts = defn.get("smarts")
        if smarts:
            result["smarts"].append(
                UnivariateSMARTS(
                    smarts=smarts,
                    prior_type=type_key,
                    description=defn.get("description", ""),
                    priority=50,  # Default priority for loaded patterns
                )
            )

        # Store the prior definition (without smarts, for the prior module)
        result["priors"][type_key] = {
            "description": defn.get("description", ""),
            "means_deg": defn.get("means_deg", []),
            "concentrations": defn.get("concentrations", []),
            "weights": defn.get("weights", []),
        }

    logger.info(
        f"Loaded {len(result['priors'])} univariate priors with "
        f"{len(result['smarts'])} SMARTS patterns from {filepath}"
    )

    return result


def load_bivariate_priors_from_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load bivariate prior definitions from a JSON file.

    Expected format:
    {
        "type_id": {
            "smarts": "[CX4:1]-[CX4:2]-[CX4:3]-[CX4:4]-[CX4:5]",
            "topology": "adjacent",
            "description": "...",
            "components": [
                {"mu1_deg": ..., "mu2_deg": ..., "kappa1": ..., "kappa2": ..., "correlation": ...},
                ...
            ],
            "weights": [...]
        },
        ...
    }

    Each entry contains the SMARTS pattern, topology, and fitted parameters.

    Args:
        filepath: Path to the JSON file

    Returns:
        Dict with keys:
            - "priors": {type_id: prior_definition} - the fit parameters
            - "smarts": List[BivariateSMARTS] - SMARTS patterns for matching
    """
    filepath = Path(filepath)

    with filepath.open("r") as f:
        data = json.load(f)

    result = {
        "priors": {},
        "smarts": [],
    }

    for type_id, defn in data.items():
        type_key = _try_int(type_id)

        # Extract SMARTS pattern from the entry
        smarts = defn.get("smarts")
        if smarts:
            topology_str = defn.get("topology", "adjacent")
            topology = BivariateTopology(topology_str)

            result["smarts"].append(
                BivariateSMARTS(
                    smarts=smarts,
                    prior_type=type_key,
                    topology=topology,
                    description=defn.get("description", ""),
                    priority=50,  # Default priority for loaded patterns
                )
            )

        # Store the prior definition
        result["priors"][type_key] = {
            "description": defn.get("description", ""),
            "components": defn.get("components", []),
            "weights": defn.get("weights", []),
        }

    logger.info(
        f"Loaded {len(result['priors'])} bivariate priors with "
        f"{len(result['smarts'])} SMARTS patterns from {filepath}"
    )

    return result


def create_prior_module(
    mol: Chem.Mol,
    dihedrals: List[Tuple[int, int, int, int]],
    univariate_file: Optional[Union[str, Path]] = None,
    bivariate_file: Optional[Union[str, Path]] = None,
    fallback_type: PriorTypeId = "sp3_sp3",
    max_concentration: Optional[float] = DEFAULT_MAX_CONCENTRATION,
    background_weight: float = DEFAULT_BACKGROUND_WEIGHT,
) -> DihedralPriorModule:
    """
    Create a DihedralPriorModule for a molecule.

    Args:
        mol: RDKit molecule
        dihedrals: List of dihedral atom tuples (from detect_dihedrals)
        univariate_file: Optional JSON file with univariate priors
        bivariate_file: Optional JSON file with bivariate priors
        fallback_type: Default prior type for unmatched dihedrals
        max_concentration: Cap on von Mises concentration (kappa); see
            DihedralPriorModule. ``None`` disables the cap.
        background_weight: Uniform background weight mixed into each univariate
            prior; see DihedralPriorModule. 0.0 disables it.

    Returns:
        Configured DihedralPriorModule
    """
    # Load custom priors if provided
    univariate_priors = {}
    bivariate_priors = {}
    custom_uni_smarts = []
    custom_bi_smarts = []

    if univariate_file is not None:
        loaded = load_univariate_priors_from_json(univariate_file)
        univariate_priors = loaded["priors"]
        custom_uni_smarts = loaded["smarts"]

    if bivariate_file is not None:
        loaded = load_bivariate_priors_from_json(bivariate_file)
        bivariate_priors = loaded["priors"]
        custom_bi_smarts = loaded["smarts"]

    # Combine built-in and custom SMARTS
    all_uni_smarts = BUILTIN_UNIVARIATE_SMARTS + custom_uni_smarts

    # Match patterns to dihedrals
    matcher = DihedralPriorMatcher(all_uni_smarts, custom_bi_smarts)
    univariate_assignments, bivariate_assignments = matcher.assign_all(mol, dihedrals)

    logger.info(
        f"Assigned {len(univariate_assignments)} univariate, "
        f"{len(bivariate_assignments)} bivariate priors to {len(dihedrals)} dihedrals"
    )

    return DihedralPriorModule(
        dim=len(dihedrals),
        univariate_assignments=univariate_assignments,
        bivariate_assignments=bivariate_assignments,
        univariate_priors=univariate_priors,
        bivariate_priors=bivariate_priors,
        fallback_type=fallback_type,
        input_in_degrees=False,  # PiBO operates in [0,1] normalized space
        max_concentration=max_concentration,
        background_weight=background_weight,
    )
