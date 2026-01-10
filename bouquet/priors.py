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
    try:
        return int(value)
    except (ValueError, TypeError):
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

BUILTIN_BIVARIATE_PRIORS: Dict[PriorTypeId, Dict[str, Any]] = {
    # Generic correlated pattern
    "generic_correlated": {
        "description": "Generic correlated adjacent dihedrals",
        "components": [
            {"mu1_deg": 60.0, "mu2_deg": 60.0, "kappa1": 4.0, "kappa2": 4.0, "correlation": 1.0},
            {"mu1_deg": 180.0, "mu2_deg": 180.0, "kappa1": 5.0, "kappa2": 5.0, "correlation": 1.0},
            {"mu1_deg": 300.0, "mu2_deg": 300.0, "kappa1": 4.0, "kappa2": 4.0, "correlation": 1.0},
        ],
        "weights": [0.3, 0.4, 0.3],
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

# Generic 2D patterns (low priority)
BUILTIN_BIVARIATE_SMARTS: List[BivariateSMARTS] = [
    # Adjacent sp3-sp3 bonds
    BivariateSMARTS(
        smarts="[CX4:1]-[CX4:2]-[CX4:3]-[CX4:4]-[CX4:5]",
        prior_type="generic_correlated",
        description="Adjacent sp3-sp3 bonds",
        priority=5,
    ),
]


# ============================================================================
# Bivariate Von Mises Mixture
# ============================================================================


class BivariateVonMisesMixture(nn.Module):
    """Mixture of bivariate von Mises distributions for correlated angles."""

    def __init__(self, components: List[Dict[str, float]], weights: List[float]):
        super().__init__()

        weights_t = torch.tensor(weights, dtype=torch.float64)
        if weights_t.sum() == 0:
            raise ValueError(f"Weights must sum to a non-zero value")
        weights_t = weights_t / weights_t.sum()
        self.register_buffer("weights", weights_t)
        self.register_buffer("log_weights", torch.log(weights_t))

        mu1 = [math.radians(c["mu1_deg"]) for c in components]
        mu2 = [math.radians(c["mu2_deg"]) for c in components]
        k1 = [c["kappa1"] for c in components]
        k2 = [c["kappa2"] for c in components]
        corr = [c.get("correlation", 0.0) for c in components]

        self.register_buffer("mu1", torch.tensor(mu1, dtype=torch.float64))
        self.register_buffer("mu2", torch.tensor(mu2, dtype=torch.float64))
        self.register_buffer("kappa1", torch.tensor(k1, dtype=torch.float64))
        self.register_buffer("kappa2", torch.tensor(k2, dtype=torch.float64))
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

    Usage with PiBO:
        prior_module = DihedralPriorModule(...)
        pibo_acqf = PriorGuidedAcquisitionFunction(
            acq_function=base_acqf,
            prior_module=prior_module,
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
        """
        super().__init__()

        self.dim = dim
        self.fallback_type = fallback_type
        self.input_in_degrees = input_in_degrees
        self.univariate_assignments = univariate_assignments
        self.bivariate_assignments = bivariate_assignments

        # Merge built-in with custom priors
        self.univariate_priors = {**BUILTIN_UNIVARIATE_PRIORS, **univariate_priors}
        self.bivariate_priors = {**BUILTIN_BIVARIATE_PRIORS, **bivariate_priors}

        # Build distributions
        self._build_univariate()
        self._build_bivariate()

    @cached_property
    def bivariate_dims(self) -> Set[int]:
        """Dimensions handled by bivariate priors."""
        return {d for pair in self.bivariate_assignments for d in pair}

    def _build_univariate(self):
        """Build 1D von Mises mixture distributions."""
        self.univariate_dists: Dict[int, Optional[MixtureSameFamily]] = {}

        for d in range(self.dim):
            if d in self.bivariate_dims:
                continue
            self.univariate_dists[d] = self._create_univariate_dist(d)

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
        Compute log probability of X under the prior.

        Args:
            X: Shape (..., q, dim) - dihedral values

        Returns:
            Log probability, shape (..., q)
        """
        log_prob = torch.zeros(X.shape[:-1], dtype=X.dtype, device=X.device)

        # Univariate contributions
        for d, dist in self.univariate_dists.items():
            if dist is None:
                continue
            angle = self._to_radians(X[..., d])
            log_prob = log_prob + dist.log_prob(angle)

        # Bivariate contributions
        for (d1, d2), dist in self.bivariate_dists.items():
            angle1 = self._to_radians(X[..., d1])
            angle2 = self._to_radians(X[..., d2])
            log_prob = log_prob + dist.log_prob(angle1, angle2)

        return log_prob

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
                if len(match) >= 4 and _matches_dihedral(match[:4], dihedral):
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


def load_priors_from_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load prior definitions from a JSON file.

    Expected format:
    {
        "univariate": {
            "type_id": {
                "description": "...",
                "means_deg": [...],
                "concentrations": [...],
                "weights": [...]
            },
            ...
        },
        "bivariate": {
            "type_id": {
                "description": "...",
                "components": [
                    {"mu1_deg": ..., "mu2_deg": ..., "kappa1": ..., "kappa2": ..., "correlation": ...},
                    ...
                ],
                "weights": [...]
            },
            ...
        },
        "univariate_smarts": [
            {"smarts": "...", "prior_type": "...", "description": "...", "priority": 50},
            ...
        ],
        "bivariate_smarts": [
            {"smarts": "...", "prior_type": "...", "description": "...", "priority": 50},
            ...
        ]
    }

    Returns:
        Dict with keys: univariate, bivariate, univariate_smarts, bivariate_smarts
    """
    filepath = Path(filepath)

    with filepath.open("r") as f:
        data = json.load(f)

    result = {
        "univariate": {},
        "bivariate": {},
        "univariate_smarts": [],
        "bivariate_smarts": [],
    }

    # Parse univariate priors
    for type_id, defn in data.get("univariate", {}).items():
        result["univariate"][_try_int(type_id)] = defn

    # Parse bivariate priors
    for type_id, defn in data.get("bivariate", {}).items():
        result["bivariate"][_try_int(type_id)] = defn

    # Parse SMARTS patterns
    for entry in data.get("univariate_smarts", []):
        result["univariate_smarts"].append(
            UnivariateSMARTS(
                smarts=entry["smarts"],
                prior_type=_try_int(entry["prior_type"]),
                description=entry.get("description", ""),
                priority=entry.get("priority", 50),
            )
        )

    for entry in data.get("bivariate_smarts", []):
        # Parse topology, defaulting to ADJACENT for backward compatibility
        topology_str = entry.get("topology", "adjacent")
        topology = BivariateTopology(topology_str)

        result["bivariate_smarts"].append(
            BivariateSMARTS(
                smarts=entry["smarts"],
                prior_type=_try_int(entry["prior_type"]),
                topology=topology,
                description=entry.get("description", ""),
                priority=entry.get("priority", 50),
            )
        )

    logger.info(
        f"Loaded {len(result['univariate'])} univariate, "
        f"{len(result['bivariate'])} bivariate priors, "
        f"{len(result['univariate_smarts'])} 1D SMARTS, "
        f"{len(result['bivariate_smarts'])} 2D SMARTS from {filepath}"
    )

    return result


def create_prior_module(
    mol: Chem.Mol,
    dihedrals: List[Tuple[int, int, int, int]],
    priors_file: Optional[Union[str, Path]] = None,
    fallback_type: PriorTypeId = "sp3_sp3",
) -> DihedralPriorModule:
    """
    Create a DihedralPriorModule for a molecule.

    Args:
        mol: RDKit molecule
        dihedrals: List of dihedral atom tuples (from detect_dihedrals)
        priors_file: Optional JSON file with custom priors
        fallback_type: Default prior type for unmatched dihedrals

    Returns:
        Configured DihedralPriorModule
    """
    # Load custom priors if provided
    univariate_priors = {}
    bivariate_priors = {}
    custom_uni_smarts = []
    custom_bi_smarts = []

    if priors_file is not None:
        loaded = load_priors_from_json(priors_file)
        univariate_priors = loaded["univariate"]
        bivariate_priors = loaded["bivariate"]
        custom_uni_smarts = loaded["univariate_smarts"]
        custom_bi_smarts = loaded["bivariate_smarts"]

    # Combine built-in and custom SMARTS
    all_uni_smarts = BUILTIN_UNIVARIATE_SMARTS + custom_uni_smarts
    all_bi_smarts = BUILTIN_BIVARIATE_SMARTS + custom_bi_smarts

    # Match patterns to dihedrals
    matcher = DihedralPriorMatcher(all_uni_smarts, all_bi_smarts)
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
        input_in_degrees=True,  # bouquet uses degrees internally
    )
