#!/usr/bin/env python3
"""Tests for the dihedral prior path (angle convention, topologies, pair order).

These cover the three things that can silently mis-apply a fitted prior without
raising anything: the degrees->radians convention, the atom maps that define each
bivariate topology, and the orientation of a matched dihedral pair.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bouquet.priors import (
    BivariateAssignment,
    BivariateSMARTS,
    BivariateTopology,
    BivariateVonMisesMixture,
    DihedralPriorModule,
    DihedralPriorMatcher,
    TOPOLOGY_DEFINITIONS,
)

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402


def _grid_peaks(values: torch.Tensor, grid: torch.Tensor) -> list:
    """Local maxima of a periodic 1D scan, in the units of ``grid``."""
    n = len(values)
    return [
        float(grid[i])
        for i in range(n)
        if values[i] > values[i - 1] and values[i] > values[(i + 1) % n]
    ]


# ---------------------------------------------------------------- convention


def test_univariate_peaks_at_declared_means():
    """A declared mean must be scored as a maximum at that same angle.

    ``_to_radians`` used to compute ``radians(x) - pi`` while the means went in
    unshifted, so sp3_sp3's 60/180/300 actually peaked at 240/0/120.
    """
    module = DihedralPriorModule(
        dim=1,
        univariate_assignments={0: "sp3_sp3"},
        bivariate_assignments={},
        univariate_priors={},
        bivariate_priors={},
        max_concentration=None,
    )
    grid = torch.arange(0.0, 360.0, 1.0)
    log_prob = module(grid.unsqueeze(-1))

    peaks = _grid_peaks(log_prob, grid)
    assert sorted(peaks) == pytest.approx([60.0, 180.0, 300.0], abs=1.0)


def test_to_radians_maps_zero_to_zero():
    """0 deg in is 0 rad out -- the input is the dihedral handed to set_dihedral."""
    module = DihedralPriorModule(
        dim=1,
        univariate_assignments={0: "sp3_sp3"},
        bivariate_assignments={},
        univariate_priors={},
        bivariate_priors={},
    )
    x = torch.tensor([0.0, 90.0, 180.0, 270.0])
    got = module._to_radians(x)
    expect = torch.tensor([0.0, math.pi / 2, -math.pi, -math.pi / 2])
    assert torch.allclose(got, expect, atol=1e-9)


def test_normalized_input_matches_degrees():
    """The [0, 1) branch must agree with the degrees branch on the same angles."""
    deg = DihedralPriorModule(
        dim=1,
        univariate_assignments={0: "sp3_sp3"},
        bivariate_assignments={},
        univariate_priors={},
        bivariate_priors={},
        input_in_degrees=True,
    )
    norm = DihedralPriorModule(
        dim=1,
        univariate_assignments={0: "sp3_sp3"},
        bivariate_assignments={},
        univariate_priors={},
        bivariate_priors={},
        input_in_degrees=False,
    )
    x = torch.arange(0.0, 360.0, 7.0)
    assert torch.allclose(deg._to_radians(x), norm._to_radians(x / 360.0), atol=1e-9)


def test_peak_modes_agree_with_forward():
    """Seeding and scoring must not disagree: every reported peak is a maximum."""
    module = DihedralPriorModule(
        dim=1,
        univariate_assignments={0: "sp3_sp3"},
        bivariate_assignments={},
        univariate_priors={},
        bivariate_priors={},
        max_concentration=None,
    )
    axes, _uniform = module.peak_modes()
    (_dims, candidates), = axes

    for (mu,), _w in candidates:
        here = module(torch.tensor([[mu]]))
        for offset in (-15.0, 15.0):
            nearby = module(torch.tensor([[(mu + offset) % 360.0]]))
            assert here > nearby


def test_bivariate_peaks_at_declared_means():
    """Same convention check for the 2D factor, including a correlated component."""
    priors = {
        "test2d": {
            "components": [
                {
                    "mu1_deg": 62.0,
                    "mu2_deg": -118.0,
                    "kappa1": 8.0,
                    "kappa2": 8.0,
                    "correlation": -3.0,
                }
            ],
            "weights": [1.0],
        }
    }
    module = DihedralPriorModule(
        dim=2,
        univariate_assignments={},
        bivariate_assignments={(0, 1): "test2d"},
        univariate_priors={},
        bivariate_priors=priors,
        max_concentration=None,
    )
    grid = torch.arange(0.0, 360.0, 2.0)
    g1, g2 = torch.meshgrid(grid, grid, indexing="ij")
    X = torch.stack([g1.reshape(-1), g2.reshape(-1)], dim=-1)
    best = int(torch.argmax(module(X)))

    assert float(X[best, 0]) == pytest.approx(62.0, abs=2.0)
    assert float(X[best, 1]) % 360.0 == pytest.approx(242.0, abs=2.0)  # -118 mod 360


# ---------------------------------------------------------------- topologies


@pytest.mark.parametrize(
    "topology,min_atoms,dih1,dih2",
    [
        (BivariateTopology.ADJACENT, 5, (0, 1, 2, 3), (1, 2, 3, 4)),
        (BivariateTopology.SKIP_ONE, 6, (0, 1, 2, 3), (2, 3, 4, 5)),
        (BivariateTopology.W_SHAPE, 6, (0, 1, 2, 3), (3, 2, 4, 5)),
        (BivariateTopology.Y_SHAPE, 6, (0, 1, 2, 3), (4, 3, 2, 5)),
    ],
)
def test_topology_atom_maps(topology, min_atoms, dih1, dih2):
    """Pin the atom maps: these mirror PAIRS in the torsion-library generator.

    A fitted prior is measured with those exact quartets, so a change here
    re-points every fitted mixture at a different pair of angles.
    """
    assert TOPOLOGY_DEFINITIONS[topology] == (min_atoms, dih1, dih2)


def test_y_shape_matches_branched_skeleton():
    """y_shape's two dihedrals sit on adjacent bonds off a branch point."""
    _min_atoms, dih1, dih2 = TOPOLOGY_DEFINITIONS[BivariateTopology.Y_SHAPE]
    # Shared central atom, distinct central bonds -- this is what separates
    # y_shape from adjacent (a linear 5-atom chain).
    assert set(dih1[1:3]) & set(dih2[1:3]) == {2}
    assert set(dih1[1:3]) != set(dih2[1:3])


# ---------------------------------------------------------------- pair order


def _matcher(topology: BivariateTopology, smarts: str) -> DihedralPriorMatcher:
    return DihedralPriorMatcher(
        univariate_smarts=[],
        bivariate_smarts=[
            BivariateSMARTS(smarts=smarts, prior_type="p", topology=topology)
        ],
    )


def test_pair_order_follows_topology_not_dihedral_index():
    """The key is (dim_of_dih1, dim_of_dih2) -- not sorted.

    ``forward`` feeds ``X[..., d1]`` as phi1, so sorting the key applies the
    transposed mixture whenever the topology's second dihedral happens to come
    first in the caller's dihedral list.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCCC"))
    smarts = "[C:1][C:2][C:3][C:4][C:5]"
    matcher = _matcher(BivariateTopology.ADJACENT, smarts)

    match = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))[0]
    dih1 = tuple(match[i] for i in (0, 1, 2, 3))
    dih2 = tuple(match[i] for i in (1, 2, 3, 4))

    # Present the dihedrals in the *reverse* of the topology's order.
    dihedrals = [dih2, dih1]
    result = matcher.match_bivariate(mol, dihedrals)

    assert set(result) == {(1, 0)}, "orientation must survive the dihedral ordering"
    assert result[(1, 0)].prior_type == "p"


def test_pair_claimed_once_per_orientation():
    """One physical pair must not be recorded twice under two orientations."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCCC"))
    smarts = "[C:1][C:2][C:3][C:4][C:5]"
    matcher = _matcher(BivariateTopology.ADJACENT, smarts)

    match = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))[0]
    dihedrals = [
        tuple(match[i] for i in (0, 1, 2, 3)),
        tuple(match[i] for i in (1, 2, 3, 4)),
    ]
    result = matcher.match_bivariate(mol, dihedrals)

    assert len(result) == 1
    assert set(result) in ({(0, 1)}, {(1, 0)})


def test_transposed_assignment_changes_the_density():
    """Guards the reason order matters: an asymmetric mixture is not symmetric."""
    priors = {
        "asym": {
            "components": [
                {
                    "mu1_deg": 60.0,
                    "mu2_deg": 180.0,
                    "kappa1": 20.0,
                    "kappa2": 2.0,
                    "correlation": 1.5,
                }
            ],
            "weights": [1.0],
        }
    }
    kwargs = dict(
        dim=2,
        univariate_assignments={},
        univariate_priors={},
        bivariate_priors=priors,
        max_concentration=None,
    )
    forward = DihedralPriorModule(bivariate_assignments={(0, 1): "asym"}, **kwargs)
    transposed = DihedralPriorModule(bivariate_assignments={(1, 0): "asym"}, **kwargs)

    X = torch.tensor([[60.0, 180.0]])
    assert not torch.allclose(forward(X), transposed(X))


# ---------------------------------------------------------------- kappa cap


def test_max_concentration_clamps_bivariate_kappa():
    """A near-delta histogram fit becomes a finite-width basin for the optimizer."""
    mixture = BivariateVonMisesMixture(
        components=[
            {"mu1_deg": 0.0, "mu2_deg": 0.0, "kappa1": 9000.0, "kappa2": 3.0}
        ],
        weights=[1.0],
        max_concentration=50.0,
    )
    assert float(mixture.kappa1[0]) == pytest.approx(50.0)
    assert float(mixture.kappa2[0]) == pytest.approx(3.0)


# ---------------------------------------------------------------- terminal offset


def _embedded(smiles: str, seed: int = 3):
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def test_offset_is_invariant_under_rotation():
    """The offset between two terminal choices is fixed by rigid local geometry.

    This is what makes the correction exact rather than an approximation: the
    four atoms of a quartet are joined by direct bonds, so no rotatable bond
    lies between them and no torsion can change their relative arrangement.
    """
    from rdkit.Chem import rdMolTransforms as rmt

    from bouquet.setup import detect_dihedrals

    mol = _embedded("CC(c1ccccc1)c1ccccc1")
    conf = mol.GetConformer()
    smarts = "[a:1][c:2]!@[C&X4&H1:3]([a:4][a:5])[#1:6]"  # y1665
    match = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))[0]
    q_hist = tuple(match[i] for i in (0, 1, 2, 3))

    dihedrals = [tuple(d.chain) for d in detect_dihedrals(mol)]
    q_bq = next(d for d in dihedrals if set(d[1:3]) == set(q_hist[1:3]))

    offsets = []
    for angle in range(0, 360, 20):
        rmt.SetDihedralDeg(conf, *q_bq, float(angle))
        raw = rmt.GetDihedralDeg(conf, *q_bq) - rmt.GetDihedralDeg(conf, *q_hist)
        offsets.append((raw + 180.0) % 360.0 - 180.0)  # wrap, as the code does
    spread = max(offsets) - min(offsets)
    assert spread < 1e-6, f"offset drifted by {spread} deg under rotation"


def test_hydrogen_terminal_pattern_still_matches():
    """A prior whose SMARTS ends on H must still find a pair.

    bouquet's rotor list is heavy-atom only, so requiring an exact quartet made
    this whole class of fitted priors unable to fire.
    """
    from bouquet.setup import detect_dihedrals

    mol = _embedded("CC(c1ccccc1)c1ccccc1")
    smarts = "[a:1][c:2]!@[C&X4&H1:3]([a:4][a:5])[#1:6]"  # y1665
    matcher = _matcher(BivariateTopology.Y_SHAPE, smarts)

    dihedrals = [tuple(d.chain) for d in detect_dihedrals(mol)]
    result = matcher.match_bivariate(mol, dihedrals)

    assert result, "a hydrogen-terminated quartet must still match its rotor"
    assignment = next(iter(result.values()))
    assert isinstance(assignment, BivariateAssignment)
    assert assignment.offset1_deg != 0.0 or assignment.offset2_deg != 0.0


def test_offset_puts_the_mode_where_the_prior_expects_it():
    """Scoring must peak at the fitted mean expressed in bouquet's own frame."""
    from rdkit.Chem import rdMolTransforms as rmt

    from bouquet.setup import detect_dihedrals

    mol = _embedded("CC(c1ccccc1)c1ccccc1")
    conf = mol.GetConformer()
    smarts = "[a:1][c:2]!@[C&X4&H1:3]([a:4][a:5])[#1:6]"
    matcher = _matcher(BivariateTopology.Y_SHAPE, smarts)
    dihedrals = [tuple(d.chain) for d in detect_dihedrals(mol)]
    assignments = matcher.match_bivariate(mol, dihedrals)
    (d1, d2), assignment = next(iter(assignments.items()))

    priors = {
        "p": {
            "components": [
                {
                    "mu1_deg": 75.0,
                    "mu2_deg": 200.0,
                    "kappa1": 30.0,
                    "kappa2": 30.0,
                    "correlation": 0.0,
                }
            ],
            "weights": [1.0],
        }
    }
    module = DihedralPriorModule(
        dim=len(dihedrals),
        univariate_assignments={},
        bivariate_assignments=assignments,
        univariate_priors={},
        bivariate_priors=priors,
        fallback_type="uniform",
        max_concentration=None,
    )

    # Drive the molecule to the geometry the prior calls its mode, then read the
    # search coordinates off that geometry.
    match = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))[0]
    _m, i1, i2 = TOPOLOGY_DEFINITIONS[BivariateTopology.Y_SHAPE]
    q1 = tuple(match[i] for i in i1)
    q2 = tuple(match[i] for i in i2)
    rmt.SetDihedralDeg(conf, *q1, 75.0)
    rmt.SetDihedralDeg(conf, *q2, 200.0)

    X = torch.zeros(1, len(dihedrals), dtype=torch.float64)
    for k, q in enumerate(dihedrals):
        X[0, k] = rmt.GetDihedralDeg(conf, *q) % 360.0

    at_mode = module(X)
    for delta in (-40.0, 40.0):
        off = X.clone()
        off[0, d1] = (off[0, d1] + delta) % 360.0
        assert at_mode > module(off), f"prior did not peak at its mode ({delta=})"

    # And the offsets are what carried it there.
    assert abs(assignment.offset1_deg) > 1.0
