"""Tests for bouquet.rings.select (which ring-state seeds bouquet actually starts from).

Pure metadata/geometry logic -- no QM, no RDKit. The behaviour that matters is that a rule
never returns fewer seeds than the budget allows, and that the non-energy rules actually
reorder rather than quietly degrading to the energy ranking.
"""

from __future__ import annotations

import numpy as np
import pytest

from bouquet.rings.select import RULES, select_seeds


def _seed(state, energy, cluster=1, coords=None):
    return {"ring_state": {str(k): v for k, v in state.items()},
            "rel_energy_kcal": energy, "cluster_size": cluster,
            "coords": coords if coords is not None else np.zeros((4, 3)).tolist()}


def _spread_seeds():
    """Six seeds of one ring system; index 5 is the geometric outlier but highest energy."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(4, 3))
    seeds = []
    for i in range(5):
        seeds.append(_seed({0: i}, energy=float(i), coords=(base + 0.01 * i).tolist()))
    seeds.append(_seed({0: 5}, energy=9.0, coords=(base + 6.0).tolist()))
    return seeds


RING_IDX = [0, 1, 2, 3]


@pytest.mark.parametrize("rule", RULES)
def test_every_rule_spends_the_whole_budget_exactly_once(rule):
    seeds = _spread_seeds()
    got = select_seeds(seeds, 4, rule, RING_IDX)
    assert len(got) == 4
    assert len(set(got)) == 4, "a seed was selected twice"
    assert all(0 <= i < len(seeds) for i in got)


@pytest.mark.parametrize("rule", RULES)
def test_budget_larger_than_the_seed_set_is_clamped(rule):
    seeds = _spread_seeds()
    assert len(select_seeds(seeds, 99, rule, RING_IDX)) == len(seeds)
    assert select_seeds(seeds, 0, rule, RING_IDX) == []


def test_energy_rule_is_the_energy_ranking():
    seeds = _spread_seeds()
    assert select_seeds(seeds, 3, "energy") == [0, 1, 2]


def test_coverage_prefers_an_uncovered_system_state_over_a_lower_energy_duplicate():
    # System 1 stays in state 0 for the three cheapest seeds; only seed 3 moves it.
    seeds = [_seed({0: 0, 1: 0}, 0.0), _seed({0: 1, 1: 0}, 1.0),
             _seed({0: 2, 1: 0}, 2.0), _seed({0: 0, 1: 1}, 8.0)]
    got = select_seeds(seeds, 2, "coverage")
    assert 3 in got, "coverage ignored the only seed covering system 1's second state"
    assert got != select_seeds(seeds, 2, "energy")


def test_geometry_reaches_the_outlier_that_energy_ranking_buries():
    """The m124/m077 case: the useful seed is geometrically distinct but high in energy."""
    seeds = _spread_seeds()
    assert 5 not in select_seeds(seeds, 3, "energy")
    assert 5 in select_seeds(seeds, 3, "geometry", RING_IDX)


def test_hybrid_keeps_both_the_cheapest_and_the_outlier():
    seeds = _spread_seeds()
    got = select_seeds(seeds, 4, "hybrid", RING_IDX)
    assert 0 in got, "hybrid dropped the lowest-energy seed"
    assert 5 in got, "hybrid dropped the geometric outlier"


def test_geometry_rules_require_ring_idx():
    seeds = _spread_seeds()
    for rule in ("geometry", "hybrid"):
        with pytest.raises(ValueError, match="ring_idx"):
            select_seeds(seeds, 2, rule)


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError, match="unknown rule"):
        select_seeds(_spread_seeds(), 2, "nope")
