"""Tests for bouquet.rings.mtd (the ring-MTD driver).

Fast tests cover the gate and walker seeding (no QM). The end-to-end run is @slow -- it
runs real GFN-FF metadynamics -- and is the §11.3 integration check that perceive -> seed
-> walk -> harvest wires together and produces stereo-valid, labelled ring-state seeds.
"""

from __future__ import annotations

import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402

from bouquet.rings.harvest import HarvestResult  # noqa: E402
from bouquet.rings.mtd import (  # noqa: E402
    RingMTDConfig, _adapt_k_hill, _better, _verdict, run_ring_mtd, seed_walkers, should_run,
)
from bouquet.rings.systems import perceive_ring_systems  # noqa: E402


# --- gate ------------------------------------------------------------------------

def test_should_run_gate():
    assert should_run(Chem.MolFromSmiles("C1CCCCC1"))          # cyclohexane
    assert should_run(Chem.MolFromSmiles("C1CC2(CC1)CCCCC2"))  # spiro
    assert not should_run(Chem.MolFromSmiles("c1ccccc1"))      # aromatic only
    assert not should_run(Chem.MolFromSmiles("CCCCO"))         # acyclic


# --- walker seeding (ETKDG + MMFF only, no QM) ------------------------------------

def test_seed_walkers_count_and_stamp():
    smi = "CC1CCCCC1"
    molh = Chem.AddHs(Chem.MolFromSmiles(smi))
    systems = perceive_ring_systems(molh)
    cfg = RingMTDConfig(n_walkers=4, max_embeddings=20, seed=1)
    starts, mult = seed_walkers(molh, systems, cfg, charge=0)
    assert len(starts) == 4                       # cyclically padded to n_walkers
    assert all(len(s) == molh.GetNumAtoms() for s in starts)
    assert isinstance(mult, int) and mult >= 1
    # charge/spin stamped so xtb reads the state per-eval
    assert all(s.get_initial_charges().sum() == pytest.approx(0.0) for s in starts)


@pytest.mark.parametrize("smi", [
    # m077_d04, m095_d04, m109_d05, m120_d05 -- every one permutes through a SMILES
    # round-trip, and every one showed a k-independent stereo-reject rate because of it.
    "O=C1N2c3c(C(=O)[C@@]2(O)CCN(c2c1cccc2)C(=O)OC)cccc3",
    "C[C@@H]1CCCC(=O)CCC/C=C\\c2cc3c(c(c2C(=O)O1)O)[C@H](CC(=O)O3)c1cccc(c1)O",
    "COc1ccc(cc1O[C@H]2C[C@@H]3CC[C@H]2C3)C4CNC(=O)NC4",
    "C1CC2(C1)NCC[C@@H]2N[C@@H]1CCC2(C1)OCCO2",
])
def test_seed_walkers_preserves_caller_atom_order(smi):
    """Walker starts must be indexed like the caller's molh -- no SMILES round-trip.

    Regression for RING_MTD_BENCH §9.6: seed_walkers used to rebuild the molecule from
    ``MolToSmiles(mol, canonical=False)``, which does NOT preserve atom order (a SMILES is
    a DFS traversal). ``systems``' ring indices and the stereo signature are indices into
    the caller's molh, so a permuted start silently reads the wrong atoms.
    """
    molh = Chem.AddHs(Chem.MolFromSmiles(smi))
    systems = perceive_ring_systems(molh)
    cfg = RingMTDConfig(n_walkers=1, max_embeddings=4, seed=0)
    starts, _mult = seed_walkers(molh, systems, cfg, charge=0)
    expected = [a.GetAtomicNum() for a in molh.GetAtoms()]
    assert list(starts[0].get_atomic_numbers()) == expected
    # The bug is invisible in the element sequence alone for some molecules; the geometry
    # must also reproduce molh's own bonding, so check a bonded pair stays bonded.
    b = molh.GetBondWithIdx(0)
    i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
    assert starts[0].get_distance(i, j) < 2.0


def test_run_ring_mtd_acyclic_is_empty():
    """No QM: acyclic short-circuits before any walker runs.

    This is the ONE empty result that is legitimate (the gate declined) and so must not
    raise RingMTDError -- everything else empty means the bias destroyed the run.
    """
    res = run_ring_mtd("CCCCO")
    assert res.seeds == []
    assert res.diagnostics.get("skipped")


# --- adaptive bias controller (no QM: pure decision logic) ------------------------

def test_verdict_classifies_both_failure_modes():
    cfg = RingMTDConfig()
    strong = {"stereo_reject_rate": 1.0, "n_clusters": 0}      # m077/m095's Stage-0 signature
    weak = {"stereo_reject_rate": 0.0, "n_clusters": 1}        # m009/m146/m147's
    ok = {"stereo_reject_rate": 0.0, "n_clusters": 12}
    assert _verdict(strong, cfg) == "too_strong"
    assert _verdict(weak, cfg) == "too_weak"
    assert _verdict(ok, cfg) == "ok"
    # A rate above harvest's 5% ALARM but below the action ceiling is not worth a re-run.
    assert _verdict({"stereo_reject_rate": 0.10, "n_clusters": 9}, cfg) == "ok"


def test_adapt_k_hill_brackets_and_converges():
    """Opposite verdicts must bracket, not oscillate -- the reason this bisects."""
    cfg = RingMTDConfig()
    lo, hi = cfg.k_hill_bounds[0], float("inf")

    k1, lo, hi = _adapt_k_hill(0.5, "too_strong", lo, hi, cfg)
    assert k1 < 0.5 and hi == 0.5           # capped from above
    k2, lo, hi = _adapt_k_hill(k1, "too_weak", lo, hi, cfg)
    assert lo == k1 and k1 < k2 < 0.5       # floored from below, next probe inside the bracket
    k3, lo, hi = _adapt_k_hill(k2, "too_strong", lo, hi, cfg)
    assert k1 < k3 < k2                     # bracket keeps shrinking, no return to 0.5


def test_adapt_k_hill_respects_bounds():
    cfg = RingMTDConfig(k_hill_bounds=(0.1, 1.0))
    k, _lo, _hi = _adapt_k_hill(1.0, "too_weak", 1.0, float("inf"), cfg)
    assert k == 1.0                          # escalation clamped at the ceiling
    k, _lo, _hi = _adapt_k_hill(0.1, "too_strong", 0.0, 0.1, cfg)
    assert k == 0.1                          # back-off clamped at the floor


def test_better_prefers_healthy_then_more_states():
    ok_small = HarvestResult(seeds=[1], diagnostics={"stereo_reject_rate": 0.0})
    bad_big = HarvestResult(seeds=[1, 2, 3], diagnostics={"stereo_reject_rate": 0.9})
    # A healthy round beats an unhealthy one even with fewer seeds ...
    assert _better(ok_small, "ok", bad_big, "too_strong")
    assert not _better(bad_big, "too_strong", ok_small, "ok")
    # ... and among equals, more ring states wins.
    ok_big = HarvestResult(seeds=[1, 2, 3], diagnostics={"stereo_reject_rate": 0.0})
    assert _better(ok_big, "ok", ok_small, "ok")


# --- end-to-end (slow: real GFN-FF MD) -------------------------------------------

@pytest.mark.slow
def test_run_ring_mtd_cyclohexane_end_to_end():
    cfg = RingMTDConfig(n_walkers=2, t_equil_ps=0.5, t_run_ps=2.0,
                        deposit_interval_ps=0.5, snapshot_interval_ps=0.5,
                        n_processes=1, seed=1)
    res = run_ring_mtd("C1CCCCC1", cfg)
    assert res.seeds, "expected at least one ring-state seed"
    # every seed carries a per-system state label and a normalised weight
    sid = 0
    for s in res.seeds:
        assert sid in s.ring_state
    assert abs(sum(s.weight for s in res.seeds) - 1.0) < 1e-6
    assert res.seeds[0].rel_energy_kcal == pytest.approx(0.0, abs=1e-9)
    d = res.diagnostics
    assert d["n_walkers"] == 2 and d["n_ring_systems"] == 1
    assert "stereo_reject_rate" in d


@pytest.mark.slow
def test_permuting_molecule_harvests_without_stereo_rejects():
    """Regression: m095_d04 rejected 100% of its frames at every hill height.

    Cause was neither the bias nor the surface -- ``seed_walkers`` rebuilt the molecule from
    a SMILES, which permuted its atoms relative to the caller's ``molh``, so the stereo
    signature was read off the wrong atoms. This molecule is one of the permuting cases
    (see ``test_seed_walkers_preserves_caller_atom_order``); end-to-end it must now harvest
    seeds with essentially no rejects.
    """
    smi = ("C[C@@H]1CCCC(=O)CCC/C=C\\c2cc3c(c(c2C(=O)O1)O)[C@H](CC(=O)O3)"
           "c1cccc(c1)O")
    cfg = RingMTDConfig(n_walkers=2, t_equil_ps=0.5, t_run_ps=3.0,
                        deposit_interval_ps=0.5, snapshot_interval_ps=0.5,
                        n_processes=1, seed=0)
    res = run_ring_mtd(smi, cfg)
    assert res.seeds, "correct atom order must yield seeds (was 0 -> RingMTDError)"
    assert res.diagnostics["stereo_reject_rate"] < 0.25, res.diagnostics


# --- §11.3 physics gate (slow: real GFN-FF MD) -----------------------------------
#
# These check that the driver reproduces textbook ring energetics, not just that it
# runs. GFN-FF gets all three right (verified deterministic across seeds):
#   * cyclohexane twist-boat ~5.2 kcal/mol above the chair (expt ~5.5),
#   * trans-decalin is conformationally locked (one low ring state),
#   * trans-decalin sits ~2.2 kcal/mol below cis (expt ~2.7).
#
# Decalin label subtlety: for these SMILES the bridgehead H-C-C-H dihedral (NOT the CIP
# labels, which are unreliable for this symmetric fused system) is the ground truth --
# [C@H]..[C@H] is CIS (~-44 deg), [C@H]..[C@@H] is TRANS (180 deg). Do not "fix" them.
_CIS_DECALIN = "C1CC[C@H]2CCCC[C@H]2C1"
_TRANS_DECALIN = "C1CC[C@H]2CCCC[C@@H]2C1"
_EV_TO_KCAL = 23.0605


def _gate_cfg(seed: int = 1) -> RingMTDConfig:
    """Lean config that reliably crosses the chair-flip barrier (~18 s/molecule)."""
    return RingMTDConfig(n_walkers=2, n_processes=1, t_equil_ps=0.5, t_run_ps=6.0,
                         deposit_interval_ps=1.0, snapshot_interval_ps=0.25, seed=seed)


@pytest.mark.slow
def test_cyclohexane_twist_boat_energy():
    """Cyclohexane yields a second ring state (twist-boat) 4-8 kcal/mol above the chair."""
    res = run_ring_mtd("C1CCCCC1", _gate_cfg())
    assert res.seeds[0].rel_energy_kcal == pytest.approx(0.0, abs=1e-9)  # chair is the min
    excited = [s.rel_energy_kcal for s in res.seeds if s.rel_energy_kcal > 1.0]
    assert any(4.0 <= e <= 8.0 for e in excited), f"no twist-boat in 4-8 kcal: {excited}"


@pytest.mark.slow
def test_trans_decalin_is_conformationally_locked():
    """trans-decalin cannot ring-flip: exactly one ring state within a tight window.

    (Higher twist-boat excited states may appear above ~5 kcal/mol; the point is that
    there is no low-lying second chair -- unlike cis, which is flexible.)"""
    res = run_ring_mtd(_TRANS_DECALIN, _gate_cfg())
    low = [s for s in res.seeds if s.rel_energy_kcal <= 3.0]
    assert len(low) == 1, f"trans-decalin should have one low ring state, got {low}"


@pytest.mark.slow
def test_decalin_trans_below_cis_and_cis_more_flexible():
    """GFN-FF ranks the diastereomers correctly (trans < cis) and cis is more flexible."""
    trans = run_ring_mtd(_TRANS_DECALIN, _gate_cfg())
    cis = run_ring_mtd(_CIS_DECALIN, _gate_cfg())
    delta = (cis.seeds[0].energy_eV - trans.seeds[0].energy_eV) * _EV_TO_KCAL
    assert delta > 0.5, f"trans should sit below cis, got cis - trans = {delta:.2f} kcal"
    # cis interconverts between chair-chair forms + twist-boats -> more ring states
    assert len(cis.seeds) > len(trans.seeds)
