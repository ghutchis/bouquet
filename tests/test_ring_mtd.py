"""Tests for bouquet.rings.mtd (the ring-MTD driver).

Fast tests cover the gate and walker seeding (no QM). The end-to-end run is @slow -- it
runs real GFN-FF metadynamics -- and is the §11.3 integration check that perceive -> seed
-> walk -> harvest wires together and produces stereo-valid, labelled ring-state seeds.
"""

from __future__ import annotations

import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402

from bouquet.rings.mtd import (  # noqa: E402
    RingMTDConfig, run_ring_mtd, seed_walkers, should_run,
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
    starts, mult = seed_walkers(smi, systems, cfg, charge=0)
    assert len(starts) == 4                       # cyclically padded to n_walkers
    assert all(len(s) == molh.GetNumAtoms() for s in starts)
    assert isinstance(mult, int) and mult >= 1
    # charge/spin stamped so xtb reads the state per-eval
    assert all(s.get_initial_charges().sum() == pytest.approx(0.0) for s in starts)


def test_run_ring_mtd_acyclic_is_empty():
    """No QM: acyclic short-circuits before any walker runs."""
    res = run_ring_mtd("CCCCO")
    assert res.seeds == []
    assert res.diagnostics.get("skipped")


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
