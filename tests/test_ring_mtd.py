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
