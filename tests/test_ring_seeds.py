"""Tests for bouquet.rings.seeds (JSON contract) and bouquet.rings.cli.

All fast: the seed schema is exercised with a hand-built HarvestResult, and the CLI is
driven on an acyclic molecule (which the gate declines, so no metadynamics runs).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from bouquet.rings.cli import build_config, main  # noqa: E402
from bouquet.rings.harvest import HarvestResult, RingStateSeed  # noqa: E402
from bouquet.rings.mtd import RingMTDConfig  # noqa: E402
from bouquet.rings.seeds import SCHEMA_VERSION, read_seeds, write_seeds  # noqa: E402
from bouquet.rings.systems import perceive_ring_systems  # noqa: E402
from bouquet.setup import mol_to_ase_atoms  # noqa: E402


def _fake_result(molh, cids):
    seeds = [
        RingStateSeed(atoms=mol_to_ase_atoms(molh, cids[0]), energy_eV=-1000.0,
                      ring_state={0: 0}, cluster_size=3, rel_energy_kcal=0.0, weight=0.7),
        RingStateSeed(atoms=mol_to_ase_atoms(molh, cids[1]), energy_eV=-999.9,
                      ring_state={0: 1}, cluster_size=1, rel_energy_kcal=0.5, weight=0.3),
    ]
    return HarvestResult(seeds=seeds, diagnostics={"n_walkers": 2, "n_clusters": 2,
                                                   "stereo_reject_rate": 0.0})


def test_seed_roundtrip(tmp_path):
    # The serialized atom identities must track the SEED-COORDINATE order (whatever it is),
    # not an order re-derived from the SMILES at read time. Build the seeds in an order that
    # deliberately differs from AddHs(canonical-SMILES) to prove readback uses the stored one.
    mol = Chem.MolFromSmiles("OC1CCC(N)CC1")
    seed_mol = Chem.AddHs(mol)                                   # the order seeds are built in
    coord_order = [a.GetAtomicNum() for a in seed_mol.GetAtoms()]
    canon_order = [a.GetAtomicNum()
                   for a in Chem.AddHs(Chem.MolFromSmiles(Chem.MolToSmiles(mol))).GetAtoms()]
    assert coord_order != canon_order, "pick a molecule whose canonical AddHs order differs"

    p = AllChem.ETKDGv3(); p.randomSeed = 1
    cids = AllChem.EmbedMultipleConfs(seed_mol, numConfs=2, params=p)
    AllChem.MMFFOptimizeMoleculeConfs(seed_mol)
    systems = perceive_ring_systems(seed_mol)
    result = _fake_result(seed_mol, cids)

    out = tmp_path / "seeds.json"
    write_seeds(out, mol, systems, result, RingMTDConfig(), wall_seconds=1.2)

    data = json.loads(out.read_text())
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["molecule"]["n_atoms"] == seed_mol.GetNumAtoms()
    # stored atomic_numbers track the seed-coordinate order, NOT an order re-derived from SMILES
    assert data["molecule"]["atomic_numbers"] == coord_order
    assert data["ring_systems"][0]["n_states_found"] == 2      # states 0 and 1 seen
    assert data["seeds"][0]["boltzmann_weight_298K"] == 0.7
    assert data["method"]["wall_seconds"] == 1.2

    rs = read_seeds(out)
    assert rs.smiles
    assert rs.ring_states() == [{0: 0}, {0: 1}]
    confs = rs.conformers()
    assert len(confs) == 2
    assert len(confs[0]) == seed_mol.GetNumAtoms()
    # atom identities and coords both come back in the stored (coordinate) order
    assert list(confs[0].get_atomic_numbers()) == coord_order
    np.testing.assert_allclose(confs[0].get_positions(),
                               result.seeds[0].atoms.get_positions(), atol=1e-6)


def test_build_config_overrides_only_given_flags():
    args = build_config_args(walkers=8, engine="gfn2", topology_check=True)
    cfg = build_config(args)
    assert cfg.n_walkers == 8
    assert cfg.engine == "gfn2"
    assert cfg.topology_check is True
    # untouched flags keep defaults
    assert cfg.temperature_K == RingMTDConfig().temperature_K
    assert cfg.k_hill_kcal == RingMTDConfig().k_hill_kcal


def test_cli_acyclic_writes_empty_seed_set(tmp_path):
    out = tmp_path / "acyclic.json"
    rc = main(["--smiles", "CCCCO", "--out", str(out), "--quiet"])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["seeds"] == []
    assert data["diagnostics"].get("skipped")


def test_cli_bad_smiles_returns_error(tmp_path):
    rc = main(["--smiles", "not_a_smiles)))", "--out", str(tmp_path / "x.json"), "--quiet"])
    assert rc == 2


# -- helper to fabricate a parsed-args namespace for build_config -----------------
class build_config_args:
    def __init__(self, **kw):
        defaults = dict(engine=None, temperature=None, t_run_ps=None, t_equil_ps=None,
                        walkers=None, processes=None, k_hill=None, energy_window=None,
                        topology_check=None, seed=None)
        defaults.update(kw)
        self.__dict__.update(defaults)


# --- selection at consumption time ------------------------------------------------

def _seed_file(tmp_path, n=6):
    """A seed file with n distinct ring states, energy-ranked, for selection tests."""
    mol = Chem.MolFromSmiles("OC1CCC(N)CC1")
    molh = Chem.AddHs(mol)
    p = AllChem.ETKDGv3(); p.randomSeed = 3
    cids = list(AllChem.EmbedMultipleConfs(molh, numConfs=n, params=p))
    AllChem.MMFFOptimizeMoleculeConfs(molh)
    seeds = [RingStateSeed(atoms=mol_to_ase_atoms(molh, c), energy_eV=-1000.0 + 0.1 * i,
                           ring_state={0: i}, cluster_size=1,
                           rel_energy_kcal=float(i), weight=1.0 / len(cids))
             for i, c in enumerate(cids)]
    result = HarvestResult(seeds=seeds, diagnostics={"n_walkers": 1, "n_clusters": len(cids)})
    out = tmp_path / "sel.json"
    write_seeds(out, mol, perceive_ring_systems(molh), result, RingMTDConfig())
    return out, len(cids)


def test_conformers_still_returns_the_whole_record(tmp_path):
    """The file stays the complete set -- that is what keeps rules scorable offline."""
    out, n = _seed_file(tmp_path)
    assert len(read_seeds(out).conformers()) == n


def test_select_returns_a_budgeted_subset(tmp_path):
    out, n = _seed_file(tmp_path)
    s = read_seeds(out)
    for rule in ("energy", "hybrid"):
        got = s.select(3, rule)
        assert len(got) == 3
        assert all(len(c) == len(s.conformers()[0]) for c in got)
    assert len(s.select(99, "hybrid")) == n     # budget above the pool is clamped


def test_select_rules_can_differ(tmp_path):
    """hybrid must not silently degrade to the energy prefix on a multi-state file."""
    out, _n = _seed_file(tmp_path)
    s = read_seeds(out)
    from bouquet.rings.select import select_seeds
    e = select_seeds(s.seeds, 3, "energy", s.ring_idx)
    h = select_seeds(s.seeds, 3, "hybrid", s.ring_idx)
    assert e == [0, 1, 2]
    assert set(h) != set(e), "hybrid picked the energy prefix; geometry had no effect"


def test_get_conformers_from_file_applies_the_budget(tmp_path):
    """The path bouquet actually takes for --conformer-file."""
    from bouquet.setup import get_conformers_from_file

    out, n = _seed_file(tmp_path)
    assert len(get_conformers_from_file(str(out))) == n                     # default: all
    assert len(get_conformers_from_file(str(out), max_conformers=2)) == 2
    got = get_conformers_from_file(str(out), max_conformers=2, select_rule="energy")
    assert len(got) == 2
    assert all(c.info.get("charge") == 0 for c in got)
