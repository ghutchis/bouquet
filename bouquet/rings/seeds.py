"""Read/write the ring-seed JSON contract (``ring_seeds/1.0``).

Serializes a :class:`bouquet.rings.harvest.HarvestResult` into the seed file bouquet
consumes, and reads it back into ASE structures. The one invariant that matters: ``coords``
are in the molecule's input-atom order (RDKit ``AddHs`` order), because bouquet indexes its
torsions off exactly that ordering -- so a seed's coordinates drop straight into
``run_optimization(initial_conformers=...)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "ring_seeds/1.0"
_EV_PER_HARTREE = 27.211386245988


def _jsonable(x):
    """Coerce numpy scalars/arrays in a diagnostics dict to plain JSON types."""
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def seeds_to_dict(mol, systems, result, config, wall_seconds: float | None = None) -> dict:
    """Build the ``ring_seeds/1.0`` dictionary from a run's molecule, systems, and result."""
    from rdkit import Chem

    molh = Chem.AddHs(mol)
    n_states = {s.id: 0 for s in systems}
    for seed in result.seeds:
        for sid, st in seed.ring_state.items():
            n_states[sid] = max(n_states[sid], int(st) + 1)

    return {
        "schema_version": SCHEMA_VERSION,
        "molecule": {
            "smiles": Chem.MolToSmiles(mol),
            "inchikey": Chem.MolToInchiKey(mol),
            "charge": Chem.GetFormalCharge(mol),
            "n_atoms": molh.GetNumAtoms(),
        },
        "method": {
            "engine": config.engine,
            "temperature_K": config.temperature_K,
            "t_run_ps": config.t_run_ps,
            "n_walkers": result.diagnostics.get("n_walkers", config.n_walkers),
            "k_hill_kcal": config.k_hill_kcal,
            "wall_seconds": wall_seconds,
        },
        "ring_systems": [{
            "id": s.id,
            "ring_idx": s.ring_idx.tolist(),
            "dedup_idx": s.dedup_idx.tolist(),
            "spiro_partners": list(s.spiro_partners),
            "n_states_found": n_states[s.id],
        } for s in systems],
        "seeds": [{
            "seed_id": i,
            "ring_state": {str(k): int(v) for k, v in seed.ring_state.items()},
            "energy_hartree": seed.energy_eV / _EV_PER_HARTREE,
            "rel_energy_kcal": seed.rel_energy_kcal,
            "boltzmann_weight_298K": seed.weight,
            "cluster_size": seed.cluster_size,
            "coords": seed.atoms.get_positions().tolist(),
        } for i, seed in enumerate(result.seeds)],
        "diagnostics": _jsonable(result.diagnostics),
    }


def write_seeds(path, mol, systems, result, config, wall_seconds: float | None = None) -> None:
    """Write the ring-seed JSON for a run to ``path``."""
    d = seeds_to_dict(mol, systems, result, config, wall_seconds=wall_seconds)
    Path(path).write_text(json.dumps(d, indent=2))


@dataclass
class RingSeeds:
    """Parsed ring-seed file. ``conformers()`` returns the seeds as ASE structures."""

    data: dict

    @property
    def smiles(self) -> str:
        return self.data["molecule"]["smiles"]

    @property
    def seeds(self) -> list[dict]:
        return self.data["seeds"]

    def ring_states(self) -> list[dict[int, int]]:
        """Per-seed ``{ring_system_id: state_id}`` (JSON string keys cast back to int)."""
        return [{int(k): v for k, v in s["ring_state"].items()} for s in self.seeds]

    def conformers(self):
        """Seed geometries as ASE ``Atoms`` in input-atom order, ready for bouquet."""
        from ase import Atoms
        from rdkit import Chem

        molh = Chem.AddHs(Chem.MolFromSmiles(self.smiles))
        numbers = [a.GetAtomicNum() for a in molh.GetAtoms()]
        return [Atoms(numbers=numbers, positions=np.array(s["coords"], dtype=float))
                for s in self.seeds]


def read_seeds(path) -> RingSeeds:
    """Load a ring-seed JSON; warns on a schema-version mismatch but still parses."""
    import logging
    data = json.loads(Path(path).read_text())
    got = data.get("schema_version")
    if got != SCHEMA_VERSION:
        logging.getLogger(__name__).warning(
            "ring-seed schema %s != expected %s", got, SCHEMA_VERSION)
    return RingSeeds(data)
