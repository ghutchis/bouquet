"""Ring-conformer seeding for bouquet via RMSD-biased metadynamics.

Bouquet's Bayesian optimization searches acyclic torsions only; ring pucker is fixed by
whatever ETKDG produced at t=0. For spiro/polycyclic/macrocyclic systems ETKDG can miss
the ring conformer of the global minimum entirely (measured: 0 of 1000 ETKDG conformers
within 0.25 A of the CREST ring for the hard cases), so the BO cannot recover it. This
subpackage discovers ring conformers with a ring-atom RMSD bias (CREST's mechanism,
narrowed to ring atoms) and hands bouquet a small set of ranked ring-state seeds.

Calculators come from :class:`bouquet.calculator.CalculatorFactory` (``gfnff``, ``gfn2``,
...) -- the driver uses it directly, like the rest of bouquet.

Top-level entry point: :func:`bouquet.rings.mtd.run_ring_mtd`.
"""

from __future__ import annotations

from bouquet.rings.bias import RMSDBias, kabsch_rotation, rmsd_sq_and_grad
from bouquet.rings.harvest import HarvestResult, RingStateSeed, harvest
from bouquet.rings.mtd import RingMTDConfig, run_ring_mtd, should_run
from bouquet.rings.seeds import RingSeeds, read_seeds, write_seeds
from bouquet.rings.systems import (
    RingSystem,
    perceive_ring_systems,
    ring_state_distance,
)

__all__ = [
    "RMSDBias",
    "kabsch_rotation",
    "rmsd_sq_and_grad",
    "RingSystem",
    "perceive_ring_systems",
    "ring_state_distance",
    "harvest",
    "HarvestResult",
    "RingStateSeed",
    "run_ring_mtd",
    "RingMTDConfig",
    "should_run",
    "read_seeds",
    "write_seeds",
    "RingSeeds",
]
