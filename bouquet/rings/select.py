"""Choosing WHICH ring-state seeds bouquet actually starts a BO from.

The metadynamics returns the *product* of per-ring-system states -- 41 seeds on a 4-system
spiro, ~100 with the current pre-filter -- and bouquet must run a torsional BO from each one
it is given, so the seed set has to be cut to a handful.

**Energy rank is the right default; these alternatives are not.** On the Stage-0 hard core
the hit sat at energy rank 9 twice (m077, m124), which suggested otherwise -- but on the full
166-molecule Stage-1 population the hit is the lowest-energy seed far more often than not
(median energy rank 0, 77 of 81 hits inside a k=8 energy prefix), and the geometry-aware
rules displace good seeds: `energy` 77/81, `hybrid` 74/81, `geometry` 73/81. The Stage-0
subset was selected for exactly the pathology that puts hits at deep ranks, so it could not
generalize. Kept here because they remain the right tool if a future seed set really is
energy-uninformative, and because keeping them makes that testable in one command.

These rules all reorder the SAME seeds -- no extra sampling, no extra quenching -- so any of
them can be scored offline against a finished run (`scripts/ring_seed_selection.py`).

A seed here is any mapping with ``ring_state`` ({system_id: state_id}) and
``rel_energy_kcal``; ``cluster_size`` is used by one rule. That is exactly the
``ring_seeds/1.0`` seed record, so both the live `HarvestResult` and a written seed file can
be fed in.
"""

from __future__ import annotations

RULES = ("energy", "coverage", "novelty", "diversity", "population", "geometry", "hybrid")


def _state(seed) -> dict:
    """``{system_id: state_id}`` with integer keys (JSON turns them into strings)."""
    rs = seed["ring_state"] if isinstance(seed, dict) else seed.ring_state
    return {int(k): int(v) for k, v in rs.items()}


def _energy(seed) -> float:
    return float(seed["rel_energy_kcal"] if isinstance(seed, dict) else seed.rel_energy_kcal)


def _population(seed) -> int:
    if isinstance(seed, dict):
        return int(seed.get("cluster_size", 1))
    return int(getattr(seed, "cluster_size", 1))


def _ring_coords(seed, ring_idx):
    import numpy as np
    c = seed["coords"] if isinstance(seed, dict) else seed.atoms.get_positions()
    return np.asarray(c, dtype=float)[list(ring_idx)]


def select_seeds(seeds, k: int, rule: str = "coverage", ring_idx=None) -> list[int]:
    """Indices of the ``k`` seeds to hand bouquet, in the order they should be used.

    ``energy``     -- lowest relative energy first. The incumbent, and the baseline the
                      others have to beat.
    ``coverage``   -- round-robin over ring systems: each turn, take the lowest-energy seed
                      that shows a state not yet covered *for that system*. Spends the budget
                      on distinct ring geometry rather than on re-ranking one basin.
    ``novelty``    -- lowest-energy seed that introduces a new state in ANY system, then fill
                      by energy. Like ``coverage`` but without the per-system fairness.
    ``diversity``  -- farthest-point in state space (Hamming distance between state tuples),
                      seeded from the lowest-energy seed.
    ``population`` -- largest cluster first: a state the metadynamics revisited often is a
                      wider basin. Tie-broken by energy.
    ``geometry``   -- farthest-point on actual ring-atom GEOMETRY (needs ``ring_idx`` and
                      seeds carrying coordinates). The label-based rules above all degenerate
                      to ``energy`` on a single-ring-system molecule -- every seed is a
                      distinct state of that one system, so "novel state" orders them by
                      energy and nothing moves. That is most of the benchmark's
                      hit-carrying molecules, and it is why m077's rank-9 hit survives no
                      label rule. Geometry keeps discriminating there.
    ``hybrid``     -- half the budget by energy, half by geometric farthest-point. Pure
                      ``geometry`` trades near-in hits for far-out ones; this keeps both.

    Ties everywhere break toward lower energy, so every rule degrades to ``energy`` when the
    molecule has a single ring system in a single state.
    """
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")
    if rule in ("geometry", "hybrid") and ring_idx is None:
        raise ValueError(f"rule {rule!r} needs ring_idx (and seeds carrying coords)")
    n = len(seeds)
    k = min(k, n)
    by_energy = sorted(range(n), key=lambda i: _energy(seeds[i]))
    if k <= 0:
        return []
    if rule == "energy":
        return by_energy[:k]
    if rule == "population":
        return sorted(range(n), key=lambda i: (-_population(seeds[i]), _energy(seeds[i])))[:k]
    if rule in ("geometry", "hybrid"):
        from bouquet.rings.systems import _kabsch_rmsd
        pts = [_ring_coords(seeds[i], ring_idx) for i in range(n)]
        # `hybrid` spends the first half of the budget on the energy ranking and the rest on
        # geometric spread. Pure `geometry` starts from the single lowest-energy seed and
        # then jumps straight to the extremes, which recovers the far-out hits (m124, m077 at
        # energy rank 9) but drops near-in ones that energy would have kept (m020 at rank 1).
        # Reserving half the budget for each keeps both kinds.
        n_energy = max(1, k // 2) if rule == "hybrid" else 1
        picked = list(by_energy[:n_energy])
        dist = [min(_kabsch_rmsd(pts[i], pts[p]) for p in picked) for i in range(n)]
        while len(picked) < k:
            i = max((j for j in range(n) if j not in picked),
                    key=lambda j: (dist[j], -_energy(seeds[j])))
            picked.append(i)
            dist = [min(dist[j], _kabsch_rmsd(pts[j], pts[i])) for j in range(n)]
        return picked

    states = [_state(seeds[i]) for i in range(n)]
    system_ids = sorted({sid for s in states for sid in s})

    if rule == "diversity":
        picked = [by_energy[0]]
        keys = [tuple(s.get(sid) for sid in system_ids) for s in states]

        def hamming(a, b):
            return sum(x != y for x, y in zip(a, b))

        dist = [hamming(keys[i], keys[picked[0]]) for i in range(n)]
        while len(picked) < k:
            # Farthest from everything picked so far; ties to lower energy.
            i = max((i for i in range(n) if i not in picked),
                    key=lambda i: (dist[i], -_energy(seeds[i])))
            picked.append(i)
            dist = [min(dist[j], hamming(keys[j], keys[i])) for j in range(n)]
        return picked

    picked: list[int] = []
    if rule == "novelty":
        seen: dict[int, set] = {sid: set() for sid in system_ids}
        for i in by_energy:                      # one pass, lowest energy first
            if len(picked) >= k:
                break
            if any(states[i].get(sid) not in seen[sid] for sid in system_ids):
                picked.append(i)
                for sid in system_ids:
                    seen[sid].add(states[i].get(sid))
    else:                                        # coverage: per-system round-robin
        seen = {sid: set() for sid in system_ids}
        progress = True
        while len(picked) < k and progress:
            progress = False
            for sid in system_ids:
                if len(picked) >= k:
                    break
                nxt = next((i for i in by_energy
                            if i not in picked and states[i].get(sid) not in seen[sid]), None)
                if nxt is None:
                    continue
                picked.append(nxt)
                for s2 in system_ids:            # a seed covers a state in EVERY system
                    seen[s2].add(states[nxt].get(s2))
                progress = True

    # Every rule must return exactly k seeds when k are available: top up by energy once the
    # rule runs out of new states, so a budget is never spent short.
    for i in by_energy:
        if len(picked) >= k:
            break
        if i not in picked:
            picked.append(i)
    return picked
