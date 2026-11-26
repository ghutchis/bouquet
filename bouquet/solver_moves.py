"""Collective (non-local) proposal moves for high-dimensional conformer search.

Two alternatives to a standard axis-wise BO step, both a committed geometry change
followed by an UNCONSTRAINED relaxation so the dihedrals can slide along the curved
fold valley a dihedral-pinned step cannot cross:

* Phase 2.5 low-mode / basin-hopping (:func:`_low_mode_move`) -- a kick along a
  data-derived (position-PCA) soft mode;
* Phase 3 category-tied move (:func:`_category_move`) -- a low-dimensional move over
  per-SMARTS-category dihedral values (a chemistry-defined REMBO embedding).

Both reach back into :mod:`bouquet.surrogate` for the reduced-space selector but only
reference :class:`bouquet.solver.OptimizationState` for typing (see TYPE_CHECKING).
"""

from __future__ import annotations

# See bouquet/__init__.py for the lazy-import rationale.
__lazy_modules__ = ["numpy", "torch"]

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator

from bouquet.assess import _reset_calculators, evaluate_energy, relax_structure
from bouquet.config import RELAX_FAILURE_ENERGY_EV
from bouquet.setup import DihedralInfo, bonds_broken, bonds_formed
from bouquet.surrogate import _select_next_points_botorch

if TYPE_CHECKING:
    from bouquet.solver import OptimizationState


# Phase 2.5 low-mode move relaxation budgets. The constrained pre-relax removes the
# worst clashes at the kicked dihedrals; the UNCONSTRAINED relax then lets every DOF
# (dihedrals included) slide to the nearest local minimum -- the step that bends a
# straight kick onto the curved fold valley (the straight-line dihedral path is
# clash-barrier-blocked; see the Phase 2.4 diagnostics in bouquet_hdbo_plan.md).
_LOWMODE_CONSTRAINED_STEPS = 20
# Unconstrained relax budget for a collective (low-mode / category) move.
_LOWMODE_FREE_STEPS = 100


def _collective_kick_relax(
    target_deg: np.ndarray,
    anchor: Atoms,
    dihedrals: list[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
) -> tuple[float, Atoms]:
    """Shared collective-move evaluation: set the proposed dihedrals and short CONSTRAINED
    relax (clash cleanup), then release the constraint and relax UNCONSTRAINED to a local
    minimum -- so the geometry can slide along the curved fold valley a standard, dihedral-
    pinned BO step cannot cross. Used by both the low-mode kick and the category broadcast.
    Returns (energy [eV], relaxed atoms)."""
    _, atoms = evaluate_energy(
        target_deg, anchor, dihedrals, calc, relaxCalc,
        relax=True, steps=_LOWMODE_CONSTRAINED_STEPS,
    )
    atoms.set_constraint()
    return relax_structure(atoms, calc, relaxCalc, _LOWMODE_FREE_STEPS)


def _collective_clash_reject(state: OptimizationState, target_deg: np.ndarray) -> bool:
    """True if a proposed collective geometry fails the clash feasibility region.

    Collective moves must be screened by the SAME region the acquisition optimizer
    uses. They were not, originally, and that was the one hole left in the guards:
    a low-mode kick is a committed geometry change followed by an UNCONSTRAINED
    relaxation, which is precisely where a clashed structure collapses -- and a
    collapsed structure poisons xtb's restart wavefunction for the remainder of the
    run (see assess._reset_calculators). Low-mode auto-enables at
    HIGH_D_DIHEDRAL_THRESHOLD, so the hole opened exactly at the high d it was
    meant to help: two d=12 molecules in the kernel A/B collapsed mid-run and
    stayed collapsed in every arm, reporting energies near -6000 kcal/mol.
    """
    if state.clash_feasibility is None:
        return False
    feasible = bool(
        state.clash_feasibility(np.asarray(target_deg, dtype=float)[None, :])[0]
    )
    if not feasible:
        state.n_collective_clash_rejects += 1
    return not feasible


def _collective_result(
    state: OptimizationState,
    atoms: Atoms,
    energy: float,
    dihedrals: list[DihedralInfo],
    calc: Calculator | None = None,
    relaxCalc: Calculator | None = None,
) -> tuple[np.ndarray, Atoms, float] | None:
    """Finish a collective move: reject it (return None) if the UNCONSTRAINED relax
    changed connectivity, else return (relaxed dihedrals [deg], atoms, energy).

    The same species guard ``_evaluate_point`` applies to ordinary BO steps, and for
    the same reason -- but it matters MORE here, because a collective move relaxes
    without dihedral constraints and so has far more freedom to collapse. Bond
    FORMATION is checked by default; bond BREAKING stays behind --retain-bonds, as
    on the BO path. A rejected geometry also resets the calculators, since it may
    have left a broken SCF solution that would corrupt every later evaluation.
    """
    rejected = False
    if state.required_bonds is not None and energy < RELAX_FAILURE_ENERGY_EV:
        if state.check_bond_forms and bonds_formed(atoms, state.required_bonds):
            state.n_bond_forms += 1
            rejected = True
        elif state.check_bond_breaks and bonds_broken(atoms, state.required_bonds):
            state.n_bond_breaks += 1
            rejected = True
    if rejected or energy >= RELAX_FAILURE_ENERGY_EV:
        _reset_calculators(*(c for c in (calc, relaxCalc) if c is not None))
    if rejected:
        return None
    new_coords = np.array([di.get_angle(atoms) for di in dihedrals])
    return new_coords, atoms, energy


def _low_mode_move(
    state: OptimizationState,
    dihedrals: list[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
    rng: np.random.Generator,
) -> tuple[np.ndarray, Atoms, float] | None:
    """One low-mode / basin-hopping move: kick the incumbent along a soft (position-PCA)
    direction, then relax UNCONSTRAINED, returning (relaxed dihedrals [deg], relaxed
    atoms, energy [eV]).

    Unlike a standard BO step -- which relaxes only the non-dihedral DOF with the
    dihedrals pinned -- the unconstrained relaxation lets the dihedrals move, so the
    geometry can slide along the curved fold valley instead of stalling on the
    straight-line clash ridge (Kolossvary-Guida low-mode search). Returns ``None`` to
    fall back to a standard step (too few elite points, or a move that broke bonds)."""
    from bouquet.subspace import LowEnergySubspace

    coords = state.observed_coords.detach().cpu().numpy()
    energies = state.observed_energies.detach().cpu().numpy()
    # Clamp to the torsion count: the PCA basis has at most `d` columns, so a larger
    # k would let the `rng.integers(k)` mode pick index past the available columns.
    k = min(state.lowmode_modes, coords.shape[1])
    ss = LowEnergySubspace(
        n_dihedrals=coords.shape[1],
        symmetry_orders=getattr(state, "symmetry_orders", None),
    )
    ss.update(coords, energies)
    if ss.n_elite < k + 1:
        return None

    # Anchor = the incumbent's actual 3D structure (lowest-energy observation).
    best_idx = int(state.observed_energies.argmin().item())
    anchor = state.observed_atoms[best_idx]

    # Kick direction: a random one of the k soft modes (random sign), amplitude
    # ~ lowmode_kick_deg RMS/dih.
    #  "pca" -- top-k position-PCA of the low-energy set (the default and benchmark
    #           winner: adaptive directions that re-aim as relaxation walks the incumbent).
    #  "enm" -- softest elastic-network modes projected into torsion space (bouquet.enm);
    #           DORMANT: data-independent, lost to PCA in the Phase-C benchmark.
    V = None
    if state.lowmode_kick_dir == "enm":
        from bouquet.enm import enm_dihedral_modes
        chains = [tuple(di.chain) for di in dihedrals]
        V = enm_dihedral_modes(anchor.get_positions(), chains, k)
        if V.shape[1] == 0:                      # no usable ENM mode
            V = None
    if V is None:                                # PCA (default, or ENM fallback)
        _, V = ss.pca_basis(k)                   # (d, k) tangent (radian) directions
    direction = V[:, rng.integers(V.shape[1])]
    nrm = float(np.linalg.norm(direction))
    if nrm < 1e-12:
        return None
    direction = direction / nrm
    if rng.random() < 0.5:
        direction = -direction
    # incumbent + amp*direction, wrapped to torsion degrees (reuse LowEnergySubspace.line).
    amp = math.radians(state.lowmode_kick_deg) * math.sqrt(direction.shape[0])
    target_deg = ss.line(direction, np.array([amp]))[0]

    if _collective_clash_reject(state, target_deg):
        return None
    energy, atoms = _collective_kick_relax(target_deg, anchor, dihedrals, calc, relaxCalc)
    return _collective_result(state, atoms, energy, dihedrals, calc, relaxCalc)


def _build_category_groups(
    category_assignments: dict | None, n_dihedrals: int
) -> list[list[int]]:
    """Partition dihedral indices into tied categories for the category move.

    Two dihedrals share a group iff the SMARTS matcher assigned them the same
    *specific* torsion-library category (an integer type id) -- i.e. they are the same
    chemical rotor (e.g. the k-th dihedral of every monomer in a foldamer). The generic
    builtin fallbacks (``sp3_sp3``/``sp3_sp2``, string type ids), unassigned dihedrals,
    and bivariate-pair dihedrals are NOT a shared rotor, so each becomes its own singleton
    group -- matching the ``max_spec`` in scripts/cat_suitability.py and the
    ``CAT_MAXSPEC_THRESHOLD`` auto-enable calibration. A homopolymer foldamer thus collapses
    to a handful of groups (the repeat unit); a molecule with no specific repeats stays at
    one group per dihedral (the move degenerates to ordinary REMBO). Returns the groups as
    sorted lists of indices, ordered by first member.

    ``category_assignments`` is the ``{dihedral_index: type_id}`` map from
    :func:`bouquet.priors.assign_categories` (torlib SMARTS ids) -- independent of the
    fitted priors, so categories are available with PiBO steering off.
    """
    key_to_members: dict = {}
    assignments = category_assignments or {}
    for d in range(n_dihedrals):
        t = assignments.get(d)
        # Only an integer (fitted-library) id is a real shared rotor; generic string ids
        # and unassigned/bivariate dihedrals get their own singleton group.
        key = ("cat", t) if isinstance(t, int) else ("solo", d)
        key_to_members.setdefault(key, []).append(d)
    groups = sorted(key_to_members.values(), key=lambda m: m[0])
    return groups


def _sample_category_z_from_prior(
    state: OptimizationState, rng: np.random.Generator
) -> np.ndarray:
    """Draw one reduced-space point (one value per category group, degrees) from the
    prior, for warming up the reduced-space GP. Each group's value is sampled from a
    mode of that category's fitted 1D von Mises mixture (weighted by mixture weight,
    with a small angular jitter); a uniform (no-preference) category draws uniformly.
    """
    groups = state.category_groups or []
    pm = state.prior_module
    uni_dists = getattr(pm, "univariate_dists", {}) if pm is not None else {}
    z = np.empty(len(groups), dtype=float)
    for gi, members in enumerate(groups):
        dist = uni_dists.get(members[0])
        if dist is None:
            z[gi] = rng.uniform(0.0, 360.0)
            continue
        locs = dist.component_distribution.loc.detach().cpu().numpy()      # radians
        probs = dist.mixture_distribution.probs.detach().cpu().numpy()
        probs = probs / probs.sum()
        k = int(rng.choice(len(locs), p=probs))
        z[gi] = (math.degrees(float(locs[k])) + rng.normal(0.0, 15.0)) % 360.0
    return z


def _category_move(
    state: OptimizationState,
    dihedrals: list[DihedralInfo],
    calc: Calculator,
    relaxCalc: Calculator,
    rng: np.random.Generator,
) -> tuple[np.ndarray, Atoms, float] | None:
    """One category-tied collective move: choose a per-category value vector (from the
    prior while the reduced buffer is small, else by reduced-space LogEI), set every
    dihedral in a category to its category's value on top of the incumbent, then
    constrained + UNCONSTRAINED relax -- returning (relaxed dihedrals [deg], relaxed
    atoms, energy [eV]).

    This is the chemistry-defined counterpart of :func:`_low_mode_move`: instead of
    kicking along a data-derived PCA direction, it moves along the fixed block-constant
    embedding implied by the SMARTS categories, so a correlated fold is one reduced-space
    point and is reachable from the very first move. The reduced point is recorded in
    ``state._cat_Z/_cat_Y`` (the low-D BO buffer) before the connectivity check, so even a
    rejected move informs the reduced GP. Returns ``None`` to fall back to a standard step
    (no groups, or a move that broke bonds under --retain-bonds).

    NOTE (prototype): the reduced coordinates use period 360 for every category. Aromatic
    categories are physically 180-periodic (theta == theta+180), so their search domain is
    2x larger than necessary -- correct, just not maximally efficient. Per-category period
    is the obvious next refinement.
    """
    groups = state.category_groups
    if not groups:
        return None

    best_idx = int(state.observed_energies.argmin().item())
    anchor = state.observed_atoms[best_idx]

    # Reduced-space point z (one value per category), in degrees. Below the warmup the
    # prior seeds it; after that the reduced-space GP picks it -- the same value-only
    # periodic-GP + LogEI selector the outer loop uses, just over the n_group-dim buffer.
    if len(state._cat_Z) < state.category_min_moves:
        z = _sample_category_z_from_prior(state, rng)
    else:
        z = _select_next_points_botorch(
            torch.tensor(np.asarray(state._cat_Z), dtype=torch.float64, device=state.device),
            torch.tensor(np.asarray(state._cat_Y), dtype=torch.float64, device=state.device),
            acq_num_restarts=state.acq_num_restarts,
            acq_raw_samples=state.acq_raw_samples,
        )

    # Broadcast: every dihedral takes its group's value (groups partition all indices).
    target_deg = np.empty(len(dihedrals), dtype=float)
    for gi, members in enumerate(groups):
        target_deg[members] = z[gi] % 360.0

    if _collective_clash_reject(state, target_deg):
        # A clashed z still teaches the low-D GP -- record it at the failure energy
        # rather than evaluating it, matching the "record before the connectivity
        # check" intent below without paying for (or risking) the evaluation.
        state._cat_Z.append([float(v) for v in z])
        state._cat_Y.append(float(RELAX_FAILURE_ENERGY_EV - state.start_energy))
        return None

    energy, atoms = _collective_kick_relax(target_deg, anchor, dihedrals, calc, relaxCalc)
    result = _collective_result(state, atoms, energy, dihedrals, calc, relaxCalc)

    # Record the reduced-space observation (energy relative to the run's start).
    # If the species guard rejected a collapsed geometry, record the failure energy
    # rather than the bogus negative collapse energy so it cannot poison the low-D GP.
    recorded_energy = energy if result is not None else RELAX_FAILURE_ENERGY_EV
    state._cat_Z.append([float(v) for v in z])
    state._cat_Y.append(float(recorded_energy - state.start_energy))
    return result
