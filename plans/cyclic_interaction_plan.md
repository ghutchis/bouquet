# Plan 2: Cyclic Molecules & Interaction-Stabilized Ring Conformations

**Problem.** On `stopbench-500`, flexible-ring molecules lose to CREST 2–3× more often than acyclic molecules (spiro 32%, macrocycles 36%, fused/simple 17–18% vs. acyclic 12%; see `bouquet/rings/RING_MTD_BENCH.md`). The decoupled ring-MTD module (`bouquet/rings/`) failed its promotion gate (recall 56.8% vs. ETKDG 60.1%; end-to-end 37/85 vs. 44/85).

**Why the previous strategy missed CREST's ring conformers:**
1. **Intramolecular-interaction stabilization:** Unusual ring puckers (boats, twist-boats, inverted medium rings/macrocycles) are often 5–10 kcal/mol uphill on the bare ring and only become low-energy when locked in by an **intramolecular H-bond, salt bridge, or $\pi$-interaction** with an exocyclic substituent or across the ring. Ring-only MTD (`bias_idx = ring_idx`) with arbitrary exocyclic rotamers rarely forms those interactions, and energy-ranking seeds *before* optimizing exocyclic torsions discards the right ring pucker (e.g., rank #26/57 on `m047_d02`).
2. **Broken ring-seed handoff (`setup.py:418`):** Passing a ring-seed `.json` to `bouquet --conformer-file` currently extracts only *acyclic dihedral angles* and applies them to the single `state.start_atoms` geometry — silently discarding every seed's ring pucker during BO.
3. **Force-field vs. QM ranking mismatch:** GFN-FF can mis-rank or destabilize electronically stabilized ring conformers (anomeric effects, intramolecular H-bonds) relative to GFN2.

---

## Phase 1 — Interaction-Guided Distance-Geometry Seeding *(Start Here — Fastest, Highest Signal)*

**Core idea.** Rather than blindly heating a ring in MD hoping it stumbles into an intramolecularly H-bonded pucker, **construct those conformations directly** in milliseconds using RDKit's distance-geometry bounds matrix.

### Algorithm (`bouquet/rings/interaction_seeds.py` or extension to `setup.py`)
1. **Perceive intramolecular interaction pairs** (reusing the polar-H / donor–acceptor logic in `bouquet/clash.py`):
   - **H-bond pairs:** Polar H (`H` bonded to `N, O, S, F`) paired with H-bond acceptors (`N, O, F, S` with lone pairs, excluding the donor's own heavy atom), separated by $\ge 4$ bonds (`path >= 4`) where at least one atom is on or within 2 bonds of a flexible ring (or both are on a macrocycle/medium ring of size $\ge 7$).
   - **Salt-bridge / charged pairs:** Oppositely formally charged atoms (e.g. `[NH3+]` / `[O-]`) separated by $\ge 4$ bonds.
   - **Optional (Phase 1b) $\pi\text{–}\pi$ / transannular pairs:** Centroid-atom pairs between two aromatic rings separated by a flexible ring or short linker.
2. **Feasibility pre-filter on pairs:** Keep pairs whose topological separation (`path`) is geometrically capable of forming an intramolecular contact (e.g., `4 <= path <= 10`, or cross-ring pairs in rings $\ge 7$ atoms) and cap the number of targeted pairs (e.g., top 8–12 pairs prioritized by strong donor–acceptor chemistry: charged > OH/NH…O=C > OH/NH…N/O).
3. **Constrained ETKDG embedding:**
   - Obtain `bm = rdDistGeom.GetMoleculeBoundsMatrix(mol)` and smooth via `rdDistGeom.DoTriangleSmoothing(bm)`.
   - For each candidate interaction pair $(h, a)$ (and small subsets of 2 compatible pairs when multiple exist):
     - Set upper bound `bm[min(h, a), max(h, a)] = 2.4` Å (H-bond `H…A`) and lower bound `bm[max(h, a), min(h, a)] = 1.6` Å (or heavy-atom `D…A` in `[2.5, 3.2]` Å).
     - Re-run `DoTriangleSmoothing(bm)`; skip if triangle-infeasible (fast geometric rejection of impossible H-bonds).
     - Call `AllChem.EmbedMultipleConfs` using the modified bounds matrix (e.g., 4–8 conformers per feasible pair, with `useSmallRingTorsions=True`, `useMacrocycleTorsions=True`).
4. **Unconstrained quench & pool merge:**
   - Merge unconstrained ETKDG embeddings + interaction-constrained embeddings.
   - Quick unconstrained quench on `relaxCalc` (GFN-FF) or MMFF, stereo-check (`stereo_retained` so constrained embedding cannot invert a stereocenter), deduplicate by ring+first-shell iRMSD, and score top diverse candidates with `calc` (GFN2 single-point).
- **Cost:** Tens of milliseconds for embedding + a handful of cheap GFN-FF quenches (~1% of ring-MTD cost).
- **Test:** Run the exact `scripts/ring_mtd_bench.py` basin-iRMSD recall harness (0.125 Å threshold) on the 166 flexible-ring benchmark set, comparing `etkdg` vs. `etkdg + interaction_seeds` vs. `mtd`.

---

## Phase 2 — Ring–Exocyclic Genetic Crossing (GC)

**Core idea.** Even when the right ring pucker *is* already in the candidate pool (`RING_MISS__IN_SEED_POOL` and `RING_FLIP_MISS` — 6 of 27 classified CREST wins in `RING_MTD_BENCH §1.3`), it is often paired with a high-energy exocyclic rotamer, so `select_initial_structure` picks a different ring pucker with a prettier exocyclic tail. Crossing ring scaffolds with exocyclic torsion states decouples the two.

### Algorithm
1. **Extract two pools** (either at initialization from diverse ETKDG + interaction seeds, or mid-run after initial BO has found good exocyclic folds):
   - **Ring scaffold pool** $\{R_1, \dots, R_K\}$ ($K \le 8$): deduplicated by ring+first-shell iRMSD (`fine_dedup_A = 0.20` Å), selected for geometric spread + energy (`hybrid` or `geometry` from `rings/select.py`).
   - **Exocyclic torsion pool** $\{\boldsymbol{\theta}_1, \dots, \boldsymbol{\theta}_M\}$ ($M \le 6$): lowest-energy distinct acyclic dihedral vectors found so far.
2. **Cross (`R_i × theta_j`):**
   - For each pair $(R_i, \boldsymbol{\theta}_j)$, build the grafted geometry using `TorsionGeometry(R_i, dihedrals).build(theta_j)`.
   - Screen immediately with `ClashDetector` (milliseconds).
   - Quench surviving non-clashing crosses unconstrained on `relaxCalc` (GFN-FF, 30–50 steps) with stereo + species guards; score top-energy novel crosses on `calc` (GFN2).
3. **Why this also helps acyclic molecules:** See Plan 1 Phase 3 — the same GC helper (`bouquet/crossover.py`) can handle both graph-cut torsion crossover (acyclic) and scaffold × torsion crossover (cyclic).

---

## Phase 3 — Joint `(ring_scaffold, torsion)` BO (Fixing the Multi-Ring Handoff)

**Why necessary.** Currently `run_optimization()` holds a single `state.start_atoms`, so all BO evaluations inherit one ring pucker. Running $K$ completely separate BOs multiplies cost by $K$, whereas throwing away ring puckers (current `--conformer-file` behavior) wastes the seeding work.

### Two progressive implementations (start with 3a, test 3b only if 3a saturates)

- **3a. Multi-Armed Bandit / Shared-Prior Ring Scheduler (simplest, low risk):**
  - Maintain $K$ ring scaffolds $\{R_1, \dots, R_K\}$ (typically $K = 3\text{–}6$, from Phase 1 + Phase 2).
  - Allocate BO budget dynamically across scaffolds using UCB / Thompson sampling on each scaffold's best energy + uncertainty, **sharing**:
    - the PiBO torsion priors,
    - GC-transferred exocyclic torsions (whenever scaffold $r$ finds a new global-best $\boldsymbol{\theta}^*$, test $\boldsymbol{\theta}^*$ on the other active scaffolds via Phase 2).
  - Crucially, a scaffold whose bare ring was $+5$ kcal/mol uphill gets a small protected warmup (e.g. 5–8 steps + best exocyclic cross) before the bandit prunes it — giving interaction-stabilized puckers a fair chance to fold and drop to the global minimum.

- **3b. Product-Kernel Categorical × Periodic GP (`RING_MTD_v2 §9`):**
  - Single GP over $(r, \boldsymbol{\theta})$ with covariance $k((r, \boldsymbol{\theta}), (r', \boldsymbol{\theta}')) = k_{\text{cat}}(r, r') \cdot k_{\text{per}}(\boldsymbol{\theta}, \boldsymbol{\theta}')$ (using BoTorch's `CategoricalKernel` or a learned low-rank correlation matrix over ring scaffolds).
  - Allows observations on scaffold $r$ to directly inform the torsional landscape of scaffold $r'$, scaling cleanly to larger $K$ (e.g., $K = 10\text{–}16$ on spiro/macrocyclic systems).
  - Acquisition maximizes `LogEI` over $\boldsymbol{\theta}$ for each scaffold $r$ (or top-3 promising scaffolds) and picks $\arg\max_{r, \boldsymbol{\theta}} \alpha(r, \boldsymbol{\theta})$.

---

## Phase 4 — Interaction-Preserving Ring Kick (Fallback for Macrocycles / Medium Rings)

*Only pursue if Phases 1–3 still miss CREST on macrocycles ($\ge 10$-membered) or spiro systems.*
- Starting from the incumbent **folded** conformer (whose intramolecular H-bonds are already formed), run a short 2–5 ps high-T Langevin kick + quench on ring atoms while keeping **weak flat-bottom harmonic tethers** on detected H-bond contacts (`d < 2.8 Å`), so the ring explores alternate puckers *conditioned on preserving the stabilizing interaction*.

---

## Benchmark Protocol & Acceptance Criteria

- **Population:** The 166 flexible-ring molecules with CREST references in `stopbench-500` (stratified by `SIMPLE`, `FUSED`, `SPIRO`, `BRIDGED_CAGE`, `MACROCYCLE`), keeping controls (`NO_FLEX_RING` and already-solved ring molecules) to ensure no speed or quality regression.
- **Stage A (Pool recall — cheap, no BO):** Measure 0.125 Å relaxed ring-iRMSD recall (`scripts/ring_mtd_bench.py`) for:
  1. `etkdg` (production 16 embeddings),
  2. `etkdg_100` (cheap embedding-count control),
  3. `etkdg + interaction_seeds` (Phase 1),
  4. `etkdg + interaction_seeds + GC` (Phase 1 + Phase 2).
- **Stage B (End-to-end BO):** Run 3-seed paired BO on the 85-molecule Stage-2 set from `RING_MTD_BENCH §9.19`, measuring:
  - CREST-match rate (within 0.5 kcal/mol of $E^*_{\text{CREST}}$),
  - Paired mean/median energy delta vs. baseline,
  - Total wall-clock vs. baseline and vs. CREST.
- **Promotion gate:** Must beat both production `etkdg` *and* `etkdg_100` on recall AND improve end-to-end CREST-match rate (strictly beating the 44/85 baseline where ring-MTD regressed to 37/85) at $< 20\%$ total wall-clock overhead.
