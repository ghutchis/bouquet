# Plan 1: Improving Medium-Dimension Acyclic Conformer Search ($d = 6\text{–}9$)

**Problem.** Bouquet matches or beats CREST on small acyclic molecules ($d \le 5$), and high-dimensional collective moves (dim-scaled lengthscale prior + PCA low-mode + SMARTS category moves) deliver clear gains at $d \ge 12$. In the intermediate regime ($d = 6\text{–}9$), however, bouquet lags CREST: the molecule is too irregular for global PCA fold modes or repeating-unit category ties, while the rotamer space ($\sim 3^6\text{–}3^9$) is too large for an isotropic product kernel over constrained evaluations to reliably locate narrow **correlated-dihedral canyons** (e.g., intramolecular H-bonds, salt bridges, or $\pi$-stacking that couple 2–4 contiguous torsions).

**Prior lessons not to re-derive** (see `bouquet/surrogate.py` and `bouquet_hdbo_plan.md`):
- First-order **additive** kernels improved held-out prediction but *hurt* end-to-end search (+0.41 kcal/mol) via over-exploitation (shrinking $\sigma$ stopped EI from exploring).
- Pairwise **trellis** ($\theta_i \pm \theta_j$) terms lost to the plain product kernel and learned symmetric weights.
- **ENM** kick directions lost to data-derived PCA; **ARD** alone tied with `product`.
- **Bounded-early gradients** (`use_gradients`, $S \approx 50$) and the **clash constraint** (`clash.py`) both gave real wins.

---

## Phase 0 — Re-benchmark Bivariate PiBO Priors (Post-Bugfix Baseline)

**Why first (zero new code).** `DihedralPriorModule.forward()` previously subtracted `offset1_deg` / `offset2_deg` (degrees) directly from normalized `[0, 1]` coordinates when `input_in_degrees=False` (always true in PiBO mode). Because ~64% of matched 2D pairs in ZINC-like molecules carry a non-zero terminal-atom offset (`priors.py`), every such bivariate prior was misaligned during search.

- **Action:** Re-run `scripts/sweep_bivariate.py` on the $d = 6\text{–}9$ acyclic set using the fixed `bouquet/priors.py`.
- **Also verify:** Measure the hit rate of `bivariate_priors` on `stopbench-500` ($d = 6\text{–}9$) and check whether low coverage (missing adjacent/skip-one patterns) or prior weight (`prior_exponent` / `prior_decay`) limits impact.
- **Gate:** Record new paired baseline vs. `product` + univariate `--priors` before stacking Phases 1–4.

---

## Phase 1 — Exact Graph-Symmetry Rotor Periodicity ($C_2$ / $C_3$ Folding)

### Why ARD did not already handle this
1. In the default `kernel_arm="product"`, `ProductStructureKernel` wraps a single scalar `PeriodicKernel` (`period_length` shape `(1, 1)`), constrained by `NormalPrior(1.0, 0.1)`.
2. Even under `kernel_arm="ard"` (`period_length` shape `(1, d)`), `NormalPrior(1.0, 0.1)` imposes a $5\sigma$ penalty ($-12.5$ nats) on `period = 0.5` ($C_2$) and a $6.7\sigma$ penalty ($-22.2$ nats) on `period = 1/3` ($C_3$), preventing MLL from ever discovering sub-turn periods.
3. In `GradientEnhancedPeriodicGP`, `period_length` is explicitly broadcast to `1.0` and frozen (`requires_grad_(False)`).
4. Even if MLL *could* learn `period = 0.5`, it would still waste evaluations visiting both $\theta$ and $\theta + 180^\circ$ first — and `optimize_acqf` bounds, Sobol pools, and ensemble basin distances all remain $[0, 1]^d$ ($360^\circ$).

### Why exact graph symmetry beats inferring period from 1D/2D priors
A *meta*-substituted phenyl ring (`m-Cl-C6H4-`) or a chiral `sp3-sp3` bond has a symmetric-looking 1D histogram prior across a dataset, yet $\theta$ and $\theta + 180^\circ$ are **chemically distinct conformers** with different energies. Folding by prior appearance would silently cut away true minima. By contrast, **exact molecular graph symmetry** guarantees $E(\theta + 360^\circ / m) \equiv E(\theta)$ identically (zero approximation, iRMSD $= 0.000\text{ \AA}$).

### Detection rule (exact, rigid local symmetry)
For each rotatable bond `(a1, a2)` with rotating fragment `group` (containing `a2`) and stationary side `rest` (containing `a1`), compute `ranks = Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=True)`. Check symmetry on **either** side of the bond (if either side has exact $C_m$ symmetry about the bond axis, the torsion is $360^\circ / m$-periodic):
- **Terminal $C_3$ tops** ($m = 3$, period $120^\circ$): e.g., $-\text{CH}_3$, $-\text{CF}_3$, $-\text{CCl}_3$, $-\text{NH}_3^+$, $-\text{C(CH}_3)_3$ — atom has 3 non-bond neighbors with identical `ranks`, and each neighbor subtree has no internal rotatable bonds or flexible rings (or is itself a $C_3$-symmetric terminal top like $-\text{CH}_3$ in *tert*-butyl, which is symmetric after 10-step relaxation, though conservative v1 can restrict to leaf/rigid neighbors).
- **Planar $C_2$ rotors** ($m = 2$, period $180^\circ$): e.g., $-\text{NO}_2$, $-\text{CO}_2^-$, $-\text{C}_6\text{H}_5$, *para*-substituted phenyl, 3,5-disubstituted phenyl, symmetric pyrimidine — bond atom has 2 non-bond neighbors with identical `ranks`, and the entire fragment on that side contains **no internal rotatable bonds** and **no non-aromatic puckerable rings** (so a $180^\circ$ flip cannot swap two differently puckered or differently rotated sub-branches).

### Integration into `bouquet`
- Add `symmetry_order: int = 1` to `DihedralInfo` (detected in `bouquet/setup.py`).
- **Representation:** Either:
  - *(Cleanest)* Map each dihedral $j$ into $[0, 1/m_j)$ with `bounds[1, j] = 1.0 / m_j`, fold observed coordinates `x_j = (deg_j % (360.0 / m_j)) / 360.0`, and set fixed `period_length[…, j] = 1.0 / m_j` on `PeriodicKernel` / `PeriodicKernelGrad` (keeping `dE/dtheta` chain rule untouched).
  - Update `_draw_feasible_pool`, `_cert_sobol_pool`, and `ensemble._periodic_min_dist_sq` to wrap modulo `360.0 / m_j`.
- **Validation test:** Verify on 50 diverse molecules from `stopbench-500` that rotating any $m_j > 1$ bond by $360^\circ / m_j$ preserves GFN2 energy to $< 10^{-4}$ eV and iRMSD $< 10^{-3}\text{ \AA}$.
- **Expected gain:** Shrinks search volume by $2^{n_{C_2}} 3^{n_{C_3}}$ (typically 2×–16× on drug-like $d = 6\text{–}9$ molecules) with zero extra runtime cost.

---

## Phase 2 — Memetic "Relax-and-Relabel" BO Steps (Conquered Correlated Canyons)

**Hypothesis.** The core reason CREST beats BO on $d = 6\text{–}9$ correlated folds is **unconstrained local relaxation**: CREST relaxes every candidate unconstrained so any geometry inside a correlated H-bond/fold basin slides down the diagonal valley into the minimum. Bouquet pins all dihedrals during search (`FixInternals`), so near-miss proposals get scored on steep canyon walls instead of at the minimum.

### Mechanism
Keep standard constrained BO as the backbone, and add a lightweight memetic polish trigger:
- **Trigger:** Whenever a BO step lands within `polish_window_kcal` (e.g., 6–10 kcal/mol) of the current best — or periodically every $K$ steps (e.g., every 8–10 steps) starting from the best *unpolished* low-energy observation.
- **Move:**
  1. Remove `FixInternals` and run a short **unconstrained** relaxation on `relaxCalc` (e.g., 30–50 ABC-FIRE steps, cheap GFN-FF).
  2. Apply the existing **species guard** (`bonds_formed` + optional `bonds_broken`).
  3. Read back the relaxed torsion vector $\boldsymbol{\theta}_{\text{rel}} = [d.\text{get\_angle}(\text{atoms})]$.
  4. If $\boldsymbol{\theta}_{\text{rel}}$ moved meaningfully from any existing observation (e.g., wrapped RMS torsion distance $\ge 10^\circ$), score with `calc` (and project `dE/dtheta` if gradients enabled) and **append $(\boldsymbol{\theta}_{\text{rel}}, E_{\text{rel}})$ to `OptimizationState`**.
- **Why this works with the GP:** The GP directly receives the exact canyon-floor coordinates and energies (and near-zero gradients at the minimum), letting subsequent BO steps avoid re-exploring that basin and focus elsewhere.
- **Arms to A/B test:**
  1. `baseline`: standard constrained BO (current default).
  2. `polish_triggered`: unconstrained 40-step GFN-FF polish when $E \le E_{\text{best}} + 8\text{ kcal/mol}$ (capped at e.g. 20% of budget).
  3. `polish_replace`: instead of adding a second observation, evaluate every $k$-th BO step *unconstrained* (30–50 steps) and record its relaxed $\boldsymbol{\theta}_{\text{rel}}$ (zero extra `calc` calls).

---

## Phase 3 — Graph-Aware Genetic Crossing (GC) in Torsion Space

**Motivation.** CREST's Genetic Crossing (GC) step is one of its strongest engines on *both* cyclic and medium-$d$ acyclic molecules: if conformer A found the optimal fold for the left half of a molecule and conformer B found the optimal fold for the right half, standard single-point BO must stumble onto the joint combination by chance, whereas GC synthesizes it directly in one move.

### Graph-partitioned crossover (better than random coordinate crossover)
Random uniform crossover of dihedral indices breaks local correlated motifs (splitting an H-bonded turn in half). Instead, partition dihedrals along the **molecular bonding graph**:
1. Take the top-$K$ distinct low-energy conformers from `OptimizationState` (deduplicated by wrapped torsion RMSD $\ge 25^\circ$, e.g. $K = 8\text{–}12$).
2. Pick a random parent pair $(A, B)$ (Boltzmann- or rank-weighted) and choose a **cut bond** $b$ (a non-terminal rotatable bond that splits the rotatable-bond tree into two connected subtrees $L$ and $R$).
3. Form offspring $\boldsymbol{\theta}_{\text{child}}$ taking $\theta_j^A$ for $j \in L$, $\theta_j^B$ for $j \in R$, and optionally setting the cut dihedral $\theta_b$ to $\theta_b^A$, $\theta_b^B$, or a prior peak / small perturbation.
4. **Fast screening:** Generate a pool of 20–50 candidate crosses via vectorized `TorsionGeometry` + `ClashDetector` (milliseconds), optionally rank clash-free offspring by either **GP posterior mean / LogEI** or a fast **GFN-FF single-point / 20-step quench**, and promote only the top 1–2 novel (non-duplicate) offspring to full `calc` evaluation (unconstrained polish + species guard) and append them to `OptimizationState`.

### Scheduling
- Run one lightweight GC burst (generating 2–4 evaluated offspring) once at mid-budget (e.g. 40% and 75% of budget), or as a periodic collective move alongside standard BO once $\ge 6$ distinct low-energy basins exist.
- Because the offspring are clash-screened and GP-ranked before touching `calc`, the added QM cost is tiny (2–6 evaluations total).

---

## Phase 4 — Diverse Force-Field Pre-Seeding & Local Basin Polish

- **FF pre-seeding:** Instead of `init_steps = 5` blind Gaussian draws around 1 ETKDG geometry, generate 64–128 ETKDG conformers, run fast MMFF/GFN-FF optimization, cluster by symmetry-reduced torsion RMSD, and seed BO with the `init_steps` (e.g. 8–12) lowest-energy *distinct* clusters. (Measure whether the GFN-FF cost is negligible relative to GFN2 BO steps.)
- **Optional gradient-guided local trust region (only if Phases 1–3 still leave a gap):** Around the top-2 distinct basins in the second half of the budget, alternate global LogEI with a local periodic box ($\pm 45^\circ$ shrinking on stalls) using the gradient-enhanced GP.

---

## Benchmark Protocol & Acceptance Criteria

- **Dataset:** Acyclic / `NO_FLEX_RING` subset of `smiles/stopbench-500.csv` with $d \in [6, 9]$ (plus a $d \in [3, 5]$ non-regression control set and $d \in [10, 12]$ spillover check).
- **Pairing:** Common random numbers (CRN), $\ge 3$ seeds per molecule, fixed GFN2 energy / GFN-FF optimizer, comparing against existing CREST references in `stop_bench_gfn2/`.
- **Primary metrics:**
  1. Fraction of molecules reaching within 0.5 kcal/mol (and 0.1 kcal/mol) of CREST $E^*$ at matched evaluation budget and at matched wall-clock.
  2. Paired median $\Delta E = E_{\text{arm}} - E_{\text{baseline}}$ (Wilcoxon signed-rank $p < 0.05$).
  3. Wall-clock overhead ratio vs. baseline.
- **Promotion gates:**
  - **Phase 1 (Symmetry folding):** Ship as default if 100% energy/iRMSD invariance passes and search quality is non-inferior or better on $d = 3\text{–}9$.
  - **Phase 2 (Relax-and-relabel) & Phase 3 (Graph GC):** Promote each independently if it closes $\ge 25\%$ of the remaining $d = 6\text{–}9$ CREST gap with $< 15\%$ wall-clock overhead and zero regression at $d \le 5$.
