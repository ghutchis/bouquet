# Plan 3: Efficient Boltzmann Ensemble Discovery

**Problem.** Bouquet's primary loop is tuned to find a single global minimum (`LogEI`). The existing `--ensemble` pipeline (`bouquet/ensemble.py`, `bouquet/solver_ensemble.py`) runs a full global-min search first, then appends sequential `q = 1` level-set / annealed-boundary exploration steps (refitting a fresh GP on every step), and finally runs unconstrained tight L-BFGS on all harvested candidates. While functional, it has four efficiency bottlenecks:
1. **Misallocated energy weighting:** Flat level-set exploration ($P(E \le E_{\min} + \Delta)\,\sigma$) treats a $+0.5$ kcal/mol basin (43% relative weight at 298 K) identically to a $+9.0$ kcal/mol basin ($<10^{-6}$ weight).
2. **Two-phase budget waste:** Spending the main budget on pure `LogEI` over-samples the global minimum before exploration even begins.
3. **Late-only basin resolution:** Running only constrained relaxations during exploration means torsion-space basin dedup (`ENSEMBLE_BASIN_DEG = 40°`) does not know which points actually collapse into the *same* local minimum until tight-opt at the end.
4. **High-level evaluation cost & symmetry duplicates:** Every exploration step pays full `calc` cost sequentially (`q = 1`) and explores full $360^\circ$ ranges even on $C_2$/$C_3$ symmetric rotors (whose duplicates `irmsd` later discards).

---

## Phase 1 — Boltzmann-MGF Acquisition & Partition-Function Stopping Rule

### Closed-form Boltzmann-weighted acquisition
Under the minimization-sense selection GP posterior $E(x) \sim \mathcal{N}(\mu(x), \sigma^2(x))$, the expected Boltzmann weight at effective inverse temperature $\beta_{\text{eff}} = (k_B T_{\text{eff}})^{-1}$ has an **exact, analytic, smooth closed form** via the log-normal moment generating function:

$$\log \mathbb{E}\!\left[e^{-\beta_{\text{eff}}(E(x) - E_{\min})}\right] = -\beta_{\text{eff}}(\mu(x) - E_{\min}) + \frac{1}{2}\beta_{\text{eff}}^2 \sigma^2(x)$$

Combined with the smooth squared-distance diversity penalty (`_diversity_factor`, fixed to avoid `sqrt(0)` NaNs):

$$\alpha_{\text{Boltz}}(x) = \left[-\beta_{\text{eff}}(\mu(x) - E_{\min}) + \frac{1}{2}\beta_{\text{eff}}^2 \sigma^2(x)\right] + \log\bigl(1 - \lambda\,\text{proximity}(x)\bigr)$$

*(working in log-space avoids float underflow/overflow and gives clean gradients for L-BFGS-B.)*

### Why this is cleaner than `levelset` / `hybrid`
- **Single-phase capable:** Automatically prioritizes finding the global minimum *and* all thermally populated conformers in order of Boltzmann weight — no hard switch between `LogEI` and level-set needed (though a brief initial `LogEI` warmup can be A/B tested).
- **Natural temperature tuning:**
  - $T_{\text{eff}} = 298\text{–}400\text{ K}$: laser-focused on the conformers that actually dominate room-temperature observables (NMR, optical rotation, Boltzmann-averaged energies; 99% of weight lives within $\sim 2.7$ kcal/mol).
  - $T_{\text{eff}} = 700\text{–}1000\text{ K}$ ($k_B T_{\text{eff}} \approx 1.4\text{–}2.0$ kcal/mol): naturally covers the 3–6 kcal/mol DFT re-ranking window without wasting evaluations at 9–10 kcal/mol.
- **Principled stopping rule (undiscovered partition-function bound):**
  Over the fixed clash-free Sobol pool $\mathcal{P}$, estimate the max remaining unvisited Boltzmann contribution:
  $$w_{\text{rem}} = \max_{x \in \mathcal{P}_{\text{unvisited}}} \exp\!\left(-\beta(\mu(x) - E_{\min}) + \frac{1}{2}\beta^2 \sigma^2(x)\right)$$
  Stop exploration once $w_{\text{rem}} / Z_{\text{found}} < \varepsilon$ (e.g., $\varepsilon = 0.01\text{–}0.02$) for $k$ consecutive steps — directly targeting the actual quantity of interest (missing Boltzmann population) rather than an arbitrary global $\max \sigma$.

---

## Phase 2 — Symmetry-Reduced Basin Accounting & Online Quench-to-Basin

1. **Symmetry-aware torsion distance (reusing Plan 1 Phase 1):**
   - Wrap `_periodic_min_dist_sq` per dimension by $360^\circ / m_j$ (for $C_2$/$C_3$ symmetric rotors) and restrict acquisition bounds to $[0, 1/m_j]$.
   - Prevents ensemble exploration from wasting evaluations re-discovering flipped phenyls or rotated methyls that `irmsd` merges at the end.
2. **Online quench-to-basin (replacing blind constrained-only exploration):**
   - Whenever an exploration candidate lands within the discovery window (e.g., $\Delta E_{\text{const}} \le 8$ kcal/mol), immediately run a fast unconstrained `relaxCalc` quench (50–100 steps, or tight L-BFGS when `relaxCalc` is cheap) and deduplicate the resulting *true local minimum* online via `_rmsd` (iRMSD).
   - **Three compounding benefits:**
     1. `minima_deg` repels *actual* discovered basins (at their true relaxed torsions $\boldsymbol{\theta}_{\min}$) rather than arbitrary constrained points.
     2. You know the exact running count of distinct conformers and running Boltzmann partition function $Z_{\text{found}}$ at every step (enabling the Phase 1 stopping rule).
     3. Discovered minima are already relaxed, eliminating most redundant work in `tight_optimize_and_weight` at the end.
3. **Reuse GC (Plan 1 Phase 3 / Plan 2 Phase 2) as an ensemble generator:**
   - Crossing the top-$K$ discovered basins in torsion space (and across ring scaffolds for cyclic molecules) followed by clash-screen + GFN-FF quench is an exceptionally cheap way to generate combinatorial conformer ensembles (e.g., 2 independent functional groups with 3 rotamers each $\rightarrow$ $3 \times 3 = 9$ thermally accessible conformers discovered in one burst instead of 9 BO cycles).

---

## Phase 3 — Batched Acquisition ($q = 4\text{–}8$) & Parallel Evaluation

**Bottleneck.** `_perform_ensemble_exploration` currently fits a fresh `SingleTaskGP` (100 Adam steps) for every single `q = 1` proposal. At $n = 200\text{–}500$, GP fitting + `optimize_acqf` becomes a major fraction of wall time and prevents parallelizing calculator calls.

### Implementation
- **Sequential local penalization (Kriging Believer / Gonzales-style batching):**
  1. Fit the selection GP **once** per batch (or freeze hypers for 3–5 batches once $n > 50$, mirroring `gradfreeze`).
  2. Propose $x^{(1)} = \arg\max \alpha_{\text{Boltz}}(x; \text{basins})$.
  3. Temporarily append $x^{(1)}$ to the basin-repulsion set `minima_deg` (with $\lambda = 1.0$) and optimize $\alpha_{\text{Boltz}}$ again to get $x^{(2)}, \dots, x^{(q)}$ ($q = 4\text{–}8$).
  4. Evaluate all $q$ proposals **in parallel** across worker processes (each with `OMP_NUM_THREADS=1` for xTB, matching `rings/mtd.py`).
- **Expected gain:** $q\times$ fewer GP refits and near-$q\times$ wall-clock speedup on multi-core nodes with equal or better basin diversity.

---

## Phase 4 — Multi-Fidelity $\Delta$-Learning Funnel (Cheap Exploration → Expensive QM Ensemble)

**Motivation.** For expensive targets (`gfn2` on large molecules, `r2scan3c`, `b3lyp`, `wb97x`), doing all ensemble exploration directly at the high level of theory is prohibitive — yet ranking directly by GFN-FF/MMFF has 2–4 kcal/mol errors.

### Architecture
1. **Stage 1 — Exhaustive low-fidelity basin harvest (`relaxCalc`, e.g. GFN-FF or AIMNet2):**
   - Run fast Phase 1–3 ensemble discovery + ETKDG/interaction seeding + GC entirely on the cheap calculator to obtain a deduplicated pool of local minima $\{M_1, \dots, M_L\}$ within a generous low-level window (e.g., 10 kcal/mol, typically $L = 30\text{–}300$ conformers).
2. **Stage 2 — $\Delta$-GP prioritized high-level promotion (`calc`, e.g. GFN2 / DFT):**
   - Score a diverse initial subset (e.g., 10–15 minima spanning low energy + geometry spread) at `calc` to observe $\Delta E_i = E_{\text{high}}(M_i) - E_{\text{low}}(M_i)$.
   - Fit a periodic GP to the residual $\Delta E(\boldsymbol{\theta})$ (much smoother than $E_{\text{high}}$ itself).
   - Predict $E_{\text{high}}(M_j) = E_{\text{low}}(M_j) + \hat{\mu}_\Delta(\boldsymbol{\theta}_j) \pm \hat{\sigma}_\Delta(\boldsymbol{\theta}_j)$ for all remaining un-evaluated low-level minima, and select the next batch using the **Phase 1 Boltzmann-MGF acquisition over the discrete pool $\{M_j\}$** (instantaneous — no continuous `optimize_acqf` needed).
   - Stop once the predicted Boltzmann weight of all remaining un-evaluated minima falls below threshold $\varepsilon$, then tight-optimize only the surviving high-level window conformers.

---

## Benchmark Protocol & Acceptance Criteria

- **Datasets:** `cysteine_bench/`, `ensemble_bench/` (low/medium $d$), and `ensemble_high_d_bench/` (high $d$), with CREST ensembles (re-optimized on the same level of theory) as ground truth.
- **Metrics (beyond raw conformer count):**
  1. **Boltzmann-weighted recall** (fraction of true CREST 298 K Boltzmann population captured by the bouquet ensemble; primary metric).
  2. **Window recall** at both **3.0 kcal/mol** (thermally populated) and **6.0 kcal/mol** (DFT re-ranking window), using iRMSD $< 0.125\text{ \AA}$.
  3. **Ensemble property accuracy:** Error in Boltzmann-averaged energy $\langle E \rangle$ and conformational entropy $S_{\text{conf}} = -R \sum_i p_i \ln p_i$ vs. CREST.
  4. **Efficiency:** High-level (`gfn2`) evaluations and wall-clock time to reach 90% and 95% Boltzmann population recall vs. current `--ensemble` (`levelset` and `hybrid`) and vs. CREST.
- **Promotion gates:**
  - **Phase 1 (Boltzmann-MGF):** Must match or exceed `levelset` on 3 kcal/mol & Boltzmann recall AND beat `levelset` on 6 kcal/mol recall (or match `hybrid` at $\ge 30\%$ fewer evaluations).
  - **Phase 2–3 (Online quench + GC + batch $q$):** $\ge 2\times$ wall-clock speedup at equal or better Boltzmann recall.
  - **Phase 4 ($\Delta$-learning funnel):** $\ge 3\times$ reduction in high-level (`gfn2`/DFT) evaluations at $\ge 95\%$ Boltzmann population recall.
