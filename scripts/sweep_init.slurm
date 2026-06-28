#!/bin/bash
# SLURM array submit for sweep_init.py: one task per seed.
#
# Each array task runs the FULL molecule set at ONE seed and writes its own
# output CSV (plus _traj sibling) into $RESULTS. Tasks never share a file, so
# there is no append race on the network drive. After the array drains, merge
# with scripts/merge_sweep.py and run analyze/traj on the merged pair.
#
# Why per-seed:
#   - load is auto-balanced (every task runs the identical molecule set);
#   - analyze/traj pair by (molecule, seed), so each pair lives inside one task
#     -- the split is statistically identical to a single-node run;
#   - a dead node re-runs only its own missing trials via --resume.
#
# Submit:
#   sbatch scripts/sweep_init.slurm
# Merge + analyze when SLURM reports all array tasks COMPLETED:
#   python scripts/merge_sweep.py "$RESULTS"/init_seed*.csv -o "$RESULTS"/init_all.csv
#   python scripts/sweep_init.py analyze "$RESULTS"/init_all.csv
#   python scripts/sweep_init.py traj    "$RESULTS"/init_all_traj.csv --plot-dir "$RESULTS"/plots
#
#SBATCH --job-name=sweep_init
#SBATCH --array=0-4                 # one task per seed; keep in sync with SEEDS below
#SBATCH --cpus-per-task=16          # = --workers; each trial is single-threaded
#SBATCH --time=24:00:00
#SBATCH --output=logs/sweep_init_%A_%a.out
#SBATCH --error=logs/sweep_init_%A_%a.err

set -euo pipefail

# ---- Edit these for your cluster -------------------------------------------
BOUQUET=/path/to/bouquet                    # repo root
RESULTS=/net/results/init_bench             # network dir; per-task files, no sharing
INPUT="$BOUQUET/big_benchmark.csv"          # smiles,name CSV for the larger set
PRIORS="$BOUQUET/gfn2_priors.json"          # fitted prior at repo root
SEEDS=(1 2 3 4 5)           # length must match --array range
# ----------------------------------------------------------------------------

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
mkdir -p "$RESULTS" "$BOUQUET/logs"

cd "$BOUQUET"

# --workers = allocated CPUs: trials run single-threaded (default --single-thread
# pins BLAS to 1), so one worker per core with no oversubscription.
# --resume makes a requeued/failed task pick up only its missing (seed) trials.
# Drop --no-use-gradients (and set --energy == --optimizer) only if you want the
# gradient-enhanced GP; here gfn2/gfnff is a mixed surface, so gradients stay off.
# NOTE: this seeds initial points from gfn2_priors.json with PiBO acquisition OFF
# (sweep_init's peaks arm passes --prior-exponent 0), so it tests init quality of
# the fitted prior, not the PiBO search
pixi run python scripts/sweep_init.py run \
    --input "$INPUT" \
    --output "$RESULTS/init_seed${SEED}.csv" \
    --priors-file "$PRIORS" \
    --no-use-gradients \
    --energy gfn2 --optimizer gfnff \
    --seeds "$SEED" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --timeout 1800 \
    --resume

echo "Task ${SLURM_ARRAY_TASK_ID} (seed ${SEED}) done -> $RESULTS/init_seed${SEED}.csv"
