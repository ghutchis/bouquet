# bouquet/scripts

Helper scripts for building dihedral priors and for benchmarking the bouquet
conformer search. These are research/utility scripts run from the repository
root (most expect `torlib.txt`, `gfn2_priors.json`, and a `data/` tree to be
present there);

There are two rough pipelines here:

1. **Prior construction** — turn crystal/QM geometries into per-SMARTS torsion
   histograms, then fit those histograms to von Mises mixture priors.
2. **Benchmarking** — run `bouquet.cli` over a molecule set across seeds (and
   across prior configurations) and summarize how well it finds low-energy
   conformers.

## Prior construction

### `create-torsion-histograms.py`
Builds per-torsion angle histograms from a library of structures. Loops over
`*/*.sdf` files (connectivity) paired with matching `.xyz` files (conformer
geometry), matches every SMARTS in `torlib.txt` (from the [TorLib 2020
paper](https://doi.org/10.1021/acs.jcim.2c00043)), and accumulates a 360-bin
(1°) histogram per pattern into `torsions/tl{index}.txt`. When several patterns
hit the same four atoms, the most specific (later) one wins. Adapted from
[Peter Schmidtke's COD torsion-angle
post](https://pschmidtke.github.io/blog/rdkit/crystallography/small%20molecule%20xray/xray/database/2021/01/25/cod-and-torsion-angles.html).

No CLI args — edit the input glob, `torlib.txt` path, and `out_template` at the
top of the file. A per-file 60 s timeout guards against hangs.

### `assemble_univariate_priors.py`
Fits the histograms above into a JSON prior file usable by bouquet. For each
SMARTS pattern it loads histogram counts from one or more data directories
(default `data/cod`, `data/zinc`, `data/pubchemqc`), fits a von Mises mixture
(peak detection + adaptive component selection, scored by R² and Bhattacharyya
coefficient), and falls back to bouquet's generic `sp3_sp3` / `sp3_sp2` priors
when a fit is close enough to generic. Can emit diagnostic plots and a
diagnostics CSV.

```bash
python scripts/assemble_univariate_priors.py \
    --torlib torlib.txt \
    --data-dirs data/cod data/zinc data/pubchemqc \
    --output univariate_priors.json \
    --max-components 5 --min-r2 0.85 --min-counts 100 \
    [--plot-dir plots/] [--diagnostics-csv diag.csv] [-v]
```

### `dihedral_matcher.py`
Standalone diagnostic: given a single molecule and a SMARTS pattern file, report
which dihedral in the structure matches which pattern and its current angle.
Most-specific (later) pattern wins per dihedral; dihedrals are normalized so
`(a,b,c,d)` and `(d,c,b,a)` collapse to one. Useful for sanity-checking pattern
coverage.

```bash
python scripts/dihedral_matcher.py molecule.mol patterns.txt -o results.csv [--add-hydrogens]
```

## Benchmarking

### `batch.py`
Runs `bouquet.cli` (`--auto --relax`, with `gfn2_priors.json`) over a
CSV/TSV of `smiles,name` molecules across several random seeds, parsing each
run's log for step counts and constrained/unconstrained `E−E0` energies. Writes
one wide row per molecule (five metrics per seed). Resumable, with optional
parallel workers and selectable energy/optimizer methods.

```bash
python scripts/batch.py --input mols.csv --output results.csv \
    --seeds 1234,12345,3141,314159,42 --workers 8 [--energy gfn2] [--resume] [-v]
```

### `sweep_priors.py`
Compares prior configurations against a no-prior baseline. Runs the same
`bouquet.cli` invocation as `batch.py` for every (configuration × molecule ×
seed) and writes a tidy long-format CSV, then analyzes it. The configurations
(concentration cap, background weight, exponent) live in `CONFIGURATIONS` near
the top of the file; reuses `batch.parse_log_output`. Resumable at
(config, name, seed) granularity.

```bash
# 1. run the sweep
python scripts/sweep_priors.py run --input mols.csv --output sweep.csv \
    --seeds 1234,12345,3141,314159,42 --workers 8 [--configs noprior,cap20_bg0.1]

# 2. analyze: per-config step/trapping + energy summaries and a paired
#    win/loss comparison vs the no-prior baseline
python scripts/sweep_priors.py analyze sweep.csv
```

The `analyze` step excludes physically broken geometries (`|E−E0|` above
`--max-abs-e`, default 5 eV) from energy aggregates and restricts cross-molecule
comparisons to the set of molecules present in every configuration.

### `analyze_bouquet.py`
Post-processes one or more `batch.py`-style result CSVs (e.g.
`platinum-ei.csv`, `platinum-lei.csv`) into a single comparison table: for each
molecule, the minimum unconstrained `E−E0` across all seeds, one column per
input file (column suffix taken from the `{prefix}-{suffix}.csv` filename).

```bash
python scripts/analyze_bouquet.py platinum-ei.csv platinum-lei.csv -o bouquet_analysis.csv
```
