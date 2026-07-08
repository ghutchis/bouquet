# 🌻 BOUQUET 💐

Generate beautiful ensembles of molecular geometries
**B**ayesian **O**ptimization **U**sing **QU**antum **E**nergy **T**ool

This repo contains code for optimizing conformers using Bayesian optimization for active learning and quantum chemistry computations.

*Background*

Conformers define the different geometries with the same molecular bonding graph but different coordinates.
Finding the lowest-energy conformation is a common task in molecular modeling, and one that often requires significant time to solve.
We implement optimal experimental design techniques to solve this problem
following [recent](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0354-7) [work](https://pubs.acs.org/doi/full/10.1021/acs.jctc.0c00648)
that uses Bayesian optimization find optimize dihedral angles.

## Installation

We recommend use of [`pixi`](https://pixi.prefix.dev/):
```bash
pixi install
```

Many parts of the code are available through `pip`:
```bash
pip install '.[all]'
```

## Use

`bouquet` provides a simple interface to the code. To optimize cysteine with default arguments.

```bash
bouquet --smiles "C([C@@H](C(=O)O)N)S"
```

This will produce a folder in the `solutions` directory containing the optimized geometry
(`final.xyz`) and many other files for debugging.

### Common options

Call `bouquet --help` for the full list. The most useful flags:

**Input**

| Flag | Default | Description |
| --- | --- | --- |
| `--smiles` | — | SMILES string of the molecule to optimize |
| `--file` | — | Read the starting structure from a file instead of a SMILES string |
| `--conformer-file` | — | File of multiple conformers to use as initial guesses (overrides `--init-steps`) |
| `--name` | SMILES / file name | Output name for the run directory |
| `--seed` | random | Random seed for reproducible runs |

**Search budget**

| Flag | Default | Description |
| --- | --- | --- |
| `--auto` | off | Scale the number of steps to the number of detected dihedrals |
| `--num-steps` | `32` | Number of Bayesian-optimization steps |
| `--init-steps` | `5` | Number of initial guesses before optimization begins |

**Energy / geometry**

| Flag | Default | Description |
| --- | --- | --- |
| `--energy` | `gfn2` | Energy method (`aimnet2`, `ani`, `b3lyp`, `b97`, `gfn0`, `gfn2`, `gfnff`, `mmff`, `uff`) |
| `--optimizer` | `gfnff` | Method used for geometry relaxation |
| `--relax` | on | Relax non-dihedral degrees of freedom before scoring each energy |

