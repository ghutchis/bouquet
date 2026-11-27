"""Shared helpers for the illustrative BO figures (``plot_bo_1d.py`` / ``plot_bo_2d.py``).

* :class:`SliceSurface` -- a real-molecule dihedral slice built by
  ``scripts/build_bo_slice.py``: energies and torsion gradients looked up on the
  slice grid (BO proposals are grid points, so every evaluation is an exact
  table lookup of bouquet's own constrained-relaxed objective).
* :func:`draw_molecule` -- 2D depiction with the scanned rotor bond(s) highlighted.
* :func:`plot_convergence` -- best-so-far gap to the slice minimum vs evaluations,
  median and interquartile band over several initial-point seeds, both arms.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

EV2KCAL = 23.060541945

C_VALUE = "#1f77b4"
C_GRAD = "#d62728"


class SliceSurface:
    """A 1D or 2D dihedral slice loaded from a ``build_bo_slice.py`` ``.npz``."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        d = np.load(self.path, allow_pickle=False)
        self.grid_deg = d["grid_deg"].astype(float)  # (M,)
        self.energy_grid = d["energy_eV"].astype(float)  # (M,) or (M, M), eV
        self.grad_grid = d["grad_eV_rad"].astype(float)  # (..., k), eV/rad
        self.anchor_deg = d["anchor_deg"].astype(float)
        self.chains = d["chains"].astype(int)
        self.smiles = str(d["smiles"])
        self.mol_id = str(d["mol_id"])
        self.method = str(d["method"])
        self.relax_steps = int(d["relax_steps"])
        self.ndim = self.energy_grid.ndim
        self.step = 360.0 / len(self.grid_deg)

    def index(self, deg) -> np.ndarray:
        """Nearest grid index (periodic) for angle(s) ``deg``."""
        return np.rint(np.asarray(deg, dtype=float) % 360.0 / self.step).astype(int) % len(
            self.grid_deg)

    def snap(self, deg) -> np.ndarray:
        """Snap angle(s) to the nearest grid point."""
        return self.grid_deg[self.index(deg)]

    def energy(self, *deg) -> np.ndarray:
        """Relative energy (eV) at grid angle(s); one argument per scanned rotor."""
        return self.energy_grid[tuple(self.index(x) for x in deg)]

    def gradient(self, *deg) -> np.ndarray:
        """dE/dtheta (eV/rad) at grid angle(s); trailing axis = scanned rotors."""
        g = self.grad_grid[tuple(self.index(x) for x in deg)]
        return g[..., 0] if self.ndim == 1 else g

    def global_min_deg(self) -> np.ndarray:
        idx = np.unravel_index(int(self.energy_grid.argmin()), self.energy_grid.shape)
        return self.grid_deg[list(idx)]

    def label(self) -> str:
        return f"{self.mol_id} ({self.method.upper()}, constrained relaxation)"


def draw_molecule(surface: SliceSurface, size=(520, 400)) -> np.ndarray:
    """RGBA image of the molecule with the scanned rotor bond(s) highlighted.

    The rotor chains index the reference ``.xyz`` atom order, so the depiction is
    built from that geometry (bond orders from ``rdDetermineBonds``) rather than
    from the SMILES, and hydrogens are removed while tracking original indices.
    """
    import ase.io
    from io import BytesIO
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    from bouquet.setup import get_initial_structure_from_file
    import matplotlib.image as mpimg

    xyz = surface.path.with_name(f"{surface.mol_id}.xyz")
    clean = Path(tempfile.mkdtemp()) / xyz.name
    ase.io.write(clean, ase.io.read(xyz, index=0), format="xyz", comment=surface.mol_id)
    _, mol = get_initial_structure_from_file(str(clean))
    for a in mol.GetAtoms():
        a.SetIntProp("orig", a.GetIdx())
    heavy = Chem.RemoveHs(mol)
    orig2new = {a.GetIntProp("orig"): a.GetIdx() for a in heavy.GetAtoms()}
    AllChem.Compute2DCoords(heavy)

    colors = [(0.84, 0.15, 0.16), (0.12, 0.47, 0.71)]
    hl_atoms, hl_bonds, atom_cols, bond_cols = [], [], {}, {}
    for k, chain in enumerate(surface.chains):
        i, j = orig2new[int(chain[1])], orig2new[int(chain[2])]
        b = heavy.GetBondBetweenAtoms(i, j).GetIdx()
        col = colors[k % len(colors)]
        hl_bonds.append(b)
        bond_cols[b] = col
        for a in (i, j):
            hl_atoms.append(a)
            atom_cols[a] = col

    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 2
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, heavy, highlightAtoms=hl_atoms, highlightBonds=hl_bonds,
        highlightAtomColors=atom_cols, highlightBondColors=bond_cols,
    )
    drawer.FinishDrawing()
    return mpimg.imread(BytesIO(drawer.GetDrawingText()), format="png")


def plot_convergence(histories: dict, out_path: Path, title: str, n0: int = 1,
                     thr_kcal: float = 1.0):
    """Best-so-far energy gap vs evaluations, median + IQR over seeds, both arms.

    ``histories[name]`` is an (n_seeds, n_evals) array of best-so-far relative
    energies in kcal/mol (relative to the slice minimum, so 0 = found it); column
    0 is the state after the ``n0`` initial evaluations.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.0), constrained_layout=True)
    styles = {"no_gradients": (C_VALUE, "Value-only GP"),
              "gradients": (C_GRAD, "Gradient-enhanced GP")}
    for name, H in histories.items():
        col, lab = styles[name]
        n = np.arange(n0, n0 + H.shape[1])
        med = np.median(H, axis=0)
        lo, hi = np.percentile(H, [25, 75], axis=0)
        ax.fill_between(n, lo, hi, color=col, alpha=0.18, lw=0, step="post")
        ax.step(n, med, where="post", color=col, lw=2.0, label=f"{lab} (median)")
    ax.axhline(thr_kcal, color="0.5", ls=":", lw=1.2)
    ax.set_xlabel("Energy evaluations")
    ax.set_ylabel("Best found − slice minimum (kcal/mol)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(n0, n0 + max(H.shape[1] for H in histories.values()) - 1)
    n_seeds = next(iter(histories.values())).shape[0]
    ax.set_title(f"{title}\n{n_seeds} random initial designs; band = IQR", fontsize=10)
    ax.legend(loc="upper right")
    fig.savefig(out_path)
    plt.close(fig)

    for name, H in histories.items():
        hit = [(np.flatnonzero(h <= thr_kcal)[0] + n0) if (h <= thr_kcal).any() else np.nan
               for h in H]
        hit = np.array(hit, dtype=float)
        print(f"  {name:13s}: within {thr_kcal:g} kcal/mol in "
              f"{np.sum(~np.isnan(hit))}/{len(hit)} seeds; "
              f"median evals {np.nanmedian(hit) if (~np.isnan(hit)).any() else float('nan'):.0f}")
