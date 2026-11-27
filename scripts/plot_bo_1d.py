"""Illustrative 1D Bayesian-optimization figures: value-only vs gradient-enhanced GP.

Generates a publication-quality frame (PDF) for every observation added to a 1D
torsion-scan BO run, for both arms:

  * ``no_gradients`` -- the value-only periodic GP (``SingleTaskGP`` +
    :func:`bouquet.solver._periodic_covar_module`), and
  * ``gradients``   -- the gradient-enhanced periodic GP
    (:func:`bouquet.solver._fit_gradient_gp`), which also conditions on the
    analytic torsion gradient ``dE/dtheta`` at each point.

Both arms share the same synthetic periodic energy surface and the same two
initial points, so the only difference between the two sequences is whether the
surrogate sees gradients. The figures show, frame by frame, that the gradient
arm collapses its posterior uncertainty and locates the global torsion minimum
in fewer evaluations.

The surrogate construction (the ``/360`` input normalization, the
maximize sign-flip, the value standardization, and the matching ``2*pi/std``
gradient scaling) mirrors :func:`bouquet.solver._select_next_points_botorch`
exactly, so the plotted GP *is* the acquisition GP. The next point per frame is
the argmax of LogExpectedImprovement on the plotted dense grid (a 1D analogue of
the solver's ``optimize_acqf``).

With ``--slice`` the synthetic surface is replaced by a real-molecule dihedral
slice built by ``scripts/build_bo_slice.py`` (bouquet's own constrained-relaxed
GFN2 objective, the other rotors anchored at the CREST global minimum); energies
and gradients are then exact grid lookups. ``--composite`` adds a multi-frame
summary panel and ``--seeds`` a convergence curve over random initial designs.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. pixi run python scripts/plot_bo_1d.py

    # real-molecule slice, manuscript output
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. pixi run python scripts/plot_bo_1d.py \\
        --slice data/bo_slices/m116_d05_slice_2.npz \\
        --out ~/Devel/bouquet-manuscript/figures/bo_1d \\
        --n-total 20 --y-max 12 --init 126.5 177 --composite 2 4 5 20 --seeds 40
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from botorch.acquisition.analytic import LogExpectedImprovement

from bouquet.config import ENERGY_CLIP_OFFSET
from bouquet.surrogate import _fit_gradient_gp, _fit_value_gp, make_covar_module

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bo_slice_common import SliceSurface, draw_molecule, plot_convergence  # noqa: E402

torch.set_default_dtype(torch.float64)

EV2KCAL = 23.060541945  # eV -> kcal/mol
DEG2RAD = math.pi / 180.0

# ---------------------------------------------------------------------------
# Synthetic 1D torsion surface (energies in eV, angle in degrees).
# A tilted three-fold profile: a deep global well near ~240 deg, a shallower
# secondary well near ~120 deg, and a high barrier near 0/360 deg -- enough
# structure that the optimizer must explore before committing.
# ---------------------------------------------------------------------------
_A3, _A1, _PHI = 0.5, 0.5, 3.5  # eV amplitudes and 1-fold phase (rad)


def _raw_energy_rad(t: np.ndarray | float):
    return _A3 * (1.0 - np.cos(3.0 * t)) + _A1 * (1.0 - np.cos(t - _PHI))


# Offset so the global minimum sits at 0 eV (a constant shift; gradients unchanged).
_FINE = np.linspace(0.0, 2.0 * math.pi, 20001)
_EMIN = float(_raw_energy_rad(_FINE).min())


def true_energy(deg: np.ndarray | float):
    """Relative torsion energy (eV) at angle ``deg`` (degrees)."""
    t = np.asarray(deg, dtype=float) * DEG2RAD
    return _raw_energy_rad(t) - _EMIN


def true_gradient(deg: np.ndarray | float):
    """Analytic dE/dtheta (eV/rad) -- the signal the gradient arm conditions on."""
    t = np.asarray(deg, dtype=float) * DEG2RAD
    return _A3 * 3.0 * np.sin(3.0 * t) + _A1 * np.sin(t - _PHI)


# ---------------------------------------------------------------------------
# Surrogate fit + dense-grid posterior + acquisition (mirrors the solver).
# ---------------------------------------------------------------------------
class ArmResult:
    """Posterior and acquisition on the plotting grid for one BO frame."""

    def __init__(self, mean_eV, sigma_eV, ei, next_deg, best_eV):
        self.mean_eV = mean_eV  # (M,) posterior mean energy, eV
        self.sigma_eV = sigma_eV  # (M,) posterior std, eV
        self.ei = ei  # (M,) expected improvement (acq.), arbitrary scale
        self.next_deg = next_deg  # argmax-EI angle (next point to evaluate)
        self.best_eV = best_eV  # best observed energy so far, eV


def fit_arm(
    obs_deg: np.ndarray,
    obs_eV: np.ndarray,
    obs_grad: np.ndarray,
    use_gradients: bool,
    grid_deg: np.ndarray,
) -> ArmResult:
    """Fit the (value-only or gradient-enhanced) GP and evaluate it on ``grid_deg``.

    The standardization here is identical to
    ``bouquet.solver._select_next_points_botorch`` so the posterior we plot is the
    one acquisition actually sees; we then map it back to physical energy for the
    figure.
    """
    train_X = torch.tensor(obs_deg, dtype=torch.float64).unsqueeze(-1)  # (n,1) deg
    raw_y = torch.tensor(obs_eV, dtype=torch.float64)  # (n,) relative eV
    x = train_X / 360.0

    # maximize sign-flip + energy clamp (inactive here; mirrors the solver)
    energy_cap = ENERGY_CLIP_OFFSET + torch.log10(torch.clamp(raw_y, min=1.0))
    neg = (-1.0 * torch.minimum(raw_y, energy_cap)).unsqueeze(-1)  # (n,1)
    neg_mean = neg.mean(dim=0, keepdim=True)
    y_std = neg.std(dim=0, keepdim=True)
    y_std = torch.where(y_std >= 1e-9, y_std, torch.ones_like(y_std))
    train_y = (neg - neg_mean) / y_std  # standardized, "maximize" sense

    if use_gradients:
        grad = torch.tensor(obs_grad, dtype=torch.float64).unsqueeze(-1)  # (n,1) eV/rad
        gp = _fit_gradient_gp(x, train_y, raw_y, energy_cap, grad, y_std)
    else:
        gp = _fit_value_gp(x, train_y, covar_module=make_covar_module(1))
    gp.eval()

    gx = torch.tensor(grid_deg / 360.0, dtype=torch.float64).unsqueeze(-1)  # (M,1)
    with torch.no_grad():
        post = gp.posterior(gx)
        m_std = post.mean.reshape(-1)
        sd_std = post.variance.clamp_min(0.0).sqrt().reshape(-1)

    ys = float(y_std)
    nm = float(neg_mean)
    # invert standardization + sign flip: E = -(m_std*std + mean)
    mean_eV = -(m_std * ys + nm)
    sigma_eV = sd_std * ys

    # LogExpectedImprovement on the same grid -> exp -> argmax = next point.
    acqf = LogExpectedImprovement(gp, best_f=train_y.max(), maximize=True)
    with torch.no_grad():
        log_ei = acqf(gx.unsqueeze(1))  # (M,1,1) -> (M,)
    # Exclude grid points already evaluated (selected angles are grid points and
    # get appended verbatim) so EI can't re-propose a sampled angle and stall.
    step = 360.0 / len(grid_deg)
    gap = np.abs(grid_deg[:, None] - obs_deg[None, :])
    gap = np.minimum(gap, 360.0 - gap)  # periodic distance
    sampled = torch.from_numpy((gap < step / 2).any(axis=1))
    log_ei = log_ei.masked_fill(sampled, float("-inf"))
    ei = log_ei.exp()
    next_deg = float(grid_deg[int(ei.argmax())])

    return ArmResult(
        mean_eV=mean_eV.numpy(),
        sigma_eV=sigma_eV.numpy(),
        ei=ei.numpy(),
        next_deg=next_deg,
        best_eV=float(raw_y.min()),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,  # editable text in Illustrator/Inkscape
            "ps.fonttype": 42,
        }
    )


C_TRUE = "#888888"
C_MEAN = "#1f77b4"
C_BAND = "#1f77b4"
C_OBS = "#111111"
C_NEXT = "#d62728"
C_ACQ = "#2ca02c"


def plot_frame(
    grid_deg: np.ndarray,
    true_eV: np.ndarray,
    res: ArmResult,
    obs_deg: np.ndarray,
    obs_eV: np.ndarray,
    obs_grad: np.ndarray,
    use_gradients: bool,
    n_obs: int,
    n_init: int,
    ylim: tuple[float, float],
    out_path: Path,
    show_next: bool = True,
):
    """Render one BO frame (posterior + acquisition) to a PDF."""
    fig = plt.figure(figsize=(7.0, 5.2), constrained_layout=True)
    gs = GridSpec(2, 1, height_ratios=[3.0, 1.0], figure=fig, hspace=0.05)
    ax = fig.add_subplot(gs[0])
    axa = fig.add_subplot(gs[1], sharex=ax)

    true_k = true_eV * EV2KCAL
    mean_k = res.mean_eV * EV2KCAL
    sig_k = res.sigma_eV * EV2KCAL
    obs_k = obs_eV * EV2KCAL

    # --- top: surrogate over the true surface ---
    ax.fill_between(
        grid_deg,
        mean_k - 2.0 * sig_k,
        mean_k + 2.0 * sig_k,
        color=C_BAND,
        alpha=0.18,
        lw=0,
        label="GP 95% CI",
    )
    ax.plot(grid_deg, true_k, color=C_TRUE, ls="--", lw=1.8, label="True energy")
    ax.plot(grid_deg, mean_k, color=C_MEAN, label="GP mean")

    # observations; split initial vs BO-acquired for a subtle visual cue
    init_mask = np.arange(len(obs_deg)) < n_init
    ax.scatter(
        obs_deg[init_mask], obs_k[init_mask], s=55, color=C_OBS, zorder=5,
        edgecolors="white", linewidths=0.6, label="Initial points",
    )
    if (~init_mask).any():
        ax.scatter(
            obs_deg[~init_mask], obs_k[~init_mask], s=55, color=C_OBS, zorder=5,
            marker="o", edgecolors="white", linewidths=0.6, label="Acquired points",
        )

    # gradient ticks: short segments with the observed dE/dtheta slope
    if use_gradients:
        dx = 11.0  # half-length of the slope tick, in degrees
        for xd, yk, g in zip(obs_deg, obs_k, obs_grad):
            slope_k_per_deg = g * EV2KCAL * DEG2RAD  # eV/rad -> kcal/mol per deg
            ax.plot(
                [xd - dx, xd + dx],
                [yk - slope_k_per_deg * dx, yk + slope_k_per_deg * dx],
                color=C_NEXT, lw=1.6, alpha=0.85, zorder=4,
            )

    if show_next:
        ax.axvline(res.next_deg, color=C_NEXT, ls=":", lw=1.6, alpha=0.9)

    ax.set_ylim(*ylim)
    ax.set_xlim(0, 360)
    ax.set_ylabel("Relative energy (kcal/mol)")
    ax.set_xticks(np.arange(0, 361, 60))
    plt.setp(ax.get_xticklabels(), visible=False)

    arm = "Gradient-enhanced GP" if use_gradients else "Value-only GP"
    ax.set_title(f"{arm}   —   {n_obs} evaluations   (best: {res.best_eV*EV2KCAL:.2f} kcal/mol)")
    ax.legend(loc="upper center", ncol=3, columnspacing=1.2, handletextpad=0.5)

    # --- bottom: acquisition ---
    ei = res.ei
    ei_plot = ei / ei.max() if ei.max() > 0 else ei
    axa.fill_between(grid_deg, 0, ei_plot, color=C_ACQ, alpha=0.30, lw=0)
    axa.plot(grid_deg, ei_plot, color=C_ACQ, lw=1.6)
    if show_next:
        axa.axvline(res.next_deg, color=C_NEXT, ls=":", lw=1.6, alpha=0.9)
        axa.scatter([res.next_deg], [1.0], marker="v", s=60, color=C_NEXT,
                    zorder=6, clip_on=False, label="Next point")
        axa.legend(loc="upper right")
    axa.set_ylim(0, 1.08)
    axa.set_yticks([])
    axa.set_xlim(0, 360)
    axa.set_xlabel("Dihedral angle (degrees)")
    axa.set_ylabel("EI", rotation=0, labelpad=14, va="center")
    axa.set_xticks(np.arange(0, 361, 60))

    fig.savefig(out_path)
    plt.close(fig)




def plot_composite(
    grid_deg: np.ndarray,
    true_eV: np.ndarray,
    frames: dict,
    show_n: list[int],
    n_init: int,
    ylim: tuple[float, float],
    out_path: Path,
    mol_img: np.ndarray | None = None,
    xlabel: str = "Dihedral angle (degrees)",
):
    """Summary figure: posterior at selected evaluation counts, both arms.

    Rows are the two arms, columns the requested evaluation counts; an optional
    molecule depiction (scanned bond highlighted) occupies a leading column.
    """
    names = [("no_gradients", "Value-only GP"), ("gradients", "Gradient-enhanced GP")]
    ncol = len(show_n) + (1 if mol_img is not None else 0)
    fig, axes = plt.subplots(
        2, ncol, figsize=(3.3 * ncol, 5.6), constrained_layout=True, squeeze=False,
        gridspec_kw={"width_ratios": ([1.15] if mol_img is not None else []) + [1] * len(show_n)},
    )
    c0 = 0
    if mol_img is not None:
        gs = axes[0, 0].get_gridspec()
        for a in axes[:, 0]:
            a.remove()
        axm = fig.add_subplot(gs[:, 0])
        axm.imshow(mol_img)
        axm.axis("off")
        c0 = 1

    true_k = true_eV * EV2KCAL
    for r, (name, label) in enumerate(names):
        by_n = {f["n"]: f for f in frames[name]}
        for c, n in enumerate(show_n):
            ax = axes[r, c0 + c]
            f = by_n[n]
            res = f["res"]
            mean_k = res.mean_eV * EV2KCAL
            sig_k = res.sigma_eV * EV2KCAL
            ax.fill_between(grid_deg, mean_k - 2 * sig_k, mean_k + 2 * sig_k,
                            color=C_BAND, alpha=0.18, lw=0)
            ax.plot(grid_deg, true_k, color=C_TRUE, ls="--", lw=1.4)
            ax.plot(grid_deg, mean_k, color=C_MEAN, lw=1.6)
            obs_k = f["obs_eV"] * EV2KCAL
            ax.scatter(f["obs_deg"], obs_k, s=28, color=C_OBS, zorder=5,
                       edgecolors="white", linewidths=0.5)
            if name == "gradients":
                dx = 11.0
                for xd, yk, g in zip(f["obs_deg"], obs_k, f["obs_grad"]):
                    s = g * EV2KCAL * DEG2RAD
                    ax.plot([xd - dx, xd + dx], [yk - s * dx, yk + s * dx],
                            color=C_NEXT, lw=1.3, alpha=0.85, zorder=4)
            if n < max(by_n):
                ax.axvline(res.next_deg, color=C_NEXT, ls=":", lw=1.3)
            ax.set_xlim(0, 360)
            ax.set_ylim(*ylim)
            ax.set_xticks(np.arange(0, 361, 90))
            ax.text(0.03, 0.96, f"n = {n}\nbest {res.best_eV * EV2KCAL:.2f}",
                    transform=ax.transAxes, va="top", fontsize=9)
            if r == 0:
                ax.set_title(f"{n} evaluations", fontsize=11)
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel(xlabel)
            if c == 0:
                ax.set_ylabel(f"{label}\nE (kcal/mol)")
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# BO driver
# ---------------------------------------------------------------------------
def run_arm(
    use_gradients: bool,
    init_deg: list[float],
    n_total: int,
    grid_deg: np.ndarray,
    true_eV: np.ndarray,
    ylim: tuple[float, float],
    out_dir: Path | None,
    energy_fn=true_energy,
    grad_fn=true_gradient,
) -> tuple[list[tuple[int, float]], list[dict]]:
    """Run one BO arm; write a frame per evaluation unless ``out_dir`` is None.

    Returns the (n_obs, best_eV) history and the per-frame records (posterior +
    observations) used for the composite figure.
    """
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    n_init = len(init_deg)

    obs_deg = np.array(init_deg, dtype=float)
    obs_eV = np.asarray(energy_fn(obs_deg), dtype=float)
    obs_grad = np.asarray(grad_fn(obs_deg), dtype=float)

    history: list[tuple[int, float]] = []
    frames: list[dict] = []
    while True:
        n = len(obs_deg)
        res = fit_arm(obs_deg, obs_eV, obs_grad, use_gradients, grid_deg)
        history.append((n, res.best_eV))
        frames.append(dict(n=n, res=res, obs_deg=obs_deg.copy(), obs_eV=obs_eV.copy(),
                           obs_grad=obs_grad.copy()))
        last = n >= n_total
        if out_dir is not None:
            plot_frame(
                grid_deg, true_eV, res, obs_deg, obs_eV, obs_grad,
                use_gradients, n, n_init, ylim,
                out_dir / f"frame_{n:02d}.pdf",
                show_next=not last,
            )
        if last:
            break
        # evaluate the proposed point and append (true energy + gradient)
        nd = res.next_deg
        obs_deg = np.append(obs_deg, nd)
        obs_eV = np.append(obs_eV, energy_fn(nd))
        obs_grad = np.append(obs_grad, grad_fn(nd))

    return history, frames


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures/bo_1d", help="output directory")
    ap.add_argument("--n-total", type=int, default=20, help="total evaluations per arm")
    ap.add_argument(
        "--init", type=float, nargs="+", default=None,
        help="initial dihedral angles (degrees); both arms share these "
             "(default: 60 150; with --slice, snapped to the slice grid)",
    )
    ap.add_argument("--slice", default=None,
                    help="real-molecule slice .npz from scripts/build_bo_slice.py")
    ap.add_argument("--y-max", type=float, default=None,
                    help="upper energy axis limit, kcal/mol (default: surface max + 4)")
    ap.add_argument("--composite", type=int, nargs="+", default=None,
                    help="evaluation counts to show in composite.pdf")
    ap.add_argument("--seeds", type=int, default=0,
                    help="also run this many random initial designs -> convergence.pdf")
    ap.add_argument("--seed-n-init", type=int, default=2,
                    help="initial points per random design (--seeds)")
    ap.add_argument("--seed-min-kcal", type=float, default=1.0,
                    help="random designs draw initial points only where E exceeds this "
                         "(kcal/mol above the minimum), so no design starts already within "
                         "the 1 kcal/mol success threshold")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    init = args.init if args.init is not None else [60.0, 150.0]
    if args.n_total < len(init):
        ap.error(
            f"--n-total ({args.n_total}) must be >= the number of initial points "
            f"({len(init)}: {init})"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    _style()

    surface = None
    if args.slice:
        surface = SliceSurface(args.slice)
        if surface.ndim != 1:
            ap.error(f"{args.slice} is a {surface.ndim}D slice; use plot_bo_2d.py")
        grid_deg = surface.grid_deg
        true_eV = surface.energy_grid
        energy_fn, grad_fn = surface.energy, surface.gradient
        init = [float(x) for x in surface.snap(init)]
    else:
        grid_deg = np.linspace(0.0, 360.0, 1000, endpoint=False)
        true_eV = true_energy(grid_deg)
        energy_fn, grad_fn = true_energy, true_gradient
    true_k = true_eV * EV2KCAL
    y_max = args.y_max if args.y_max is not None else float(true_k.max()) + 4.0
    ylim = (-1.5, y_max)

    out_root = Path(args.out).expanduser()
    where = f" [{surface.label()}]" if surface else ""
    print(f"Global minimum near {grid_deg[int(true_eV.argmin())]:.1f} deg "
          f"(0.00 kcal/mol){where}; initial points at {init} deg\n")

    hist, frames = {}, {}
    for use_grad, name in [(False, "no_gradients"), (True, "gradients")]:
        # reseed so the two arms are paired (same acquisition-optimizer randomness)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        hist[name], frames[name] = run_arm(
            use_grad, list(init), args.n_total, grid_deg, true_eV, ylim,
            out_root / name, energy_fn, grad_fn,
        )
        print(f"[{name}] wrote {len(hist[name])} frames to {out_root / name}")

    # convergence summary: first eval within 1 kcal/mol of the global minimum
    thr_eV = 1.0 / EV2KCAL
    print("\nCalls to within 1 kcal/mol of the global minimum:")
    for name in ("no_gradients", "gradients"):
        reached = next((n for n, b in hist[name] if b <= thr_eV), None)
        msg = f"{reached} evaluations" if reached else f"not within budget ({args.n_total})"
        print(f"  {name:13s}: {msg}")

    if args.composite:
        bad = [n for n in args.composite if not len(init) <= n <= args.n_total]
        if bad:
            ap.error(f"--composite counts {bad} outside [{len(init)}, {args.n_total}]")
        mol_img = draw_molecule(surface) if surface else None
        plot_composite(grid_deg, true_eV, frames, args.composite, len(init), ylim,
                       out_root / "composite.pdf", mol_img=mol_img)
        print(f"\nwrote {out_root / 'composite.pdf'}")

    if args.seeds:
        rng = np.random.default_rng(args.seed)
        H = {"no_gradients": [], "gradients": []}
        designs = []
        for s in range(args.seeds):
            init_s = list(rng.choice(grid_deg[true_k > args.seed_min_kcal],
                                     size=args.seed_n_init, replace=False))
            designs.append(init_s)
            for use_grad, name in [(False, "no_gradients"), (True, "gradients")]:
                torch.manual_seed(args.seed + s)
                h, _ = run_arm(use_grad, init_s, args.n_total, grid_deg, true_eV, ylim,
                               None, energy_fn, grad_fn)
                H[name].append([b * EV2KCAL for _, b in h])
        H = {k: np.array(v) for k, v in H.items()}
        np.savez(out_root / "convergence.npz", designs_deg=np.array(designs), **H)
        title = surface.label() if surface else "Synthetic 1D surface"
        print(f"\nConvergence over {args.seeds} random designs "
              f"(n_init={args.seed_n_init}, initial E > {args.seed_min_kcal:g} kcal/mol):")
        plot_convergence(H, out_root / "convergence.pdf", f"1D: {title}",
                         n0=args.seed_n_init)
        print(f"wrote {out_root / 'convergence.pdf'}")


if __name__ == "__main__":
    main()
