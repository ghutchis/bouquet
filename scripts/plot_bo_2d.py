"""Illustrative 2D Bayesian-optimization figures: value-only vs gradient-enhanced GP.

The 2D analogue of ``scripts/plot_bo_1d.py``. Two coupled torsions span a
periodic energy surface; for every observation added to a BO run we render a
2x2 frame (PDF) -- the true surface, the GP posterior mean, the posterior
uncertainty, and the acquisition function -- for both arms:

  * ``no_gradients`` -- the value-only periodic GP (``SingleTaskGP`` +
    :func:`bouquet.solver._periodic_covar_module`), and
  * ``gradients``   -- the gradient-enhanced periodic GP
    (:func:`bouquet.solver._fit_gradient_gp`), which also conditions on the
    analytic torsion gradient ``(dE/dtheta_1, dE/dtheta_2)`` at each point. That
    gradient is a 2D vector, drawn as a downhill arrow at each observation.

Both arms share the same surface and the same initial points, so the only
difference is whether the surrogate sees gradients. The figures show the
gradient arm collapsing its uncertainty across the whole torus -- not just at
the sampled points -- and locating the global basin in far fewer evaluations.

The surrogate construction mirrors
``bouquet.solver._select_next_points_botorch`` exactly (the ``/360``
normalization, the maximize sign-flip, the value standardization, and the
matching ``2*pi/std`` gradient scaling), so the plotted GP *is* the acquisition
GP. The next point is the argmax of LogExpectedImprovement on the plotted grid.

With ``--slice`` the synthetic surface is replaced by a real-molecule 2D dihedral
slice built by ``scripts/build_bo_slice.py`` (bouquet's own constrained-relaxed
GFN2 objective, the other rotors anchored at the CREST global minimum); energies
and gradients are then exact grid lookups and the slice grid is the plotting /
acquisition grid. ``--composite`` adds a multi-frame summary panel and
``--seeds`` a convergence curve over random initial designs.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. pixi run python scripts/plot_bo_2d.py

    # real-molecule slice, manuscript output
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. pixi run python scripts/plot_bo_2d.py \\
        --slice data/bo_slices/m139_d06_slice_1_2.npz \\
        --out ~/Devel/bouquet-manuscript/figures/bo_2d \\
        --n-total 30 --e-max 12 --init 225 325 180 315 305 85 \\
        --composite 3 10 20 30 --seeds 20
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

from botorch.acquisition.analytic import LogExpectedImprovement

from bouquet.config import ENERGY_CLIP_OFFSET
from bouquet.surrogate import _fit_gradient_gp, _fit_value_gp, make_covar_module

# bo_slice_common is imported only on the --slice / --seeds paths that use it, so the
# synthetic-surface default runs without it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

torch.set_default_dtype(torch.float64)

EV2KCAL = 23.060541945  # eV -> kcal/mol
DEG2RAD = math.pi / 180.0

# ---------------------------------------------------------------------------
# Synthetic 2D torsion surface (energies in eV, angles in degrees).
# A sum of periodic (von Mises-like) wells on the torus: a deep global basin
# plus two shallower competing basins, so the optimizer must explore in 2D
# before committing. Smooth and analytically differentiable everywhere.
# ---------------------------------------------------------------------------
_KAPPA = 0.8
# (center_theta1_deg, center_theta2_deg, depth_eV)
_WELLS = [
    (250.0, 280.0, 0.060),  # global basin (deepest)
    (90.0, 110.0, 0.045),   # secondary
    (340.0, 30.0, 0.038),   # tertiary
]


def _raw_energy_rad(t1, t2):
    e = np.zeros(np.broadcast(t1, t2).shape)
    for c1, c2, d in _WELLS:
        e = e - d * np.exp(_KAPPA * (np.cos(t1 - c1 * DEG2RAD) + np.cos(t2 - c2 * DEG2RAD)))
    return e


# Offset so the global minimum sits at 0 eV (constant shift; gradients unchanged).
_g = np.linspace(0.0, 2.0 * math.pi, 721)
_T1, _T2 = np.meshgrid(_g, _g, indexing="ij")
_EMIN = float(_raw_energy_rad(_T1, _T2).min())


def true_energy(deg1, deg2):
    """Relative energy (eV) at angles (deg1, deg2) in degrees."""
    t1 = np.asarray(deg1, dtype=float) * DEG2RAD
    t2 = np.asarray(deg2, dtype=float) * DEG2RAD
    return _raw_energy_rad(t1, t2) - _EMIN


def true_gradient(deg1, deg2):
    """Analytic (dE/dtheta_1, dE/dtheta_2) in eV/rad -- the gradient signal."""
    t1 = np.asarray(deg1, dtype=float) * DEG2RAD
    t2 = np.asarray(deg2, dtype=float) * DEG2RAD
    g1 = np.zeros(np.broadcast(t1, t2).shape)
    g2 = np.zeros(np.broadcast(t1, t2).shape)
    for c1, c2, d in _WELLS:
        w = d * np.exp(_KAPPA * (np.cos(t1 - c1 * DEG2RAD) + np.cos(t2 - c2 * DEG2RAD)))
        g1 = g1 + w * _KAPPA * np.sin(t1 - c1 * DEG2RAD)
        g2 = g2 + w * _KAPPA * np.sin(t2 - c2 * DEG2RAD)
    return np.stack([g1, g2], axis=-1)


# ---------------------------------------------------------------------------
# Surrogate fit + grid posterior + acquisition (mirrors the solver, d=2).
# ---------------------------------------------------------------------------
class ArmResult:
    def __init__(self, mean_eV, sigma_eV, ei, next_deg, best_eV):
        self.mean_eV = mean_eV  # (M,) posterior mean energy, eV
        self.sigma_eV = sigma_eV  # (M,) posterior std, eV
        self.ei = ei  # (M,) expected improvement, arbitrary scale
        self.next_deg = next_deg  # (2,) argmax-EI angle, the next point
        self.best_eV = best_eV  # best observed energy so far, eV


def fit_arm(
    obs_deg: np.ndarray,  # (n, 2)
    obs_eV: np.ndarray,  # (n,)
    obs_grad: np.ndarray,  # (n, 2) eV/rad
    use_gradients: bool,
    query_deg: np.ndarray,  # (M, 2)
) -> ArmResult:
    """Fit the GP and evaluate posterior + acquisition at ``query_deg``.

    Standardization is identical to ``_select_next_points_botorch`` so the
    plotted posterior is the acquisition GP; we then map it back to energy.
    """
    train_X = torch.tensor(obs_deg, dtype=torch.float64)  # (n, 2) deg
    raw_y = torch.tensor(obs_eV, dtype=torch.float64)  # (n,) eV
    x = train_X / 360.0

    energy_cap = ENERGY_CLIP_OFFSET + torch.log10(torch.clamp(raw_y, min=1.0))
    neg = (-1.0 * torch.minimum(raw_y, energy_cap)).unsqueeze(-1)  # (n,1)
    neg_mean = neg.mean(dim=0, keepdim=True)
    y_std = neg.std(dim=0, keepdim=True)
    y_std = torch.where(y_std >= 1e-9, y_std, torch.ones_like(y_std))
    train_y = (neg - neg_mean) / y_std

    if use_gradients:
        grad = torch.tensor(obs_grad, dtype=torch.float64)  # (n, 2) eV/rad
        gp = _fit_gradient_gp(x, train_y, raw_y, energy_cap, grad, y_std)
    else:
        gp = _fit_value_gp(x, train_y, covar_module=make_covar_module(2))
    gp.eval()

    # Batch query points as (M, 1, d) so each is its own q=1 posterior -- avoids
    # forming the full M x M covariance (essential for a dense 2D grid).
    qx = torch.tensor(query_deg / 360.0, dtype=torch.float64).unsqueeze(1)  # (M,1,2)
    with torch.no_grad():
        post = gp.posterior(qx)
        m_std = post.mean.reshape(-1)
        sd_std = post.variance.clamp_min(0.0).sqrt().reshape(-1)

    ys = float(y_std)
    nm = float(neg_mean)
    mean_eV = -(m_std * ys + nm)  # invert standardization + sign flip
    sigma_eV = sd_std * ys

    acqf = LogExpectedImprovement(gp, best_f=train_y.max(), maximize=True)
    with torch.no_grad():
        log_ei = acqf(qx)  # (M,1,2) -> (M,)
    # Exclude grid points already evaluated (selected points are grid rows and get
    # appended verbatim) so EI can't re-propose a sampled point and stall.
    n_grid = round(math.sqrt(len(query_deg)))
    step = 360.0 / n_grid
    d = np.abs(query_deg[:, None, :] - obs_deg[None, :, :])  # (M, n, 2)
    d = np.minimum(d, 360.0 - d)  # periodic per axis
    sampled = (d < step / 2).all(axis=2).any(axis=1)  # (M,)
    if not sampled.all():  # keep at least one candidate if every grid point is taken
        log_ei = log_ei.masked_fill(torch.from_numpy(sampled), float("-inf"))
    ei = log_ei.exp()
    next_deg = query_deg[int(ei.argmax())].copy()

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
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.linewidth": 0.9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


C_OBS = "#111111"
C_NEXT = "#d62728"
C_GRAD = "#d62728"


def _overlay_points(ax, obs_deg, n_init, next_deg=None, star_color=C_NEXT):
    init = obs_deg[:n_init]
    acq = obs_deg[n_init:]
    ax.scatter(init[:, 0], init[:, 1], s=45, c=C_OBS, edgecolors="white",
               linewidths=0.7, zorder=5)
    if len(acq):
        ax.scatter(acq[:, 0], acq[:, 1], s=45, c=C_OBS, marker="o",
                   edgecolors="white", linewidths=0.7, zorder=5)
    if next_deg is not None:
        ax.scatter([next_deg[0]], [next_deg[1]], s=180, marker="*",
                   c=star_color, edgecolors="white", linewidths=0.8, zorder=6)


def _capped_arrows(g, grad_scale, max_deg=45.0):
    """Downhill arrow components (-gradient), with drawn length capped at ``max_deg``.

    Display only: near clash walls the torsion gradient is orders of magnitude
    above the median, and an uncapped arrow spans the whole torus. Direction is
    preserved; the GP itself always conditions on the true gradient.
    """
    v = -np.asarray(g, dtype=float)
    length = np.linalg.norm(v, axis=-1, keepdims=True) / grad_scale  # degrees
    factor = np.minimum(1.0, max_deg / np.maximum(length, 1e-12))
    return v * factor


def _axfmt(ax, xlabel=True, ylabel=True):
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ax.set_xticks(np.arange(0, 361, 90))
    ax.set_yticks(np.arange(0, 361, 90))
    ax.set_aspect("equal")
    if xlabel:
        ax.set_xlabel(r"Dihedral $\theta_1$ (degrees)")
    if ylabel:
        ax.set_ylabel(r"Dihedral $\theta_2$ (degrees)")


def plot_frame(
    T1, T2, true_k, res, obs_deg, obs_grad, use_gradients, n_obs, n_init,
    e_levels, sigma_vmax, grad_scale, out_path, show_next=True,
):
    """Render one 2x2 BO frame to a PDF."""
    M = T1.shape
    mean_k = (res.mean_eV * EV2KCAL).reshape(M)
    sig_k = (res.sigma_eV * EV2KCAL).reshape(M)
    ei = res.ei.reshape(M)
    nd = res.next_deg if show_next else None

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.6), constrained_layout=True)
    (ax_t, ax_m), (ax_s, ax_a) = axes

    arm = "Gradient-enhanced GP" if use_gradients else "Value-only GP"
    fig.suptitle(
        f"{arm}   —   {n_obs} evaluations   "
        f"(best: {res.best_eV * EV2KCAL:.2f} kcal/mol)",
        fontsize=14,
    )

    # --- true surface ---
    cf = ax_t.contourf(T1, T2, true_k, levels=e_levels, cmap="viridis", extend="max")
    ax_t.contour(T1, T2, true_k, levels=e_levels, colors="white",
                 linewidths=0.3, alpha=0.4)
    ax_t.set_title("True energy surface")
    if use_gradients:
        # downhill arrows (-gradient) at each observation
        v = _capped_arrows(obs_grad, grad_scale)  # (n,2), from eV/rad
        ax_t.quiver(
            obs_deg[:, 0], obs_deg[:, 1], v[:, 0], v[:, 1],
            color=C_GRAD, angles="xy", scale_units="xy", scale=grad_scale,
            width=0.006, zorder=4,
        )
    _overlay_points(ax_t, obs_deg, n_init, nd)
    _axfmt(ax_t, xlabel=False)
    fig.colorbar(cf, ax=ax_t, shrink=0.85, label="kcal/mol")

    # --- posterior mean ---
    cf = ax_m.contourf(T1, T2, mean_k, levels=e_levels, cmap="viridis", extend="both")
    ax_m.contour(T1, T2, mean_k, levels=e_levels, colors="white",
                 linewidths=0.3, alpha=0.4)
    ax_m.set_title("GP posterior mean")
    _overlay_points(ax_m, obs_deg, n_init, nd)
    _axfmt(ax_m, xlabel=False, ylabel=False)
    fig.colorbar(cf, ax=ax_m, shrink=0.85, label="kcal/mol")

    # --- posterior uncertainty ---
    cf = ax_s.contourf(T1, T2, sig_k, levels=np.linspace(0, sigma_vmax, 21),
                       cmap="magma", extend="max")
    ax_s.set_title(r"GP uncertainty ($\sigma$)")
    _overlay_points(ax_s, obs_deg, n_init, nd)
    _axfmt(ax_s)
    fig.colorbar(cf, ax=ax_s, shrink=0.85, label="kcal/mol")

    # --- acquisition ---
    ei_plot = ei / ei.max() if ei.max() > 0 else ei
    cf = ax_a.contourf(T1, T2, ei_plot, levels=np.linspace(0, 1, 21), cmap="cividis")
    ax_a.set_title("Acquisition (EI, per-frame scale)")
    _overlay_points(ax_a, obs_deg, n_init, nd)
    _axfmt(ax_a, ylabel=False)
    fig.colorbar(cf, ax=ax_a, shrink=0.85, label="relative EI")

    fig.savefig(out_path)
    plt.close(fig)


def plot_composite(
    T1, T2, true_k, frames, show_n, n_init, e_levels, out_path, mol_img=None,
):
    """Summary figure: GP posterior mean at selected evaluation counts, both arms.

    Rows are the two arms, columns the requested evaluation counts; a leading
    column shows the true surface (top) and the molecule with the scanned bonds
    highlighted (bottom), when a depiction is supplied.
    """
    names = [("no_gradients", "Value-only GP"), ("gradients", "Gradient-enhanced GP")]
    lead = 1
    ncol = len(show_n) + lead
    fig, axes = plt.subplots(2, ncol, figsize=(3.0 * ncol + 0.8, 6.4),
                             constrained_layout=True, squeeze=False)
    M = T1.shape

    ax = axes[0, 0]
    cf = ax.contourf(T1, T2, true_k, levels=e_levels, cmap="viridis", extend="max")
    gmin = np.unravel_index(int(true_k.argmin()), M)
    ax.scatter([T1[gmin]], [T2[gmin]], marker="x", s=60, c="white", lw=2, zorder=6)
    ax.set_title("True energy surface", fontsize=11)
    _axfmt(ax, xlabel=False)
    ax = axes[1, 0]
    if mol_img is not None:
        ax.imshow(mol_img)
        ax.set_title(r"$\theta_1$ red, $\theta_2$ blue", fontsize=10)
    ax.axis("off")

    for r, (name, label) in enumerate(names):
        by_n = {f["n"]: f for f in frames[name]}
        for c, n in enumerate(show_n):
            ax = axes[r, lead + c]
            f = by_n[n]
            res = f["res"]
            mean_k = (res.mean_eV * EV2KCAL).reshape(M)
            ax.contourf(T1, T2, mean_k, levels=e_levels, cmap="viridis", extend="both")
            ax.contour(T1, T2, mean_k, levels=e_levels, colors="white",
                       linewidths=0.3, alpha=0.4)
            if name == "gradients":
                v = _capped_arrows(f["obs_grad"], f["grad_scale"])
                ax.quiver(f["obs_deg"][:, 0], f["obs_deg"][:, 1], v[:, 0], v[:, 1],
                          color=C_GRAD, angles="xy", scale_units="xy",
                          scale=f["grad_scale"], width=0.008, zorder=4)
            nd = res.next_deg if n < max(by_n) else None
            _overlay_points(ax, f["obs_deg"], n_init, nd)
            ax.scatter([T1[gmin]], [T2[gmin]], marker="x", s=50, c="white", lw=1.8, zorder=6)
            _axfmt(ax, xlabel=(r == 1), ylabel=False)
            ax.set_title(f"{n} evaluations (best {res.best_eV * EV2KCAL:.2f})"
                         if r == 0 else f"best {res.best_eV * EV2KCAL:.2f} kcal/mol",
                         fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{label}\n" + r"$\theta_2$ (degrees)")
    fig.colorbar(cf, ax=axes[:, -1], shrink=0.7, label="kcal/mol")
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# BO driver
# ---------------------------------------------------------------------------
def run_arm(
    use_gradients, init_deg, n_total, T1, T2, query_deg, true_k,
    e_levels, sigma_vmax, grad_scale, out_dir,
    energy_fn=true_energy, grad_fn=true_gradient,
) -> tuple[list[tuple[int, float]], list[dict]]:
    """Run one BO arm; write a frame per evaluation unless ``out_dir`` is None.

    Returns the (n_obs, best_eV) history and the per-frame records used for the
    composite figure.
    """
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    n_init = len(init_deg)

    obs_deg = np.array(init_deg, dtype=float)  # (n,2)
    obs_eV = np.asarray(energy_fn(obs_deg[:, 0], obs_deg[:, 1]), dtype=float)
    obs_grad = np.asarray(grad_fn(obs_deg[:, 0], obs_deg[:, 1]), dtype=float)

    history: list[tuple[int, float]] = []
    frames: list[dict] = []
    while True:
        n = len(obs_deg)
        res = fit_arm(obs_deg, obs_eV, obs_grad, use_gradients, query_deg)
        history.append((n, res.best_eV))
        frames.append(dict(n=n, res=res, obs_deg=obs_deg.copy(), obs_grad=obs_grad.copy(),
                           grad_scale=grad_scale))
        last = n >= n_total
        if out_dir is not None:
            plot_frame(
                T1, T2, true_k, res, obs_deg, obs_grad, use_gradients, n, n_init,
                e_levels, sigma_vmax, grad_scale, out_dir / f"frame_{n:02d}.pdf",
                show_next=not last,
            )
        if last:
            break
        nd = res.next_deg
        obs_deg = np.vstack([obs_deg, nd])
        obs_eV = np.append(obs_eV, energy_fn(nd[0], nd[1]))
        obs_grad = np.vstack([obs_grad, grad_fn(nd[0], nd[1])])

    return history, frames


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures/bo_2d")
    ap.add_argument("--n-total", type=int, default=24, help="total evaluations per arm")
    ap.add_argument("--n-grid", type=int, default=80,
                    help="contour grid resolution per axis (synthetic surface only)")
    ap.add_argument(
        "--init", type=float, nargs="+", default=None,
        help="flat list of initial (theta1, theta2) pairs in degrees; both arms share "
             "these (default: 30 60 150 40 60 300; with --slice, snapped to the grid)",
    )
    ap.add_argument("--slice", default=None,
                    help="real-molecule 2D slice .npz from scripts/build_bo_slice.py")
    ap.add_argument("--e-max", type=float, default=None,
                    help="top of the energy colour scale, kcal/mol (default: surface max)")
    ap.add_argument("--composite", type=int, nargs="+", default=None,
                    help="evaluation counts to show in composite.pdf")
    ap.add_argument("--seeds", type=int, default=0,
                    help="also run this many random initial designs -> convergence.pdf")
    ap.add_argument("--seed-n-init", type=int, default=3,
                    help="initial points per random design (--seeds)")
    ap.add_argument("--seed-min-kcal", type=float, default=1.0,
                    help="random designs draw initial points only where E exceeds this "
                         "(kcal/mol above the minimum), so no design starts already within "
                         "the 1 kcal/mol success threshold")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    init_flat = args.init if args.init is not None else [30, 60, 150, 40, 60, 300]
    if len(init_flat) % 2 != 0:
        ap.error(
            f"--init must be a flat list of (theta1, theta2) pairs (even length); "
            f"got {len(init_flat)} values: {init_flat}"
        )
    num_pairs = len(init_flat) // 2
    if args.n_total < num_pairs:
        ap.error(
            f"--n-total ({args.n_total}) must be >= the number of initial pairs "
            f"({num_pairs}) from --init"
        )
    init = np.array(init_flat, dtype=float).reshape(-1, 2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    _style()

    surface = None
    if args.slice:
        from bo_slice_common import SliceSurface

        surface = SliceSurface(args.slice)
        if surface.ndim != 2:
            ap.error(f"{args.slice} is a {surface.ndim}D slice; use plot_bo_1d.py")
        g = surface.grid_deg
        energy_fn, grad_fn = surface.energy, surface.gradient
        init = surface.snap(init)
    else:
        g = np.linspace(0.0, 360.0, args.n_grid, endpoint=False)
        energy_fn, grad_fn = true_energy, true_gradient
    T1, T2 = np.meshgrid(g, g, indexing="ij")
    query_deg = np.column_stack([T1.ravel(), T2.ravel()])  # (M,2)
    true_k = energy_fn(T1, T2) * EV2KCAL
    e_max = args.e_max if args.e_max is not None else float(true_k.max())
    e_levels = np.linspace(0.0, e_max, 21)

    gmin = np.unravel_index(int(true_k.argmin()), true_k.shape)
    where = f" [{surface.label()}]" if surface else ""
    print(f"Global minimum near ({g[gmin[0]]:.0f}, {g[gmin[1]]:.0f}) deg{where}; "
          f"initial points:\n{init}\n")

    # Reference uncertainty scale: max sigma from the initial-only value GP, so the
    # sigma colorbar is shared across all frames AND both arms (shows the collapse).
    ref = fit_arm(init, energy_fn(init[:, 0], init[:, 1]),
                  grad_fn(init[:, 0], init[:, 1]), False, query_deg)
    sigma_vmax = float((ref.sigma_eV * EV2KCAL).max())

    # Quiver scale: map a *typical* (median) gradient magnitude to ~30 deg, so
    # arrows on the gentle well flanks -- including the initial points -- are
    # clearly visible. Using the global max instead lets the few steep ridges
    # dominate the scale and shrinks every other arrow to near-invisibility;
    # the median is robust to those outliers (steeper points just draw longer).
    gmag = np.linalg.norm(grad_fn(query_deg[:, 0], query_deg[:, 1]), axis=-1)
    grad_scale = float(np.median(gmag)) / 30.0

    out_root = Path(args.out).expanduser()
    hist, frames = {}, {}
    for use_grad, name in [(False, "no_gradients"), (True, "gradients")]:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        hist[name], frames[name] = run_arm(
            use_grad, [row for row in init], args.n_total, T1, T2, query_deg,
            true_k, e_levels, sigma_vmax, grad_scale, out_root / name,
            energy_fn, grad_fn,
        )
        print(f"[{name}] wrote {len(hist[name])} frames to {out_root / name}")

    thr_eV = 1.0 / EV2KCAL
    print("\nCalls to within 1 kcal/mol of the global minimum:")
    for name in ("no_gradients", "gradients"):
        reached = next((n for n, b in hist[name] if b <= thr_eV), None)
        msg = f"{reached} evaluations" if reached else f"not within budget ({args.n_total})"
        print(f"  {name:13s}: {msg}")

    if args.composite:
        bad = [n for n in args.composite if not num_pairs <= n <= args.n_total]
        if bad:
            ap.error(f"--composite counts {bad} outside [{num_pairs}, {args.n_total}]")
        if surface:
            from bo_slice_common import draw_molecule
        mol_img = draw_molecule(surface) if surface else None
        plot_composite(T1, T2, true_k, frames, args.composite, num_pairs, e_levels,
                       out_root / "composite.pdf", mol_img=mol_img)
        print(f"\nwrote {out_root / 'composite.pdf'}")

    if args.seeds:
        from bo_slice_common import plot_convergence

        rng = np.random.default_rng(args.seed)
        H = {"no_gradients": [], "gradients": []}
        designs = []
        for s in range(args.seeds):
            allowed = np.flatnonzero(true_k.ravel() > args.seed_min_kcal)
            pick = rng.choice(allowed, size=args.seed_n_init, replace=False)
            init_s = [row for row in query_deg[pick]]
            designs.append(query_deg[pick])
            for use_grad, name in [(False, "no_gradients"), (True, "gradients")]:
                torch.manual_seed(args.seed + s)
                h, _ = run_arm(use_grad, init_s, args.n_total, T1, T2, query_deg, true_k,
                               e_levels, sigma_vmax, grad_scale, None, energy_fn, grad_fn)
                H[name].append([b * EV2KCAL for _, b in h])
            print(f"  seed {s + 1}/{args.seeds} done", flush=True)
        H = {k: np.array(v) for k, v in H.items()}
        np.savez(out_root / "convergence.npz", designs_deg=np.array(designs), **H)
        title = surface.label() if surface else "Synthetic 2D surface"
        print(f"\nConvergence over {args.seeds} random designs "
              f"(n_init={args.seed_n_init}, initial E > {args.seed_min_kcal:g} kcal/mol):")
        plot_convergence(H, out_root / "convergence.pdf", f"2D: {title}",
                         n0=args.seed_n_init)
        print(f"wrote {out_root / 'convergence.pdf'}")


if __name__ == "__main__":
    main()
