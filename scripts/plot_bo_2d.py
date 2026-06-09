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

Run:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. pixi run python scripts/plot_bo_2d.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.optim.fit import fit_gpytorch_mll_torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from bouquet.solver import _periodic_covar_module, _fit_gradient_gp

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

    energy_cap = 2.0 + torch.log10(torch.clamp(raw_y, min=1.0))
    neg = (-1.0 * torch.minimum(raw_y, energy_cap)).unsqueeze(-1)  # (n,1)
    neg_mean = neg.mean(dim=0, keepdim=True)
    y_std = neg.std(dim=0, keepdim=True)
    y_std = torch.where(y_std >= 1e-9, y_std, torch.ones_like(y_std))
    train_y = (neg - neg_mean) / y_std

    if use_gradients:
        grad = torch.tensor(obs_grad, dtype=torch.float64)  # (n, 2) eV/rad
        gp = _fit_gradient_gp(x, train_y, raw_y, energy_cap, grad, y_std)
    else:
        gp = SingleTaskGP(x, train_y, covar_module=_periodic_covar_module(2))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll_torch(
            mll, step_limit=200, optimizer=lambda p: torch.optim.Adam(p, lr=0.01)
        )
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
    cf = ax_t.contourf(T1, T2, true_k, levels=e_levels, cmap="viridis")
    ax_t.contour(T1, T2, true_k, levels=e_levels, colors="white",
                 linewidths=0.3, alpha=0.4)
    ax_t.set_title("True energy surface")
    if use_gradients:
        # downhill arrows (-gradient) at each observation
        g = obs_grad  # (n,2) eV/rad
        ax_t.quiver(
            obs_deg[:, 0], obs_deg[:, 1], -g[:, 0], -g[:, 1],
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


# ---------------------------------------------------------------------------
# BO driver
# ---------------------------------------------------------------------------
def run_arm(
    use_gradients, init_deg, n_total, T1, T2, query_deg, true_k,
    e_levels, sigma_vmax, grad_scale, out_dir,
) -> list[tuple[int, float]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_init = len(init_deg)

    obs_deg = np.array(init_deg, dtype=float)  # (n,2)
    obs_eV = true_energy(obs_deg[:, 0], obs_deg[:, 1])
    obs_grad = true_gradient(obs_deg[:, 0], obs_deg[:, 1])

    history: list[tuple[int, float]] = []
    while True:
        n = len(obs_deg)
        res = fit_arm(obs_deg, obs_eV, obs_grad, use_gradients, query_deg)
        history.append((n, res.best_eV))
        last = n >= n_total
        plot_frame(
            T1, T2, true_k, res, obs_deg, obs_grad, use_gradients, n, n_init,
            e_levels, sigma_vmax, grad_scale, out_dir / f"frame_{n:02d}.pdf",
            show_next=not last,
        )
        if last:
            break
        nd = res.next_deg
        obs_deg = np.vstack([obs_deg, nd])
        obs_eV = np.append(obs_eV, true_energy(nd[0], nd[1]))
        obs_grad = np.vstack([obs_grad, true_gradient(nd[0], nd[1])])

    return history


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figures/bo_2d")
    ap.add_argument("--n-total", type=int, default=24, help="total evaluations per arm")
    ap.add_argument("--n-grid", type=int, default=80, help="contour grid resolution per axis")
    ap.add_argument(
        "--init", type=float, nargs="+", default=[30, 60, 150, 40, 60, 300],
        help="flat list of initial (theta1, theta2) pairs in degrees; both arms share these",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    init = np.array(args.init, dtype=float).reshape(-1, 2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    _style()

    # dense grid for contours / argmax acquisition
    g = np.linspace(0.0, 360.0, args.n_grid, endpoint=False)
    T1, T2 = np.meshgrid(g, g, indexing="ij")
    query_deg = np.column_stack([T1.ravel(), T2.ravel()])  # (M,2)
    true_k = true_energy(T1, T2) * EV2KCAL
    e_levels = np.linspace(0.0, float(true_k.max()), 21)

    gmin = np.unravel_index(int(true_k.argmin()), true_k.shape)
    print(f"Global minimum near ({g[gmin[0]]:.0f}, {g[gmin[1]]:.0f}) deg; "
          f"initial points:\n{init}\n")

    # Reference uncertainty scale: max sigma from the initial-only value GP, so the
    # sigma colorbar is shared across all frames AND both arms (shows the collapse).
    ref = fit_arm(init, true_energy(init[:, 0], init[:, 1]),
                  true_gradient(init[:, 0], init[:, 1]), False, query_deg)
    sigma_vmax = float((ref.sigma_eV * EV2KCAL).max())

    # Quiver scale: map a *typical* (median) gradient magnitude to ~30 deg, so
    # arrows on the gentle well flanks -- including the initial points -- are
    # clearly visible. Using the global max instead lets the few steep ridges
    # dominate the scale and shrinks every other arrow to near-invisibility;
    # the median is robust to those outliers (steeper points just draw longer).
    gmag = np.linalg.norm(true_gradient(query_deg[:, 0], query_deg[:, 1]), axis=-1)
    grad_scale = float(np.median(gmag)) / 30.0

    out_root = Path(args.out)
    hist = {}
    for use_grad, name in [(False, "no_gradients"), (True, "gradients")]:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        hist[name] = run_arm(
            use_grad, [row for row in init], args.n_total, T1, T2, query_deg,
            true_k, e_levels, sigma_vmax, grad_scale, out_root / name,
        )
        print(f"[{name}] wrote {len(hist[name])} frames to {out_root / name}")

    thr_eV = 1.0 / EV2KCAL
    print("\nCalls to within 1 kcal/mol of the global minimum:")
    for name in ("no_gradients", "gradients"):
        reached = next((n for n, b in hist[name] if b <= thr_eV), None)
        msg = f"{reached} evaluations" if reached else f"not within budget ({args.n_total})"
        print(f"  {name:13s}: {msg}")


if __name__ == "__main__":
    main()
