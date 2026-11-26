"""Gradient-enhanced periodic GP kernel for torsion-angle Bayesian optimization.

bouquet's surrogate is a product of per-torsion periodic kernels (see
:func:`bouquet.solver._periodic_covar_module`). To feed analytic torsion
gradients ``dE/dtheta`` (see :mod:`bouquet.gradients`) into the GP we need the
*gradient-enhanced* covariance: the joint covariance of the function values and
their first derivatives. gpytorch ships :class:`~gpytorch.kernels.RBFKernelGrad`
but no periodic analogue, so this module supplies it.

Closed forms
------------
The base (product) periodic kernel, matching gpytorch's ``PeriodicKernel``
parameterization (per-dim lengthscale ``l_m`` and period ``p_m``). Note that,
following gpytorch, the ``sin^2`` term is divided by ``l_m`` to the *first*
power (not ``l_m^2``), so this shares hyperparameter semantics with the existing
``PeriodicKernel`` surrogate:

    u_m   = pi * (x1_m - x2_m) / p_m
    K     = prod_m exp(-2 sin^2(u_m) / l_m)

The blocks of the gradient-enhanced covariance are (derivations validated
against autograd in ``tests/test_periodic_grad_kernel.py``):

    dK/dx1_m            = K * g_m                     g_m = -(2 pi / (l_m p_m)) sin(2 u_m)
    dK/dx2_m            = -K * g_m
    d2K/dx1_m dx2_m'    = K * (delta_mm' D_m - g_m g_m')   D_m = (4 pi^2/(l_m p_m^2)) cos(2 u_m)

Layout
------
For inputs ``x1`` (n1 x d) and ``x2`` (n2 x d) the kernel returns an
``n1 (d+1) x n2 (d+1)`` matrix in gpytorch's interleaved (MultiTask) ordering:
per point, the value is followed by the ``d`` partial derivatives
``[f, df/dx_1, ..., df/dx_d]``. This matches ``RBFKernelGrad`` so the existing
gradient-GP machinery (and the perfect-shuffle convention) applies unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from gpytorch.kernels.periodic_kernel import PeriodicKernel

PI = math.pi


def _shuffle_index(n: int, d: int, device, dtype=torch.long) -> torch.Tensor:
    """Perfect-shuffle permutation mapping grouped -> interleaved ordering.

    Grouped order is ``[f(0..n-1), df/dx_1(0..n-1), ..., df/dx_d(0..n-1)]``;
    interleaved order is ``[f_0, df/dx_1|_0, ..., df/dx_d|_0, f_1, ...]``. Same
    convention as :class:`gpytorch.kernels.RBFKernelGrad`.
    """
    return (
        torch.arange(n * (d + 1), device=device)
        .view(d + 1, n)
        .t()
        .reshape(n * (d + 1))
    )


class PeriodicKernelGrad(PeriodicKernel):
    r"""Periodic kernel that also models value/derivative covariances.

    A drop-in periodic analogue of :class:`gpytorch.kernels.RBFKernelGrad`. Like
    that kernel it carries **no** output scale; wrap it in a
    :class:`~gpytorch.kernels.ScaleKernel` to add one. Length scale and period
    follow :class:`~gpytorch.kernels.PeriodicKernel`; pass ``ard_num_dims=d`` for
    per-torsion hyperparameters or leave it unset to share one set across all
    torsions (mirroring the current ``ProductStructureKernel`` surrogate).
    """

    def __init__(self, symmetry_orders: Sequence[int] | None = None, **kwargs):
        super().__init__(**kwargs)
        if symmetry_orders is not None and any(m > 1 for m in symmetry_orders):
            sym_sq = torch.tensor(
                [float(m * m) for m in symmetry_orders], dtype=torch.float64
            ).view(1, 1, -1)
            self.register_buffer("_sym_sq", sym_sq)
        else:
            self._sym_sq = None

    def _perdim_params(self, d: int):
        """Per-dimension lengthscale and period, each broadcastable to (..., 1, 1, d)."""
        # PeriodicKernel stores both as (*batch, 1, ard_num_dims); ard is 1 or d.
        ell = self.lengthscale[..., 0, :]  # (*batch, ard)
        period = self.period_length[..., 0, :]  # (*batch, ard)
        ell = ell.unsqueeze(-2).unsqueeze(-2)  # (*batch, 1, 1, ard)
        period = period.unsqueeze(-2).unsqueeze(-2)  # (*batch, 1, 1, ard)
        if self._sym_sq is not None:
            ell = ell * self._sym_sq.to(ell)
        return ell, period

    # ---- block accessors ----------------------------------------------------
    # The augmented covariance is assembled from three blocks. Exposing them
    # separately is what lets a caller build ONLY the sub-matrix it conditions on:
    # with a gradient window, `forward` would materialize the full
    # n1(d+1) x n2(d+1) matrix just to have most of it discarded, which at d=8 and
    # n=2138 is 2.76 GB plus a 2.18 GB Hessian intermediate -- enough to OOM a node
    # even though the retained matrix is 40x smaller. See
    # GradientEnhancedPeriodicGP._kept_train_cov.

    def _terms(self, x1, x2, need_curv: bool = True):
        """Shared per-dimension intermediates: value kernel, dK/dx factor, curvature."""
        d = x1.shape[-1]
        ell, period = self._perdim_params(d)
        u = PI * (x1.unsqueeze(-2) - x2.unsqueeze(-3)) / period
        k11 = torch.exp((-2.0 * torch.sin(u).pow(2) / ell).sum(dim=-1))
        g = -(2.0 * PI) / (ell * period) * torch.sin(2.0 * u)
        curv = (
            (4.0 * PI**2) / (ell * period.pow(2)) * torch.cos(2.0 * u)
            if need_curv
            else None
        )
        return k11, g, curv

    def value_value(self, x1, x2):
        """cov(f_i, f_j), shape (..., n1, n2).

        Accumulated one torsion at a time rather than via :meth:`_terms`, which
        would materialize an (n1, n2, d) tensor -- 293 MB at d=8 with n1=n2=2138,
        several times over for the delta/sin/square chain. This block is the one
        evaluated on the FULL point set (the others only touch the windowed
        subset), so it is where that intermediate actually hurts; accumulating
        costs d passes over an (n1, n2) buffer instead.
        """
        d = x1.shape[-1]
        ell, period = self._perdim_params(d)
        ell_v, per_v = ell[..., 0, 0, :], period[..., 0, 0, :]
        acc = None
        for m in range(d):
            lm = ell_v[..., m % ell_v.shape[-1]][..., None, None]
            pm = per_v[..., m % per_v.shape[-1]][..., None, None]
            u = PI * (x1[..., m].unsqueeze(-1) - x2[..., m].unsqueeze(-2)) / pm
            term = -2.0 * torch.sin(u).pow(2) / lm
            acc = term if acc is None else acc + term
        return torch.exp(acc)

    def value_grad(self, x1, x2):
        """cov(f_i, df_j/dx_m), shape (..., n1, n2*d), columns grouped m-major."""
        k11, g, _ = self._terms(x1, x2, need_curv=False)
        n1, n2, d = g.shape[-3], g.shape[-2], g.shape[-1]
        batch = g.shape[:-3]
        return (-k11.unsqueeze(-1) * g).transpose(-1, -2).reshape(*batch, n1, n2 * d)

    def grad_grad(self, x1, x2):
        """cov(df_i/dx_m, df_j/dx_m'), shape (..., n1*d, n2*d)."""
        k11, g, curv = self._terms(x1, x2)
        n1, n2, d = g.shape[-3], g.shape[-2], g.shape[-1]
        batch = g.shape[:-3]
        hess = k11.unsqueeze(-1).unsqueeze(-1) * (
            torch.diag_embed(curv) - g.unsqueeze(-1) * g.unsqueeze(-2)
        )
        b = len(batch)
        perm = list(range(b)) + [b + 2, b, b + 3, b + 1]
        return hess.permute(*perm).reshape(*batch, n1 * d, n2 * d)

    def forward(self, x1, x2, diag=False, **params):
        batch_shape = x1.shape[:-2]
        n1, d = x1.shape[-2:]
        n2 = x2.shape[-2]
        ell, period = self._perdim_params(d)  # (..., 1, 1, d) each (or ard=1)

        if diag:
            if not (n1 == n2 and torch.equal(x1, x2)):
                raise RuntimeError("diag=True only works when x1 == x2")
            # u = 0 along the diagonal: value 1, gradient variance D_m = 4 pi^2/(l^2 p^2).
            value = torch.ones(*batch_shape, n1, device=x1.device, dtype=x1.dtype)
            D = (4.0 * PI**2) / (ell * period.pow(2))  # (..., 1, 1, ard); ard is 1 or d
            # Broadcast per-dim variances from ard (shared: 1, or ARD: d) to d.
            D = D.squeeze(-2).squeeze(-2).expand(*batch_shape, d)  # (*batch, d)
            grad = D.unsqueeze(-2).expand(*batch_shape, n1, d)  # (*batch, n1, d)
            # grouped [value(n1), grad dim-major (d*n1)] then interleave
            k_diag = torch.cat(
                [value, grad.transpose(-1, -2).reshape(*batch_shape, n1 * d)], dim=-1
            )
            pi = _shuffle_index(n1, d, x1.device)
            return k_diag[..., pi]

        K11, g, D = self._terms(x1, x2)

        K11u = K11.unsqueeze(-1)  # (..., n1, n2, 1)

        # value-grad block: cov(f_i, df_j/dx2_m) = -K * g_m ; columns grouped m-major
        vg = (-K11u * g).transpose(-1, -2).reshape(*batch_shape, n1, n2 * d)
        # grad-value block: cov(df_i/dx1_m, f_j) = K * g_m ; rows grouped m-major
        gv = (K11u * g).movedim(-1, -3).reshape(*batch_shape, n1 * d, n2)
        # Hessian: cov(df_i/dx1_m, df_j/dx2_m') = K (delta_mm' D_m - g_m g_m')
        gg = g.unsqueeze(-1) * g.unsqueeze(-2)  # (..., n1, n2, d, d)
        hess = K11.unsqueeze(-1).unsqueeze(-1) * (torch.diag_embed(D) - gg)
        # reorder (i, j, m, m') -> (m, i, m', j) then flatten to (d*n1, d*n2)
        b = len(batch_shape)
        perm = list(range(b)) + [b + 2, b, b + 3, b + 1]
        hess = hess.permute(*perm).reshape(*batch_shape, n1 * d, n2 * d)

        # Every quadrant below is written, so no need to zero-initialize.
        K = torch.empty(
            *batch_shape, n1 * (d + 1), n2 * (d + 1), device=x1.device, dtype=x1.dtype
        )
        K[..., :n1, :n2] = K11
        K[..., :n1, n2:] = vg
        K[..., n1:, :n2] = gv
        K[..., n1:, n2:] = hess

        if n1 == n2 and torch.equal(x1, x2):
            K = 0.5 * (K + K.transpose(-1, -2))

        pi1 = _shuffle_index(n1, d, x1.device)
        pi2 = _shuffle_index(n2, d, x1.device)
        return K[..., pi1, :][..., :, pi2]

    def num_outputs_per_input(self, x1, x2):
        return x1.size(-1) + 1
