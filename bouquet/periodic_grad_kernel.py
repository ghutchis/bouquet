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
    that kernel it carries **no** outputscale; wrap it in a
    :class:`~gpytorch.kernels.ScaleKernel` to add one. Lengthscale and period
    follow :class:`~gpytorch.kernels.PeriodicKernel`; pass ``ard_num_dims=d`` for
    per-torsion hyperparameters or leave it unset to share one set across all
    torsions (mirroring the current ``ProductStructureKernel`` surrogate).
    """

    def _perdim_params(self, d: int):
        """Per-dimension lengthscale and period, each broadcastable to (..., 1, 1, d)."""
        # PeriodicKernel stores both as (*batch, 1, ard_num_dims); ard is 1 or d.
        ell = self.lengthscale[..., 0, :]  # (*batch, ard)
        period = self.period_length[..., 0, :]  # (*batch, ard)
        ell = ell.unsqueeze(-2).unsqueeze(-2)  # (*batch, 1, 1, ard)
        period = period.unsqueeze(-2).unsqueeze(-2)  # (*batch, 1, 1, ard)
        return ell, period

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
            D = D.reshape(*batch_shape, -1).expand(*batch_shape, d)  # (*batch, d)
            grad = D.unsqueeze(-2).expand(*batch_shape, n1, d)  # (*batch, n1, d)
            # grouped [value(n1), grad dim-major (d*n1)] then interleave
            k_diag = torch.cat(
                [value, grad.transpose(-1, -2).reshape(*batch_shape, n1 * d)], dim=-1
            )
            pi = _shuffle_index(n1, d, x1.device)
            return k_diag[..., pi]

        delta = x1.unsqueeze(-2) - x2.unsqueeze(-3)  # (..., n1, n2, d)
        u = PI * delta / period
        sin_u = torch.sin(u)
        sin_2u = torch.sin(2.0 * u)
        cos_2u = torch.cos(2.0 * u)

        # Value block K_11 = prod_m exp(-2 sin^2 u_m / l_m)  (l to first power, per gpytorch)
        K11 = torch.exp((-2.0 * sin_u.pow(2) / ell).sum(dim=-1))  # (..., n1, n2)

        # Per-dim first-derivative factor g_m (= dK/dx1_m / K) and curvature D_m.
        g = -(2.0 * PI) / (ell * period) * sin_2u  # (..., n1, n2, d)
        D = (4.0 * PI**2) / (ell * period.pow(2)) * cos_2u  # (..., n1, n2, d)

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
