"""Validation for bouquet.periodic_grad_kernel.PeriodicKernelGrad.

The core check mirrors the Phase-0 gradient gate: the closed-form
gradient-enhanced covariance (value / first-derivative / Hessian blocks) must
match autograd of the scalar product-periodic kernel to machine precision, in
gpytorch's interleaved [f, df/dx_1, ..., df/dx_d] ordering. Also checks the
value block against the stock PeriodicKernel, the diag path, and PSD.
"""

import math

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")

from torch.func import jacrev  # noqa: E402

from bouquet.periodic_grad_kernel import PeriodicKernelGrad  # noqa: E402

torch.manual_seed(0)
PI = math.pi


def _scalar_kernel(ell: torch.Tensor, period: torch.Tensor):
    """Scalar product-periodic kernel k(a, b) matching PeriodicKernelGrad."""

    def k(a, b):
        # gpytorch PeriodicKernel divides sin^2 by lengthscale to the first power.
        u = PI * (a - b) / period
        return torch.exp((-2.0 * torch.sin(u).pow(2) / ell).sum())

    return k


def _autograd_augmented(X: torch.Tensor, ell: torch.Tensor, period: torch.Tensor):
    """Reference (n(d+1), n(d+1)) interleaved covariance built from autograd."""
    n, d = X.shape
    k = _scalar_kernel(ell, period)
    gA = jacrev(k, argnums=0)
    gB = jacrev(k, argnums=1)
    H = jacrev(jacrev(k, argnums=0), argnums=1)
    blk = d + 1
    K = torch.zeros(n * blk, n * blk, dtype=X.dtype)
    for i in range(n):
        for j in range(n):
            xi, xj = X[i], X[j]
            block = torch.zeros(blk, blk, dtype=X.dtype)
            block[0, 0] = k(xi, xj)
            block[0, 1:] = gB(xi, xj)  # cov(f_i, df_j/dx_m)
            block[1:, 0] = gA(xi, xj)  # cov(df_i/dx_m, f_j)
            block[1:, 1:] = H(xi, xj)  # cov(df_i/dx_m, df_j/dx_m')
            K[i * blk : (i + 1) * blk, j * blk : (j + 1) * blk] = block
    return K


def _make_kernel(d, ell, period):
    kern = PeriodicKernelGrad(ard_num_dims=d).double()
    kern.lengthscale = ell.reshape(1, d)
    kern.period_length = period.reshape(1, d)
    return kern


@pytest.mark.parametrize("d", [1, 2, 3])
def test_closed_form_matches_autograd_shared_params(d):
    X = torch.rand(5, d, dtype=torch.float64) * (2 * PI)
    ell = torch.full((d,), 0.7, dtype=torch.float64)
    period = torch.full((d,), 2 * PI, dtype=torch.float64)
    kern = _make_kernel(d, ell, period)
    K = kern.forward(X, X)
    ref = _autograd_augmented(X, ell, period)
    assert torch.allclose(K, ref, atol=1e-9), (K - ref).abs().max()


@pytest.mark.parametrize("d", [2, 3])
def test_closed_form_matches_autograd_ard(d):
    """Per-torsion (ARD) lengthscales and periods."""
    X = torch.rand(4, d, dtype=torch.float64) * (2 * PI)
    ell = 0.4 + torch.rand(d, dtype=torch.float64)
    period = (2 * PI) * (0.5 + torch.rand(d, dtype=torch.float64))
    kern = _make_kernel(d, ell, period)
    K = kern.forward(X, X)
    ref = _autograd_augmented(X, ell, period)
    assert torch.allclose(K, ref, atol=1e-9), (K - ref).abs().max()


def test_offdiagonal_x1_neq_x2():
    """Cross-covariance between distinct input sets (no symmetrization path)."""
    d = 2
    ell = torch.tensor([0.6, 0.9], dtype=torch.float64)
    period = torch.tensor([2 * PI, 2 * PI], dtype=torch.float64)
    X1 = torch.rand(3, d, dtype=torch.float64) * (2 * PI)
    X2 = torch.rand(4, d, dtype=torch.float64) * (2 * PI)
    kern = _make_kernel(d, ell, period)
    K = kern.forward(X1, X2)
    # reference: same block formulas, rectangular
    k = _scalar_kernel(ell, period)
    gA, gB = jacrev(k, 0), jacrev(k, 1)
    H = jacrev(jacrev(k, 0), 1)
    blk = d + 1
    ref = torch.zeros(3 * blk, 4 * blk, dtype=torch.float64)
    for i in range(3):
        for j in range(4):
            b = torch.zeros(blk, blk, dtype=torch.float64)
            b[0, 0] = k(X1[i], X2[j])
            b[0, 1:] = gB(X1[i], X2[j])
            b[1:, 0] = gA(X1[i], X2[j])
            b[1:, 1:] = H(X1[i], X2[j])
            ref[i * blk : (i + 1) * blk, j * blk : (j + 1) * blk] = b
    assert torch.allclose(K, ref, atol=1e-9), (K - ref).abs().max()


def test_value_block_matches_plain_periodic_kernel():
    """The f-f sub-block must equal the stock PeriodicKernel."""
    d = 3
    ell = torch.tensor([0.7, 1.1, 0.5], dtype=torch.float64)
    period = torch.tensor([2 * PI, 2 * PI, 2 * PI], dtype=torch.float64)
    X = torch.rand(6, d, dtype=torch.float64) * (2 * PI)
    kern = _make_kernel(d, ell, period)
    K = kern.forward(X, X)
    value = K[:: d + 1, :: d + 1]  # interleaved -> value rows/cols are every (d+1)

    plain = gpytorch.kernels.PeriodicKernel(ard_num_dims=d).double()
    plain.lengthscale = ell.reshape(1, d)
    plain.period_length = period.reshape(1, d)
    Kplain = plain.forward(X, X)
    assert torch.allclose(value, Kplain, atol=1e-9), (value - Kplain).abs().max()


def test_diag_matches_full_diagonal():
    d = 2
    ell = torch.tensor([0.6, 0.8], dtype=torch.float64)
    period = torch.tensor([2 * PI, 1.5 * PI], dtype=torch.float64)
    X = torch.rand(5, d, dtype=torch.float64) * (2 * PI)
    kern = _make_kernel(d, ell, period)
    full = kern.forward(X, X)
    diag = kern.forward(X, X, diag=True)
    assert torch.allclose(diag, torch.diagonal(full), atol=1e-9)


def test_diag_matches_full_diagonal_shared_hypers():
    """Shared (ard=1) hyperparameters with d>1 -- the production configuration."""
    d = 3
    kern = PeriodicKernelGrad().double()  # ard_num_dims defaults to 1 (shared)
    kern.lengthscale = torch.tensor([[0.7]], dtype=torch.float64)
    kern.period_length = torch.tensor([[2 * PI]], dtype=torch.float64)
    X = torch.rand(4, d, dtype=torch.float64) * (2 * PI)
    full = kern.forward(X, X)
    diag = kern.forward(X, X, diag=True)
    assert torch.allclose(diag, torch.diagonal(full), atol=1e-9)


def test_augmented_matrix_is_psd():
    d = 2
    ell = torch.tensor([0.7, 0.7], dtype=torch.float64)
    period = torch.tensor([2 * PI, 2 * PI], dtype=torch.float64)
    X = torch.rand(8, d, dtype=torch.float64) * (2 * PI)
    kern = _make_kernel(d, ell, period)
    K = kern.forward(X, X)
    N = K.shape[0]
    # small jitter, as any gradient-GP needs; must factor cleanly
    torch.linalg.cholesky(K + 1e-6 * torch.eye(N, dtype=torch.float64))
    eigmin = torch.linalg.eigvalsh(K).min()
    assert eigmin > -1e-8, eigmin
