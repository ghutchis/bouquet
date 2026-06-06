"""Tests for bouquet.gradient_gp.GradientEnhancedPeriodicGP.

Three properties:

1. The value posterior (mean + variance) matches an independent autograd-built
   reference of the gradient-enhanced GP -- conditioning on [E, dE/dtheta] and
   reading off only the E marginal.
2. As the gradient-noise -> infinity (gradients ignored) the value posterior
   collapses onto the plain value-only periodic GP. This is the "drops in
   unchanged" guarantee: with no usable gradient information the surrogate is
   exactly the existing one.
3. It is a valid BoTorch single-output model: LogExpectedImprovement and
   optimize_acqf run on it and return an in-bounds candidate.
"""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")

from botorch.acquisition.analytic import LogExpectedImprovement  # noqa: E402
from botorch.optim import optimize_acqf  # noqa: E402
from torch.func import jacrev  # noqa: E402

from bouquet.gradient_gp import GradientEnhancedPeriodicGP, _softplus_inv  # noqa: E402

torch.manual_seed(0)
PI = math.pi


def _set_hypers(model, outputscale, lengthscale, period, value_noise, grad_noise, mean):
    model.kernel.lengthscale = torch.tensor([[lengthscale]], dtype=torch.float64)
    model.kernel.period_length = torch.tensor([[period]], dtype=torch.float64)
    model.raw_outputscale.data = torch.tensor(_softplus_inv(outputscale))
    model.raw_value_noise.data = torch.tensor(_softplus_inv(value_noise))
    model.raw_grad_noise.data = torch.tensor(_softplus_inv(grad_noise))
    model.mean_const.data = torch.tensor(float(mean))
    model._condition()


def _reference_value_posterior(X, y, g, Xs, s, ell, p, nv, ng, mu, mask=None):
    """Independent gradient-GP value posterior via autograd-built blocks.

    ``mask`` (n,) bool selects which points contribute gradient observations;
    None means all do.
    """
    n, d = X.shape
    if mask is None:
        mask = torch.ones(n, dtype=torch.bool)

    def k(a, b):  # unscaled scalar kernel
        u = PI * (a - b) / p
        return torch.exp((-2.0 * torch.sin(u).pow(2) / ell).sum())

    gB = jacrev(k, argnums=1)
    H = jacrev(jacrev(k, argnums=0), argnums=1)
    blk = d + 1

    # Kept interleaved indices: value always, gradients only for masked points.
    keep = []
    for i in range(n):
        keep.append(i * blk)
        if bool(mask[i]):
            keep.extend(range(i * blk + 1, i * blk + blk))

    # Full augmented training covariance (scaled), then restrict to kept entries.
    K = torch.zeros(n * blk, n * blk, dtype=X.dtype)
    for i in range(n):
        for j in range(n):
            b = torch.zeros(blk, blk, dtype=X.dtype)
            b[0, 0] = k(X[i], X[j])
            b[0, 1:] = jacrev(k, 1)(X[i], X[j])
            b[1:, 0] = jacrev(k, 0)(X[i], X[j])
            b[1:, 1:] = H(X[i], X[j])
            K[i * blk : (i + 1) * blk, j * blk : (j + 1) * blk] = b
    K = s * K
    noise = torch.zeros(n * blk, dtype=X.dtype)
    noise[0::blk] = nv
    noise[[r for r in range(n * blk) if r % blk != 0]] = ng
    K = K + torch.diag(noise)

    ystack = torch.cat([y.reshape(n, 1), g], dim=1).reshape(-1, 1)
    mstack = torch.zeros(n * blk, 1, dtype=X.dtype)
    mstack[0::blk] = mu

    keep = torch.tensor(keep)
    Kk = K[keep][:, keep] + 1e-6 * torch.eye(len(keep), dtype=X.dtype)
    L = torch.linalg.cholesky(Kk)
    alpha = torch.cholesky_solve((ystack - mstack)[keep], L)

    means, varis = [], []
    for xs in Xs:
        kc_full = torch.zeros(1, n * blk, dtype=X.dtype)
        for j in range(n):
            kc_full[0, j * blk] = s * k(xs, X[j])
            kc_full[0, j * blk + 1 : (j + 1) * blk] = s * gB(xs, X[j])
        kc = kc_full[:, keep]
        m = mu + (kc @ alpha).reshape(())
        v = s * k(xs, xs) - (kc @ torch.cholesky_solve(kc.t(), L)).reshape(())
        means.append(m)
        varis.append(v)
    return torch.stack(means), torch.stack(varis)


def test_value_posterior_matches_autograd_reference():
    d, n = 2, 6
    X = torch.rand(n, d, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    g = torch.randn(n, d, dtype=torch.float64)
    s, ell, p, nv, ng, mu = 1.7, 0.6, 1.0, 1e-2, 2e-2, 0.3

    model = GradientEnhancedPeriodicGP(X, y.reshape(-1, 1), g, period=p)
    _set_hypers(model, s, ell, p, nv, ng, mu)

    Xs = torch.rand(5, d, dtype=torch.float64)
    post = model.posterior(Xs.unsqueeze(-2))  # (5, 1, d) -> q=1
    mean = post.mean.reshape(-1)
    var = post.variance.reshape(-1)

    ref_mean, ref_var = _reference_value_posterior(X, y, g, Xs, s, ell, p, nv, ng, mu)
    assert torch.allclose(mean, ref_mean, atol=1e-6), (mean - ref_mean).abs().max()
    assert torch.allclose(var, ref_var, atol=1e-6), (var - ref_var).abs().max()


def test_grad_mask_excludes_gradient_rows():
    """Per-point grad_mask: masked points contribute value only, not gradient."""
    d, n = 2, 6
    X = torch.rand(n, d, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    g = torch.randn(n, d, dtype=torch.float64)
    mask = torch.tensor([True, False, True, True, False, True])
    # NaN in the excluded gradient rows must not affect the result.
    g_in = g.clone()
    g_in[~mask] = float("nan")
    s, ell, p, nv, ng, mu = 1.4, 0.7, 1.0, 1e-2, 3e-2, -0.2

    model = GradientEnhancedPeriodicGP(
        X, y.reshape(-1, 1), g_in, grad_mask=mask, period=p
    )
    _set_hypers(model, s, ell, p, nv, ng, mu)

    Xs = torch.rand(5, d, dtype=torch.float64)
    post = model.posterior(Xs.unsqueeze(-2))
    ref_mean, ref_var = _reference_value_posterior(
        X, y, g, Xs, s, ell, p, nv, ng, mu, mask=mask
    )
    assert torch.allclose(post.mean.reshape(-1), ref_mean, atol=1e-6)
    assert torch.allclose(post.variance.reshape(-1), ref_var, atol=1e-6)


def test_all_false_mask_is_value_only_gp():
    """grad_mask all-False is exactly the plain value-only periodic GP."""
    d, n = 2, 7
    X = torch.rand(n, d, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    g = torch.full((n, d), float("nan"), dtype=torch.float64)
    s, ell, p, nv, mu = 1.1, 0.9, 1.0, 1e-2, 0.0

    model = GradientEnhancedPeriodicGP(
        X, y.reshape(-1, 1), g, grad_mask=torch.zeros(n, dtype=torch.bool), period=p
    )
    _set_hypers(model, s, ell, p, nv, grad_noise=1e-2, mean=mu)

    def k(a, b):
        u = PI * (a - b) / p
        return torch.exp((-2.0 * torch.sin(u).pow(2) / ell).sum())

    Kvv = s * torch.stack([torch.stack([k(X[i], X[j]) for j in range(n)]) for i in range(n)])
    Kvv = Kvv + nv * torch.eye(n, dtype=X.dtype) + 1e-6 * torch.eye(n, dtype=X.dtype)
    Lv = torch.linalg.cholesky(Kvv)
    alpha_v = torch.cholesky_solve(y.reshape(-1, 1), Lv)

    Xs = torch.rand(4, d, dtype=torch.float64)
    post = model.posterior(Xs.unsqueeze(-2))
    for idx, xs in enumerate(Xs):
        kv = s * torch.stack([k(xs, X[j]) for j in range(n)]).reshape(1, -1)
        m_ref = (kv @ alpha_v).reshape(())
        v_ref = s * k(xs, xs) - (kv @ torch.cholesky_solve(kv.t(), Lv)).reshape(())
        assert torch.allclose(post.mean.reshape(-1)[idx], m_ref, atol=1e-6)
        assert torch.allclose(post.variance.reshape(-1)[idx], v_ref, atol=1e-6)


def test_reduces_to_value_only_gp_when_gradients_ignored():
    """grad_noise -> infinity must recover the plain value-only periodic GP."""
    d, n = 2, 7
    X = torch.rand(n, d, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    g = torch.randn(n, d, dtype=torch.float64)
    s, ell, p, nv, mu = 1.3, 0.8, 1.0, 1e-2, 0.0

    model = GradientEnhancedPeriodicGP(X, y.reshape(-1, 1), g, period=p)
    _set_hypers(model, s, ell, p, nv, grad_noise=1e8, mean=mu)

    # Value-only reference GP with the same value kernel and noise.
    def k(a, b):
        u = PI * (a - b) / p
        return torch.exp((-2.0 * torch.sin(u).pow(2) / ell).sum())

    Kvv = s * torch.stack([torch.stack([k(X[i], X[j]) for j in range(n)]) for i in range(n)])
    Kvv = Kvv + nv * torch.eye(n, dtype=X.dtype) + 1e-6 * torch.eye(n, dtype=X.dtype)
    Lv = torch.linalg.cholesky(Kvv)
    alpha_v = torch.cholesky_solve(y.reshape(-1, 1), Lv)

    Xs = torch.rand(4, d, dtype=torch.float64)
    post = model.posterior(Xs.unsqueeze(-2))
    for idx, xs in enumerate(Xs):
        kv = s * torch.stack([k(xs, X[j]) for j in range(n)]).reshape(1, -1)
        m_ref = (kv @ alpha_v).reshape(())
        v_ref = s * k(xs, xs) - (kv @ torch.cholesky_solve(kv.t(), Lv)).reshape(())
        assert torch.allclose(post.mean.reshape(-1)[idx], m_ref, atol=1e-4)
        assert torch.allclose(post.variance.reshape(-1)[idx], v_ref, atol=1e-4)


def test_botorch_acquisition_integration():
    """LogExpectedImprovement + optimize_acqf run and return an in-bounds point."""
    d, n = 2, 8
    X = torch.rand(n, d, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    g = 0.1 * torch.randn(n, d, dtype=torch.float64)

    model = GradientEnhancedPeriodicGP(X, y.reshape(-1, 1), g, period=1.0)
    model.fit(steps=30)

    acqf = LogExpectedImprovement(model, best_f=y.max(), maximize=True)
    bounds = torch.zeros(2, d, dtype=torch.float64)
    bounds[1] = 1.0
    candidate, value = optimize_acqf(
        acqf, bounds=bounds, q=1, num_restarts=4, raw_samples=16
    )
    assert candidate.shape == (1, d)
    assert torch.isfinite(value).all()
    assert (candidate >= 0).all() and (candidate <= 1).all()
