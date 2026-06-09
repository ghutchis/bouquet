"""Phase-1 gate: decide the gradient-enhanced GP kernel route (a) vs (b).

This is a *synthetic* validation harness, not a unit test and not part of the
shipped package. It answers one question before we commit to the harder
value-posterior BoTorch wrapper:

    Which periodic gradient-enhanced GP kernel should bouquet use?

    (a) sin/cos input embedding + RBF      -> k(d) = exp(-(1 - cos d) / l^2)
    (b) native periodic kernel             -> k(d) = exp(-2 sin^2(pi d / P) / l^2)

Both are smooth periodic kernels, so we can build *both* gradient-enhanced GPs
in identical inference code and let the kernel be the only variable. We obtain
every first/second cross-derivative the gradient-enhanced covariance needs from
``torch.func`` autograd -- for a validation gate we only need the blocks to be
*correct*, not in closed form (the closed forms are only needed later, for the
production gpytorch kernel, and only for route (b)).

What we measure, mirroring the Phase-0 finite-difference gate:

  1. Posterior-mean RMSE vs. training size n (data efficiency). A gradient-
     enhanced GP must beat the value-only GP at equal n or the effort is moot.
  2. Posterior-gradient error vs. the analytic gradient (the Phase-0 tolerance,
     lifted to the GP posterior).
  3. Conditioning of the gradient-augmented covariance (the practical failure
     mode that separates the two routes more than accuracy does).

Run:  KMP_DUPLICATE_LIB_OK=TRUE pixi run python scripts/grad_kernel_gate.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch
from torch.func import jacrev, vmap

torch.set_default_dtype(torch.float64)
TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Synthetic periodic test functions (torsion-like Fourier potentials).
# Each returns a scalar; gradients come from autograd so "truth" is exact.
# ---------------------------------------------------------------------------
def f_1d(x: torch.Tensor) -> torch.Tensor:
    t = x[0]
    return torch.cos(t) + 0.6 * torch.cos(2 * t - 0.7) + 0.3 * torch.cos(3 * t + 1.2)


def f_2d(x: torch.Tensor) -> torch.Tensor:
    t1, t2 = x[0], x[1]
    return torch.cos(t1) + torch.cos(t2) + 0.5 * torch.cos(t1 - t2)


def f_3d(x: torch.Tensor) -> torch.Tensor:
    t1, t2, t3 = x[0], x[1], x[2]
    return (
        torch.cos(t1)
        + 0.7 * torch.cos(2 * t2)
        + torch.cos(t3)
        + 0.4 * torch.cos(t1 - t3)
    )


@dataclass
class TestProblem:
    name: str
    d: int
    fn: Callable[[torch.Tensor], torch.Tensor]


PROBLEMS = [
    TestProblem("1D (3-term Fourier)", 1, f_1d),
    TestProblem("2D (coupled)", 2, f_2d),
    TestProblem("3D (coupled)", 3, f_3d),
]


def eval_with_grad(fn: Callable, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized (value, gradient) of ``fn`` over a batch of points X (n, d)."""
    vals = vmap(fn)(X)
    grads = vmap(jacrev(fn))(X)
    return vals, grads


# ---------------------------------------------------------------------------
# Per-dim periodic kernels. d = x - x' (radians). Product over dims gives the
# full kernel; outputscale and lengthscale are applied in the GP wrapper.
# ---------------------------------------------------------------------------
def _embedding_perdim(d: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:
    # (a) RBF on (cos, sin): ||e(t)-e(t')||^2 = 2 - 2 cos d  ->  exp(-(1-cos d)/l^2)
    return torch.exp(-(1.0 - torch.cos(d)) / (ell**2))


def _periodic_perdim(d: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:
    # (b) gpytorch-style periodic kernel, period = 2*pi
    s = torch.sin(math.pi * d / TWO_PI)
    return torch.exp(-2.0 * s**2 / (ell**2))


def make_kernel(perdim: Callable) -> Callable:
    """Build k(x1, x2, outputscale, lengthscale) -> scalar from a per-dim factor."""

    def k(x1: torch.Tensor, x2: torch.Tensor, os: torch.Tensor, ell: torch.Tensor):
        factors = perdim(x1 - x2, ell)
        return os * torch.prod(factors)

    return k


# ---------------------------------------------------------------------------
# Gradient-enhanced exact GP. Observation block per point is [f, df/dx_1, ...].
# Every kernel block comes from autograd of the scalar kernel k(x1, x2), but
# evaluated *batched* over all point pairs at once via vmap (the Python-loop
# version was ~100x slower). Posterior gradients are assembled from the same
# kernel-derivative primitives rather than nesting autograd at predict time.
# ---------------------------------------------------------------------------
class GradGP:
    def __init__(self, kernel: Callable, with_grad: bool):
        self.kernel = kernel
        self.with_grad = with_grad

    def _primitives(self, A, B, os, ell):
        """Batched kernel and its derivatives over all pairs (A_i, B_j).

        Returns (kv, dB, dA, h) with shapes (p,q), (p,q,d), (p,q,d), (p,q,d,d).
        dB = d k / d B  (grad wrt 2nd arg),  dA = d k / d A,  h = d^2k / dA dB.
        """
        k = lambda a, b: self.kernel(a, b, os, ell)
        p, q, d = A.shape[0], B.shape[0], A.shape[1]
        Af = A[:, None, :].expand(p, q, d).reshape(-1, d)
        Bf = B[None, :, :].expand(p, q, d).reshape(-1, d)
        kv = vmap(k)(Af, Bf).reshape(p, q)
        if not self.with_grad:
            return kv, None, None, None
        dB = vmap(jacrev(k, argnums=1))(Af, Bf).reshape(p, q, d)
        dA = vmap(jacrev(k, argnums=0))(Af, Bf).reshape(p, q, d)
        h = vmap(jacrev(jacrev(k, argnums=0), argnums=1))(Af, Bf).reshape(p, q, d, d)
        return kv, dB, dA, h

    def _assemble(self, kv, dB, dA, h):
        """Stack per-pair (1+d, 1+d) blocks into a full (p(1+d), q(1+d)) matrix."""
        p, q = kv.shape
        if not self.with_grad:
            return kv
        d = dB.shape[-1]
        M = torch.empty(p, 1 + d, q, 1 + d)
        M[:, 0, :, 0] = kv          # cov(f_i, f_j)
        M[:, 0, :, 1:] = dB         # cov(f_i, g_j)
        M[:, 1:, :, 0] = dA.permute(0, 2, 1)   # cov(g_i, f_j)
        M[:, 1:, :, 1:] = h.permute(0, 2, 1, 3)  # cov(g_i, g_j)
        return M.reshape(p * (1 + d), q * (1 + d))

    def _full_K(self, X, os, ell):
        return self._assemble(*self._primitives(X, X, os, ell))

    @staticmethod
    def _stack_targets(vals, grads, with_grad):
        if not with_grad:
            return vals.reshape(-1, 1)
        n, d = grads.shape
        return torch.cat([vals.reshape(n, 1), grads], dim=1).reshape(-1, 1)

    # ---- fit / predict -------------------------------------------------------
    def fit(self, X, vals, grads, steps=150, lr=0.05):
        y = self._stack_targets(vals, grads, self.with_grad)
        self._y, self._X = y, X
        log_os = torch.zeros(1, requires_grad=True)
        log_ell = torch.zeros(1, requires_grad=True)
        log_noise = torch.full((1,), math.log(1e-3), requires_grad=True)
        opt = torch.optim.Adam([log_os, log_ell, log_noise], lr=lr)
        N = y.shape[0]
        for _ in range(steps):
            opt.zero_grad()
            os = torch.exp(log_os)
            ell = torch.exp(log_ell).clamp(min=1e-2)
            noise = torch.exp(log_noise).clamp(min=1e-6)
            K = self._full_K(X, os, ell) + noise * torch.eye(N)
            L = torch.linalg.cholesky(K)
            alpha = torch.cholesky_solve(y, L)
            nll = 0.5 * (y * alpha).sum() + torch.log(torch.diagonal(L)).sum()
            nll.backward()
            opt.step()
        with torch.no_grad():
            self.os = torch.exp(log_os).detach()
            self.ell = torch.exp(log_ell).clamp(min=1e-2).detach()
            self.noise = torch.exp(log_noise).clamp(min=1e-6).detach()
            K = self._full_K(X, self.os, self.ell) + self.noise * torch.eye(N)
            self._L = torch.linalg.cholesky(K)
            self._alpha = torch.cholesky_solve(y, self._L)
            self._condK = torch.linalg.cond(K).item()
        return self

    def posterior_mean_and_grad(self, Xstar):
        """Posterior mean and its gradient at test points, both in closed form.

        mean(x*)      = cov(f(x*), train) @ alpha
        grad mean(x*) = cov(grad f(x*), train) @ alpha
        Both cross-covariances reuse the kernel-derivative primitives between
        the test points (A = Xstar) and the training points (B = X).
        """
        with torch.no_grad():
            kv, dB, dA, h = self._primitives(Xstar, self._X, self.os, self.ell)
            m, n = kv.shape
            d = Xstar.shape[1]
            if self.with_grad:
                # cross(f*, [f_j, g_j]) : (m, n*(1+d))
                cf = torch.empty(m, n, 1 + d)
                cf[:, :, 0] = kv
                cf[:, :, 1:] = dB
                mean = (cf.reshape(m, n * (1 + d)) @ self._alpha).reshape(m)
                # cross(g*_a, [f_j, g_j]) : (m, d, n*(1+d))
                cg = torch.empty(m, d, n, 1 + d)
                cg[:, :, :, 0] = dA.permute(0, 2, 1)
                cg[:, :, :, 1:] = h.permute(0, 2, 1, 3)
                gmean = (cg.reshape(m, d, n * (1 + d)) @ self._alpha).reshape(m, d)
            else:
                mean = (kv @ self._alpha).reshape(m)
                # value-only GP: still differentiate the cross-cov wrt x* for
                # the posterior-mean gradient (dA = d k(x*, X_j) / d x*).
                k = lambda a, b: self.kernel(a, b, self.os, self.ell)
                Af = Xstar[:, None, :].expand(m, n, d).reshape(-1, d)
                Bf = self._X[None, :, :].expand(m, n, d).reshape(-1, d)
                dA = vmap(jacrev(k, argnums=0))(Af, Bf).reshape(m, n, d)
                gmean = (dA.permute(0, 2, 1) @ self._alpha).reshape(m, d)
        return mean, gmean


# ---------------------------------------------------------------------------
# Sampling / metrics
# ---------------------------------------------------------------------------
def sobol_like(n, d, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, d, generator=g) * TWO_PI


def dense_grid(d, per_dim=24):
    axis = torch.linspace(0, TWO_PI, per_dim + 1)[:-1]
    mesh = torch.meshgrid(*([axis] * d), indexing="ij")
    return torch.stack([m.reshape(-1) for m in mesh], dim=1)


def rmse(a, b):
    return torch.sqrt(torch.mean((a - b) ** 2)).item()


KERNELS = {
    "value-only (periodic)": (make_kernel(_periodic_perdim), False),
    "(a) embedding+RBF": (make_kernel(_embedding_perdim), True),
    "(b) native periodic": (make_kernel(_periodic_perdim), True),
}


def run_problem(problem: TestProblem, n_values: list[int], seeds: list[int]):
    Xtest = dense_grid(problem.d, per_dim=16 if problem.d <= 2 else 8)
    ytest, gtest = eval_with_grad(problem.fn, Xtest)
    results: dict[str, dict[int, dict[str, list[float]]]] = {
        name: {n: {"mean": [], "grad": [], "cond": []} for n in n_values}
        for name in KERNELS
    }
    for n in n_values:
        for seed in seeds:
            Xtr = sobol_like(n, problem.d, seed)
            ytr, gtr = eval_with_grad(problem.fn, Xtr)
            for name, (kern, with_grad) in KERNELS.items():
                gp = GradGP(kern, with_grad).fit(Xtr, ytr, gtr)
                mean, gmean = gp.posterior_mean_and_grad(Xtest)
                results[name][n]["mean"].append(rmse(mean, ytest))
                results[name][n]["grad"].append(rmse(gmean, gtest))
                results[name][n]["cond"].append(gp._condK)
    return results


def fmt(vals: list[float]) -> str:
    t = torch.tensor(vals)
    return f"{t.mean():.4f}+-{t.std():.4f}"


def kernel_identity_check():
    """Show analytically/numerically that (a) and (b) are the SAME kernel.

    embedding-RBF:  exp(-(1 - cos d) / l^2)
    native periodic: exp(-2 sin^2(d/2) / l^2)   and   2 sin^2(d/2) = 1 - cos d
    => identical at equal lengthscale. So accuracy/conditioning cannot separate
    the two routes; only the *implementation* differs.
    """
    d = torch.linspace(-math.pi, math.pi, 401)
    ell = torch.tensor(0.8)
    a = torch.exp(-(1.0 - torch.cos(d)) / ell**2)
    b = torch.exp(-2.0 * torch.sin(math.pi * d / TWO_PI) ** 2 / ell**2)
    print("=" * 78)
    print("Kernel identity  (a) embedding-RBF  vs  (b) native periodic")
    print("=" * 78)
    print(f"  max |k_a(d) - k_b(d)| over [-pi, pi] at equal lengthscale: "
          f"{(a - b).abs().max().item():.2e}")
    print("  => same function; the gate's matching rows below are exact, not a bug.\n")


def cost_model():
    """The thing that actually differs: gradient-augmented matrix size and the
    nature of the gradient observation, per torsion dimension d, n points."""
    print("=" * 78)
    print("What differs between the routes (per d torsions, n points)")
    print("=" * 78)
    print("  (b) native PeriodicKernelGrad:")
    print("      - covariance dim N = n*(1 + d)        [1 gradient slot / torsion]")
    print("      - gradient obs = dE/dtheta, exactly what gradients.py returns")
    print("      - drops straight into a periodic value-posterior wrapper")
    print("  (a) stock gpytorch RBFKernelGrad on (cos,sin) embedding:")
    print("      - input dim doubles: d torsions -> 2d embedding coords")
    print("      - covariance dim N = n*(1 + 2d)       [2 gradient slots / torsion]")
    print("      - RBFKernelGrad wants d/d(cos), d/d(sin) SEPARATELY, but only the")
    print("        tangential combo (-sin, cos).dE/dtheta is physical; the radial")
    print("        partial is off-manifold and must be faked (tangential lift).")
    print("      - same kernel as (b), obtained less efficiently + an approximation\n")


def main():
    kernel_identity_check()
    cost_model()
    n_values = [6, 10, 16]
    seeds = list(range(6))
    for problem in PROBLEMS:
        print("\n" + "=" * 78)
        print(f"{problem.name}   (d={problem.d}, {len(seeds)} seeds)")
        print("=" * 78)
        res = run_problem(problem, n_values, seeds)
        for metric, label in [("mean", "posterior-mean RMSE"), ("grad", "gradient RMSE")]:
            print(f"\n  {label}:")
            header = "    {:<24}".format("kernel") + "".join(
                f"n={n:<14}" for n in n_values
            )
            print(header)
            for name in KERNELS:
                row = "    {:<24}".format(name)
                for n in n_values:
                    row += "{:<16}".format(fmt(res[name][n][metric]))
                print(row)
        print("\n  covariance condition number (median over seeds, largest n):")
        for name in KERNELS:
            c = torch.tensor(res[name][n_values[-1]]["cond"]).median().item()
            print(f"    {name:<24} {c:.2e}")


if __name__ == "__main__":
    main()
