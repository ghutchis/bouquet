"""Gradient-enhanced periodic GP that exposes a function-*value* posterior.

This is the integration layer for gradient-enhanced BO (see
``docs/gradient_enhanced_bo.md``). It trains a GP on stacked ``[E, dE/dtheta]``
observations using :class:`bouquet.periodic_grad_kernel.PeriodicKernelGrad`, but
exposes only the marginal posterior over the function value ``E``. Acquisition
functions that expect a single-output model -- BoTorch's
``LogExpectedImprovement`` and the PiBO ``PriorGuidedAcquisitionFunction`` -- can
therefore use it unchanged: they never see the derivative outputs, which exist
only to sharpen the value posterior.

Conventions
-----------
The model is unit-agnostic: it takes inputs ``X`` (n x d), values ``Y`` (n x 1)
and gradients ``dY/dX`` (n x d) that must all be expressed in the *same*
coordinate system and scaling. The caller (the solver) is responsible for the
``train_X / 360`` normalization, the maximization sign flip, and standardizing
``Y`` (with the matching ``1/std`` factor on the gradients).

Period note: for inputs normalized so that a full turn is 1.0 (the solver's
``/360`` convention) the correct periodic ``period_length`` is ``1.0``, not the
``360`` used by the legacy value-only ``_periodic_covar_module`` -- see the
module docstring of ``periodic_grad_kernel`` and the project notes.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from bouquet.periodic_grad_kernel import PeriodicKernelGrad

_JITTER = 1e-6


def _softplus_inv(x: float) -> float:
    # Stable inverse of softplus: for large x, softplus(x) ~= x.
    if x > 20.0:
        return x
    return math.log(math.expm1(x))


class GradientEnhancedPeriodicGP(Model):
    """Derivative-informed periodic GP with a value-only posterior.

    Args:
        train_X: Training inputs, shape (n, d).
        train_Y: Training values, shape (n, 1) or (n,).
        train_grad: Training gradients dY/dX, shape (n, d).
        period: Periodic kernel period in the input coordinate (1.0 for
            full-turn-normalized inputs).
        value_noise / grad_noise: Initial homoskedastic noise on the value and
            gradient observations.
    """

    num_outputs = 1

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        train_grad: torch.Tensor,
        grad_mask: torch.Tensor | None = None,
        period: float = 1.0,
        value_noise: float = 1e-2,
        grad_noise: float = 1e-2,
    ) -> None:
        super().__init__()
        train_Y = train_Y.reshape(-1, 1)
        n, d = train_X.shape
        if train_grad.shape != (n, d):
            raise ValueError(
                f"train_grad shape {tuple(train_grad.shape)} != (n, d) = {(n, d)}"
            )
        self._d = d
        self._train_X = train_X
        # Per-point gradient mask: every point contributes its value; only masked
        # points contribute gradient observations (failed/NaN gradients -> False).
        if grad_mask is None:
            grad_mask = torch.ones(n, dtype=torch.bool, device=train_X.device)
        self._grad_mask = grad_mask
        # Zero out excluded (possibly NaN) gradient rows so they can't poison the
        # stack even though _keep drops them anyway.
        train_grad = torch.where(grad_mask.unsqueeze(-1), train_grad, 0.0)
        # Interleaved targets [y_0, g_0,1..d, y_1, ...] matching the kernel layout.
        self._train_stack = torch.cat([train_Y, train_grad], dim=1).reshape(-1, 1)
        # Indices into the full interleaved vector that we actually condition on.
        blk = d + 1
        keep = []
        for i in range(n):
            keep.append(i * blk)  # value entry (always kept)
            if bool(grad_mask[i]):
                keep.extend(range(i * blk + 1, i * blk + blk))  # gradient entries
        self._keep = torch.tensor(keep, dtype=torch.long, device=train_X.device)

        self.kernel = PeriodicKernelGrad().to(train_X)
        self.kernel.period_length = torch.full((1, 1), float(period)).to(train_X)
        self.kernel.lengthscale = torch.full((1, 1), 1.0).to(train_X)
        # The /360 input normalization fixes the period exactly (full turn = 1.0),
        # and the gradient unit-scaling in the solver assumes it, so the period is
        # a structural constant, not a fitted hyperparameter -- freeze it.
        self.kernel.raw_period_length.requires_grad_(False)

        # Free hyperparameters (softplus for the positive ones). The noises seed
        # raw_value_noise/raw_grad_noise via _softplus_inv, which is only defined
        # for positive inputs, so require them strictly positive.
        if value_noise <= 0 or grad_noise <= 0:
            raise ValueError(
                f"value_noise and grad_noise must be > 0 (got value_noise="
                f"{value_noise}, grad_noise={grad_noise})"
            )
        self.raw_outputscale = torch.nn.Parameter(torch.tensor(_softplus_inv(1.0)))
        self.raw_value_noise = torch.nn.Parameter(
            torch.tensor(_softplus_inv(value_noise))
        )
        self.raw_grad_noise = torch.nn.Parameter(
            torch.tensor(_softplus_inv(grad_noise))
        )
        self.mean_const = torch.nn.Parameter(torch.zeros(()))
        self.to(train_X)
        self._condition()

    # ---- hyperparameter accessors -------------------------------------------
    @property
    def outputscale(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_outputscale)

    @property
    def value_noise(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_value_noise)

    @property
    def grad_noise(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_grad_noise)

    # ---- covariance helpers --------------------------------------------------
    def _aug(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Outputscale-scaled augmented covariance, shape (|A|(d+1), |B|(d+1))."""
        return self.outputscale * self.kernel.forward(A, B)

    def _kept_column(self, value_fill, grad_fill) -> torch.Tensor:
        """Per-kept-entry vector with ``value_fill`` on value entries and
        ``grad_fill`` on gradient entries (used for the noise and mean stacks)."""
        n, d = self._train_X.shape
        col = torch.empty(n, d + 1, dtype=self._train_X.dtype, device=self._train_X.device)
        col[:, 0] = value_fill
        col[:, 1:] = grad_fill
        return col.reshape(-1)[self._keep]

    # ---- conditioning / fitting ---------------------------------------------
    def _kept_train_cov(self) -> torch.Tensor:
        """Augmented training covariance restricted to the kept entries, with
        per-entry observation noise and jitter added to the diagonal."""
        keep = self._keep
        K = self._aug(self._train_X, self._train_X)[keep][:, keep]
        K.diagonal().add_(self._kept_column(self.value_noise, self.grad_noise) + _JITTER)
        return K

    def _factorize(self):
        """Cholesky factor, residual, and alpha for the current hyperparameters."""
        L = torch.linalg.cholesky(self._kept_train_cov())
        resid = self._train_stack[self._keep] - self._kept_column(self.mean_const, 0.0).unsqueeze(-1)
        alpha = torch.cholesky_solve(resid, L)
        return L, resid, alpha

    def _condition(self) -> None:
        """Factorize the (kept) training covariance and cache L and alpha."""
        self._L, _, self._alpha = self._factorize()

    def fit(self, steps: int = 200, lr: float = 0.05) -> GradientEnhancedPeriodicGP:
        """Fit hyperparameters by maximizing the (augmented) marginal likelihood."""
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            L, resid, alpha = self._factorize()
            nll = 0.5 * (resid * alpha).sum() + torch.log(torch.diagonal(L)).sum()
            nll.backward()
            opt.step()
        with torch.no_grad():
            self._condition()
        return self

    # ---- value posterior -----------------------------------------------------
    def posterior(
        self,
        X: torch.Tensor,
        output_indices=None,
        observation_noise: bool = False,
        posterior_transform=None,
        **kwargs,
    ) -> GPyTorchPosterior:
        d = self._d
        # Acquisition optimizers may hand us inputs in a different dtype (e.g.
        # float32 bounds); compute in the training dtype to avoid mismatches.
        X = X.to(self._train_X)
        bshape = X.shape[:-2]
        q = X.shape[-2]
        Xf = X.reshape(-1, d)  # (M, d), M = prod(bshape) * q
        M = Xf.shape[0]

        # cross-cov between test VALUES and the kept training [value, grad] entries.
        kc = self._aug(Xf, self._train_X)[0 :: (d + 1), :][:, self._keep]  # (M, Nkeep)
        mean = self.mean_const + (kc @ self._alpha).reshape(M)  # (M,)

        # Acquisition uses q=1, so only the marginal value variance is needed.
        # The periodic value self-covariance is 1, so the prior variance is the
        # outputscale; this avoids forming the full M x M posterior covariance.
        v = torch.cholesky_solve(kc.transpose(-1, -2), self._L)  # (Nkeep, M)
        var = self.outputscale - (kc * v.transpose(-1, -2)).sum(-1)  # (M,)
        if observation_noise:
            var = var + self.value_noise
        var = var.clamp_min(_JITTER)

        mean_b = mean.reshape(*bshape, q)
        cov_b = torch.diag_embed(var.reshape(*bshape, q))  # (*b, q, q)
        mvn = MultivariateNormal(mean_b, cov_b)
        posterior = GPyTorchPosterior(distribution=mvn)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior
