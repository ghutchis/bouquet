"""The GP covariance arms: the shipped product kernel and its ARD variant.

Only two arms survive. A first-order additive component and two pairwise
("trellis") variants were built, measured and removed -- see
``bouquet.surrogate.COVAR_ARMS`` for why, in short: the additive term won held-out
prediction decisively and then made the end-to-end SEARCH worse, so predictive
accuracy and search utility are anti-correlated for this surrogate.
"""

import math

import pytest
import torch

from bouquet.gradient_gp import GRAD_KERNEL_ARMS, GradientEnhancedPeriodicGP
from bouquet.surrogate import COVAR_ARMS, _fit_value_gp, make_covar_module


@pytest.fixture(autouse=True)
def double_precision():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    yield
    torch.set_default_dtype(old)


@pytest.fixture
def X():
    return torch.rand(24, 5, dtype=torch.double)


class TestValueKernelArms:
    def test_every_arm_builds_a_valid_covariance(self, X):
        for arm in COVAR_ARMS:
            K = make_covar_module(X.shape[1], arm=arm)(X, X).to_dense()
            assert K.shape == (len(X), len(X)), arm
            assert torch.linalg.eigvalsh(K).min() > -1e-8, arm

    def test_ard_has_one_lengthscale_per_torsion(self, X):
        d = X.shape[1]
        assert make_covar_module(d, arm="ard").base_kernel.lengthscale.numel() == d
        assert (
            make_covar_module(d, arm="product").base_kernel.base_kernel.lengthscale.numel()
            == 1
        )

    def test_ard_reduces_to_product_when_lengthscales_are_equal(self, X):
        """ARD is a strict generalization, so equal lengthscales must recover product."""
        d = X.shape[1]
        ard, prod = make_covar_module(d, arm="ard"), make_covar_module(d, arm="product")
        with torch.no_grad():
            ard.outputscale = prod.outputscale
            ard.base_kernel.lengthscale = torch.full(
                (1, d), float(prod.base_kernel.base_kernel.lengthscale)
            )
            ard.base_kernel.period_length = torch.full(
                (1, 1), float(prod.base_kernel.base_kernel.period_length)
            )
        assert torch.allclose(ard(X, X).to_dense(), prod(X, X).to_dense(), atol=1e-10)

    def test_ard_falls_back_below_two_torsions(self):
        """One torsion has nothing to differentiate, so ARD is the product kernel."""
        assert not hasattr(make_covar_module(1, arm="ard"), "kernels")

    def test_unknown_arm_rejected(self):
        with pytest.raises(ValueError):
            make_covar_module(4, arm="additive")   # removed; must not silently pass

    def test_fits_through_the_standard_gp_path(self, X):
        d = X.shape[1]
        torch.manual_seed(0)
        y = torch.sin(2 * math.pi * X[:, 0])[:, None].double()
        for arm in COVAR_ARMS:
            gp = _fit_value_gp(X, y, covar_module=make_covar_module(d, arm=arm))
            gp.eval()
            with torch.no_grad():
                pred = gp.likelihood(gp(X)).mean
            assert torch.isfinite(pred).all(), arm


class TestGradientKernelArms:
    def test_both_arms_fit_and_predict(self):
        torch.manual_seed(0)
        n, d = 18, 4
        X, Y, G = torch.rand(n, d), torch.randn(n, 1), torch.randn(n, d)
        for arm in GRAD_KERNEL_ARMS:
            gp = GradientEnhancedPeriodicGP(X, Y, G, period=1.0, kernel_arm=arm)
            gp.fit(steps=10)
            assert torch.isfinite(gp.posterior(torch.rand(4, d)).mean).all(), arm

    def test_ard_gives_the_gradient_kernel_per_torsion_lengthscales(self):
        n, d = 12, 4
        X, Y, G = torch.rand(n, d), torch.randn(n, 1), torch.randn(n, d)
        prod = GradientEnhancedPeriodicGP(X, Y, G, period=1.0, kernel_arm="product")
        ard = GradientEnhancedPeriodicGP(X, Y, G, period=1.0, kernel_arm="ard")
        assert prod.kernel.lengthscale.numel() == 1
        assert ard.kernel.lengthscale.numel() == d

    def test_unknown_arm_rejected(self):
        n, d = 8, 3
        X, Y, G = torch.rand(n, d), torch.randn(n, 1), torch.randn(n, d)
        with pytest.raises(ValueError):
            GradientEnhancedPeriodicGP(X, Y, G, period=1.0, kernel_arm="ard_additive")
