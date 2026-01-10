import pytest
from tinygrad.tensor import Tensor as T

from tinyflow.path.scheduler import (
    CosineScheduler,
    LinearScheduler,
    LinearVarPresScheduler,
    PolynomialScheduler,
)


class TestLinearScheduler:
    def test_boundary_conditions(self):
        """Test boundary conditions: alpha(0)=0, alpha(1)=1, sigma(0)=1, sigma(1)=0"""
        scheduler = LinearScheduler()
        t0, t1 = T.zeros(1), T.ones(1)

        assert abs(scheduler.alpha_t(t0).numpy()[0]) < 1e-5
        assert abs(scheduler.alpha_t(t1).numpy()[0] - 1.0) < 1e-5
        assert abs(scheduler.sigma_t(t0).numpy()[0] - 1.0) < 1e-5
        assert abs(scheduler.sigma_t(t1).numpy()[0]) < 1e-5


class TestPolynomialScheduler:
    def test_boundary_conditions(self):
        """Test boundary conditions"""
        scheduler = PolynomialScheduler(n=2)
        t0, t1 = T.zeros(1), T.ones(1)

        assert abs(scheduler.alpha_t(t0).numpy()[0]) < 1e-5
        assert abs(scheduler.alpha_t(t1).numpy()[0] - 1.0) < 1e-5
        assert abs(scheduler.sigma_t(t0).numpy()[0] - 1.0) < 1e-5
        assert abs(scheduler.sigma_t(t1).numpy()[0]) < 1e-5


class TestLinearVarPresScheduler:
    def test_variance_preservation(self):
        """Test that alpha_t^2 + sigma_t^2 = 1"""
        scheduler = LinearVarPresScheduler()
        t_vals = T.linspace(0, 0.99, 10)

        for i in range(len(t_vals)):
            t = t_vals[i : i + 1]
            alpha = scheduler.alpha_t(t).numpy()[0]
            sigma = scheduler.sigma_t(t).numpy()[0]
            variance_sum = alpha**2 + sigma**2
            assert abs(variance_sum - 1.0) < 1e-4


class TestCosineScheduler:
    def test_boundary_conditions(self):
        """Test boundary conditions"""
        scheduler = CosineScheduler()
        t0, t1 = T.zeros(1), T.ones(1)

        assert abs(scheduler.alpha_t(t0).numpy()[0]) < 1e-5
        assert abs(scheduler.alpha_t(t1).numpy()[0] - 1.0) < 1e-4
        assert abs(scheduler.sigma_t(t0).numpy()[0] - 1.0) < 1e-4
        assert abs(scheduler.sigma_t(t1).numpy()[0]) < 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
