import pytest
from tinygrad.tensor import Tensor as T

from tinyflow.path.scheduler import (
    CosineScheduler,
    LinearScheduler,
    LinearVarPresScheduler,
    PolynomialScheduler,
)


class TestLinearScheduler:
    def test_boundary_alpha_t(self):
        """Test that alpha_t satisfies boundary conditions"""
        scheduler = LinearScheduler()

        t0 = T.zeros(1)
        t1 = T.ones(1)

        alpha_0 = scheduler.alpha_t(t0).numpy()
        alpha_1 = scheduler.alpha_t(t1).numpy()

        assert abs(alpha_0[0]) < 1e-5, f"alpha(0) should be 0, got {alpha_0[0]}"
        assert abs(alpha_1[0] - 1.0) < 1e-5, f"alpha(1) should be 1, got {alpha_1[0]}"

    def test_boundary_sigma_t(self):
        """Test that sigma_t satisfies boundary conditions"""
        scheduler = LinearScheduler()

        t0 = T.zeros(1)
        t1 = T.ones(1)

        sigma_0 = scheduler.sigma_t(t0).numpy()
        sigma_1 = scheduler.sigma_t(t1).numpy()

        assert abs(sigma_0[0] - 1.0) < 1e-5, f"sigma(0) should be 1, got {sigma_0[0]}"
        assert abs(sigma_1[0]) < 1e-5, f"sigma(1) should be 0, got {sigma_1[0]}"

    def test_monotonicity(self):
        """Test that alpha_t increases and sigma_t decreases"""
        scheduler = LinearScheduler()

        t_vals = T.linspace(0, 1, 10)
        alpha_vals = scheduler.alpha_t(t_vals).numpy()
        sigma_vals = scheduler.sigma_t(t_vals).numpy()

        # Check alpha is increasing
        for i in range(len(alpha_vals) - 1):
            assert alpha_vals[i] <= alpha_vals[i + 1], "alpha_t should be increasing"

        # Check sigma is decreasing
        for i in range(len(sigma_vals) - 1):
            assert sigma_vals[i] >= sigma_vals[i + 1], "sigma_t should be decreasing"


class TestPolynomialScheduler:
    def test_boundary_conditions_n2(self):
        """Test boundary conditions for polynomial order 2"""
        scheduler = PolynomialScheduler(n=2)

        t0 = T.zeros(1)
        t1 = T.ones(1)

        alpha_0 = scheduler.alpha_t(t0).numpy()
        alpha_1 = scheduler.alpha_t(t1).numpy()
        sigma_0 = scheduler.sigma_t(t0).numpy()
        sigma_1 = scheduler.sigma_t(t1).numpy()

        assert abs(alpha_0[0]) < 1e-5, "alpha(0) should be 0"
        assert abs(alpha_1[0] - 1.0) < 1e-5, "alpha(1) should be 1"
        assert abs(sigma_0[0] - 1.0) < 1e-5, "sigma(0) should be 1"
        assert abs(sigma_1[0]) < 1e-5, "sigma(1) should be 0"

    def test_monotonicity(self):
        """Test that alpha_t increases and sigma_t decreases"""
        scheduler = PolynomialScheduler(n=3)

        t_vals = T.linspace(0, 1, 10)
        alpha_vals = scheduler.alpha_t(t_vals).numpy()
        sigma_vals = scheduler.sigma_t(t_vals).numpy()

        # Check alpha is increasing
        for i in range(len(alpha_vals) - 1):
            assert alpha_vals[i] <= alpha_vals[i + 1], "alpha_t should be increasing"

        # Check sigma is decreasing
        for i in range(len(sigma_vals) - 1):
            assert sigma_vals[i] >= sigma_vals[i + 1], "sigma_t should be decreasing"


class TestLinearVarPresScheduler:
    def test_boundary_conditions(self):
        """Test boundary conditions"""
        scheduler = LinearVarPresScheduler()

        t0 = T.zeros(1)
        t1 = T.ones(1) * 0.99  # Avoid singularity at t=1

        alpha_0 = scheduler.alpha_t(t0).numpy()
        alpha_1 = scheduler.alpha_t(t1).numpy()
        sigma_0 = scheduler.sigma_t(t0).numpy()

        assert abs(alpha_0[0]) < 1e-5, "alpha(0) should be 0"
        assert abs(alpha_1[0] - 0.99) < 1e-4, "alpha(0.99) should be ~0.99"
        assert abs(sigma_0[0] - 1.0) < 1e-5, "sigma(0) should be 1"

    def test_no_division_by_zero(self):
        """Test that epsilon prevents division by zero at t=1"""
        scheduler = LinearVarPresScheduler(eps=1e-8)

        t1 = T.ones(1)  # t=1

        # Should not raise an error due to epsilon
        sigma_t_dot = scheduler.sigma_t_dot(t1).numpy()

        # Should be a finite value
        assert not (sigma_t_dot[0] != sigma_t_dot[0]), "sigma_t_dot should not be NaN"
        assert abs(sigma_t_dot[0]) < 1e8, "sigma_t_dot should not be infinite"

    def test_variance_preservation(self):
        """Test that alpha_t^2 + sigma_t^2 = 1"""
        scheduler = LinearVarPresScheduler()

        t_vals = T.linspace(0, 0.99, 10)
        for i in range(len(t_vals)):
            t = t_vals[i : i + 1]
            alpha = scheduler.alpha_t(t).numpy()[0]
            sigma = scheduler.sigma_t(t).numpy()[0]

            variance_sum = alpha**2 + sigma**2
            assert abs(variance_sum - 1.0) < 1e-4, (
                f"Variance preservation failed at t={t.numpy()[0]}: "
                f"alpha^2 + sigma^2 = {variance_sum}"
            )


class TestCosineScheduler:
    def test_boundary_conditions(self):
        """Test boundary conditions"""
        scheduler = CosineScheduler()

        t0 = T.zeros(1)
        t1 = T.ones(1)

        alpha_0 = scheduler.alpha_t(t0).numpy()
        alpha_1 = scheduler.alpha_t(t1).numpy()
        sigma_0 = scheduler.sigma_t(t0).numpy()
        sigma_1 = scheduler.sigma_t(t1).numpy()

        assert abs(alpha_0[0]) < 1e-5, "alpha(0) should be ~0"
        assert abs(alpha_1[0] - 1.0) < 1e-4, "alpha(1) should be ~1"
        assert abs(sigma_0[0] - 1.0) < 1e-4, "sigma(0) should be ~1"
        assert abs(sigma_1[0]) < 1e-4, "sigma(1) should be ~0"

    def test_smooth_transition(self):
        """Test that cosine scheduler provides smooth transition"""
        scheduler = CosineScheduler()

        t_vals = T.linspace(0, 1, 100)
        alpha_vals = scheduler.alpha_t(t_vals).numpy()

        # Check for smoothness by checking that changes are gradual
        diffs = [abs(alpha_vals[i + 1] - alpha_vals[i]) for i in range(len(alpha_vals) - 1)]
        max_diff = max(diffs)

        # With 100 points, max difference should be small
        assert max_diff < 0.02, f"Cosine scheduler should be smooth, max_diff={max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
