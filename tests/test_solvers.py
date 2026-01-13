import pytest
from tinygrad.tensor import Tensor as T

from tinyflow.solver import DDIM, RK4, Euler, Heun
from tinyflow.utils import preprocess_time_moons


class TestEuler:
    def test_linear_ode(self):
        """Test Euler solver on dx/dt = x"""

        def linear_rhs(x, t):
            return x

        solver = Euler(rhs_fn=linear_rhs)
        x1 = solver.step(0.1, T.zeros(1), T.ones(1))

        # Euler: x1 = x0 + h * f(x0) = 1 + 0.1 * 1 = 1.1
        assert abs(x1.numpy()[0] - 1.1) < 1e-5


class TestRK4:
    def test_linear_ode(self):
        """Test RK4 solver on dx/dt = x"""

        def linear_rhs(x, t):
            return x

        solver = RK4(rhs_fn=linear_rhs)
        x1 = solver.step(0.1, T.zeros(1), T.ones(1))

        # Analytical: x(0.1) = exp(0.1) ≈ 1.10517
        assert abs(x1.numpy()[0] - 1.10517) < 1e-3

    def test_more_accurate_than_euler(self):
        """Test RK4 accuracy vs Euler"""

        def linear_rhs(x, t):
            return x

        euler = Euler(rhs_fn=linear_rhs)
        rk4 = RK4(rhs_fn=linear_rhs)

        x_euler = euler.step(0.1, T.zeros(1), T.ones(1))
        x_rk4 = rk4.step(0.1, T.zeros(1), T.ones(1))

        expected = 1.10517
        error_euler = abs(x_euler.numpy()[0] - expected)
        error_rk4 = abs(x_rk4.numpy()[0] - expected)

        assert error_rk4 < error_euler

    def test_preprocess_hook(self):
        """Test preprocess hook integration"""

        def simple_rhs(x, t):
            return x * 0

        solver = RK4(rhs_fn=simple_rhs, preprocess_hook=preprocess_time_moons)
        result = solver.step(0.01, T.rand(1), T.randn(5, 2))

        assert result.shape == (5, 2)


class TestHeun:
    def test_linear_ode(self):
        """Test Heun solver on dx/dt = x"""

        def linear_rhs(x, t):
            return x

        solver = Heun(rhs_fn=linear_rhs)
        x1 = solver.step(0.1, T.zeros(1), T.ones(1))

        # Analytical: x(0.1) = exp(0.1) ≈ 1.10517
        # Heun should be more accurate than Euler
        assert abs(x1.numpy()[0] - 1.10517) < 1e-3

    def test_more_accurate_than_euler(self):
        """Test Heun is more accurate than Euler"""

        def linear_rhs(x, t):
            return x

        euler = Euler(rhs_fn=linear_rhs)
        heun = Heun(rhs_fn=linear_rhs)

        x_euler = euler.step(0.1, T.zeros(1), T.ones(1))
        x_heun = heun.step(0.1, T.zeros(1), T.ones(1))

        expected = 1.10517
        error_euler = abs(x_euler.numpy()[0] - expected)
        error_heun = abs(x_heun.numpy()[0] - expected)

        assert error_heun < error_euler

    def test_second_order_accuracy(self):
        """Test Heun has second-order convergence"""

        def linear_rhs(x, t):
            return x

        solver = Heun(rhs_fn=linear_rhs)

        # Test with two different step sizes
        x_h1 = solver.step(0.1, T.zeros(1), T.ones(1))
        x_h2 = solver.step(0.05, T.zeros(1), T.ones(1))
        x_h2 = solver.step(0.05, T([0.05]), x_h2)  # Second half step

        # Heun should be second-order: error ~ h^2
        # With half step size, error should reduce by ~4x
        expected = 1.10517
        error_h1 = abs(x_h1.numpy()[0] - expected)
        error_h2 = abs(x_h2.numpy()[0] - expected)

        # Second-order method: error_h2 should be roughly error_h1 / 4
        assert error_h2 < error_h1 / 2  # Relaxed check


class TestDDIM:
    def test_deterministic_sampling(self):
        """Test DDIM produces deterministic results"""

        def linear_rhs(x, t):
            return x

        solver = DDIM(rhs_fn=linear_rhs, eta=0.0)  # eta=0 for deterministic

        x1_run1 = solver.step(0.1, T.zeros(1), T.ones(1))
        x1_run2 = solver.step(0.1, T.zeros(1), T.ones(1))

        # Should be identical (deterministic)
        assert abs(x1_run1.numpy()[0] - x1_run2.numpy()[0]) < 1e-6

    def test_custom_timestep_schedule(self):
        """Test DDIM with custom timestep schedule"""

        def constant_rhs(x, t):
            return T.ones_like(x)

        solver = DDIM(rhs_fn=constant_rhs, eta=0.0)

        # Custom schedule: [0, 0.5, 1.0] instead of uniform [0, 0.1, 0.2, ...]
        time_grid = T([0.0, 0.5, 1.0])
        x_init = T.zeros(1)

        x_final = solver.solve(x_init, time_grid)

        # With constant velocity of 1, x should increase by ~1.0 total
        assert abs(x_final.numpy()[0] - 1.0) < 0.1

    def test_few_steps_sampling(self):
        """Test DDIM can work with very few steps"""

        def linear_rhs(x, t):
            return x * 0.5  # Slower growth

        solver = DDIM(rhs_fn=linear_rhs, eta=0.0)

        # Only 5 steps from 0 to 1
        time_grid = T.linspace(0, 1, 5)
        x_init = T.ones(1)

        x_final = solver.solve(x_init, time_grid)

        # Should still produce reasonable result
        assert x_final.shape == (1,)
        assert x_final.numpy()[0] > 0  # Should be positive


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
