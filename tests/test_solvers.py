import pytest
from tinygrad.tensor import Tensor as T

from tinyflow.solver import RK4, Euler
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
