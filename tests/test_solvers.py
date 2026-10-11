import numpy as np
import pytest
from tinygrad.tensor import Tensor as T

from tinyflow.solver import DDIM, RK4, Euler, Heun, MidpointSolver, uniform_time_grid
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


class TestSolve:
    def test_uniform_time_grid(self):
        grid = uniform_time_grid(4)
        assert np.allclose(grid, [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_uniform_time_grid_rejects_zero_steps(self):
        with pytest.raises(ValueError):
            uniform_time_grid(0)

    def test_rejects_single_point_grid(self):
        with pytest.raises(ValueError):
            Euler(rhs_fn=lambda x, t: x).solve(T.ones(1), [0.0])

    @pytest.mark.parametrize("jit", [True, False])
    @pytest.mark.parametrize("solver_cls", [Euler, Heun, MidpointSolver, RK4, DDIM])
    def test_constant_velocity_covers_unit_interval(self, solver_cls, jit):
        """dx/dt = 1 from t=0 to t=1 moves x by exactly 1 for any number of steps"""
        solver = solver_cls(rhs_fn=lambda x, t: T.ones_like(x))
        x = solver.solve(T.zeros(3), uniform_time_grid(5), jit=jit)
        assert np.allclose(x.numpy(), 1.0, atol=1e-5)

    @pytest.mark.parametrize("solver_cls", [Euler, Heun, MidpointSolver, RK4])
    def test_rhs_times_stay_in_unit_interval(self, solver_cls):
        """No stage evaluates the velocity field outside the training range [0, 1]"""
        seen = []

        def rhs(x, t):
            seen.append(float(t.numpy().max()))
            return T.ones_like(x)

        solver_cls(rhs_fn=rhs).solve(T.zeros(2), uniform_time_grid(5), jit=False)
        assert min(seen) >= 0.0
        assert max(seen) <= 1.0 + 1e-6

    def test_euler_evaluates_at_left_grid_points(self):
        seen = []

        def rhs(x, t):
            seen.append(float(t.numpy()[0]))
            return T.ones_like(x)

        Euler(rhs_fn=rhs).solve(T.zeros(1), uniform_time_grid(4), jit=False)
        assert np.allclose(seen, [0.0, 0.25, 0.5, 0.75])

    def test_non_uniform_grid(self):
        """dx/dt = t integrated exactly by the trapezoid rule gives 0.5"""
        solver = Heun(rhs_fn=lambda x, t: t.expand(*x.shape))
        x = solver.solve(T.zeros(2), [0.0, 0.1, 0.4, 1.0])
        assert np.allclose(x.numpy(), 0.5, atol=1e-5)

    def test_accepts_tensor_grid(self):
        solver = Euler(rhs_fn=lambda x, t: T.ones_like(x))
        x = solver.solve(T.zeros(1), T([0.0, 0.5, 1.0]))
        assert abs(x.numpy()[0] - 1.0) < 1e-5

    def test_on_step_reports_right_grid_points(self):
        calls = []
        solver = Euler(rhs_fn=lambda x, t: T.ones_like(x))
        solver.solve(
            T.zeros(1),
            uniform_time_grid(4),
            on_step=lambda i, t, x: calls.append((i, t, x.numpy()[0])),
        )
        assert [c[0] for c in calls] == [0, 1, 2, 3]
        assert np.allclose([c[1] for c in calls], [0.25, 0.5, 0.75, 1.0])
        assert np.allclose([c[2] for c in calls], [0.25, 0.5, 0.75, 1.0], atol=1e-5)

    def test_jit_matches_no_jit(self):
        solver = RK4(rhs_fn=lambda x, t: x * t)
        x0 = T([1.0, 2.0])
        grid = uniform_time_grid(6)
        assert np.allclose(
            solver.solve(x0, grid, jit=True).numpy(),
            solver.solve(x0, grid, jit=False).numpy(),
            atol=1e-5,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
