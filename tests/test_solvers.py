import pytest
from tinygrad.tensor import Tensor as T
from tinyflow.solver import Euler, RK4
from tinyflow.utils import preprocess_time_moons


class TestEuler:
    def test_implements_required_methods(self):
        """Test that Euler implements step and sample methods"""

        def simple_rhs(x, t):
            return x * 0.1

        solver = Euler(rhs_fn=simple_rhs)
        assert hasattr(solver, "step"), "Euler should have step method"
        assert hasattr(solver, "sample"), "Euler should have sample method"

    def test_linear_ode(self):
        """Test Euler solver on simple linear ODE: dx/dt = x"""

        def linear_rhs(x, t):
            return x

        solver = Euler(rhs_fn=linear_rhs)

        x0 = T.ones(1)
        t = T.zeros(1)
        h = 0.1

        x1 = solver.step(h, t, x0)

        # Euler method: x1 = x0 + h * f(x0, t0) = 1 + 0.1 * 1 = 1.1
        expected = 1.1
        diff = abs(x1.numpy()[0] - expected)
        assert diff < 1e-5, f"Expected {expected}, got {x1.numpy()[0]}"

    def test_correct_argument_order(self):
        """Test that solver passes arguments in correct order (x, t)"""
        call_log = []

        def rhs_tracker(x, t):
            call_log.append(("x_shape", x.shape, "t_shape", t.shape))
            return x * 0

        solver = Euler(rhs_fn=rhs_tracker)
        x = T.randn(5, 2)
        t = T.rand(5, 1)
        h = 0.01

        solver.step(h, t, x)

        assert len(call_log) == 1, "RHS should be called once"
        assert call_log[0][1] == (5, 2), "First arg should be x with shape (5, 2)"
        assert call_log[0][3] == (5, 1), "Second arg should be t with shape (5, 1)"

    def test_sample_equals_step(self):
        """Test that sample method equals step for Euler"""

        def simple_rhs(x, t):
            return x * 0.5

        solver = Euler(rhs_fn=simple_rhs)

        x = T.randn(10, 2)
        t = T.rand(10, 1)
        h = 0.01

        result_step = solver.step(h, t, x)
        result_sample = solver.sample(h, t, x)

        diff = (result_step - result_sample).abs().max().numpy()
        assert diff < 1e-6, "sample() and step() should give same result for Euler"


class TestRK4:
    def test_implements_required_methods(self):
        """Test that RK4 implements step and sample methods"""

        def simple_rhs(x, t):
            return x * 0.1

        solver = RK4(rhs_fn=simple_rhs)
        assert hasattr(solver, "step"), "RK4 should have step method"
        assert hasattr(solver, "sample"), "RK4 should have sample method"

    def test_linear_ode(self):
        """Test RK4 solver on simple linear ODE: dx/dt = x"""

        def linear_rhs(x, t):
            return x

        solver = RK4(rhs_fn=linear_rhs)

        x0 = T.ones(1)
        t = T.zeros(1)
        h = 0.1

        x1 = solver.step(h, t, x0)

        # RK4 should be more accurate than Euler
        # Analytical solution: x(t) = exp(t), so x(0.1) = exp(0.1) ≈ 1.10517
        expected = 1.10517
        diff = abs(x1.numpy()[0] - expected)
        assert diff < 1e-3, f"RK4 should be accurate, diff={diff}"

    def test_more_accurate_than_euler(self):
        """Test that RK4 is more accurate than Euler for same step size"""

        def linear_rhs(x, t):
            return x

        euler = Euler(rhs_fn=linear_rhs)
        rk4 = RK4(rhs_fn=linear_rhs)

        x0 = T.ones(1)
        t = T.zeros(1)
        h = 0.1

        x_euler = euler.step(h, t, x0)
        x_rk4 = rk4.step(h, t, x0)

        # Analytical solution: x(0.1) = exp(0.1) ≈ 1.10517
        expected = 1.10517

        error_euler = abs(x_euler.numpy()[0] - expected)
        error_rk4 = abs(x_rk4.numpy()[0] - expected)

        assert error_rk4 < error_euler, (
            f"RK4 should be more accurate than Euler. "
            f"Euler error: {error_euler}, RK4 error: {error_rk4}"
        )

    def test_correct_argument_order(self):
        """Test that solver passes arguments in correct order (x, t)"""
        call_count = [0]

        def rhs_tracker(x, t):
            call_count[0] += 1
            # Verify shapes to ensure correct argument order
            assert x.shape == (5, 2), f"First arg should be x with shape (5, 2), got {x.shape}"
            assert t.shape == (5, 1), f"Second arg should be t with shape (5, 1), got {t.shape}"
            return x * 0

        solver = RK4(rhs_fn=rhs_tracker)
        x = T.randn(5, 2)
        t = T.rand(5, 1)
        h = 0.01

        solver.step(h, t, x)

        assert call_count[0] == 4, "RK4 should call RHS 4 times (k1, k2, k3, k4)"

    def test_preprocess_hook(self):
        """Test that preprocess hook is called correctly"""
        hook_calls = []

        def track_hook(t, rhs_prev):
            hook_calls.append((t.shape, rhs_prev.shape))
            return preprocess_time_moons(t, rhs_prev)

        def simple_rhs(x, t):
            return x * 0

        solver = RK4(rhs_fn=simple_rhs, preprocess_hook=track_hook)

        x = T.randn(5, 2)
        t = T.rand(1)
        h = 0.01

        solver.step(h, t, x)

        # RK4 should call hook 4 times (for k1, k2, k3, k4)
        assert len(hook_calls) == 4, f"Preprocess hook should be called 4 times, got {len(hook_calls)}"

    def test_sample_equals_step(self):
        """Test that sample method equals step for RK4"""

        def simple_rhs(x, t):
            return x * 0.5

        solver = RK4(rhs_fn=simple_rhs)

        x = T.randn(10, 2)
        t = T.rand(10, 1)
        h = 0.01

        result_step = solver.step(h, t, x)
        result_sample = solver.sample(h, t, x)

        diff = (result_step - result_sample).abs().max().numpy()
        assert diff < 1e-6, "sample() and step() should give same result for RK4"

    def test_batch_processing(self):
        """Test that RK4 works correctly with batched inputs"""

        def simple_rhs(x, t):
            return x * 0.1

        solver = RK4(rhs_fn=simple_rhs)

        batch_size = 8
        dim = 4
        x = T.randn(batch_size, dim)
        t = T.rand(batch_size, 1)
        h = 0.01

        result = solver.step(h, t, x)

        assert result.shape == (batch_size, dim), f"Output shape should match input shape"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
