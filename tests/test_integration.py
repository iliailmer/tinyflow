import numpy as np
import pytest
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor as T

from tinyflow.losses import mse
from tinyflow.nn import NeuralNetwork
from tinyflow.path import AffinePath, OptimalTransportPath
from tinyflow.path.scheduler import LinearScheduler
from tinyflow.solver import RK4, Euler
from tinyflow.utils import preprocess_time_moons


class TestEndToEndTraining:
    def test_simple_2d_training(self):
        """Test that a simple 2D model can train for a few iterations without errors"""
        # Setup
        model = NeuralNetwork(2, 2, time_embed_dim=4)
        path = AffinePath(scheduler=LinearScheduler())
        optim = Adam(get_parameters(model), lr=0.001)

        # Create simple synthetic data
        x_1 = T.randn(10, 2)
        x_0 = T.randn(10, 2)
        t = T.rand(10, 1) * 0.99

        # Training loop
        T.training = True
        losses = []

        for _ in range(3):
            x_t, dx_t = path.sample(x_1=x_1, t=t, x_0=x_0)
            out = model(x_t, t)

            optim.zero_grad()
            loss = mse(out, dx_t)
            loss.backward()
            optim.step()

            losses.append(loss.item())

        # Verify
        assert len(losses) == 3, "Should have 3 loss values"
        assert all(not np.isnan(loss_val) for loss_val in losses), "Loss should not be NaN"
        assert all(not np.isinf(loss_val) for loss_val in losses), "Loss should not be infinite"

    def test_optimal_transport_training(self):
        """Test training with OptimalTransportPath"""
        model = NeuralNetwork(2, 2, time_embed_dim=4)
        path = OptimalTransportPath(sigma_min=0.0)
        optim = Adam(get_parameters(model), lr=0.001)

        x_1 = T.randn(10, 2)
        x_0 = T.randn(10, 2)
        t = T.rand(10, 1) * 0.99

        T.training = True
        losses = []

        for _ in range(3):
            x_t, dx_t = path.sample(x_1=x_1, t=t, x_0=x_0)
            out = model(x_t, t)

            optim.zero_grad()
            loss = mse(out, dx_t)
            loss.backward()
            optim.step()

            losses.append(loss.item())

        assert len(losses) == 3
        assert all(not np.isnan(loss_val) for loss_val in losses)

    def test_solver_integration_euler(self):
        """Test that Euler solver works with trained model"""
        model = NeuralNetwork(2, 2, 4)
        path = AffinePath(scheduler=LinearScheduler())
        optim = Adam(get_parameters(model), lr=0.001)

        # Quick training
        x_1 = T.randn(10, 2)
        x_0 = T.randn(10, 2)
        t = T.rand(10, 1) * 0.99

        T.training = True
        for _ in range(2):
            x_t, dx_t = path.sample(x_1=x_1, t=t, x_0=x_0)
            out = model(x_t, t)
            optim.zero_grad()
            loss = mse(out, dx_t)
            loss.backward()
            optim.step()

        # Test solver
        T.training = False
        solver = Euler(rhs_fn=model)

        x_init = T.randn(5, 2)
        t_curr = T.zeros(5, 1)
        h_step = 0.1

        x_next = solver.sample(h_step, t_curr, x_init)

        assert x_next.shape == (5, 2)
        assert not np.isnan(x_next.numpy()).any()

    def test_solver_integration_rk4(self):
        """Test that RK4 solver works with trained model"""
        model = NeuralNetwork(2, 2, 4)
        path = AffinePath(scheduler=LinearScheduler())
        optim = Adam(get_parameters(model), lr=0.001)

        # Quick training
        x_1 = T.randn(10, 2)
        x_0 = T.randn(10, 2)
        t = T.rand(10, 1) * 0.99

        T.training = True
        for _ in range(2):
            x_t, dx_t = path.sample(x_1=x_1, t=t, x_0=x_0)
            out = model(x_t, t)
            optim.zero_grad()
            loss = mse(out, dx_t)
            loss.backward()
            optim.step()

        # Test solver with preprocess hook
        T.training = False
        solver = RK4(rhs_fn=model, preprocess_hook=preprocess_time_moons)

        x_init = T.randn(5, 2)
        t_curr = T.zeros(1)
        h_step = 0.1

        x_next = solver.sample(h_step, t_curr, x_init)

        assert x_next.shape == (5, 2)
        assert not np.isnan(x_next.numpy()).any()

    def test_multi_step_sampling(self):
        """Test that we can sample multiple steps through time"""
        model = NeuralNetwork(2, 2, 4)
        path = AffinePath(scheduler=LinearScheduler())
        optim = Adam(get_parameters(model), lr=0.01)

        # Train a bit
        x_1 = T.randn(20, 2)
        x_0 = T.randn(20, 2)

        T.training = True
        for _ in range(5):
            t = T.rand(20, 1) * 0.99
            x_t, dx_t = path.sample(x_1=x_1, t=t, x_0=x_0)
            out = model(x_t, t)
            optim.zero_grad()
            loss = mse(out, dx_t)
            loss.backward()
            optim.step()

        # Sample through time
        T.training = False
        solver = RK4(rhs_fn=model, preprocess_hook=preprocess_time_moons)

        x = T.randn(10, 2)
        h_step = 0.1
        time_steps = 10

        trajectory = [x.numpy().copy()]

        for step in range(time_steps):
            t = T.zeros(1) + step * h_step
            x = solver.sample(h_step, t, x)
            trajectory.append(x.numpy().copy())

        assert len(trajectory) == time_steps + 1
        assert all(traj.shape == (10, 2) for traj in trajectory)
        assert all(not np.isnan(traj).any() for traj in trajectory)

    def test_gradient_flow(self):
        """Test that gradients flow correctly through the network"""
        model = NeuralNetwork(2, 2, 4)
        path = AffinePath(scheduler=LinearScheduler())
        optim = Adam(get_parameters(model), lr=0.001)

        x_1 = T.randn(10, 2)
        x_0 = T.randn(10, 2)
        t = T.rand(10, 1) * 0.99

        T.training = True

        # Get initial weights (first layer)
        initial_weight = model.layer0.weight.numpy().copy()

        # Training step
        x_t, dx_t = path.sample(x_1=x_1, t=t, x_0=x_0)
        out = model(x_t, t)
        optim.zero_grad()
        loss = mse(out, dx_t)
        loss.backward()
        optim.step()

        # Check that weights changed
        final_weight = model.layer0.weight.numpy()
        weight_diff = np.abs(final_weight - initial_weight).max()

        assert weight_diff > 1e-6, "Weights should change after training step"

    def test_time_clamping_prevents_singularity(self):
        """Test that clamping t prevents singularities in training"""
        model = NeuralNetwork(2, 2, 4)
        path = AffinePath(scheduler=LinearScheduler())
        optim = Adam(get_parameters(model), lr=0.001)

        x_1 = T.randn(100, 2)
        x_0 = T.randn(100, 2)

        T.training = True
        losses = []

        for _ in range(5):
            # Clamp t to avoid t=1.0
            t = T.rand(100, 1) * 0.99

            x_t, dx_t = path.sample(x_1=x_1, t=t, x_0=x_0)
            out = model(x_t, t)
            optim.zero_grad()
            loss = mse(out, dx_t)
            loss.backward()
            optim.step()

            losses.append(loss.item())

        # All losses should be finite
        assert all(
            not np.isnan(loss_val) for loss_val in losses
        ), "Losses should not be NaN with clamped t"
        assert all(
            not np.isinf(loss_val) for loss_val in losses
        ), "Losses should not be infinite with clamped t"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
