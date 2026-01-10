import numpy as np
import pytest
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor as T

from tinyflow.losses import mse
from tinyflow.nn import NeuralNetwork
from tinyflow.path import AffinePath
from tinyflow.path.scheduler import LinearScheduler
from tinyflow.solver import RK4
from tinyflow.utils import preprocess_time_moons


class TestEndToEndTraining:
    def test_simple_2d_training(self):
        """Test basic training loop"""
        model = NeuralNetwork(2, 2, time_embed_dim=4)
        path = AffinePath(scheduler=LinearScheduler())
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
        assert all(not np.isnan(loss_val) and not np.isinf(loss_val) for loss_val in losses)

    def test_multi_step_sampling(self):
        """Test multi-step ODE sampling with solver"""
        model = NeuralNetwork(2, 2, 4)
        path = AffinePath(scheduler=LinearScheduler())
        optim = Adam(get_parameters(model), lr=0.01)

        # Quick training
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
        for step in range(10):
            t = T.zeros(1) + step * 0.1
            x = solver.sample(0.1, t, x)

        assert x.shape == (10, 2)
        assert not np.isnan(x.numpy()).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
