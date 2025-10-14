import pytest
from tinygrad.tensor import Tensor as T
from tinyflow.path import AffinePath, OptimalTransportPath
from tinyflow.path.scheduler import LinearScheduler


class TestAffinePath:
    def test_sample_output_shape_2d(self):
        """Test that AffinePath.sample returns correct shape for 2D data"""
        path = AffinePath(scheduler=LinearScheduler())
        batch_size = 16
        dim = 2

        x_1 = T.randn(batch_size, dim)
        x_0 = T.randn(batch_size, dim)
        t = T.rand(batch_size, 1) * 0.99

        sample = path.sample(x_1=x_1, t=t, x_0=x_0)

        assert sample.x_t.shape == (batch_size, dim)
        assert sample.dx_t.shape == (batch_size, dim)

    def test_sample_output_shape_images(self):
        """Test that AffinePath.sample returns correct shape for image data"""
        path = AffinePath(scheduler=LinearScheduler())
        batch_size = 4
        channels = 1
        height = 28
        width = 28

        x_1 = T.randn(batch_size, channels, height, width)
        x_0 = T.randn(batch_size, channels, height, width)
        t = T.rand(batch_size, 1) * 0.99

        sample = path.sample(x_1=x_1, t=t, x_0=x_0)

        assert sample.x_t.shape == (batch_size, channels, height, width)
        assert sample.dx_t.shape == (batch_size, channels, height, width)

    def test_boundary_conditions_t0(self):
        """Test that at t=0, x_t equals sigma_0 * x_0 (should be close to x_0)"""
        path = AffinePath(scheduler=LinearScheduler())
        batch_size = 10
        dim = 2

        x_1 = T.randn(batch_size, dim)
        x_0 = T.randn(batch_size, dim)
        t = T.zeros(batch_size, 1)  # t=0

        sample = path.sample(x_1=x_1, t=t, x_0=x_0)

        # At t=0, LinearScheduler gives alpha_t=0, sigma_t=1
        # So x_t = 0 * x_1 + 1 * x_0 = x_0
        diff = (sample.x_t - x_0).abs().max().numpy()
        assert diff < 1e-5, f"At t=0, x_t should equal x_0, but diff={diff}"

    def test_boundary_conditions_t1(self):
        """Test that at t=1, x_t equals alpha_1 * x_1 (should be close to x_1)"""
        path = AffinePath(scheduler=LinearScheduler())
        batch_size = 10
        dim = 2

        x_1 = T.randn(batch_size, dim)
        x_0 = T.randn(batch_size, dim)
        t = T.ones(batch_size, 1) * 0.99  # Close to t=1

        sample = path.sample(x_1=x_1, t=t, x_0=x_0)

        # At t=0.99, should be very close to x_1
        diff = (sample.x_t - x_1).abs().max().numpy()
        assert diff < 0.2, f"At t=0.99, x_t should be close to x_1, but diff={diff}"


class TestOptimalTransportPath:
    def test_sample_output_shape_2d(self):
        """Test that OptimalTransportPath.sample returns correct shape for 2D data"""
        path = OptimalTransportPath(sigma_min=0.0)
        batch_size = 16
        dim = 2

        x_1 = T.randn(batch_size, dim)
        x_0 = T.randn(batch_size, dim)
        t = T.rand(batch_size, 1) * 0.99

        sample = path.sample(x_1=x_1, t=t, x_0=x_0)

        assert sample.x_t.shape == (batch_size, dim)
        assert sample.dx_t.shape == (batch_size, dim)

    def test_sample_output_shape_images(self):
        """Test that OptimalTransportPath.sample returns correct shape for image data"""
        path = OptimalTransportPath(sigma_min=0.0)
        batch_size = 4
        channels = 1
        height = 28
        width = 28

        x_1 = T.randn(batch_size, channels, height, width)
        x_0 = T.randn(batch_size, channels, height, width)
        t = T.rand(batch_size, 1) * 0.99

        sample = path.sample(x_1=x_1, t=t, x_0=x_0)

        assert sample.x_t.shape == (batch_size, channels, height, width)
        assert sample.dx_t.shape == (batch_size, channels, height, width)

    def test_boundary_conditions_t0(self):
        """Test that at t=0, x_t equals x_0"""
        path = OptimalTransportPath(sigma_min=0.0)
        batch_size = 10
        dim = 2

        x_1 = T.randn(batch_size, dim)
        x_0 = T.randn(batch_size, dim)
        t = T.zeros(batch_size, 1)  # t=0

        sample = path.sample(x_1=x_1, t=t, x_0=x_0)

        # At t=0: x_t = x_1 * 0 + (1 - 0) * x_0 = x_0
        diff = (sample.x_t - x_0).abs().max().numpy()
        assert diff < 1e-5, f"At t=0, x_t should equal x_0, but diff={diff}"

    def test_boundary_conditions_t1(self):
        """Test that at t=1, x_t equals x_1"""
        path = OptimalTransportPath(sigma_min=0.0)
        batch_size = 10
        dim = 2

        x_1 = T.randn(batch_size, dim)
        x_0 = T.randn(batch_size, dim)
        t = T.ones(batch_size, 1)  # t=1

        sample = path.sample(x_1=x_1, t=t, x_0=x_0)

        # At t=1: x_t = x_1 * 1 + (1 - 1) * x_0 = x_1
        diff = (sample.x_t - x_1).abs().max().numpy()
        assert diff < 1e-5, f"At t=1, x_t should equal x_1, but diff={diff}"

    def test_velocity_formula(self):
        """Test that velocity is x_1 - (1-sigma_min)*x_0"""
        path = OptimalTransportPath(sigma_min=0.1)
        batch_size = 10
        dim = 2

        x_1 = T.randn(batch_size, dim)
        x_0 = T.randn(batch_size, dim)
        t = T.rand(batch_size, 1) * 0.5

        sample = path.sample(x_1=x_1, t=t, x_0=x_0)

        expected_velocity = x_1 - (1 - 0.1) * x_0
        diff = (sample.dx_t - expected_velocity).abs().max().numpy()
        assert diff < 1e-5, f"Velocity formula incorrect, diff={diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
