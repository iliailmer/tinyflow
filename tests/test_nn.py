import pytest
from tinygrad.tensor import Tensor as T

from tinyflow.nn import MLP, NeuralNetwork, UNetTinygrad


class TestMLP:
    def test_output_shape(self):
        """Test MLP output shape"""
        model = MLP(2, 2, 4)
        x = T.randn(16, 2)
        t = T.rand(16, 1)
        output = model(x, t)
        assert output.shape == (16, 2)


class TestNeuralNetwork:
    def test_output_shape(self):
        """Test NeuralNetwork output shape"""
        model = NeuralNetwork(10, 10, 4)
        x = T.randn(8, 10)
        t = T.rand(8, 1)
        output = model(x, t)
        assert output.shape == (8, 10)


class TestUNetTinygrad:
    def test_output_shape_mnist(self):
        """Test UNet for MNIST-sized images"""
        model = UNetTinygrad(in_channels=1, out_channels=1)
        x = T.randn(2, 1, 28, 28)
        t = T.rand(2, 1)
        output = model(x, t)
        assert output.shape == (2, 1, 28, 28)

    def test_multi_channel(self):
        """Test UNet with RGB channels"""
        model = UNetTinygrad(in_channels=3, out_channels=3)
        x = T.randn(2, 3, 28, 28)
        t = T.rand(2, 1)
        output = model(x, t)
        assert output.shape == (2, 3, 28, 28)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
