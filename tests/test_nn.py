import pytest
from tinygrad.tensor import Tensor as T

from tinyflow.nn import MLP, NeuralNetwork, UNetCIFAR10, UNetMNIST


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


class TestUNetMNIST:
    def test_output_shape_mnist(self):
        """Test UNetMNIST for 28x28 grayscale images"""
        model = UNetMNIST(in_channels=1, out_channels=1)
        x = T.randn(2, 1, 28, 28)
        t = T.rand(2, 1)
        output = model(x, t)
        assert output.shape == (2, 1, 28, 28)


class TestUNetCIFAR10:
    def test_output_shape_cifar(self):
        """Test UNetCIFAR10 for 32x32 RGB images"""
        model = UNetCIFAR10(in_channels=3, out_channels=3)
        x = T.randn(2, 3, 32, 32)
        t = T.rand(2, 1)
        output = model(x, t)
        assert output.shape == (2, 3, 32, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
