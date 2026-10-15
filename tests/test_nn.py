import pytest
from tinygrad.tensor import Tensor as T

from tinyflow.nn import MLP, NeuralNetwork, UNet


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


class TestUNet:
    @pytest.mark.parametrize(
        ("in_channels", "image_size", "kwargs"),
        [
            (1, 28, {}),
            (3, 32, {}),
            (
                3,
                32,
                {"channels": (16, 32, 64), "blocks_per_level": 2, "attention_resolutions": (16,)},
            ),
        ],
    )
    def test_output_shape(self, in_channels, image_size, kwargs):
        model = UNet(in_channels=in_channels, image_size=image_size, **kwargs)
        x = T.randn(2, in_channels, image_size, image_size)
        output = model(x, T.rand(2, 1))
        assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
