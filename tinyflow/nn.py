# TODO: https://arxiv.org/pdf/2105.05233 <- this architecture

from loguru import logger
from tinygrad import nn
from tinygrad.tensor import Tensor

from tinyflow.nn_utils.conv import ConvBlock, ConvTransposeBlock
from tinyflow.nn_utils.time_embedding import SinusoidalTimeEmbedding, TimeEmbedding


class BaseNeuralNetwork:
    def __call__(self, x: Tensor, t: Tensor) -> Tensor:  # pyright: ignore
        raise NotImplementedError("Subclasses must implement __call__")


class MLPNetwork(BaseNeuralNetwork):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        time_embed_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "elu",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 64]

        self.activation = activation
        self.time_embed = TimeEmbedding(time_embed_dim)
        prev_dim = in_dim + time_embed_dim
        for i, hidden_dim in enumerate(hidden_dims):
            setattr(self, f"layer{i}", nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        # Output layer
        setattr(self, f"layer{len(hidden_dims)}", nn.Linear(prev_dim, out_dim))

        self.num_layers = len(hidden_dims) + 1

    def _apply_activation(self, x: Tensor) -> Tensor:
        """Apply the configured activation function."""
        if self.activation == "elu":
            return x.elu()
        if self.activation == "swish":
            return x.swish()
        if self.activation == "relu":
            return x.relu()
        logger.warning("Unknown activation function, defaulting to Swish")
        return x.swish()

    @logger.catch
    def __call__(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input state tensor, shape (batch_size, in_dim)
            t: Time tensor, shape (batch_size, 1)

        Returns:
            Predicted velocity, shape (batch_size, out_dim)
        """
        # Concatenate time with input
        t = self.time_embed(t)
        x = x.cat(t, dim=-1)

        # Forward through hidden layers with activation
        for i in range(self.num_layers - 1):
            layer = getattr(self, f"layer{i}")
            x = self._apply_activation(layer(x))

        # Final layer without activation
        final_layer = getattr(self, f"layer{self.num_layers - 1}")
        return final_layer(x)


class UNetTinygrad(BaseNeuralNetwork):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(32)  # TimeEmbedding(dim=32)

        # Encoder: +1 channel for time
        self.enc1 = ConvBlock(in_channels + self.time_embed.dim, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)

        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)

        # Decoder with skip connections
        self.dec4 = ConvTransposeBlock(512 + 256, 256, kernel_size=4)
        self.dec3 = ConvTransposeBlock(256 + 128, 128)
        self.dec2 = ConvTransposeBlock(128 + 64, 64)
        self.dec1 = ConvTransposeBlock(64 + 32, 32, stride=1, padding=1, output_padding=0)

        # Final output layer
        self.final_layer = nn.Conv2d(32, out_channels, kernel_size=1)

    def __call__(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input image tensor, shape (batch_size, in_channels, H, W)
            t: Time tensor, shape (batch_size, 1)

        Returns:
            Predicted velocity, shape (batch_size, out_channels, H, W)
        """
        # Broadcast time to spatial dimensions and concatenate with input
        t_embed = self.time_embed(t)
        x = x.cat(t_embed.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3]), dim=1)

        # Encoder path
        e1 = self.enc1(x)  # (B, 32, H, W)
        e2 = self.enc2(e1.max_pool2d((2, 2)))  # (B, 64, H/2, W/2)
        e3 = self.enc3(e2.max_pool2d((2, 2)))  # (B, 128, H/4, W/4)
        e4 = self.enc4(e3.max_pool2d((2, 2)))  # (B, 256, H/8, W/8)

        # Bottleneck
        b = self.bottleneck(e4)  # (B, 512, H/8, W/8)

        # Decoder path with skip connections
        d4 = self.dec4(b.cat(e4, dim=1))  # (B, 256, H/4, W/4)
        d3 = self.dec3(d4.cat(e3, dim=1))  # (B, 128, H/2, W/2)
        d2 = self.dec2(d3.cat(e2, dim=1))  # (B, 64, H, W)
        d1 = self.dec1(d2.cat(e1, dim=1))  # (B, 32, H, W)

        return self.final_layer(d1)


MLP = MLPNetwork  # Simple 3-layer MLP was the old default
NeuralNetwork = MLPNetwork  # Both are now the same flexible MLP


def create_mlp_simple(in_dim: int, out_dim: int, time_embed_dim: int) -> MLPNetwork:
    """
    Create simple 3-layer MLP (matches old MLP class).

    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        time_embed_dim: Time embedding dimension

    Returns:
        MLPNetwork with [64, 64, 64] hidden layers
    """
    return MLPNetwork(in_dim, out_dim, time_embed_dim, hidden_dims=[64, 64, 64], activation="elu")


def create_mlp_deep(in_dim: int, out_dim: int, time_embed_dim) -> MLPNetwork:
    """
    Create deeper MLP with bottleneck (matches old NeuralNetwork class).

    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        time_embed_dim: Time embedding dimension

    Returns:
        MLPNetwork with [64, 128, 64] hidden layers
    """
    return MLPNetwork(in_dim, out_dim, time_embed_dim, hidden_dims=[64, 128, 64], activation="elu")
