from collections.abc import Sequence
from typing import cast

from loguru import logger
from tinygrad import nn
from tinygrad.tensor import Tensor

from tinyflow.nn_utils.attention import SelfAttention
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


class UNet(BaseNeuralNetwork):
    """
    U-Net velocity model for square images.

    channels[:-1] are the widths of the encoder levels and channels[-1] is the
    width of the bottleneck. The image size is halved between levels. The decoder
    mirrors the encoder and takes a skip connection from each level.

    Args:
        in_channels: Number of image channels
        out_channels: Number of output channels, in_channels if None
        image_size: Height and width of the input images
        channels: Widths of the encoder levels, then the bottleneck
        blocks_per_level: Conv blocks at each level
        attention_resolutions: Feature map sizes that get self-attention, in the
            bottleneck and the decoder
        num_heads: Attention heads
        time_embed_dim: Dimension of the sinusoidal time embedding. An MLP maps it to a
            time vector of 4 times this size, which is added in every conv block.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int | None = None,
        image_size: int = 28,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        blocks_per_level: int = 1,
        attention_resolutions: Sequence[int] = (),
        num_heads: int = 4,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        *level_channels, bottleneck_channels = channels
        sizes = [image_size // 2**i for i in range(len(level_channels))]
        time_dim = 4 * time_embed_dim
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = [nn.Linear(time_embed_dim, time_dim), nn.Linear(time_dim, time_dim)]

        def blocks(c_in: int, c_out: int, count: int) -> list:
            return [
                ConvBlock(c_in if i == 0 else c_out, c_out, time_dim=time_dim) for i in range(count)
            ]

        def attention(c: int, size: int) -> list:
            return [SelfAttention(c, num_heads=num_heads)] if size in attention_resolutions else []

        self.encoder = []
        c_prev = in_channels
        for c in level_channels:
            self.encoder.append(blocks(c_prev, c, blocks_per_level))
            c_prev = c

        self.bottleneck = blocks(c_prev, bottleneck_channels, blocks_per_level)
        self.bottleneck += attention(bottleneck_channels, sizes[-1])
        c_prev = bottleneck_channels

        self.decoder = []
        for level in reversed(range(len(level_channels))):
            c = level_channels[level]
            if level == 0:
                level_blocks = [
                    ConvTransposeBlock(
                        c_prev + c, c, stride=1, padding=1, output_padding=0, time_dim=time_dim
                    )
                ]
                level_blocks += blocks(c, c, blocks_per_level - 1)
            else:
                out_size = sizes[level - 1]
                output_padding = out_size - (2 * sizes[level] - 1)
                level_blocks = [
                    ConvTransposeBlock(
                        c_prev + c, c, output_padding=output_padding, time_dim=time_dim
                    )
                ]
                level_blocks += blocks(c, c, blocks_per_level - 1) + attention(c, out_size)
            self.decoder.append(level_blocks)
            c_prev = c

        self.final_layer = nn.Conv2d(c_prev, out_channels or in_channels, kernel_size=1)

    def __call__(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x: Images, shape (batch_size, in_channels, image_size, image_size)
            t: Time, shape (batch_size, 1)

        Returns:
            Predicted velocity, same shape as x with out_channels channels
        """
        t_emb = self.time_mlp[1](self.time_mlp[0](self.time_embed(t)).swish())

        def apply(level_blocks: list, h: Tensor) -> Tensor:
            for block in level_blocks:
                h = block(h) if isinstance(block, SelfAttention) else block(h, t_emb)
            return h

        h = x
        skips = []
        for level, level_blocks in enumerate(self.encoder):
            if level > 0:
                h = cast(Tensor, h.max_pool2d((2, 2)))
            h = apply(level_blocks, h)
            skips.append(h)

        h = apply(self.bottleneck, h)

        for level_blocks, skip in zip(self.decoder, reversed(skips), strict=True):
            h = apply(level_blocks, h.cat(skip, dim=1))

        return self.final_layer(h)


MLP = MLPNetwork
NeuralNetwork = MLPNetwork


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
