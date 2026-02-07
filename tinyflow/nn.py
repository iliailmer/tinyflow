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


class UNetMNIST(BaseNeuralNetwork):
    """
    Modern U-Net optimized for 28x28 grayscale images (MNIST, Fashion MNIST).

    Architecture improvements:
    - GroupNorm instead of BatchNorm (stable for small batches)
    - Residual connections in all blocks (better gradient flow)
    - Swish activation (proven in diffusion models)
    - Skip connections (standard U-Net)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(64)  # Increased from 32

        self.enc1 = ConvBlock(in_channels + self.time_embed.dim, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)

        self.bottleneck = ConvBlock(256, 512)

        self.dec4 = ConvTransposeBlock(512 + 256, 256, output_padding=2)  # 3 -> 7 for odd dims
        self.dec3 = ConvTransposeBlock(256 + 128, 128)  # 7 -> 14
        self.dec2 = ConvTransposeBlock(128 + 64, 64)  # 14 -> 28
        self.dec1 = ConvTransposeBlock(64 + 32, 32, stride=1, padding=1, output_padding=0)

        self.final_layer = nn.Conv2d(32, out_channels, kernel_size=1)

    def __call__(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Forward pass for 28x28 images.

        Args:
            x: Input image tensor, shape (batch_size, in_channels, 28, 28)
            t: Time tensor, shape (batch_size, 1)

        Returns:
            Predicted velocity, shape (batch_size, out_channels, 28, 28)
        """
        t_embed = self.time_embed(t)
        x = x.cat(t_embed.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3]), dim=1)

        e1 = self.enc1(x)  # (B, 32, 28, 28)
        e2 = self.enc2(e1.max_pool2d((2, 2)))  # (B, 64, 14, 14)
        e3 = self.enc3(e2.max_pool2d((2, 2)))  # (B, 128, 7, 7)
        e4 = self.enc4(e3.max_pool2d((2, 2)))  # (B, 256, 3, 3)

        bn = self.bottleneck(e4)  # (B, 512, 3, 3)

        d4 = self.dec4(bn.cat(e4, dim=1))  # (B, 256, 7, 7)
        d3 = self.dec3(d4.cat(e3, dim=1))  # (B, 128, 14, 14)
        d2 = self.dec2(d3.cat(e2, dim=1))  # (B, 64, 28, 28)
        d1 = self.dec1(d2.cat(e1, dim=1))  # (B, 32, 28, 28)

        return self.final_layer(d1)


class UNetCIFAR10(BaseNeuralNetwork):
    """
    Modern U-Net optimized for 32x32 RGB images (CIFAR-10).

    Architecture improvements:
    - GroupNorm instead of BatchNorm (stable for small batches)
    - Residual connections in all blocks (better gradient flow)
    - Swish activation (proven in diffusion models)
    - Skip connections (standard U-Net)
    - Clean power-of-2 dimensions (no output_padding hacks needed)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(64)  # Increased from 32

        self.enc1 = ConvBlock(in_channels + self.time_embed.dim, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)

        self.bottleneck = ConvBlock(256, 512)

        self.dec4 = ConvTransposeBlock(512 + 256, 256)  # 4 -> 8
        self.dec3 = ConvTransposeBlock(256 + 128, 128)  # 8 -> 16
        self.dec2 = ConvTransposeBlock(128 + 64, 64)  # 16 -> 32
        self.dec1 = ConvTransposeBlock(64 + 32, 32, stride=1, padding=1, output_padding=0)

        self.final_layer = nn.Conv2d(32, out_channels, kernel_size=1)

    def __call__(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Forward pass for 32x32 images.

        Args:
            x: Input image tensor, shape (batch_size, in_channels, 32, 32)
            t: Time tensor, shape (batch_size, 1)

        Returns:
            Predicted velocity, shape (batch_size, out_channels, 32, 32)
        """
        t_embed = self.time_embed(t)
        x = x.cat(t_embed.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3]), dim=1)

        e1 = self.enc1(x)  # (B, 32, 32, 32)
        e2 = self.enc2(e1.max_pool2d((2, 2)))  # (B, 64, 16, 16)
        e3 = self.enc3(e2.max_pool2d((2, 2)))  # (B, 128, 8, 8)
        e4 = self.enc4(e3.max_pool2d((2, 2)))  # (B, 256, 4, 4)

        bn = self.bottleneck(e4)  # (B, 512, 4, 4)

        d4 = self.dec4(bn.cat(e4, dim=1))  # (B, 256, 8, 8)
        d3 = self.dec3(d4.cat(e3, dim=1))  # (B, 128, 16, 16)
        d2 = self.dec2(d3.cat(e2, dim=1))  # (B, 64, 32, 32)
        d1 = self.dec1(d2.cat(e1, dim=1))  # (B, 32, 32, 32)

        return self.final_layer(d1)


class UNetCIFAR10Large(BaseNeuralNetwork):
    """
    Medium U-Net for CIFAR-10 - balanced quality and memory usage.

    Improvements over base UNetCIFAR10:
    - 1.5x channel capacity: [48, 96, 192, 384, 768] (vs [32, 64, 128, 256, 512])
    - Larger time embedding: 96-dim (vs 64-dim)
    - 2 conv blocks per level (vs 1) for better feature extraction
    - Strategic attention: bottleneck (4x4) + one decoder level (8x8)
    - Fewer attention heads (4 vs 8) to reduce memory

    Model size: ~50MB (vs ~20MB for base)
    Memory: Balanced - attention only at low resolutions (cheap!)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        # Medium time embedding
        self.time_embed = SinusoidalTimeEmbedding(96)

        # Encoder: 2 conv blocks per level for better feature extraction
        # Level 1: 32x32 (48 channels)
        self.enc1a = ConvBlock(in_channels + self.time_embed.dim, 48)
        self.enc1b = ConvBlock(48, 48)

        # Level 2: 16x16 (96 channels)
        self.enc2a = ConvBlock(48, 96)
        self.enc2b = ConvBlock(96, 96)

        # Level 3: 8x8 (192 channels)
        self.enc3a = ConvBlock(96, 192)
        self.enc3b = ConvBlock(192, 192)

        # Level 4: 4x4 (384 channels)
        self.enc4a = ConvBlock(192, 384)
        self.enc4b = ConvBlock(384, 384)

        # Bottleneck: 4x4 (768 channels) + attention
        self.bottleneck_a = ConvBlock(384, 768)
        self.bottleneck_b = ConvBlock(768, 768)
        self.attn_bn = SelfAttention(768, num_heads=4)  # 4x4 is cheap!

        # Decoder: 2 conv blocks per level with skip connections
        # Level 4: 4x4 -> 8x8 (384 channels) + attention
        self.dec4 = ConvTransposeBlock(768 + 384, 384)
        self.dec4a = ConvBlock(384, 384)
        self.dec4b = ConvBlock(384, 384)
        self.attn_dec4 = SelfAttention(384, num_heads=4)  # 8x8 still reasonable

        # Level 3: 8x8 -> 16x16 (192 channels) - no attention (16x16 gets expensive)
        self.dec3 = ConvTransposeBlock(384 + 192, 192)
        self.dec3a = ConvBlock(192, 192)
        self.dec3b = ConvBlock(192, 192)

        # Level 2: 16x16 -> 32x32 (96 channels)
        self.dec2 = ConvTransposeBlock(192 + 96, 96)
        self.dec2a = ConvBlock(96, 96)
        self.dec2b = ConvBlock(96, 96)

        # Level 1: 32x32 -> 32x32 (48 channels)
        self.dec1 = ConvTransposeBlock(96 + 48, 48, stride=1, padding=1, output_padding=0)
        self.dec1a = ConvBlock(48, 48)
        self.dec1b = ConvBlock(48, 48)

        # Final output layer
        self.final_layer = nn.Conv2d(48, out_channels, kernel_size=1)

    def __call__(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Forward pass for 32x32 RGB images.

        Args:
            x: Input image tensor, shape (batch_size, 3, 32, 32)
            t: Time tensor, shape (batch_size, 1)

        Returns:
            Predicted velocity, shape (batch_size, 3, 32, 32)
        """
        # Embed time and concatenate with input
        t_embed = self.time_embed(t)
        x = x.cat(t_embed.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3]), dim=1)

        # Encoder with skip connections
        # Level 1: 32x32
        e1 = self.enc1b(self.enc1a(x))  # (B, 48, 32, 32)

        # Level 2: 16x16
        e2 = self.enc2a(e1.max_pool2d((2, 2)))  # (B, 96, 16, 16)
        e2 = self.enc2b(e2)

        # Level 3: 8x8
        e3 = self.enc3a(e2.max_pool2d((2, 2)))  # (B, 192, 8, 8)
        e3 = self.enc3b(e3)

        # Level 4: 4x4
        e4 = self.enc4a(e3.max_pool2d((2, 2)))  # (B, 384, 4, 4)
        e4 = self.enc4b(e4)

        # Bottleneck: 4x4 with attention
        bn = self.bottleneck_a(e4)  # (B, 768, 4, 4)
        bn = self.bottleneck_b(bn)
        bn = self.attn_bn(bn)  # Attention at lowest resolution (cheap!)

        # Decoder with skip connections
        # Level 4: 4x4 -> 8x8 with attention
        d4 = self.dec4(bn.cat(e4, dim=1))  # (B, 384, 8, 8)
        d4 = self.dec4b(self.dec4a(d4))
        d4 = self.attn_dec4(d4)  # Attention at 8x8 (still reasonable)

        # Level 3: 8x8 -> 16x16 (no attention - too expensive)
        d3 = self.dec3(d4.cat(e3, dim=1))  # (B, 192, 16, 16)
        d3 = self.dec3b(self.dec3a(d3))

        # Level 2: 16x16 -> 32x32
        d2 = self.dec2(d3.cat(e2, dim=1))  # (B, 96, 32, 32)
        d2 = self.dec2b(self.dec2a(d2))

        # Level 1: 32x32 -> 32x32
        d1 = self.dec1(d2.cat(e1, dim=1))  # (B, 48, 32, 32)
        d1 = self.dec1b(self.dec1a(d1))

        # Final output
        return self.final_layer(d1)


# Backward compatibility alias
UNetTinygrad = UNetMNIST


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
