"""Self-attention blocks for UNet."""

from tinygrad import nn
from tinygrad.tensor import Tensor


class SelfAttention:
    """
    Multi-head self-attention for spatial features.

    Helps capture long-range dependencies in images, critical for
    generating coherent natural images like CIFAR-10.
    """

    def __init__(self, channels: int, num_heads: int = 8, num_groups: int = 8):
        """
        Args:
            channels: Number of input channels
            num_heads: Number of attention heads
            num_groups: Number of groups for GroupNorm
        """
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        # Normalization before attention
        self.norm = nn.GroupNorm(num_groups, channels)

        # Q, K, V projections (combined for efficiency)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)

        # Output projection
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, C, H, W)
        """
        B, C, H, W = x.shape
        residual = x

        # Normalize
        x = self.norm(x)

        # Compute Q, K, V
        qkv = self.qkv(x)  # (B, 3*C, H, W)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, num_heads, H*W, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, H*W, head_dim)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, num_heads, H*W, H*W)
        attn = attn.softmax(axis=-1)

        # Apply attention to values
        out = attn @ v  # (B, num_heads, H*W, head_dim)
        out = out.transpose(2, 3).reshape(B, C, H, W)

        # Output projection
        out = self.proj(out)

        # Residual connection
        return out + residual
