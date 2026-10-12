import math

from tinygrad import nn
from tinygrad.tensor import Tensor


class TimeEmbedding:
    """
    Simple MLP-based time embedding.

    Takes a scalar time value t and produces a learned embedding.

    Args:
        dim: Dimension of the time embedding output
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.fc1 = nn.Linear(1, dim)
        self.fc2 = nn.Linear(dim, dim)

    def __call__(self, t: Tensor) -> Tensor:
        """
        Embed time value.

        Args:
            t: Time tensor of shape (batch_size, 1)

        Returns:
            Time embedding of shape (batch_size, dim)
        """
        return self.fc2(self.fc1(t).swish())


class SinusoidalTimeEmbedding:
    """
    Sinusoidal time embeddings (similar to Transformer positional encodings).

    embed[2i] = sin(scale * t * omega_i), embed[2i+1] = cos(scale * t * omega_i),
    where omega_i = 10000^(-2i/d).

    Args:
        dim: Dimension of the time embedding (should be even)
        scale: Factor applied to t. With t in [0, 1], scale=1000 spreads the
            arguments over the same range as integer diffusion timesteps;
            with scale=1 every argument is below 1 radian.
    """

    def __init__(self, dim: int, scale: float = 1000.0):
        if dim % 2 != 0:
            raise ValueError(f"Embedding dimension must be even, got {dim}")
        self.dim = dim
        self.scale = scale

    def __call__(self, t: Tensor) -> Tensor:
        """
        Compute sinusoidal time embedding.

        Args:
            t: Time tensor of shape (batch_size, 1)

        Returns:
            Time embedding of shape (batch_size, dim)
        """
        if len(t.shape) > 1:
            t = t.squeeze(-1)
        half_dim = self.dim // 2
        freq = (-math.log(10_000) * Tensor.arange(half_dim) / half_dim).exp()
        args = self.scale * t.reshape(-1, 1) * freq.reshape(1, -1)

        return Tensor.stack(args.sin(), args.cos(), dim=2).reshape(-1, self.dim)
