import math

from loguru import logger
from tinygrad import nn
from tinygrad.tensor import Tensor


class TimeEmbedding:
    """
    Simple MLP-based time embedding.

    Takes a scalar time value t and produces a learned embedding.

    TODO: Replace with sinusoidal embeddings for better temporal inductive bias.

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

    TODO: Implement this using sin/cos at multiple frequencies:
        embed[2i] = sin(t * omega_i)
        embed[2i+1] = cos(t * omega_i)
    where omega_i = 10000^(-2i/d)

    This provides better inductive bias for temporal information.

    Args:
        dim: Dimension of the time embedding (should be even)
    """

    def __init__(self, dim: int):
        if dim % 2 != 0:
            raise ValueError(f"Embedding dimension must be even, got {dim}")
        self.dim = dim
        # TODO: Implement sinusoidal embedding computation

    def __call__(self, t: Tensor) -> Tensor:
        """
        Compute sinusoidal time embedding.

        Args:
            t: Time tensor of shape (batch_size, 1)

        Returns:
            Time embedding of shape (batch_size, dim)
        """
        if self.dim % 2 != 0:
            logger.error("embedding dimension must be even")
            raise ValueError("embedding dimension must be even")
        position = Tensor.arange(t.shape[1]).unsqueeze(0)
        div = Tensor.arange(0, self.dim, 2) / self.dim * (-math.log(10000))
        embeddings = Tensor.zeros(t.shape[0], self.dim)
        embeddings[:, 0::2] = Tensor.sin(position * div)
        embeddings[:, 1::2] = Tensor.cos(position * div)
        return embeddings
