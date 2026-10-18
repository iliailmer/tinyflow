from tinygrad import nn


class ConvBlock:
    """
    Residual ConvBlock: Conv -> GroupNorm -> Swish with residual connection.

    With time_dim set, a linear projection of the time vector is added to the
    conv output, one value per channel. dropout applies in training only.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        num_groups=8,
        time_dim: int | None = None,
        dropout: float = 0.0,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_proj = nn.Linear(time_dim, out_channels) if time_dim else None
        self.dropout = dropout

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(num_groups, out_channels)

        # Projection for residual if channel dimensions don't match
        self.use_projection = in_channels != out_channels
        if self.use_projection:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def __call__(self, x, t_emb=None):
        residual = x

        out = self.conv(x)
        if self.time_proj is not None:
            assert t_emb is not None, "this block was built with time_dim and needs t_emb"
            out = out + self.time_proj(t_emb).reshape(x.shape[0], -1, 1, 1)
        out = self.norm(out).swish().dropout(self.dropout)

        # Residual connection
        if self.use_projection:
            residual = self.projection(residual)

        return out + residual


class ConvTransposeBlock:
    """Residual ConvTransposeBlock: ConvTranspose -> GroupNorm -> Swish, with time_dim as in ConvBlock."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
        num_groups=8,
        time_dim: int | None = None,
        dropout: float = 0.0,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_proj = nn.Linear(time_dim, out_channels) if time_dim else None
        self.dropout = dropout

        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding=output_padding,
        )
        self.norm = nn.GroupNorm(num_groups, out_channels)

        # Projection for residual - upsample using ConvTranspose2d
        self.use_projection = in_channels != out_channels or stride != 1
        if self.use_projection:
            self.projection = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                output_padding=output_padding,
            )

    def __call__(self, x, t_emb=None):
        residual = x

        out = self.conv(x)
        if self.time_proj is not None:
            assert t_emb is not None, "this block was built with time_dim and needs t_emb"
            out = out + self.time_proj(t_emb).reshape(x.shape[0], -1, 1, 1)
        out = self.norm(out).swish().dropout(self.dropout)

        # Residual connection (upsample residual if needed)
        if self.use_projection:
            residual = self.projection(residual)

        return out + residual
