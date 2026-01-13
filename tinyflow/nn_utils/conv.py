from tinygrad import nn


class ConvBlock:
    """Residual ConvBlock: Conv -> GroupNorm -> Swish with residual connection."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, num_groups=8):
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(num_groups, out_channels)

        # Projection for residual if channel dimensions don't match
        self.use_projection = in_channels != out_channels
        if self.use_projection:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def __call__(self, x):
        residual = x

        # Main path: Conv -> GroupNorm -> Swish
        out = self.norm(self.conv(x)).swish()

        # Residual connection
        if self.use_projection:
            residual = self.projection(residual)

        return out + residual


class ConvTransposeBlock:
    """Residual ConvTransposeBlock: ConvTranspose -> GroupNorm -> Swish with residual connection."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
        num_groups=8,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels

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

    def __call__(self, x):
        residual = x

        # Main path: ConvTranspose -> GroupNorm -> Swish
        out = self.norm(self.conv(x)).swish()

        # Residual connection (upsample residual if needed)
        if self.use_projection:
            residual = self.projection(residual)

        return out + residual
