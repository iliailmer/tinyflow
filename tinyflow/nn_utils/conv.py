from tinygrad import nn


class ConvBlock:
    """A small reusable block: Conv -> Activation -> Norm"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)

    def __call__(self, x):
        return self.norm(self.conv(x)).swish()


class ConvTransposeBlock:
    """A small reusable block: Conv -> Activation -> Norm"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
    ):
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding=output_padding,
        )
        self.norm = nn.BatchNorm2d(out_channels)

    def __call__(self, x):
        return self.norm(self.conv(x)).swish()
