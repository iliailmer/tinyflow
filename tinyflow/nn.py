"""A collection of neural networks"""

from loguru import logger
from tinygrad import nn
from tinygrad.tensor import Tensor


class BaseNeuralNetwork:
    def __call__(self, x: Tensor, t: Tensor) -> Tensor:  # pyright: ignore
        pass


class MLP(BaseNeuralNetwork):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_dim + 1, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, out_dim)

    @logger.catch
    def __call__(self, x: Tensor, t: Tensor):
        x = x.cat(t, dim=-1)
        x = self.layer1(x).elu()  # pyright: ignore
        x = self.layer2(x).elu()  # pyright: ignore
        x = self.layer3(x).elu()  # pyright: ignore
        return self.layer4(x)

    @logger.catch
    def sample(self, x: Tensor, t: Tensor, h_step) -> Tensor:
        # this is where the ODE is solved
        # d/dt x_t = u_t(x_t|x_1)
        # explicit midpoint method https://en.wikipedia.org/wiki/Midpoint_method
        t = t.reshape((1, 1))
        t = t.repeat(x.shape[0], 1)
        x_t_next = x + h_step * self(x + h_step / 2 * self(x, t), t + h_step / 2)

        return x_t_next


class NeuralNetwork(BaseNeuralNetwork):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_dim + 1, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, out_dim)

    @logger.catch
    def __call__(self, x: Tensor, t: Tensor):
        x = x.cat(t, dim=-1)
        x = self.layer1(x).elu()  # pyright: ignore
        x = self.layer2(x).elu()  # pyright: ignore
        x = self.layer3(x).elu()  # pyright: ignore
        x = self.layer4(x)
        return x


def swish(x: Tensor):
    return x.sigmoid() * x


class NeuralNetworkMNIST(BaseNeuralNetwork):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_dim + 1, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, out_dim)

    @logger.catch
    def __call__(self, x: Tensor, t: Tensor):
        x = x.cat(t, dim=-1)
        x = self.layer1(x).swish()
        x = self.layer2(x).swish()  # pyright: ignore
        x = self.layer3(x).swish()  # pyright: ignore
        x = self.layer4(x)
        return x


class NeuralNetworkCIFAR(BaseNeuralNetwork):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1
        )
        self.layer2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1
        )
        self.layer3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1
        )
        self.layer4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1
        )
        self.classifier = NeuralNetworkMNIST(64, 3072)

    def __call__(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.layer1(x).layernorm().silu()  # 16x16 # pyright: ignore
        x = self.layer2(x).layernorm().silu()  # 8x8 # pyright: ignore
        x = self.layer3(x).layernorm().silu()  # 4x4 # pyright: ignore
        x = self.layer4(x).layernorm().silu()  # 2x2 # pyright: ignore
        x = x.avg_pool2d()  # 1x1 # pyright: ignore
        x = self.classifier(x.flatten(1), t)
        return x


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


class UNetTinygrad(BaseNeuralNetwork):
    """UNet implementation in Tinygrad for CIFAR-10 modeling"""

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # Encoder (Downsampling Path)
        self.enc1 = ConvBlock(in_channels + 1, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)

        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)

        # Decoder (Upsampling Path)
        self.dec4 = ConvTransposeBlock(512 + 256, 256)  # (Upsampled + Skip)
        self.dec3 = ConvTransposeBlock(256 + 128, 128)
        self.dec2 = ConvTransposeBlock(128 + 64, 64)
        self.dec1 = ConvTransposeBlock(
            64 + 32, 32, stride=1, padding=1, output_padding=0
        )

        # Final Output Layer
        self.final_layer = nn.Conv2d(32, out_channels, kernel_size=1)

    def __call__(self, x: Tensor, t: Tensor):
        # Encoder
        x = x.cat(t, dim=1)
        e1 = self.enc1(x)  # -1, 32, 32, 32
        e2 = self.enc2(e1.max_pool2d(2))  # -1, 64, 16, 16
        e3 = self.enc3(e2.max_pool2d(2))  # -1, 128, 8, 8
        e4 = self.enc4(e3.max_pool2d(2))  # -1, 256, 4, 4

        # Bottleneck
        b = self.bottleneck(e4)  # -1, 512, 4, 4
        # Decoder with Upsampling
        d4 = self.dec4(b.cat(e4, dim=1))  # -1, 256, 8, 8
        d3 = self.dec3(d4.cat(e3, dim=1))  # -1, 128, 16, 16
        d2 = self.dec2(d3.cat(e2, dim=1))  # -1, 64, 32, 32
        d1 = self.dec1(d2.cat(e1, dim=1))  # -1, 32, 32, 32

        return self.final_layer(d1).sigmoid()  # Ensure output is in [0,1]
