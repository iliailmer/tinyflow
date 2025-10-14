# TODO: https://arxiv.org/pdf/2105.05233 <- this architecture
"""A collection of neural networks"""

from loguru import logger
from tinygrad import nn
from tinygrad.tensor import Tensor

from tinyflow.nn_utils.conv import ConvBlock, ConvTransposeBlock


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


class NeuralNetworkMNIST(BaseNeuralNetwork):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_dim + 1, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, out_dim)
        self.init_weights()

    def _init_layer(self, layer):
        layer.weight = Tensor.kaiming_normal(*layer.weight.shape)

    def init_weights(self):
        self._init_layer(self.layer1)
        self._init_layer(self.layer2)
        self._init_layer(self.layer3)
        self._init_layer(self.layer4)

    @logger.catch(reraise=True)
    def __call__(self, x: Tensor, t: Tensor):
        x = x.cat(t, dim=-1)
        x = self.layer1(x).swish()
        x = self.layer2(x).swish()  # pyright: ignore
        x = self.layer3(x).swish()  # pyright: ignore
        x = self.layer4(x)
        return x


class TimeEmbedding:
    def __init__(self, dim):
        self.fc1 = nn.Linear(1, dim)
        self.fc2 = nn.Linear(dim, dim)

    def __call__(self, t: Tensor):
        return self.fc2(self.fc1(t).swish())


class UNetTinygrad(BaseNeuralNetwork):
    """UNet implementation in Tinygrad for CIFAR-10 modeling"""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.enc1 = ConvBlock(in_channels + 1, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)

        self.bottleneck = ConvBlock(256, 512)

        self.dec4 = ConvTransposeBlock(512 + 256, 256, kernel_size=4)
        self.dec3 = ConvTransposeBlock(256 + 128, 128)
        self.dec2 = ConvTransposeBlock(128 + 64, 64)
        self.dec1 = ConvTransposeBlock(64 + 32, 32, stride=1, padding=1, output_padding=0)

        self.final_layer = nn.Conv2d(32, out_channels, kernel_size=1)

    def __call__(self, x: Tensor, t: Tensor):
        t_broadcast = t.reshape(t.shape[0], 1, 1, 1).expand(x.shape[0], 1, x.shape[2], x.shape[3])
        x = x.cat(t_broadcast, dim=1)
        e1 = self.enc1(x)  # -1, 32, 28, 28
        e2 = self.enc2(e1.max_pool2d((2, 2)))  # -1, 64, 14, 14
        e3 = self.enc3(e2.max_pool2d((2, 2)))  # -1, 128, 7, 7
        e4 = self.enc4(e3.max_pool2d((2, 2)))  # -1, 256, 3, 3

        b = self.bottleneck(e4)  # -1, 512, 3, 3

        d4 = self.dec4(b.cat(e4, dim=1))  # -1, 256, 7, 7
        d3 = self.dec3(d4.cat(e3, dim=1))  # -1, 128, 16, 16
        d2 = self.dec2(d3.cat(e2, dim=1))  # -1, 64, 32, 32
        d1 = self.dec1(d2.cat(e1, dim=1))  # -1, 32, 32, 32

        return self.final_layer(d1)
