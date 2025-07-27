from abc import ABC
from typing import Any, Callable

import numpy as np
from loguru import logger
from sklearn.datasets import make_moons
from tinygrad.nn.optim import LAMB
from tinygrad.tensor import Tensor as T
from tqdm.auto import tqdm

from tinyflow.nn import BaseNeuralNetwork
from tinyflow.path import Path
from tinyflow.utils import cifar10, mnist


class BaseTrainer(ABC):
    def __init__(
        self,
        model: BaseNeuralNetwork,
        optim: LAMB,
        loss_fn: Callable,
        path: Path,
        num_epochs: int = 10_000,
        sampling_args: dict = {},
    ):
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn
        self.path = path
        self.num_epochs = num_epochs
        self.sampling_args = sampling_args

    @logger.catch
    def train(self):
        pbar = tqdm(range(self.num_epochs))
        T.training = True
        for iter in pbar:
            x = self.sample_data()
            out, dx_t = self.epoch(x)
            self.optim.zero_grad()
            loss = self.loss_fn(out, dx_t)
            loss.backward()
            if iter % 50 == 0:
                pbar.set_description_str(
                    f"Loss: {loss.item():.4e}"  # ; grad:{self.model.layer2.weight.grad.numpy().mean():.3e}"
                )

            self.optim.step()

        return self.model

    @logger.catch
    def epoch(self, x_1):
        x_1 = T(x_1.astype("float32"))  # pyright: ignore
        t = T.rand(x_1.shape[0], 1) * 0.99  # clamping
        x_0 = T.randn(*x_1.shape)
        x_t, dx_t = self.path.sample(x_1=x_1, t=t, x_0=x_0)
        logger.info(f"t mean={t.numpy().mean():.2f}, dx_t std={dx_t.numpy().std():.3f}")
        out = self.model(x_t, t)
        return out, dx_t

    def sample_data(self) -> Any:
        pass

    def predict(self, *args, **kwargs) -> Any:
        raise NotImplementedError


class MoonsTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        optim,
        loss_fn,
        path,
        num_epochs: int = 10000,
        sampling_args: dict = {},
    ):
        super().__init__(model, optim, loss_fn, path, num_epochs, sampling_args)

    def sample_data(self):
        return make_moons(**self.sampling_args)[0]


class MNISTTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        optim,
        loss_fn,
        path,
        num_epochs: int = 10000,
        sampling_args: dict = {},
    ):
        super().__init__(model, optim, loss_fn, path, num_epochs, sampling_args)
        self.mnist = mnist()

    def sample_data(self) -> Any:
        sample_size = self.sampling_args.get("n_samples", 100)
        idx = np.random.randint(self.mnist.shape[0], size=sample_size)
        return normalize_minmax(self.mnist[idx])

    def epoch(self, x_1):
        x_1 = T(x_1.astype("float32"))
        t = T.rand(x_1.shape[0], 1)
        x_0 = T.randn(*x_1.shape)
        x_t, dx_t = self.path.sample(x_1=x_1, t=t, x_0=x_0)
        out = self.model(x_t, t)
        return out, dx_t


def normalize_minmax(x):
    return (2 * x / x.max()) - 1  # Scale [0,255] to [-1,1]


class CIFARTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        optim,
        loss_fn,
        path,
        num_epochs: int = 10000,
        sampling_args: dict = {},
    ):
        super().__init__(model, optim, loss_fn, path, num_epochs, sampling_args)
        self.cifar = cifar10()

    def epoch(self, x_1):
        x_1 = normalize_minmax(T(x_1.astype("float32"))).reshape((-1, 3, 32, 32))  # pyright: ignore
        t = T.rand(x_1.shape[0], 1, 1, 1).expand((-1, 1, 32, 32))  # pyright: ignore
        x_0 = T.randn(*x_1.shape)  # pyright: ignore
        x_t, dx_t = self.path.sample(x_1=x_1, t=t, x_0=x_0)
        out = self.model(x_t, t)
        return out, dx_t

    def sample_data(self) -> Any:
        sample_size = self.sampling_args.get("n_samples", 100)
        idx = np.random.randint(self.cifar.shape[0], size=sample_size)
        x = self.cifar[idx] / self.cifar.max()
        return x
