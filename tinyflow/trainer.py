# TODO: implement sampling with a path object
# this can also probably return a custom object
# like Sample with x_0 and x_t as attributes

from abc import ABC
from typing import Any, Callable

from sklearn.datasets import make_moons
from tinygrad.nn.optim import LAMB
from tinygrad.tensor import Tensor as T
from tqdm.auto import tqdm
from tinyflow.nn import BaseNeuralNetwork
from tinyflow.path import Path


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

    def train(self):
        pbar = tqdm(range(self.num_epochs))
        T.training = True
        for iter in pbar:
            x = self.sample_data()
            out, dx_t = self.epoch(x)
            self.optim.zero_grad()
            loss = self.loss_fn(out, dx_t)
            if iter % 50 == 0:
                pbar.set_description_str(f"Loss: {loss.item()}")
            loss.backward()
            self.optim.step()
        return self.model

    def epoch(self, x_1):
        x_1 = T(x_1.astype("float32"))  # pyright: ignore
        t = T.randn(x_1.shape[0], 1)
        x_0 = T.randn(*x_1.shape)
        x_t, dx_t = self.path.sample(x_1=x_1, t=t, x_0=x_0)
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
