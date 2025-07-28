import os
from abc import ABC
from typing import Any, Callable

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from tinygrad.nn.optim import LAMB
from tinygrad.tensor import Tensor as T
from tqdm.auto import tqdm

from tinyflow.dataloader import BaseDataloader
from tinyflow.nn import BaseNeuralNetwork
from tinyflow.path import Path


class BaseTrainer(ABC):
    def __init__(
        self,
        model: BaseNeuralNetwork,
        dataloader: BaseDataloader,
        optim: LAMB,
        loss_fn: Callable,
        path: Path,
        num_epochs: int = 10_000,
    ):
        self.model = model
        self.dataloader = dataloader
        self._data_iter = iter(dataloader)
        self.optim = optim
        self.loss_fn = loss_fn
        self.path = path
        self.num_epochs = num_epochs
        self._losses = []

    def next_batch(self):
        try:
            next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.dataloader)
            return next(self._data_iter)

    @logger.catch
    def train(self):
        pbar = tqdm(range(self.num_epochs))
        T.training = True
        for iter in pbar:
            batch = self.next_batch()
            out, dx_t = self.epoch(batch)
            self.optim.zero_grad()
            loss = self.loss_fn(out, dx_t)
            loss.backward()
            self._losses.append(loss.item())
            if iter % 50 == 0:
                pbar.set_description_str(
                    f"Loss: {loss.item():.4e}"  # ; grad:{self.model.layer2.weight.grad.numpy().mean():.3e}"
                )

            self.optim.step()

        return self.model

    @logger.catch
    def plot_loss(self, prefix: str):
        plt.figure(figsize=(10, 4))
        plt.plot(self._losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(prefix, "loss_curve.png"))
        plt.show()

    def epoch(self, x):
        raise NotImplementedError

    def sample_data(self) -> Any:
        pass

    def predict(self, *args, **kwargs) -> Any:
        raise NotImplementedError


class MNISTTrainer(BaseTrainer):
    def epoch(self, batch):
        x_batch, _ = batch
        x = T(x_batch.astype("float32"))
        t = T.rand(x.shape[0], 1) * 0.99
        x_0 = T.randn(*x.shape)
        x_t, dx_t = self.path.sample(x_1=x, t=t, x_0=x_0)
        out = self.model(x_t, t)  # pyright: ignore
        return out, dx_t
