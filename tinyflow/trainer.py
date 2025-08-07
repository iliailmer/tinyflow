import os
from abc import ABC
from typing import Any, Callable, Optional

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
            return next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.dataloader)
            return next(self._data_iter)

    @logger.catch(reraise=True)
    def train(self):
        pbar = tqdm(range(self.num_epochs))
        T.training = True
        for epoch_idx in pbar:
            mean_loss = self.epoch(epoch_idx)
            pbar.set_description(f"Loss: {mean_loss:.4f}")
        return self.model

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

    @logger.catch
    def epoch(self, epoch_idx: Optional[int]):
        raise NotImplementedError

    def sample_data(self) -> Any:
        pass

    def predict(self, *args, **kwargs) -> Any:
        raise NotImplementedError


class MNISTTrainer(BaseTrainer):
    @logger.catch(reraise=True)
    def epoch(self, epoch_idx: Optional[int]):
        mean_loss_per_epoch = 0.0
        if epoch_idx is None:
            desc = ""
        else:
            desc = f"Epoch {epoch_idx}"
        for batch in tqdm(self.dataloader, desc=desc):
            x_batch, _ = batch
            x = T(x_batch.astype("float32"))
            t = T.rand(x.shape[0], 1)
            x_0 = T.randn(*x.shape)
            x_t, dx_t = self.path.sample(x_1=x, t=t, x_0=x_0)
            out = self.model(x_t, t)  # pyright: ignore
            self.optim.zero_grad()
            loss = self.loss_fn(out, dx_t)
            loss.backward()
            mean_loss_per_epoch += loss.item()
            self._losses.append(loss.item())
            self.optim.step()
        mean_loss_per_epoch = mean_loss_per_epoch / len(self.dataloader)
        if epoch_idx is not None:
            logger.info(f"Loss: {mean_loss_per_epoch:.4f}")
        return mean_loss_per_epoch
