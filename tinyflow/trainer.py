import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from loguru import logger
from matplotlib import pyplot as plt
from tinygrad.nn.optim import LAMB
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load, safe_save
from tinygrad.tensor import Tensor as T
from tqdm.auto import tqdm

from tinyflow.dataloader import BaseDataloader
from tinyflow.nn import BaseNeuralNetwork
from tinyflow.path import Path
from tinyflow.solver import ODESolver
from tinyflow.utils import visualize_mnist


class BaseTrainer(ABC):
    def __init__(
        self,
        model: BaseNeuralNetwork,
        dataloader: BaseDataloader,
        optim: LAMB,
        loss_fn: Callable,
        path: Path,
        num_epochs: int = 10_000,
        log_interval: int = 50,
    ):
        self.model = model
        self.dataloader = dataloader
        self._data_iter = iter(dataloader)
        self.optim = optim
        self.loss_fn = loss_fn
        self.path = path
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self._losses = []

    def next_batch(self):
        try:
            return next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.dataloader)
            return next(self._data_iter)

    def save_model(self, output_path: str = "model.safetensors"):
        safe_save(get_state_dict(self.model), output_path)

    @staticmethod
    def load_model(model: BaseNeuralNetwork, model_path: str = "model.safetensors"):
        return load_state_dict(model, safe_load(model_path))

    @logger.catch(reraise=True)
    def train(self) -> BaseNeuralNetwork:
        pbar = tqdm(range(self.num_epochs))
        with T.train(True):
            for epoch_idx in pbar:
                mean_loss = self.epoch(epoch_idx)
                pbar.set_description(f"Loss: {mean_loss:.4f}")

        return self.model

    def plot_loss(self, prefix: str, log_to_mlflow: bool = True):
        _ = plt.figure(figsize=(10, 4))
        plt.plot(self._losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.grid(True)
        plt.tight_layout()

        os.makedirs(prefix, exist_ok=True)
        loss_path = os.path.join(prefix, "loss_curve.png")
        plt.savefig(loss_path)

        plt.show()
        plt.close()

    @abstractmethod
    def epoch(self, epoch_idx: int | None):
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        raise NotImplementedError


class MNISTTrainer(BaseTrainer):
    @logger.catch(reraise=True)
    def epoch(self, epoch_idx: int | None):
        mean_loss_per_epoch = 0.0
        if epoch_idx is None:
            desc = ""
        else:
            desc = f"Epoch {epoch_idx}"
        for batch in tqdm(self.dataloader, desc=desc):
            x_batch, _ = batch
            x = T(x_batch.astype("float32"))
            t = T.rand(x.shape[0], 1) * 0.99  # Clamp to avoid t=1.0 singularities
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

    def predict(self, cfg, solver: ODESolver):
        with T.train(False):
            x = T.randn(cfg.training.get("num_samples", 1), 1, 28, 28)
            h_step = cfg.training.step_size
            time_grid = T.linspace(0, 1, int(1 / h_step))

            # Generate visualization
            visualize_mnist(
                x,
                solver=solver,
                time_grid=time_grid,
                h_step=h_step,
                num_plots=cfg.training.get("num_plots", 10),
            )
