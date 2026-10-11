import os
from collections.abc import Callable

import mlflow
from loguru import logger
from matplotlib import pyplot as plt
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import Context
from tinygrad.nn.optim import LAMB
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load, safe_save
from tinygrad.tensor import Tensor as T
from tqdm.auto import tqdm

from tinyflow.dataloader import BaseDataloader
from tinyflow.nn import BaseNeuralNetwork
from tinyflow.path import Path
from tinyflow.solver import ODESolver, uniform_time_grid
from tinyflow.time_sampler import BaseTimeSampler
from tinyflow.utils import visualize_cifar10, visualize_mnist


class BaseTrainer:
    """Training loop with MLflow logging. Subclasses set the three class attributes."""

    image_shape: tuple[int, ...]
    visualize: Callable
    output_name: str

    def __init__(
        self,
        model: BaseNeuralNetwork,
        dataloader: BaseDataloader,
        optim: LAMB,
        loss_fn: Callable,
        path: Path,
        num_epochs: int = 10_000,
        log_interval: int = 50,
        lr_scheduler=None,
        time_sampler: BaseTimeSampler | None = None,
        gradient_accumulation_steps: int = 1,
    ):
        self.model = model
        self.dataloader = dataloader
        self._data_iter = iter(dataloader)
        self.optim = optim
        self.loss_fn = loss_fn
        self.path = path
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.lr_scheduler = lr_scheduler
        if time_sampler is None:
            raise ValueError("Time sampler cannot be None")
        self.time_sampler = time_sampler
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._losses = []
        self.global_step = 0
        self.accumulation_step = 0
        self._jit_train_step = TinyJit(self._train_step)
        self._jit_shape = None

    def next_batch(self):
        try:
            return next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.dataloader)
            return next(self._data_iter)

    def save_model(self, output_path: str = "model.safetensors"):
        safe_save(get_state_dict(self.model), output_path)
        print(f"✓ Model saved to: {output_path}")

    @staticmethod
    def load_model(
        model: BaseNeuralNetwork, model_path: str = "model.safetensors", strict: bool = True
    ):
        return load_state_dict(model, safe_load(model_path), strict=strict)

    @logger.catch(reraise=True)
    def train(self) -> BaseNeuralNetwork:
        pbar = tqdm(range(self.num_epochs))
        with Context(TRAINING=1):
            for epoch_idx in pbar:
                mean_loss = self.epoch(epoch_idx)
                metrics = {"loss": mean_loss}

                # Log current learning rate
                if self.lr_scheduler is not None:
                    current_lr = self.lr_scheduler.get_lr()
                    metrics["learning_rate"] = current_lr

                # Log metrics with step for proper line chart visualization
                mlflow.log_metrics(metrics, step=epoch_idx)
                pbar.set_description(f"Loss: {mean_loss:.4f}")

        return self.model

    def plot_loss(self, prefix: str):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self._losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.grid(True)
        plt.tight_layout()

        os.makedirs(prefix, exist_ok=True)
        loss_path = os.path.join(prefix, "loss_curve.png")
        plt.savefig(loss_path)
        mlflow.log_figure(fig, "loss_curve.png")
        plt.close(fig)

    def _train_step(self, x: T, t: T, x_0: T) -> T:
        x_t, dx_t = self.path.sample(x_1=x, t=t, x_0=x_0)
        self.optim.zero_grad()
        loss = self.loss_fn(self.model(x_t, t), dx_t).backward()
        # Realize the loss with the update. A loss.item() before optim.step() runs the
        # forward pass twice, and a loss.item() after it hangs the Metal GPU.
        return loss.realize(*self.optim.schedule_step())

    def _accumulation_step(self, x: T, t: T, x_0: T) -> T:
        x_t, dx_t = self.path.sample(x_1=x, t=t, x_0=x_0)
        if self.accumulation_step == 0:
            self.optim.zero_grad()

        # Scale the loss so that accumulated gradients average over the steps
        loss = (
            self.loss_fn(self.model(x_t, t), dx_t) / self.gradient_accumulation_steps
        ).backward()
        self.accumulation_step += 1
        if self.accumulation_step >= self.gradient_accumulation_steps:
            self.accumulation_step = 0
            loss.realize(*self.optim.schedule_step())
        else:
            loss.realize(*[p.grad for p in self.optim.params if p.grad is not None])
        return loss * self.gradient_accumulation_steps

    @logger.catch(reraise=True)
    def epoch(self, epoch_idx: int | None):
        mean_loss_per_epoch = 0.0
        desc = f"Epoch {epoch_idx}" if epoch_idx is not None else ""

        for x_batch, _ in tqdm(self.dataloader, desc=desc, leave=False):
            x = T(x_batch).realize()
            t = self.time_sampler.sample(x.shape[0], 1).realize()
            x_0 = T.randn(*x.shape).realize()

            if self._jit_shape is None:
                self._jit_shape = x.shape

            if self.gradient_accumulation_steps > 1:
                loss = self._accumulation_step(x, t, x_0)
            elif x.shape == self._jit_shape:
                loss = self._jit_train_step(x, t, x_0)
            else:
                # The compiled step accepts one batch shape; a smaller last batch runs without JIT
                loss = self._train_step(x, t, x_0)

            loss_value = loss.item()
            mean_loss_per_epoch += loss_value
            self._losses.append(loss_value)

            if self.accumulation_step == 0 and self.lr_scheduler is not None:
                self.lr_scheduler.step(self.global_step)
                self.optim.lr.realize()
                self.global_step += 1

        mean_loss_per_epoch = mean_loss_per_epoch / len(self.dataloader)
        if epoch_idx is not None:
            logger.info(f"Loss: {mean_loss_per_epoch:.4f}")
        return mean_loss_per_epoch

    def predict(self, cfg, solver: ODESolver):
        """Generate samples, save the snapshot figure under output_dir and log it to MLflow."""
        with Context(TRAINING=0):
            x = T.randn(cfg.training.get("n_samples", 1), *self.image_shape)
            image_path = type(self).visualize(
                x,
                solver=solver,
                time_grid=uniform_time_grid(round(1 / cfg.training.step_size)),
                num_plots=cfg.training.get("num_plots", 10),
                save_path=os.path.join(cfg.get("output_dir", "outputs"), self.output_name),
            )
            mlflow.log_artifact(image_path, "generated_images")


class MNISTTrainer(BaseTrainer):
    """Trainer for MNIST and Fashion MNIST."""

    image_shape = (1, 28, 28)
    visualize = staticmethod(visualize_mnist)
    output_name = "mnist_generation.png"


class CIFAR10Trainer(BaseTrainer):
    """Trainer for CIFAR-10 dataset."""

    image_shape = (3, 32, 32)
    visualize = staticmethod(visualize_cifar10)
    output_name = "cifar10_generation.png"
