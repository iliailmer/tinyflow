import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import mlflow
import numpy as np
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
from tinyflow.utils import visualize_cifar10, visualize_mnist

# Import metrics lazily to avoid circular imports
_fid_extractor_cache = {}


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
        lr_scheduler=None,
        gradient_accumulation_steps: int = 1,
        compute_fid: bool = False,
        fid_interval: int = 50,
        fid_num_samples: int = 500,
        dataset_name: str = "mnist",
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
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._losses = []
        self.global_step = 0
        self.accumulation_step = 0

        # FID evaluation settings
        self.compute_fid = compute_fid
        self.fid_interval = fid_interval
        self.fid_num_samples = fid_num_samples
        self.dataset_name = dataset_name
        self._fid_extractor = None

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
    def load_model(model: BaseNeuralNetwork, model_path: str = "model.safetensors"):
        return load_state_dict(model, safe_load(model_path))

    @logger.catch(reraise=True)
    def train(self) -> BaseNeuralNetwork:
        pbar = tqdm(range(self.num_epochs))
        with T.train(True):
            for epoch_idx in pbar:
                mean_loss = self.epoch(epoch_idx)
                metrics = {"loss": mean_loss}

                # Log current learning rate
                if self.lr_scheduler is not None:
                    current_lr = self.lr_scheduler.get_lr()
                    metrics["learning_rate"] = current_lr

                # Evaluate FID at intervals
                if self.compute_fid and (epoch_idx + 1) % self.fid_interval == 0:
                    fid_score = self.evaluate_fid()
                    if not np.isnan(fid_score):
                        metrics["fid"] = fid_score

                # Log metrics with step for proper line chart visualization
                mlflow.log_metrics(metrics, step=epoch_idx)
                pbar.set_description(f"Loss: {mean_loss:.4f}")

        return self.model

    def plot_loss(self, prefix: str, log_to_mlflow: bool = True):
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
        mlflow.log_figure(fig, loss_path)

    def evaluate_fid(self, solver: ODESolver | None = None) -> float:
        """
        Evaluate FID score by generating samples and comparing to real data.

        Args:
            solver: ODE solver to use for generation. If None, uses Euler solver.

        Returns:
            FID score (lower is better)
        """
        import numpy as np

        from tinyflow.metrics import calculate_fid_batched, get_feature_extractor
        from tinyflow.solver import Euler

        # Lazy load feature extractor
        if self._fid_extractor is None:
            self._fid_extractor = get_feature_extractor(self.dataset_name)

        # Check if weights are loaded
        if not self._fid_extractor._weights_loaded:
            logger.warning("FID classifier weights not loaded, skipping FID evaluation")
            return float("nan")

        # Use provided solver or create Euler solver
        if solver is None:
            solver = Euler(self.model)

        # Collect real images from dataloader
        logger.info(f"Collecting {self.fid_num_samples} real images for FID...")
        real_images = []
        real_count = 0
        for batch_images, _ in self.dataloader:
            real_images.append(batch_images)
            real_count += len(batch_images)
            if real_count >= self.fid_num_samples:
                break
        real_images = np.concatenate(real_images, axis=0)[: self.fid_num_samples]

        # Generate samples
        logger.info(f"Generating {self.fid_num_samples} samples for FID...")
        with T.train(False):
            generated_images = []
            batch_size = min(64, self.fid_num_samples)
            num_batches = (self.fid_num_samples + batch_size - 1) // batch_size

            for _ in range(num_batches):
                current_batch = min(
                    batch_size, self.fid_num_samples - len(generated_images) * batch_size
                )
                if current_batch <= 0:
                    break

                # Get input shape from real images
                sample_shape = (current_batch,) + real_images.shape[1:]
                x = T.randn(*sample_shape)

                # Integrate from t=0 to t=1
                num_steps = 20
                h = 1.0 / num_steps
                for step in range(num_steps):
                    t = step * h
                    x = solver.step(h, T.full((current_batch, 1), t), x)

                # Clamp to valid range
                x_np = x.numpy()
                x_np = np.clip(x_np, 0, 1)
                generated_images.append(x_np)

            generated_images = np.concatenate(generated_images, axis=0)[: self.fid_num_samples]

        # Calculate FID
        fid = calculate_fid_batched(
            real_images,
            generated_images,
            self._fid_extractor,
            batch_size=64,
        )

        logger.info(f"FID score: {fid:.4f}")
        return fid

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
            x = T(x_batch)
            t = T.rand(x.shape[0], 1) * 0.99
            x_0 = T.randn(*x.shape)
            x_t, dx_t = self.path.sample(x_1=x, t=t, x_0=x_0)
            out = self.model(x_t, t)

            # Zero gradients at start of accumulation
            if self.accumulation_step == 0:
                self.optim.zero_grad()

            # Scale loss by accumulation steps for proper gradient averaging
            loss = self.loss_fn(out, dx_t) / self.gradient_accumulation_steps
            loss.backward()  # Compute gradients

            loss_value = loss.item() * self.gradient_accumulation_steps  # Un-scale for logging
            mean_loss_per_epoch += loss_value
            self._losses.append(loss_value)

            self.accumulation_step += 1

            # Step optimizer when accumulation is complete
            if self.accumulation_step >= self.gradient_accumulation_steps:
                self.optim.step()  # Already realizes internally
                self.accumulation_step = 0

                # Step learning rate scheduler per optimizer step
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(self.global_step)
                    self.global_step += 1

        mean_loss_per_epoch = mean_loss_per_epoch / len(self.dataloader)
        if epoch_idx is not None:
            logger.info(f"Loss: {mean_loss_per_epoch:.4f}")
        return mean_loss_per_epoch

    def predict(self, cfg, solver: ODESolver):
        with T.train(False):
            x = T.randn(cfg.training.get("num_samples", 1), 1, 28, 28)
            h_step = cfg.training.step_size
            time_grid = T.linspace(0, 1, int(1 / h_step))

            # Generate visualization and save
            image_path = visualize_mnist(
                x,
                solver=solver,
                time_grid=time_grid,
                h_step=h_step,
                num_plots=cfg.training.get("num_plots", 10),
                save_path="outputs/mnist_generation.png",
            )

            # Log to MLflow
            try:
                mlflow.log_artifact(image_path, "generated_images")
                logger.info(f"Logged generated image to MLflow: {image_path}")
            except Exception as e:
                logger.warning(f"Failed to log image to MLflow: {e}")


class CIFAR10Trainer(BaseTrainer):
    """Trainer for CIFAR-10 dataset."""

    @logger.catch(reraise=True)
    def epoch(self, epoch_idx: int | None):
        mean_loss_per_epoch = 0.0
        desc = f"Epoch {epoch_idx}" if epoch_idx is not None else ""

        for batch in tqdm(self.dataloader, desc=desc, leave=False):
            x_batch, _ = batch
            x = T(x_batch)  # Already float32 from dataloader

            t = T.rand(x.shape[0], 1) * 0.99
            x_0 = T.randn(*x.shape)

            x_t, dx_t = self.path.sample(x_1=x, t=t, x_0=x_0)
            out = self.model(x_t, t)

            # Zero gradients at start of accumulation
            if self.accumulation_step == 0:
                self.optim.zero_grad()

            # Scale loss by accumulation steps for proper gradient averaging
            loss = self.loss_fn(out, dx_t) / self.gradient_accumulation_steps
            loss.backward()  # Compute gradients

            loss_value = loss.item() * self.gradient_accumulation_steps  # Un-scale for logging
            mean_loss_per_epoch += loss_value
            self._losses.append(loss_value)

            self.accumulation_step += 1

            # Step optimizer when accumulation is complete
            if self.accumulation_step >= self.gradient_accumulation_steps:
                self.optim.step()  # Already realizes internally
                self.accumulation_step = 0

                # Step learning rate scheduler per optimizer step
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(self.global_step)
                    self.global_step += 1

        mean_loss_per_epoch = mean_loss_per_epoch / len(self.dataloader)
        if epoch_idx is not None:
            logger.info(f"Loss: {mean_loss_per_epoch:.4f}")
        return mean_loss_per_epoch

    def predict(self, cfg, solver: ODESolver):
        """Generate CIFAR-10 samples."""
        with T.train(False):
            num_samples = cfg.training.get("num_samples", 1)
            x = T.randn(num_samples, 3, 32, 32)
            h_step = cfg.training.step_size
            time_grid = T.linspace(0, 1, int(1 / h_step))

            # Generate visualization and save
            image_path = visualize_cifar10(
                x,
                solver=solver,
                time_grid=time_grid,
                h_step=h_step,
                num_plots=cfg.training.get("num_plots", 10),
                save_path="outputs/cifar10_generation.png",
            )

            # Log to MLflow
            try:
                mlflow.log_artifact(image_path, "generated_images")
                logger.info(f"Logged generated image to MLflow: {image_path}")
            except Exception as e:
                logger.warning(f"Failed to log image to MLflow: {e}")
