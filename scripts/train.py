"""
Unified training entry point (Hydra configuration + MLflow tracking).

Supports the 2D moons toy dataset (MLP) and image datasets - MNIST, Fashion
MNIST, CIFAR-10 (UNet) - selected via `dataset=...` / `model=...` overrides.

Usage:
    # Moons dataset (fast, for testing)
    uv run scripts/train.py dataset=moons model=neural_network training=moons_default scheduler=linear

    # MNIST training with default config
    uv run scripts/train.py

    # Fashion MNIST / CIFAR-10
    uv run scripts/train.py dataset=fashion_mnist
    uv run scripts/train.py dataset=cifar10 model=unet

    # Override specific parameters
    uv run scripts/train.py scheduler=cosine optimizer.lr=0.01

    # Compare multiple schedulers (multirun)
    uv run scripts/train.py -m scheduler=linear,cosine,polynomial

    # Use a pre-configured experiment
    uv run scripts/train.py +experiment=quick_test
"""

import os

import hydra
import matplotlib.pyplot as plt
import mlflow
from _common import create_lr_scheduler, create_scheduler, create_solver, get_preprocess_hook
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.datasets import make_moons
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save
from tinygrad.tensor import Tensor as T
from tqdm.auto import tqdm

from tinyflow.dataloader import CIFAR10Loader, FashionMNISTLoader, MNISTLoader
from tinyflow.losses import mse
from tinyflow.nn import NeuralNetwork, UNetCIFAR10, UNetCIFAR10Large, UNetMNIST
from tinyflow.path import AffinePath
from tinyflow.time_sampler import BaseTimeSampler, LogitNormalSampler, UniformTimeSampler
from tinyflow.trainer import CIFAR10Trainer, MNISTTrainer
from tinyflow.utils import visualize_moons

plt.style.use("ggplot")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("flow_matching")


def create_model(cfg: DictConfig):
    """Create model from config."""
    model_type = cfg.model.type
    if model_type == "neural_network":
        return NeuralNetwork(
            in_dim=cfg.model.input_dim,
            time_embed_dim=cfg.model.time_embed_dim,
            out_dim=cfg.model.output_dim,
        )

    dataset_type = cfg.dataset.get("type", cfg.dataset.name)
    if model_type == "unet":
        if dataset_type in ["mnist", "fashion_mnist"]:
            return UNetMNIST(in_channels=1, out_channels=1)
        elif dataset_type == "cifar10":
            return UNetCIFAR10(in_channels=3, out_channels=3)
    elif model_type == "unet_large":
        if dataset_type == "cifar10":
            return UNetCIFAR10Large(in_channels=3, out_channels=3)
        else:
            raise ValueError(f"unet_large only supports cifar10, got {dataset_type}")

    raise ValueError(f"Unknown model type: {model_type} with dataset type: {dataset_type}")


def create_dataloader(cfg: DictConfig):
    """Create dataloader from config (image datasets only)."""
    dataset_type = cfg.dataset.get("type", cfg.dataset.name)
    batch_size = cfg.dataset.get("batch_size", 32)
    flatten = cfg.dataset.get("flatten", False)
    shuffle = cfg.dataset.get("shuffle", True)

    if dataset_type == "mnist":
        return MNISTLoader(flatten=flatten, batch_size=batch_size, shuffle=shuffle)
    elif dataset_type == "fashion_mnist":
        path = cfg.dataset.get("path", "dataset/fashion_mnist")
        train = cfg.dataset.get("train", True)
        return FashionMNISTLoader(
            path=path, flatten=flatten, batch_size=batch_size, shuffle=shuffle, train=train
        )
    elif dataset_type == "cifar10":
        path = cfg.dataset.get("path", "dataset/cifar10/cifar-10-batches-py")
        cache = cfg.dataset.get("cache", True)
        normalize = cfg.dataset.get("normalize", True)
        train = cfg.dataset.get("train", True)
        return CIFAR10Loader(
            path=path,
            batch_size=batch_size,
            shuffle=shuffle,
            train=train,
            cache=cache,
            normalize=normalize,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def create_time_sampler(cfg: DictConfig) -> BaseTimeSampler:
    """Create time sampler instance"""
    time_sampler_type = cfg.time_sampler.get("type")
    if time_sampler_type == "uniform":
        low: float = cfg.time_sampler.get("low")
        high: float = cfg.time_sampler.get("high")
        return UniformTimeSampler(low, high)
    if time_sampler_type == "logit_normal":
        low: float = cfg.time_sampler.get("low")
        high: float = cfg.time_sampler.get("high")
        return LogitNormalSampler(low, high)
    raise ValueError(f"Unknown time sampler: {time_sampler_type}")


def create_trainer(
    cfg: DictConfig, model, time_sampler, dataloader, optim, path, lr_scheduler=None
):
    """Create trainer from config (image datasets only)."""
    dataset_type = cfg.dataset.get("type", cfg.dataset.name)
    num_epochs = cfg.training.num_epochs
    log_interval = cfg.training.log_interval
    gradient_accumulation_steps = cfg.training.get("gradient_accumulation_steps", 1)

    if dataset_type in ["mnist", "fashion_mnist"]:
        trainer_cls = MNISTTrainer
    elif dataset_type == "cifar10":
        trainer_cls = CIFAR10Trainer
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return trainer_cls(
        model=model,
        dataloader=dataloader,
        optim=optim,
        loss_fn=mse,
        path=path,
        num_epochs=num_epochs,
        log_interval=log_interval,
        lr_scheduler=lr_scheduler,
        time_sampler=time_sampler,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )


def generate_model_name(cfg: DictConfig) -> str:
    """Generate model name from config."""
    dataset = cfg.dataset.get("type", cfg.dataset.name)
    model_type = cfg.model.type
    scheduler = cfg.scheduler.type.replace("Scheduler", "").lower()
    return f"model_{dataset}_{model_type}_{scheduler}.safetensors"


def train_moons(cfg: DictConfig, model_name: str):
    """Training loop for the 2D moons toy dataset."""
    model = create_model(cfg)
    scheduler = create_scheduler(cfg)
    path = AffinePath(scheduler=scheduler)

    optim = Adam(get_parameters(model), lr=cfg.optimizer.lr)
    lr_scheduler = create_lr_scheduler(cfg, optim)

    time_sampler = create_time_sampler(cfg)
    loss_fn = mse
    losses = []

    pbar = tqdm(range(cfg.training.num_epochs))
    T.training = True
    for iter_idx in pbar:
        x, _ = make_moons(n_samples=cfg.dataset.n_samples, noise=cfg.dataset.noise)
        x_1 = T(x.astype("float32"))  # pyright: ignore
        t = time_sampler.sample(x_1.shape[0], 1)  # T.rand(x_1.shape[0], 1) * 0.99  # clamping
        x_0 = T.randn(*x_1.shape)
        x_t, dx_t = path.sample(x_1=x_1, t=t, x_0=x_0)
        out = model(x_t, t=t)  # pyright: ignore

        optim.zero_grad()
        loss = loss_fn(out, dx_t)
        loss.backward()
        loss_val = loss.item()
        losses.append(loss_val)

        if iter_idx % cfg.training.log_interval == 0:
            desc = f"Loss: {loss_val:.4e}"
            if lr_scheduler is not None:
                desc += f" | LR: {lr_scheduler.get_lr():.6f}"
            pbar.set_description_str(desc)

        optim.step()

        if lr_scheduler is not None:
            lr_scheduler.step(iter_idx)

    if cfg.training.get("save_model", True):
        safe_save(get_state_dict(model), model_name)
        logger.info(f"✓ Model saved to: {model_name}")

    if cfg.training.get("log_artifacts", True):
        output_dir = cfg.get("output_dir", "outputs")
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(10, 4))
        plt.plot(losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_curve.png"))
        plt.close()

    if cfg.training.get("generate_samples", True):
        num_samples = cfg.training.get("n_samples", 100)
        x = T.randn(num_samples, 2)
        h_step = cfg.training.step_size
        time_grid = T.linspace(0, 1, int(1 / h_step))

        solver = create_solver(cfg, model, get_preprocess_hook(cfg))
        visualize_moons(
            x,
            solver=solver,
            time_grid=time_grid,
            h_step=h_step,
            num_plots=cfg.training.get("num_plots", 10),
        )


def train_images(cfg: DictConfig, model_name: str):
    """Training loop for image datasets (MNIST, Fashion MNIST, CIFAR-10)."""
    model = create_model(cfg)
    scheduler = create_scheduler(cfg)
    path = AffinePath(scheduler=scheduler)

    optim = Adam(get_parameters(model), lr=cfg.optimizer.lr)
    lr_scheduler = create_lr_scheduler(cfg, optim)

    dataloader = create_dataloader(cfg)
    time_sampler = create_time_sampler(cfg)
    trainer = create_trainer(cfg, model, time_sampler, dataloader, optim, path, lr_scheduler)

    dataset_name = cfg.dataset.get("type", cfg.dataset.name)
    with mlflow.start_run(run_name=dataset_name):
        mlflow.log_params(dict(cfg))
        mlflow.log_param("model_name", model_name)
        model = trainer.train()
    trainer.save_model(model_name)

    if cfg.training.get("log_artifacts", True):
        output_dir = cfg.get("output_dir", "outputs")
        os.makedirs(output_dir, exist_ok=True)
        trainer.plot_loss(output_dir, log_to_mlflow=True)

    if cfg.training.get("generate_samples", True):
        solver = create_solver(cfg, model, get_preprocess_hook(cfg))
        trainer.predict(cfg, solver)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Hydra entry point; dispatches on dataset type."""
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    if cfg.get("seed"):
        T.manual_seed(cfg.seed)

    model_name = cfg.get("model_name", generate_model_name(cfg))
    logger.info(f"\nModel will be saved as: {model_name}")

    dataset_type = cfg.dataset.get("type", cfg.dataset.name)
    if dataset_type == "moons":
        train_moons(cfg, model_name)
    else:
        train_images(cfg, model_name)


if __name__ == "__main__":
    main()
