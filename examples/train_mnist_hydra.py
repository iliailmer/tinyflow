"""
Image dataset training script with Hydra configuration and MLflow tracking.
Supports MNIST, Fashion MNIST, and CIFAR-10 datasets.
"""

import os

import hydra
import matplotlib.pyplot as plt
import mlflow
from omegaconf import DictConfig, OmegaConf
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor as T

from tinyflow.dataloader import CIFAR10Loader, FashionMNISTLoader, MNISTLoader
from tinyflow.losses import mse
from tinyflow.nn import UNetTinygrad
from tinyflow.nn_utils.lr_scheduler import (
    CosineAnnealingLR,
    StepLRScheduler,
    WarmupScheduler,
)
from tinyflow.path import AffinePath
from tinyflow.path.scheduler import (
    CosineScheduler,
    LinearScheduler,
    LinearVarPresScheduler,
    PolynomialScheduler,
)
from tinyflow.solver import RK4
from tinyflow.trainer import CIFAR10Trainer, FashionMNISTTrainer, MNISTTrainer
from tinyflow.utils import preprocess_time_cifar, preprocess_time_mnist

plt.style.use("ggplot")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("flow_matching")


def create_scheduler(cfg: DictConfig):
    """Create scheduler from config."""
    scheduler_type = cfg.scheduler.type
    if scheduler_type == "LinearScheduler":
        return LinearScheduler()
    if scheduler_type == "CosineScheduler":
        return CosineScheduler()
    if scheduler_type == "LinearVarPresScheduler":
        return LinearVarPresScheduler()
    if scheduler_type == "PolynomialScheduler":
        degree = cfg.scheduler.get("degree", 2)
        return PolynomialScheduler(degree)
    raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def create_lr_scheduler(cfg: DictConfig, optimizer):
    """Create learning rate scheduler from config."""
    if not cfg.get("lr_scheduler"):
        return None

    scheduler_type = cfg.lr_scheduler.type

    # Create base scheduler
    if scheduler_type == "NullLRScheduler":
        return None
    elif scheduler_type == "StepLRScheduler":
        base_scheduler = StepLRScheduler(
            optimizer,
            step_size=cfg.lr_scheduler.get("step_size", 1000),
            gamma=cfg.lr_scheduler.get("gamma", 0.1),
        )
    elif scheduler_type == "CosineAnnealingLR":
        base_scheduler = CosineAnnealingLR(
            optimizer,
            t_max=cfg.lr_scheduler.get("t_max", 5000),
            eta_min=cfg.lr_scheduler.get("eta_min", 0.0),
            warm=cfg.lr_scheduler.get("warm", False),
        )
    elif scheduler_type == "WarmupScheduler":
        # Create nested base scheduler
        base_cfg = cfg.lr_scheduler.base_scheduler
        if base_cfg.type == "StepLRScheduler":
            nested_scheduler = StepLRScheduler(
                optimizer,
                step_size=base_cfg.get("step_size", 1000),
                gamma=base_cfg.get("gamma", 0.1),
            )
        elif base_cfg.type == "CosineAnnealingLR":
            nested_scheduler = CosineAnnealingLR(
                optimizer,
                t_max=base_cfg.get("t_max", 5000),
                eta_min=base_cfg.get("eta_min", 0.0),
                warm=base_cfg.get("warm", False),
            )
        else:
            raise ValueError(f"Unknown base scheduler type: {base_cfg.type}")

        return WarmupScheduler(
            optimizer,
            base_scheduler=nested_scheduler,
            warmup_steps=cfg.lr_scheduler.get("warmup_steps", 500),
            warmup_start_lr=cfg.lr_scheduler.get("warmup_start_lr", 0.0),
        )
    else:
        raise ValueError(f"Unknown LR scheduler type: {scheduler_type}")

    return base_scheduler


def create_model(cfg: DictConfig):
    """Create model from config."""
    model_type = cfg.model.type
    dataset_type = cfg.dataset.get("type", cfg.dataset.name)

    if model_type == "unet":
        if "mnist" in dataset_type:
            return UNetTinygrad()
        if "cifar" in dataset_type:
            return UNetTinygrad(3, 3)

    raise ValueError(f"Unknown model type: {model_type} with dataset type: {dataset_type}")


def create_dataloader(cfg: DictConfig):
    """Create dataloader from config."""
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


def create_trainer(cfg: DictConfig, model, dataloader, optim, path, lr_scheduler=None):
    """Create trainer from config."""
    dataset_type = cfg.dataset.get("type", cfg.dataset.name)
    num_epochs = cfg.training.num_epochs
    log_interval = cfg.training.log_interval

    if dataset_type in ["mnist"]:
        return MNISTTrainer(
            model=model,
            dataloader=dataloader,
            optim=optim,
            loss_fn=mse,
            path=path,
            num_epochs=num_epochs,
            log_interval=log_interval,
            lr_scheduler=lr_scheduler,
        )
    elif dataset_type == "fashion_mnist":
        return FashionMNISTTrainer(
            model=model,
            dataloader=dataloader,
            optim=optim,
            loss_fn=mse,
            path=path,
            num_epochs=num_epochs,
            log_interval=log_interval,
            lr_scheduler=lr_scheduler,
        )
    elif dataset_type == "cifar10":
        return CIFAR10Trainer(
            model=model,
            dataloader=dataloader,
            optim=optim,
            loss_fn=mse,
            path=path,
            num_epochs=num_epochs,
            log_interval=log_interval,
            lr_scheduler=lr_scheduler,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_preprocess_hook(cfg: DictConfig):
    """Get preprocessing hook for time input based on dataset."""
    dataset_type = cfg.dataset.get("type", cfg.dataset.name)
    if dataset_type in ["mnist", "fashion_mnist"]:
        return preprocess_time_mnist
    elif dataset_type == "cifar10":
        return preprocess_time_cifar
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def generate_model_name(cfg: DictConfig) -> str:
    """Generate model name from config."""
    # Get components
    dataset = cfg.dataset.get("type", cfg.dataset.name)
    model_type = cfg.model.type
    scheduler = cfg.scheduler.type.replace("Scheduler", "").lower()

    # Build name: model_<dataset>_<model_type>_<scheduler>.safetensors
    name = f"model_{dataset}_{model_type}_{scheduler}.safetensors"
    return name


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Set random seed if specified
    if cfg.get("seed"):
        T.manual_seed(cfg.seed)

    # Generate model name (can be overridden with model_name parameter)
    model_name = cfg.get("model_name", generate_model_name(cfg))
    print(f"\nModel will be saved as: {model_name}")

    # Create model, scheduler, and path
    model = create_model(cfg)
    scheduler = create_scheduler(cfg)
    path = AffinePath(scheduler=scheduler)

    # Create optimizer
    optim = Adam(get_parameters(model), lr=cfg.optimizer.lr)

    # Create learning rate scheduler
    lr_scheduler = create_lr_scheduler(cfg, optim)

    # Create dataloader and trainer based on dataset type
    dataloader = create_dataloader(cfg)
    trainer = create_trainer(cfg, model, dataloader, optim, path, lr_scheduler)

    # Train the model
    dataset_name = cfg.dataset.get("type", cfg.dataset.name)
    with mlflow.start_run(run_name=dataset_name):
        mlflow.log_params(dict(cfg))
        mlflow.log_param("model_name", model_name)
        model = trainer.train()
    trainer.save_model(model_name)

    # Save loss plot
    if cfg.training.get("log_artifacts", True):
        output_dir = cfg.get("output_dir", "outputs")
        os.makedirs(output_dir, exist_ok=True)
        trainer.plot_loss(output_dir, log_to_mlflow=True)

    # Generate samples if requested
    if cfg.training.get("generate_samples", True):
        preprocess_hook = get_preprocess_hook(cfg)
        solver = RK4(model, preprocess_hook=preprocess_hook)
        trainer.predict(cfg, solver)


if __name__ == "__main__":
    main()
