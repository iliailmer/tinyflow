"""MNIST training script with Hydra configuration and MLflow tracking."""

import os

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor as T

from tinyflow.dataloader import MNISTLoader
from tinyflow.losses import mse
from tinyflow.nn import UNetTinygrad
from tinyflow.path import AffinePath
from tinyflow.path.scheduler import (
    CosineScheduler,
    LinearScheduler,
    LinearVarPresScheduler,
    PolynomialScheduler,
)
from tinyflow.solver import RK4
from tinyflow.trainer import MNISTTrainer
from tinyflow.utils import preprocess_time_mnist

plt.style.use("ggplot")


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


def create_model(cfg: DictConfig):
    """Create model from config."""
    model_type = cfg.model.type
    if model_type == "unet":
        return UNetTinygrad()
    raise ValueError(f"Unknown model type: {model_type}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Set random seed if specified
    if cfg.get("seed"):
        T.manual_seed(cfg.seed)

    # Create model, scheduler, and path
    model = create_model(cfg)
    scheduler = create_scheduler(cfg)
    path = AffinePath(scheduler=scheduler)

    # Create optimizer
    optim = Adam(get_parameters(model), lr=cfg.optimizer.lr)

    # Create dataloader
    dataloader = MNISTLoader(
        flatten=cfg.dataset.flatten,
        batch_size=cfg.dataset.get("batch_size", 32),
    )

    # Create trainer
    trainer = MNISTTrainer(
        model=model,
        dataloader=dataloader,
        optim=optim,
        loss_fn=mse,
        path=path,
        num_epochs=cfg.training.num_epochs,
        log_interval=cfg.training.log_interval,
    )

    # Train the model
    model = trainer.train()
    trainer.save_model()
    # Save loss plot
    if cfg.training.get("log_artifacts", True):
        output_dir = cfg.get("output_dir", "outputs")
        os.makedirs(output_dir, exist_ok=True)
        trainer.plot_loss(output_dir, log_to_mlflow=True)

    # Generate samples if requested
    if cfg.training.get("generate_samples", True):
        solver = RK4(model, preprocess_hook=preprocess_time_mnist)

        trainer.predict(cfg, solver)


if __name__ == "__main__":
    main()
