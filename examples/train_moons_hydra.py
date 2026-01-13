import os

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from sklearn.datasets import make_moons
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor as T
from tqdm.auto import tqdm

from tinyflow.losses import mse
from tinyflow.nn import NeuralNetwork
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
from tinyflow.utils import preprocess_time_moons, visualize_moons

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
    if model_type == "neural_network":
        return NeuralNetwork(
            in_dim=cfg.model.input_dim,
            time_embed_dim=cfg.model.time_embed_dim,
            out_dim=cfg.model.output_dim,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def generate_model_name(cfg: DictConfig) -> str:
    """Generate model name from config."""
    dataset = cfg.dataset.name
    model_type = cfg.model.type
    scheduler = cfg.scheduler.type.replace("Scheduler", "").lower()
    name = f"model_{dataset}_{model_type}_{scheduler}.safetensors"
    return name


def epoch(x_1, model, path):
    """Single training epoch."""
    x_1 = T(x_1.astype("float32"))  # pyright: ignore
    t = T.rand(x_1.shape[0], 1) * 0.99  # clamping
    x_0 = T.randn(*x_1.shape)
    x_t, dx_t = path.sample(x_1=x_1, t=t, x_0=x_0)
    out = model(x_t, t=t)  # pyright: ignore
    return out, dx_t


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

    loss_fn = mse
    _losses = []

    # Training loop
    pbar = tqdm(range(cfg.training.num_epochs))
    T.training = True
    for iter_idx in pbar:
        x, _ = make_moons(
            n_samples=cfg.dataset.n_samples,
            noise=cfg.dataset.noise,
        )
        out, dx_t = epoch(x, model, path)
        optim.zero_grad()
        loss = loss_fn(out, dx_t)
        loss.backward()
        loss_val = loss.item()
        _losses.append(loss_val)

        if iter_idx % cfg.training.log_interval == 0:
            desc = f"Loss: {loss_val:.4e}"
            if lr_scheduler is not None:
                current_lr = lr_scheduler.get_lr()
                desc += f" | LR: {current_lr:.6f}"
            pbar.set_description_str(desc)

        optim.step()

        # Step learning rate scheduler per iteration
        if lr_scheduler is not None:
            lr_scheduler.step(iter_idx)

    # Save model after training
    if cfg.training.get("save_model", True):
        from tinygrad.nn.state import get_state_dict, safe_save

        safe_save(get_state_dict(model), model_name)
        print(f"✓ Model saved to: {model_name}")

    # Plot loss curve after training
    if cfg.training.get("log_artifacts", True):
        output_dir = cfg.get("output_dir", "outputs")
        os.makedirs(output_dir, exist_ok=True)

        _ = plt.figure(figsize=(10, 4))
        plt.plot(_losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.grid(True)
        plt.tight_layout()

        loss_path = os.path.join(output_dir, "loss_curve.png")
        plt.savefig(loss_path)
        plt.close()

    # Generate samples after training
    if cfg.training.get("generate_samples", True):
        num_samples = cfg.training.get("n_samples", 100)
        x = T.randn(num_samples, 2)
        h_step = cfg.training.step_size
        time_grid = T.linspace(0, 1, int(1 / h_step))

        solver = RK4(model, preprocess_hook=preprocess_time_moons)
        visualize_moons(
            x,
            solver=solver,
            time_grid=time_grid,
            h_step=h_step,
            num_plots=cfg.training.get("num_plots", 10),
        )


if __name__ == "__main__":
    main()
