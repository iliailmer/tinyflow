"""Shared Hydra config helpers used by scripts/train.py and scripts/generate.py."""

import numpy as np
from omegaconf import DictConfig
from tinygrad.nn.state import safe_load
from tinygrad.tensor import Tensor as T

from tinyflow.nn import NeuralNetwork, UNetCIFAR10, UNetCIFAR10Large, UNetMNIST
from tinyflow.nn_utils.lr_scheduler import (
    CosineAnnealingLR,
    StepLRScheduler,
    WarmupScheduler,
)
from tinyflow.path.scheduler import (
    CosineScheduler,
    LinearScheduler,
    LinearVarPresScheduler,
    PolynomialScheduler,
)
from tinyflow.solver import DDIM, RK4, Euler, Heun, MidpointSolver
from tinyflow.time_sampler import (
    BaseTimeSampler,
    CosSampler,
    LogitNormalSampler,
    ModeHeavyTailSampler,
    UniformTimeSampler,
)
from tinyflow.utils import preprocess_time


def create_model(cfg: DictConfig, model_path: str | None = None):
    """Create model from config. With model_path, size the time embedding to match that checkpoint."""
    model_type = cfg.model.type
    if model_type == "neural_network":
        return NeuralNetwork(
            in_dim=cfg.model.input_dim,
            time_embed_dim=cfg.model.time_embed_dim,
            out_dim=cfg.model.output_dim,
        )

    dataset_type = cfg.dataset.get("type", cfg.dataset.name)
    if model_type == "unet" and dataset_type in ("mnist", "fashion_mnist", "cifar10"):
        model_cls, channels = (UNetCIFAR10, 3) if dataset_type == "cifar10" else (UNetMNIST, 1)
        if model_path is None:
            return model_cls(channels, channels)
        time_embed_dim = detect_time_embed_dim(model_path, in_channels=channels)
        return model_cls(channels, channels, time_embed_dim=time_embed_dim)
    if model_type == "unet_large" and dataset_type == "cifar10":
        return UNetCIFAR10Large(in_channels=3, out_channels=3)

    raise ValueError(f"Unknown model type: {model_type} with dataset type: {dataset_type}")


def create_time_sampler(cfg: DictConfig) -> BaseTimeSampler:
    """Create training time sampler from config."""
    sampler_type = cfg.time_sampler.type
    if sampler_type == "uniform":
        return UniformTimeSampler(cfg.time_sampler.low, cfg.time_sampler.high)
    if sampler_type == "logit_normal":
        return LogitNormalSampler(cfg.time_sampler.mean, cfg.time_sampler.stddev)
    if sampler_type == "mode_heavy_tail":
        return ModeHeavyTailSampler(cfg.time_sampler.scale)
    if sampler_type == "cos":
        return CosSampler()
    raise ValueError(f"Unknown time sampler: {sampler_type}")


def seed_everything(cfg: DictConfig):
    """Seed tinygrad and NumPy (dataloader shuffling, make_moons) from cfg.seed."""
    seed = cfg.get("seed")
    if seed is not None:
        T.manual_seed(seed)
        np.random.seed(seed)


def get_value_range(cfg: DictConfig) -> tuple[float, float]:
    """Range of the image values in training, from dataset.value_range. Default (0, 1)."""
    low, high = cfg.dataset.get("value_range", [0.0, 1.0])
    return float(low), float(high)


def create_scheduler(cfg: DictConfig):
    """Create flow-matching path scheduler from config."""
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


def create_solver(cfg: DictConfig, model):
    """Create ODE solver from config."""
    solver_type = cfg.solver.type
    if solver_type == "euler":
        return Euler(model, preprocess_hook=preprocess_time)
    if solver_type == "heun":
        return Heun(model, preprocess_hook=preprocess_time)
    if solver_type == "midpoint":
        return MidpointSolver(model, preprocess_hook=preprocess_time)
    if solver_type == "rk4":
        return RK4(model, preprocess_hook=preprocess_time)
    if solver_type == "ddim":
        eta = cfg.solver.get("eta", 0.0)
        return DDIM(model, preprocess_hook=preprocess_time, eta=eta)
    raise ValueError(f"Unknown solver type: {solver_type}")


def detect_time_embed_dim(model_path: str, in_channels: int) -> int:
    """Detect time_embed_dim from saved weights by inspecting enc1.conv.weight shape."""
    try:
        state = safe_load(model_path)
        key = "enc1.conv.weight"
        if key in state:
            return int(state[key].shape[1]) - in_channels
    except Exception:
        pass
    return 64  # default
