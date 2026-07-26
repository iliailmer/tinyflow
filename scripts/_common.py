"""Shared Hydra config factories used by scripts/train.py and scripts/generate.py."""

from omegaconf import DictConfig
from tinygrad.nn.state import safe_load

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
from tinyflow.utils import preprocess_time_cifar, preprocess_time_mnist, preprocess_time_moons


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


def create_solver(cfg: DictConfig, model, preprocess_hook):
    """Create ODE solver from config."""
    solver_type = cfg.solver.type
    if solver_type == "euler":
        return Euler(model, preprocess_hook=preprocess_hook)
    if solver_type == "heun":
        return Heun(model, preprocess_hook=preprocess_hook)
    if solver_type == "midpoint":
        return MidpointSolver(model, preprocess_hook=preprocess_hook)
    if solver_type == "rk4":
        return RK4(model, preprocess_hook=preprocess_hook)
    if solver_type == "ddim":
        eta = cfg.solver.get("eta", 0.0)
        return DDIM(model, preprocess_hook=preprocess_hook, eta=eta)
    raise ValueError(f"Unknown solver type: {solver_type}")


def get_preprocess_hook(cfg: DictConfig):
    """Get the time-preprocessing hook for the configured dataset."""
    dataset_type = cfg.dataset.get("type", cfg.dataset.name)
    if dataset_type == "moons":
        return preprocess_time_moons
    if dataset_type in ["mnist", "fashion_mnist"]:
        return preprocess_time_mnist
    if dataset_type == "cifar10":
        return preprocess_time_cifar
    raise ValueError(f"Unknown dataset type: {dataset_type}")


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
