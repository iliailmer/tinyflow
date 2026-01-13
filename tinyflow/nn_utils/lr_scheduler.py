import math

from tinygrad import nn


class BaseLRScheduler:
    def __init__(self, optimizer: nn.optim.Optimizer):
        self.optimizer = optimizer
        self.base_lr: float = self.optimizer.lr.numpy().item()

    def state_dict(self):
        return {"base_lr": self.base_lr}

    def load_state_dict(self, state_dict):
        self.base_lr = state_dict["base_lr"]

    def get_lr(self) -> float:
        return self.optimizer.lr.numpy().item()

    def step(self, iter: int):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(base_lr={self.base_lr})"


class NullLRScheduler(BaseLRScheduler):
    def __init__(self, optimizer: nn.optim.Optimizer):
        super().__init__(optimizer)

    def step(self, iter: int):
        new_lr = self.base_lr
        self.optimizer.lr.assign([new_lr])


class StepLRScheduler(BaseLRScheduler):
    def __init__(self, optimizer: nn.optim.Optimizer, gamma: float, step_size: int = 1000):
        super().__init__(optimizer)
        assert 0 < gamma < 1, "gamma must be within (0, 1) interval"
        self.gamma = gamma
        self.step_size = step_size

    def step(self, iter: int):
        new_lr = self.base_lr * self.gamma ** (iter // self.step_size)
        self.optimizer.lr.assign([new_lr])


class CosineAnnealingLR(BaseLRScheduler):
    def __init__(
        self,
        optimizer: nn.optim.Optimizer,
        t_max: int,
        eta_min: float = 0.0,
        warm: bool = True,
        t_mult: int = 1,
    ):
        super().__init__(optimizer)
        self.t_max = t_max
        self.eta_min = eta_min
        self.warm = warm
        self.t_mult = t_mult

    def step(self, iter: int):
        if self.warm:
            effective_iter = iter % self.t_max
        else:
            effective_iter = iter

        new_lr = (
            self.eta_min
            + (self.base_lr - self.eta_min)
            * (1 + math.cos(math.pi * effective_iter / self.t_max))
            / 2
        )
        self.optimizer.lr.assign([new_lr])

    def state_dict(self):
        return {
            "base_lr": self.base_lr,
            "t_max": self.t_max,
            "eta_min": self.eta_min,
            "warm": self.warm,
            "t_mult": self.t_mult,
        }

    def load_state_dict(self, state_dict):
        self.base_lr = state_dict["base_lr"]
        self.t_max = state_dict["t_max"]
        self.eta_min = state_dict["eta_min"]
        self.warm = state_dict["warm"]
        self.t_mult = state_dict["t_mult"]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(base_lr={self.base_lr}, t_max={self.t_max}, "
            f"eta_min={self.eta_min}, warm={self.warm}, t_mult={self.t_mult})"
        )


class WarmupScheduler(BaseLRScheduler):
    """Wrapper that adds linear warmup to any base scheduler."""

    def __init__(
        self,
        optimizer: nn.optim.Optimizer,
        base_scheduler: BaseLRScheduler,
        warmup_steps: int,
        warmup_start_lr: float = 0.0,
    ):
        super().__init__(optimizer)
        self.base_scheduler = base_scheduler
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr

    def step(self, iter: int):
        # Warmup phase: linear interpolation from warmup_start_lr to base_lr
        if iter < self.warmup_steps:
            new_lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * (
                iter / self.warmup_steps
            )
            self.optimizer.lr.assign([new_lr])
        else:
            # After warmup, delegate to base scheduler with adjusted iteration
            iter_after_warmup = iter - self.warmup_steps
            self.base_scheduler.step(iter_after_warmup)

    def state_dict(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "warmup_start_lr": self.warmup_start_lr,
            "base_scheduler": self.base_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.base_lr = state_dict["base_lr"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.warmup_start_lr = state_dict["warmup_start_lr"]
        self.base_scheduler.load_state_dict(state_dict["base_scheduler"])

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(warmup_steps={self.warmup_steps}, "
            f"warmup_start_lr={self.warmup_start_lr}, base_scheduler={self.base_scheduler})"
        )
