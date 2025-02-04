from typing import NamedTuple
from tinygrad.tensor import Tensor as T

from tinyflow.path.scheduler import BaseScheduler


class Sample(NamedTuple):
    x_t: T
    dx_t: T


class Path:
    def __init__(self, scheduler: BaseScheduler):
        self.scheduler = scheduler

    def sample(self, x_1, t, x_0) -> Sample:
        """
        Sample points at time t
        """
        return Sample(x_t=T.zeros(0), dx_t=T.zeros(0))

    def target_to_velocity(self, x_1, x_t, t):
        pass


class AffinePath(Path):
    def __init__(self, scheduler: BaseScheduler):
        super().__init__(scheduler=scheduler)

    def sample(self, x_1, t, x_0) -> Sample:
        x_t = self.scheduler.alpha_t(t) * x_1 + self.scheduler.sigma_t(t) * x_0
        dx_t = self.scheduler.alpha_t_dot(t) * x_1 + self.scheduler.sigma_t_dot(t) * x_0
        return Sample(x_t=x_t, dx_t=dx_t)


class OptimalTransportPath:
    def __init__(self, sigma_min: float = 0):
        super().__init__()
        self.sigma_min = sigma_min

    def sample(self, x_1, t, x_0):
        """
        Sample points at time t
        x_1: data points at t=1 (actual data sample)
        t: time moment
        x_0: random noise (N(0,1) sample)
        """
        x_t = x_1 * t + (1 - (1 - self.sigma_min) * t) * x_0
        dx_t = x_1 - x_0
        return Sample(x_t, dx_t)
