from typing import Callable

from loguru import logger

from tinyflow.nn import BaseNeuralNetwork
from tinyflow.solver.solver import ODESolver


def identitiy(t, rhs_prev):
    return t


class RK4(ODESolver):
    def __init__(
        self,
        rhs_fn: Callable | BaseNeuralNetwork,
        preprocess_hook: Callable = identitiy,
    ):
        super().__init__(rhs_fn)
        self.preprocess_hook = preprocess_hook

    @logger.catch
    def sample(self, h, t, rhs_prev):
        t = self.preprocess_hook(t, rhs_prev)
        return self.step(h, t, rhs_prev)

    @logger.catch
    def step(self, h, t, rhs_prev):
        k1 = self.rhs(t=t, x=rhs_prev)
        k2 = self.rhs(t=t + h / 2, x=rhs_prev + k1 * h / 2)
        k3 = self.rhs(t=t + h / 2, x=rhs_prev + k2 * h / 2)
        k4 = self.rhs(t=t + h, x=rhs_prev + k3 * h)
        return rhs_prev + h / 6 * (k1 + k2 * 2 + k3 * 2 + k4)
