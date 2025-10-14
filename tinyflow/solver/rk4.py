from typing import Callable

from loguru import logger

from tinyflow.nn import BaseNeuralNetwork
from tinyflow.solver.solver import ODESolver


def identity(t, rhs_prev):
    return t


class RK4(ODESolver):
    def __init__(
        self,
        rhs_fn: Callable | BaseNeuralNetwork,
        preprocess_hook: Callable = identity,
    ):
        super().__init__(rhs_fn)
        self.preprocess_hook = preprocess_hook

    @logger.catch(reraise=True)
    def sample(self, h, t, rhs_prev):
        # t = self.preprocess_hook(t, rhs_prev)
        return self.step(h, t, rhs_prev)

    @logger.catch(reraise=True)
    def step(self, h, t, rhs_prev):
        t1 = self.preprocess_hook(t, rhs_prev)
        k1 = self.rhs(rhs_prev, t1)

        t2 = self.preprocess_hook(t + h / 2, rhs_prev + k1 * h / 2)
        k2 = self.rhs(rhs_prev + k1 * h / 2, t2)

        t3 = self.preprocess_hook(t + h / 2, rhs_prev + k2 * h / 2)
        k3 = self.rhs(rhs_prev + k2 * h / 2, t3)

        t4 = self.preprocess_hook(t + h, rhs_prev + k3 * h)
        k4 = self.rhs(rhs_prev + k3 * h, t4)
        return rhs_prev + h / 6 * (k1 + k2 * 2 + k3 * 2 + k4)
