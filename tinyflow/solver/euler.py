from collections.abc import Callable

from tinyflow.nn import BaseNeuralNetwork
from tinyflow.solver.solver import ODESolver


def identity(t, rhs_prev):
    return t


class Euler(ODESolver):
    def __init__(
        self,
        rhs_fn: Callable | BaseNeuralNetwork,
        preprocess_hook: Callable = identity,
    ):
        super().__init__(rhs_fn)
        self.preprocess_hook = preprocess_hook

    def step(self, h, t, rhs_prev):
        t = self.preprocess_hook(t, rhs_prev)
        return rhs_prev + h * self.rhs(rhs_prev, t)

    def sample(self, h, t, rhs_prev):
        return self.step(h, t, rhs_prev)
