from collections.abc import Callable

from tinyflow.nn import BaseNeuralNetwork
from tinyflow.solver.solver import ODESolver


def identity(t, rhs_prev):
    return t


class MidpointSolver(ODESolver):
    def __init__(
        self,
        rhs_fn: Callable | BaseNeuralNetwork,
        preprocess_hook: Callable = identity,
    ):
        super().__init__(rhs_fn)
        self.preprocess_hook = preprocess_hook

    def step(self, h, t, rhs_prev):
        t1 = self.preprocess_hook(t, rhs_prev)
        k1 = self.rhs(rhs_prev, t1)
        x_mid = rhs_prev + h / 2 * k1
        t2 = self.preprocess_hook(t + h / 2, x_mid)
        return rhs_prev + h * self.rhs(x_mid, t2)

    def sample(self, h, t, rhs_prev):
        return self.step(h, t, rhs_prev)
