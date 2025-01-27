from tinyflow.nn import BaseNeuralNetwork
from tinyflow.solver.solver import ODESolver

from typing import Callable


class RK4(ODESolver):
    def __init__(self, rhs_fn: Callable | BaseNeuralNetwork):
        super().__init__(rhs_fn)

    def sample(self, h, t, rhs_prev):
        t = t.reshape((1, 1))
        t = t.repeat(rhs_prev.shape[0], 1)
        return self.step(h, t, rhs_prev)

    def step(self, h, t, rhs_prev):
        k1 = self.rhs(t=t, x=rhs_prev)
        k2 = self.rhs(t=t + h / 2, x=rhs_prev + k1 * h / 2)
        k3 = self.rhs(t=t + h / 2, x=rhs_prev + k2 * h / 2)
        k4 = self.rhs(t=t + h, x=rhs_prev + k3 * h)
        return rhs_prev + h / 6 * (k1 + k2 * 2 + k3 * 2 + k4)
