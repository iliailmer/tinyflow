from collections.abc import Callable

from tinyflow.solver.solver import ODESolver


class Euler(ODESolver):
    def __init__(self, rhs_fn: Callable):
        super().__init__(rhs_fn)

    def step(self, h, t, rhs_prev):
        return rhs_prev + h * self.rhs(rhs_prev, t)

    def sample(self, h, t, rhs_prev):
        return self.step(h, t, rhs_prev)
