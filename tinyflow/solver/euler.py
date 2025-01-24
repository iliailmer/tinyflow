from tinyflow.solver.solver import ODESolver

from typing import Callable


class Euler(ODESolver):
    def __init__(self, rhs_fn: Callable):
        super().__init__(rhs_fn)

    def step(self, h, t, rhs_prev):
        return rhs_prev + h * self.rhs(t=t, x=rhs_prev)
