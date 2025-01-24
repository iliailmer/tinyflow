from typing import Callable

from tinyflow.solver import ODESolver


class MidpointSolver(ODESolver):
    def __init__(self, rhs_fn: Callable):
        super().__init__(rhs_fn)

    def step(self, h, t, rhs_prev):
        return rhs_prev + h * self.rhs(
            t=t + h / 2, x=rhs_prev + h / 2 * self.rhs(t=t, x=rhs_prev)
        )

    def sample(self, h, t, rhs_prev):
        t = t.reshape((1, 1))
        t = t.repeat(rhs_prev.shape[0], 1)
        return self.step(h, t, rhs_prev)
