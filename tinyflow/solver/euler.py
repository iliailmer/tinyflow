from tinyflow.solver.solver import ODESolver


class Euler(ODESolver):
    def step(self, h, t, rhs_prev):
        t = self.preprocess_hook(t, rhs_prev)
        return rhs_prev + h * self.rhs(rhs_prev, t)
