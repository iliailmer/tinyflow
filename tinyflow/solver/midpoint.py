from tinyflow.solver.solver import ODESolver


class MidpointSolver(ODESolver):
    def step(self, h, t, rhs_prev):
        t1 = self.preprocess_hook(t, rhs_prev)
        k1 = self.rhs(rhs_prev, t1)
        x_mid = rhs_prev + h / 2 * k1
        t2 = self.preprocess_hook(t + h / 2, x_mid)
        return rhs_prev + h * self.rhs(x_mid, t2)
