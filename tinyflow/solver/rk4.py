from tinyflow.solver.solver import ODESolver


class RK4(ODESolver):
    def step(self, h, t, rhs_prev):
        t1 = self.preprocess_hook(t, rhs_prev)
        k1 = self.rhs(rhs_prev, t1)

        t2 = self.preprocess_hook(t + h / 2, rhs_prev + k1 * h / 2)
        k2 = self.rhs(rhs_prev + k1 * h / 2, t2)

        t3 = self.preprocess_hook(t + h / 2, rhs_prev + k2 * h / 2)
        k3 = self.rhs(rhs_prev + k2 * h / 2, t3)

        t4 = self.preprocess_hook(t + h, rhs_prev + k3 * h)
        k4 = self.rhs(rhs_prev + k3 * h, t4)
        result = rhs_prev + h / 6 * (k1 + k2 * 2 + k3 * 2 + k4)
        # Force computation and clear lazy evaluation graph
        result.realize()
        return result
