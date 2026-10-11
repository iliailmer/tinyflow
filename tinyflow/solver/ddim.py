"""DDIM-style deterministic sampler for flow matching."""

from collections.abc import Callable

from tinyflow.nn import BaseNeuralNetwork
from tinyflow.solver.solver import ODESolver, identity


class DDIM(ODESolver):
    """
    DDIM-style deterministic sampler for flow matching.

    With a velocity model, the deterministic DDIM update is the Euler step
    x(t + h) = x(t) + h * v(x(t), t), so this solver gives the same samples as Euler.

    Stochastic sampling (eta > 0) is not implemented: noise added at each step
    must be balanced by a score term in the drift, or the samples keep the noise.

    Usage:
        solver = DDIM(model)
        x = solver.solve(x_init, uniform_time_grid(20))
    """

    def __init__(
        self,
        rhs_fn: Callable | BaseNeuralNetwork,
        preprocess_hook: Callable = identity,
        eta: float = 0.0,
    ):
        """
        Args:
            rhs_fn: Velocity field function (neural network)
            preprocess_hook: Optional preprocessing for time input
            eta: Stochasticity parameter. Only 0 (deterministic) is supported.
        """
        if eta != 0:
            raise ValueError(
                f"DDIM supports only eta=0 (deterministic sampling), got eta={eta}. "
                "Stochastic sampling needs a score term in the drift, which is not implemented."
            )
        super().__init__(rhs_fn, preprocess_hook)
        self.eta = eta

    def step(self, h, t, rhs_prev):
        t_processed = self.preprocess_hook(t, rhs_prev)
        return rhs_prev + h * self.rhs(rhs_prev, t_processed)
