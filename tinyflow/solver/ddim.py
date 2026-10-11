"""
DDIM-style deterministic sampler for flow matching.

Allows for fast sampling by skipping timesteps while maintaining deterministic generation.
Can generate high-quality samples with far fewer steps (e.g., 10-50 instead of 100+).
"""

from collections.abc import Callable

from tinygrad.tensor import Tensor as T

from tinyflow.nn import BaseNeuralNetwork
from tinyflow.solver.solver import ODESolver


def identity(t, rhs_prev):
    return t


class DDIM(ODESolver):
    """
    DDIM-style deterministic sampler for flow matching.

    Unlike standard ODE solvers that use uniform timesteps, DDIM allows skipping
    steps by using a custom timestep schedule. This enables high-quality generation
    with significantly fewer function evaluations.

    Key features:
    - Deterministic sampling (no noise injection)
    - Adaptive timestep schedule
    - Works well with 10-50 steps instead of 100+
    - Popular in Stable Diffusion and modern diffusion models

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
        Initialize DDIM solver.

        Args:
            rhs_fn: Velocity field function (neural network)
            preprocess_hook: Optional preprocessing for time input
            eta: Stochasticity parameter (0 = deterministic, 1 = DDPM-like)
                 For flow matching, eta=0 (deterministic) is recommended
        """
        super().__init__(rhs_fn)
        self.preprocess_hook = preprocess_hook
        self.eta = eta

    def step(self, h, t, rhs_prev):
        """
        Perform one DDIM step with adaptive timestep.

        For flow matching, this is essentially a first-order step but allows
        for non-uniform timestep schedules.

        Args:
            h: Step size (can be variable)
            t: Current time
            rhs_prev: Current state x(t)

        Returns:
            x(t + h) using DDIM-style update
        """
        t_processed = self.preprocess_hook(t, rhs_prev)

        # Get velocity at current state
        velocity = self.rhs(rhs_prev, t_processed)

        # DDIM update (deterministic for eta=0)
        # For flow matching: x(t+h) ≈ x(t) + h * v_θ(x(t), t)
        x_next = rhs_prev + h * velocity

        # Optional: add small noise for stochastic sampling (eta > 0)
        # For flow matching, we typically keep eta=0 (deterministic)
        if self.eta > 0:
            noise_scale = self.eta * (h**0.5)
            x_next = x_next + noise_scale * T.randn(*x_next.shape)

        return x_next

    def sample(self, h, t, rhs_prev):
        return self.step(h, t, rhs_prev)
