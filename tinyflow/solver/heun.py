"""
Heun's Method (Predictor-Corrector RK2) ODE solver.

Second-order accurate solver widely used in modern diffusion models (e.g., EDM).
Provides better accuracy than Euler with minimal computational overhead.
"""

from collections.abc import Callable

from tinyflow.nn import BaseNeuralNetwork
from tinyflow.solver.solver import ODESolver


def identity(t, rhs_prev):
    return t


class Heun(ODESolver):
    """
    Heun's Method (improved Euler / RK2 predictor-corrector).

    Algorithm:
    1. Predictor: k1 = f(x, t), x_pred = x + h * k1
    2. Corrector: k2 = f(x_pred, t+h)
    3. Final: x_new = x + h * (k1 + k2) / 2

    Second-order accurate, widely used in EDM and modern diffusion samplers.
    Better quality than Euler with only 2x function evaluations per step.
    """

    def __init__(
        self,
        rhs_fn: Callable | BaseNeuralNetwork,
        preprocess_hook: Callable = identity,
    ):
        super().__init__(rhs_fn)
        self.preprocess_hook = preprocess_hook

    def step(self, h, t, rhs_prev):
        """
        Perform one Heun step.

        Args:
            h: Step size
            t: Current time
            rhs_prev: Current state x(t)

        Returns:
            x(t + h) using Heun's method
        """
        # Preprocess time input
        t_processed = self.preprocess_hook(t, rhs_prev)

        # Predictor step (Euler)
        k1 = self.rhs(rhs_prev, t_processed)
        x_pred = rhs_prev + h * k1

        # Corrector step
        t_next = t + h
        t_next_processed = self.preprocess_hook(t_next, x_pred)
        k2 = self.rhs(x_pred, t_next_processed)

        # Average of slopes (trapezoid rule)
        return rhs_prev + h * (k1 + k2) / 2

    def sample(self, h, t, rhs_prev):
        return self.step(h, t, rhs_prev)
