# tinygrad based ode solver

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import numpy as np
from tinygrad.engine.jit import TinyJit
from tinygrad.tensor import Tensor as T

from tinyflow.nn import BaseNeuralNetwork


def uniform_time_grid(num_steps: int) -> np.ndarray:
    """Return the num_steps + 1 grid points of num_steps equal steps from t=0 to t=1."""
    if num_steps < 1:
        raise ValueError(f"num_steps must be at least 1, got {num_steps}")
    return np.linspace(0.0, 1.0, num_steps + 1)


def identity(t, rhs_prev):
    return t


class ODESolver(ABC):
    def __init__(self, rhs_fn: Callable | BaseNeuralNetwork, preprocess_hook: Callable = identity):
        """
        Base class for ODESolver.
        Args:
            rhs_fn: the right hand side function of the ode
            preprocess_hook: maps (t, x) to the time input of rhs_fn
        """
        self.rhs = rhs_fn
        self.preprocess_hook = preprocess_hook

    @abstractmethod
    def step(self, h, t, rhs_prev) -> T:
        """
        Placeholder for the single ODE solver step
        """
        pass

    def solve(
        self,
        x: T,
        time_grid: Sequence[float] | np.ndarray | T,
        on_step: Callable[[int, float, T], None] | None = None,
        jit: bool = True,
    ) -> T:
        """
        Integrate from time_grid[0] to time_grid[-1].

        Args:
            x: state at time_grid[0]
            time_grid: N + 1 increasing time points for N steps
            on_step: called as on_step(i, t, x) after step i, where x is the state
                at t = time_grid[i + 1]. With jit=True the buffer behind x can be
                reused by later steps, so copy it (for example with .numpy()) to keep it.
            jit: compile the step with TinyJit

        Returns:
            State at time_grid[-1]
        """
        if isinstance(time_grid, T):
            time_grid = time_grid.numpy()
        ts = [float(t) for t in time_grid]
        if len(ts) < 2:
            raise ValueError("time_grid must have at least 2 points")
        step = TinyJit(self.step) if jit else self.step
        x = x.contiguous().realize()
        for i, (t0, t1) in enumerate(zip(ts[:-1], ts[1:], strict=True)):
            h = T([t1 - t0]).contiguous().realize()
            t = T([t0]).contiguous().realize()
            x = step(h, t, x).realize()
            if on_step is not None:
                on_step(i, t1, x)
        return x
