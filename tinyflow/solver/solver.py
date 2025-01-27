# tinygrad based ode solver

from typing import Callable
from abc import ABC, abstractmethod

from tinyflow.nn import BaseNeuralNetwork


class ODESolver(ABC):
    def __init__(self, rhs_fn: Callable | BaseNeuralNetwork):
        """
        Base class for ODESolver.
        Args:
            rhs_fn: the right hand side function of the ode
        """
        self.rhs = rhs_fn

    @abstractmethod
    def step(self, h, t, rhs_prev):
        """
        Placeholder for the single ODE solver step
        """
        pass

    @abstractmethod
    def sample(self, h, t, rhs_prev):
        pass
