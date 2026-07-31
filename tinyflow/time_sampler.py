from loguru import logger
from tinygrad.tensor import Tensor


class BaseTimeSampler:
    def __init__(self):
        pass

    def sample(self, *shape):
        """Perform sampling operation"""
        raise NotImplementedError


class UniformTimeSampler(BaseTimeSampler):
    def __init__(self, low: float = 0.0, high: float = 1.0):
        self.l = low
        self.h = high

    @logger.catch(reraise=True)
    def sample(self, *shape) -> Tensor:
        out = Tensor.rand(*shape)
        return (self.h - self.l) * out + self.l


class LogitNormalSampler(BaseTimeSampler):
    def __init__(self, mean: float = 0.0, stddev: float = 1.0):
        self.m = mean
        self.s = stddev

    @logger.catch(reraise=True)
    def sample(self, *shape) -> Tensor:
        out: Tensor = self.m + self.s * Tensor.randn(*shape)
        out: Tensor = out.sigmoid()
        return out
