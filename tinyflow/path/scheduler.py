# see https://arxiv.org/pdf/2412.06264, page 26
# requirea: alpha_0 = sigma_1 = 0; alpha_1 = sigma_0 = 1; alpha_t increases, sigma_t decreases

import math

from tinygrad.tensor import Tensor as T


class BaseScheduler:
    def __init__(self, *args, **kwargs):
        return

    def alpha_t(self, t):
        raise NotImplementedError

    def alpha_t_dot(self, t):
        raise NotImplementedError

    def sigma_t(self, t):
        raise NotImplementedError

    def sigma_t_dot(self, t):
        raise NotImplementedError


class LinearScheduler(BaseScheduler):
    def __init__(self):
        super().__init__()

    def alpha_t(self, t):
        return t

    def alpha_t_dot(self, t):
        return T.ones_like(t)

    def sigma_t(self, t):
        return 1 - t

    def sigma_t_dot(self, t):
        return -T.ones_like(t)


class PolynomialScheduler(BaseScheduler):
    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def alpha_t(self, t):
        return t**self.n

    def alpha_t_dot(self, t):
        return t ** (self.n - 1) * self.n

    def sigma_t(self, t):
        return 1 - t**self.n

    def sigma_t_dot(self, t):
        return -(t ** (self.n - 1)) * self.n


class LinearVarPresScheduler(BaseScheduler):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def alpha_t(self, t):
        return t

    def alpha_t_dot(self, t):
        return T.ones_like(t)

    def sigma_t(self, t):
        return T.sqrt(1 - t**2)

    def sigma_t_dot(self, t):
        return -t / T.sqrt(1 - t**2 + self.eps)


class CosineScheduler(BaseScheduler):
    def __init__(self):
        super().__init__()

    def alpha_t(self, t):
        return T.sin(t * 0.5 * math.pi)

    def alpha_t_dot(self, t):
        return T.cos(t * 0.5 * math.pi) * 0.5 * math.pi

    def sigma_t(self, t):
        return T.cos(t * 0.5 * math.pi)

    def sigma_t_dot(self, t):
        return -T.sin(t * 0.5 * math.pi) * 0.5 * math.pi
