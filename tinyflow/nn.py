"""A collection of neural networks"""

from tinygrad import tensor, nn

Tensor = tensor.Tensor


class MLP:
    def __init__(self, in_dim, out_dim):
        self.layer1 = nn.Linear(in_dim + 1, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, out_dim)
        self.eps = 1e-10

    def __call__(self, x: Tensor, t: Tensor):
        x = x.cat(t, dim=-1)
        x = self.layer1(x).elu()  # pyright: ignore
        x = self.layer2(x).elu()  # pyright: ignore
        x = self.layer3(x).elu()  # pyright: ignore
        return self.layer4(x)

    def sample(self, x: Tensor, t: Tensor, h_step):
        # this is where the ODE is solved
        # d/dt x_t = u_t(x_t|x_1)
        # explicit midpoint method https://en.wikipedia.org/wiki/Midpoint_method
        t = t.reshape((1, 1))
        t = t.repeat(x.shape[0], 1)
        x_t_next = x + h_step * self(x + h_step / 2 * self(x, t), t + h_step / 2)

        return x_t_next
