import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor as T
from tqdm.auto import tqdm

from tinyflow.nn import MLP

plt.style.use("ggplot")
# we have two distributions:
# distrubtion q(x) (unknown), that generates data samples (fixed in time)
# and distribution p(t, x) which is time-dependent and at t=0 is a simple (e.g. 0-1-Gaussian)
# while at t=1 p(1, x) ≈ q(x)
# The goal of flow matching is to predict the path from p(0, x) to p(1, x) for each x

# training process:
# 1. fit the neural network, that network is the velocity field
#   1.1 the network is trained to match the data distributions at random times
num_iter = 10000

model = MLP(2, 2)
optim = Adam(get_parameters(model), lr=0.01)
T.training = True
for iter in tqdm(range(num_iter)):
    x = make_moons(256, noise=0.05)[0]
    # end state, data we would like to arrive at from noise

    x_1 = T(x.astype("float32"))  # pyright: ignore
    x_0 = T.randn(*x_1.shape)  # start state, data we get from pure noise
    t = T.randn(x_1.shape[0], 1)  # time grid

    # linear path from noise to data;
    # noise at t=0, data at t=1
    # see equation 2.5 in https://arxiv.org/pdf/2412.06264
    x_t = t * x_1 + (1 - t) * x_0
    dx_t = x_1 - x_0  # see eq 2.9 on page 6 of https://arxiv.org/pdf/2412.06264
    # parametric velocity field u^theta_t(x)
    out = model(x_t, t)
    optim.zero_grad()

    loss = T.mean((out - dx_t) ** 2)  # pyright: ignore
    if iter % 50 == 0:
        print(f"Loss: {loss.item()}")
    # the reason loss has dx_t is because this is the simplest implementation of flow:
    # techincally loss is MSE(out, u_t(x_t|x_1))
    # where u_t(x|x_1) is d/dt(x_t)==d/dt(t*x_1 + (1-t)*x_0)
    loss.backward()
    optim.step()

# after training, we sample
x = T.randn(300, 2)
h_step = float(1 / 8)
fig, ax = plt.subplots(1, int(1 / h_step), figsize=(30, 4), sharex=True, sharey=True)
time_grid = T.linspace(0, 1, int(1 / h_step))

i = 0
all_xs = []
all_ys = []
for t in time_grid:
    all_xs.append(x.numpy()[:, 0])
    all_ys.append(x.numpy()[:, 1])
    ax[i].scatter(x.numpy()[:, 0], x.numpy()[:, 1], s=10)
    x = model.sample(x, t, h_step)
    i += 1
plt.tight_layout()
plt.show()

plt.savefig("moons_flow.png")
