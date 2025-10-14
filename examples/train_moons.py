import argparse
import os

from loguru import logger
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor as T
from tqdm.auto import tqdm

from tinyflow.losses import mse
from tinyflow.nn import (
    NeuralNetwork,
)
from tinyflow.path import AffinePath
from tinyflow.path.scheduler import (
    CosineScheduler,
    LinearScheduler,
    LinearVarPresScheduler,
    PolynomialScheduler,
)
from tinyflow.solver import RK4
from tinyflow.utils import (
    visualize_moons,
    preprocess_time_moons,
)


def epoch(x_1):
    x_1 = T(x_1.astype("float32"))  # pyright: ignore
    t = T.rand(x_1.shape[0], 1) * 0.99  # clamping
    x_0 = T.randn(*x_1.shape)
    x_t, dx_t = path.sample(x_1=x_1, t=t, x_0=x_0)
    out = model(x_t, t=t)  # pyright: ignore
    return out, dx_t


parser = argparse.ArgumentParser()
parser.add_argument("--poly-deg", "-n", type=int, default=2, help="Polynomial order")
parser.add_argument(
    "--scheduler", "-s", type=str, default="linear", help="Scheduler name"
)
parser.add_argument(
    "--epochs",
    "-e",
    type=int,
    default=5000,
    help="Number of iterations (epochs) training the velocity model",
)
parser.add_argument(
    "--learning-rate", "-lr", type=float, default=0.001, help="Learning Rate"
)
parser.add_argument("--step", type=float, default=0.01, help="Step size in ODE solver")
parser.add_argument(
    "--noise",
    "-nz",
    type=float,
    default=0.05,
    help="Variance of noisy points in sample",
)
parser.add_argument(
    "--n-samples",
    "-ns",
    type=int,
    default=100,
    help="Number of moons' dataset points to sample",
)

args = parser.parse_args()
schedulers = dict(
    cosine=CosineScheduler(),
    linear=LinearScheduler(),
    lvp=LinearVarPresScheduler(),
    poly=PolynomialScheduler(args.poly_deg),
)

plt.style.use("ggplot")
num_epochs = args.epochs
model = NeuralNetwork(2, 2)
optim = Adam(get_parameters(model), lr=args.learning_rate)
path = AffinePath(scheduler=schedulers[args.scheduler])
loss_fn = mse
_losses = []

# train
pbar = tqdm(range(num_epochs))
T.training = True
for iter in pbar:
    x, _ = make_moons(n_samples=args.n_samples, noise=args.noise)
    out, dx_t = epoch(x)
    optim.zero_grad()
    loss = loss_fn(out, dx_t)
    loss.backward()
    _losses.append(loss.item())
    if iter % 50 == 0:
        pbar.set_description_str(
            f"Loss: {loss.item():.4e}"  # ; grad:{self.model.layer2.weight.grad.numpy().mean():.3e}"
        )

    optim.step()

# after training, we sample

x = T.randn(100, 2)
preprocess_time = preprocess_time_moons
h_step = args.step
time_grid = T.linspace(0, 1, int(1 / h_step))


solver = RK4(model, preprocess_hook=preprocess_time)
visualize_moons(
    x,
    solver=solver,
    time_grid=time_grid,
    h_step=h_step,
    num_plots=10,
)
