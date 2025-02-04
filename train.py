import argparse

import matplotlib.pyplot as plt
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor as T

from tinyflow.losses import mse
from tinyflow.nn import NeuralNetwork
from tinyflow.path import AffinePath
from tinyflow.path.scheduler import (
    CosineScheduler,
    LinearScheduler,
    LinearVarPresScheduler,
    PolynomialScheduler,
)
from tinyflow.solver import RK4
from tinyflow.trainer import MoonsTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--poly-deg", "-n", type=int, default=2, help="Polynomial order")
parser.add_argument(
    "--scheduler", "-s", type=str, default="linear", help="Scheduler name"
)
parser.add_argument(
    "--iter",
    type=int,
    default=5000,
    help="Number of iterations training the velocity model",
)
parser.add_argument("--step", type=float, default=0.1, help="Step size in ODE solver")
args = parser.parse_args()
schedulers = dict(
    cosine=CosineScheduler(),
    linear=LinearScheduler(),
    lvp=LinearVarPresScheduler(),
    poly=PolynomialScheduler(args.poly_deg),
)


plt.style.use("ggplot")
num_iter = args.iter
model = NeuralNetwork(2, 2)
optim = Adam(get_parameters(model), lr=0.01)
path = AffinePath(scheduler=schedulers[args.scheduler])
trainer = MoonsTrainer(
    model=model,
    optim=optim,
    loss_fn=mse,
    path=path,
    num_epochs=num_iter,
    sampling_args=dict(n_samples=256, noise=0.05),
)
model = trainer.train()
# after training, we sample
x = T.randn(300, 2)
h_step = args.step
time_grid = T.linspace(0, 1, int(1 / h_step))
i = 0
solver = RK4(model)

num_plots = 10
fig, ax = plt.subplots(1, num_plots, figsize=(30, 4), sharex=True, sharey=True)
sample_every = time_grid.shape[0] // num_plots
for idx in range(int(time_grid.shape[0])):
    t = time_grid[idx]
    if (idx + 1) % sample_every == 0:
        ax[i].scatter(x.numpy()[:, 0], x.numpy()[:, 1], s=5)
        ax[i].set_title(f"Time: t={t.numpy():.2f}")
        i += 1
    x = solver.sample(h_step, t, x)
plt.tight_layout()
plt.show()
