import argparse

import matplotlib.pyplot as plt
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor as T

from tinyflow.dataloader import MNISTLoader
from tinyflow.losses import mse
from tinyflow.nn import NeuralNetworkMNIST
from tinyflow.path import AffinePath
from tinyflow.path.scheduler import (
    CosineScheduler,
    LinearScheduler,
    LinearVarPresScheduler,
    PolynomialScheduler,
)
from tinyflow.solver import RK4
from tinyflow.trainer import MNISTTrainer
from tinyflow.utils import (
    preprocess_time_mnist,
    visualize_mnist,
)

plt.style.use("ggplot")

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
args = parser.parse_args()
schedulers = dict(
    cosine=CosineScheduler(),
    linear=LinearScheduler(),
    lvp=LinearVarPresScheduler(),
    poly=PolynomialScheduler(args.poly_deg),
)
model = NeuralNetworkMNIST(28 * 28, 28 * 28)
num_epochs = args.epochs
optim = Adam(get_parameters(model), lr=args.learning_rate)
path = AffinePath(scheduler=schedulers[args.scheduler])

trainer = MNISTTrainer(
    model=model,
    dataloader=MNISTLoader(),
    optim=optim,
    loss_fn=mse,
    path=path,
    num_epochs=num_epochs,
)

model = trainer.train()
trainer.plot_loss("figures/mnist/")

# after training, we sample

x = T.randn(1, 28 * 28)
preprocess_time = preprocess_time_mnist
h_step = args.step
time_grid = T.linspace(0, 1, int(1 / h_step))

solver = RK4(model, preprocess_hook=preprocess_time)
visualize_mnist(
    x,
    solver=solver,
    time_grid=time_grid,
    h_step=h_step,
    num_plots=10,
)
