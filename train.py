import argparse

import matplotlib.pyplot as plt
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor as T

from tinyflow.losses import mse
from tinyflow.nn import (
    NeuralNetwork,
    NeuralNetworkMNIST,
    UNetTinygrad,
)
from tinyflow.path import AffinePath
from tinyflow.path.scheduler import (
    CosineScheduler,
    LinearScheduler,
    LinearVarPresScheduler,
    PolynomialScheduler,
)
from tinyflow.solver import RK4
from tinyflow.trainer import CIFARTrainer, MNISTTrainer, MoonsTrainer, normalize_minmax
from tinyflow.utils import (
    visualize,
    preprocess_time_cifar,
    preprocess_time_mnist,
    preprocess_time_moons,
)

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
parser.add_argument(
    "--dataset",
    "-ds",
    type=str,
    default="moons",
    help="Name of dataset, uses sklearn's Moons by default",
)
parser.add_argument("--step", type=float, default=0.01, help="Step size in ODE solver")
args = parser.parse_args()
schedulers = dict(
    cosine=CosineScheduler(),
    linear=LinearScheduler(),
    lvp=LinearVarPresScheduler(),
    poly=PolynomialScheduler(args.poly_deg),
)
models = dict(
    mnist=NeuralNetworkMNIST(28 * 28, 28 * 28),
    moons=NeuralNetwork(2, 2),
    cifar=UNetTinygrad(),
)

plt.style.use("ggplot")
num_epochs = args.epochs
model = models[args.dataset]
optim = Adam(get_parameters(model), lr=args.learning_rate)
path = AffinePath(scheduler=schedulers[args.scheduler])
trainers = dict(
    moons=MoonsTrainer(
        model=model,
        optim=optim,
        loss_fn=mse,
        path=path,
        num_epochs=num_epochs,
        sampling_args=dict(n_samples=256, noise=0.05),
    ),
    mnist=MNISTTrainer(
        model=model,
        optim=optim,
        loss_fn=mse,
        path=path,
        num_epochs=num_epochs,
        sampling_args=dict(n_samples=100),
    ),
    cifar=CIFARTrainer(
        model=model,
        optim=optim,
        loss_fn=mse,
        path=path,
        num_epochs=num_epochs,
        sampling_args=dict(n_samples=100),
    ),
)
trainer = trainers.get(args.dataset)
if trainer is not None:
    model = trainer.train()
else:
    raise ValueError(f"Trainer {args.dataset} is unknown")


# after training, we sample


if args.dataset == "mnist":
    x = T.randn(1, 28 * 28)
    preprocess_time = preprocess_time_mnist
elif args.dataset == "cifar":
    x = normalize_minmax(T.randn(1, 3, 32, 32))
    preprocess_time = preprocess_time_cifar
else:
    x = T.randn(100, 2)
    preprocess_time = preprocess_time_moons
h_step = args.step
time_grid = T.linspace(0, 1, int(1 / h_step))


solver = RK4(model, preprocess_hook=preprocess_time)
visualize(
    x,
    solver=solver,
    dataset=args.dataset,
    time_grid=time_grid,
    h_step=h_step,
    num_plots=10,
)
