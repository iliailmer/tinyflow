import matplotlib.pyplot as plt
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor as T

from tinyflow.losses import mse
from tinyflow.nn import NeuralNetwork
from tinyflow.solver import RK4, MidpointSolver
from tinyflow.trainer import MoonsTrainer

plt.style.use("ggplot")
num_iter = 10000
model = NeuralNetwork(2, 2)
optim = Adam(get_parameters(model), lr=0.01)

trainer = MoonsTrainer(
    model=model,
    optim=optim,
    loss_fn=mse,
    num_epochs=num_iter,
    sampling_args=dict(n_samples=256, noise=0.05),
)
model = trainer.train()
# after training, we sample
x = T.randn(300, 2)
h_step = float(1 / 50)
time_grid = T.linspace(0, 1, int(1 / h_step))
i = 0
solver = RK4(model)

num_plots = 10
fig, ax = plt.subplots(1, num_plots, figsize=(30, 4), sharex=True, sharey=True)
sample_every = time_grid.shape[0] // num_plots
for idx in range(int(time_grid.shape[0])):
    t = time_grid[idx]
    if idx % sample_every == 0:
        ax[i].scatter(x.numpy()[:, 0], x.numpy()[:, 1], s=5)
        ax[i].set_title(f"Time: t={t.numpy():.2f}")
        i += 1
    x = solver.sample(h_step, t, x)
plt.tight_layout()
plt.show()
