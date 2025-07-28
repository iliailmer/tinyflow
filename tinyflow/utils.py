import pickle

from loguru import logger
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from tinyflow.nn import Tensor


@logger.catch
def visualize_moons(x, solver, time_grid, h_step, num_plots=10):
    i = 0
    _, ax = plt.subplots(1, num_plots, figsize=(30, 4), sharex=True, sharey=True)
    sample_every = time_grid.shape[0] // num_plots
    for idx in tqdm(range(int(time_grid.shape[0]))):
        t = time_grid[idx]
        if (idx + 1) % sample_every == 0:
            ax[i].scatter(x.numpy()[:, 0], x.numpy()[:, 1], s=5)
            ax[i].set_title(f"Time: t={t.numpy():.2f}")
            i += 1
        x = solver.sample(h_step, t, x)
    plt.tight_layout()
    plt.show()


@logger.catch
def visualize_mnist(x, solver, time_grid, h_step, num_plots=10):
    i = 0
    _, ax = plt.subplots(1, num_plots, figsize=(30, 4), sharex=True, sharey=True)
    sample_every = time_grid.shape[0] // num_plots
    for idx in tqdm(range(int(time_grid.shape[0]))):
        t = time_grid[idx]
        if (idx + 1) % sample_every == 0:
            x = (x - x.min()) / (x.max() - x.min())
            ax[i].imshow(x.numpy()[0, :].reshape((28, 28)), cmap="gray")
            ax[i].axis("off")
            i += 1
        x = solver.sample(h_step, t, x)
    plt.tight_layout()
    plt.show()


def unpickle(file):
    with open(file, "rb") as fo:
        dct = pickle.load(fo, encoding="bytes")
    return dct


def preprocess_time_moons(t: Tensor, rhs_prev: Tensor):
    t = t.reshape((1, 1))
    return t.repeat(rhs_prev.shape[0], 1)


def preprocess_time_mnist(t: Tensor, rhs_prev: Tensor):
    t = t.reshape((1, 1))
    return t.repeat(rhs_prev.shape[0], 1)


def preprocess_time_cifar(t: Tensor, rhs_prev: Tensor):
    t = t.reshape((1, 1, 1, 1)).expand((1, 1, 32, 32))
    return t.repeat(rhs_prev.shape[0], 1, 1, 1)
