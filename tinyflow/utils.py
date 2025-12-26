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
        # Update x first, then visualize
        x = solver.sample(h_step, t, x)

        if (idx + 1) % sample_every == 0:
            ax[i].scatter(x.numpy()[:, 0], x.numpy()[:, 1], s=5)
            ax[i].set_title(f"Time: t={t.numpy():.2f}")
            i += 1
    plt.tight_layout()
    plt.show()


@logger.catch
def visualize_mnist(x, solver, time_grid, h_step, num_plots=10):
    i = 0
    _, ax = plt.subplots(1, num_plots, figsize=(30, 4), sharex=True, sharey=True)
    sample_every = time_grid.shape[0] // num_plots
    for idx in tqdm(range(int(time_grid.shape[0]))):
        t = time_grid[idx]
        # Update x first, then visualize
        x = solver.sample(h_step, t, x)

        # Only compute normalization when actually visualizing
        if (idx + 1) % sample_every == 0:
            x_normalized = (x - x.min()) / (x.max() - x.min())
            ax[i].imshow(x_normalized.numpy()[0, :].reshape((28, 28)), cmap="gray")
            ax[i].axis("off")
            i += 1
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


@logger.catch
def visualize_cifar10(x, solver, time_grid, h_step, num_plots=10):
    """
    Visualize CIFAR-10 generation process.

    Args:
        x: Initial noise tensor of shape (batch_size, 3, 32, 32)
        solver: ODE solver
        time_grid: Time steps
        h_step: Step size
        num_plots: Number of intermediate visualizations
    """
    import numpy as np

    i = 0
    _, ax = plt.subplots(1, num_plots, figsize=(30, 4), sharex=True, sharey=True)
    sample_every = time_grid.shape[0] // num_plots

    for idx in tqdm(range(int(time_grid.shape[0]))):
        t = time_grid[idx]
        # Update x first
        x = solver.sample(h_step, t, x)

        # Visualize at intervals
        if (idx + 1) % sample_every == 0:
            # Convert from (batch, C, H, W) to (H, W, C) for plotting
            img = x.numpy()[0, :].transpose(1, 2, 0)  # (3, 32, 32) -> (32, 32, 3)

            # Normalize to [0, 1] for display
            img_normalized = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img_normalized = np.clip(img_normalized, 0, 1)

            ax[i].imshow(img_normalized)
            ax[i].axis("off")
            ax[i].set_title(f"t={t.numpy():.2f}")
            i += 1

    plt.tight_layout()
    plt.savefig("outputs/cifar10_generation.png", dpi=150, bbox_inches="tight")
    plt.show()
