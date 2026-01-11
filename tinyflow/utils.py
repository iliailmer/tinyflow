import pickle

from loguru import logger
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from tinyflow.nn import Tensor


@logger.catch
def visualize_moons(x, solver, time_grid, h_step, num_plots=10):
    """Optimized moons visualization with minimal tensor realizations."""
    _, ax = plt.subplots(1, num_plots, figsize=(30, 4), sharex=True, sharey=True)
    sample_every = time_grid.shape[0] // num_plots

    snapshots = []
    snapshot_times = []

    for idx in tqdm(range(int(time_grid.shape[0])), desc="Generating samples"):
        t = time_grid[idx]
        x = solver.sample(h_step, t, x)

        if (idx + 1) % sample_every == 0:
            snapshots.append(x)
            snapshot_times.append(t)

    for i, (snapshot, t) in enumerate(zip(snapshots, snapshot_times, strict=False)):
        x_np = snapshot.numpy()
        ax[i].scatter(x_np[:, 0], x_np[:, 1], s=20, alpha=0.6)  # Larger points, add transparency
        ax[i].set_title(f"t={t.numpy():.2f}")
        ax[i].set_xlim(-3, 3)  # Set consistent limits
        ax[i].set_ylim(-3, 3)
        ax[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("moons_generation.png", dpi=150, bbox_inches="tight")
    print("✓ Saved moons_generation.png")
    plt.show()


@logger.catch
def visualize_mnist(x, solver, time_grid, h_step, num_plots=10):
    """Optimized MNIST visualization with minimal tensor realizations."""
    _, ax = plt.subplots(1, num_plots, figsize=(30, 4), sharex=True, sharey=True)
    sample_every = time_grid.shape[0] // num_plots

    snapshots = []
    snapshot_times = []

    for idx in tqdm(range(int(time_grid.shape[0])), desc="Generating samples"):
        t = time_grid[idx]
        x = solver.sample(h_step, t, x)

        # Store reference for visualization (still on GPU)
        if (idx + 1) % sample_every == 0:
            snapshots.append(x)
            snapshot_times.append(t)

    for i, (snapshot, t) in enumerate(zip(snapshots, snapshot_times, strict=False)):
        x_np = snapshot.numpy()[0, :].reshape((28, 28))
        x_normalized = (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-8)
        ax[i].imshow(x_normalized, cmap="gray")
        ax[i].axis("off")
        ax[i].set_title(f"t={t.numpy():.2f}")

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
    Optimized CIFAR-10 visualization with minimal tensor realizations.

    Args:
        x: Initial noise tensor of shape (batch_size, 3, 32, 32)
        solver: ODE solver
        time_grid: Time steps
        h_step: Step size
        num_plots: Number of intermediate visualizations
    """
    import numpy as np

    _, ax = plt.subplots(1, num_plots, figsize=(30, 4), sharex=True, sharey=True)
    sample_every = time_grid.shape[0] // num_plots

    snapshots = []
    snapshot_times = []

    for idx in tqdm(range(int(time_grid.shape[0])), desc="Generating samples"):
        t = time_grid[idx]
        x = solver.sample(h_step, t, x)

        if (idx + 1) % sample_every == 0:
            snapshots.append(x)
            snapshot_times.append(t)

    for i, (snapshot, t) in enumerate(zip(snapshots, snapshot_times, strict=False)):
        img = snapshot.numpy()[0, :].transpose(1, 2, 0)  # (3, 32, 32) -> (32, 32, 3)

        img_normalized = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img_normalized = np.clip(img_normalized, 0, 1)

        ax[i].imshow(img_normalized)
        ax[i].axis("off")
        ax[i].set_title(f"t={t.numpy():.2f}")

    plt.tight_layout()
    plt.savefig("outputs/cifar10_generation.png", dpi=150, bbox_inches="tight")
    plt.show()
