import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

from tinyflow.nn import Tensor


def solve_with_snapshots(x, solver, time_grid, num_snapshots: int):
    """
    Integrate over time_grid and keep evenly spaced intermediate states.

    Returns:
        List of (t, state as numpy array). The last entry is the final state.
    """
    num_steps = len(time_grid) - 1
    keep = set(np.linspace(1, num_steps, min(num_snapshots, num_steps)).round().astype(int))
    snapshots = []

    def on_step(i, t, x_t):
        if i + 1 in keep:
            snapshots.append((t, x_t.numpy()))

    solver.solve(x, time_grid, on_step=on_step)
    return snapshots


def _save_snapshot_row(snapshots, draw, save_path: str) -> str:
    _, axes = plt.subplots(
        1, len(snapshots), figsize=(30, 4), sharex=True, sharey=True, squeeze=False
    )
    for ax, (t, x_np) in zip(axes[0], snapshots, strict=True):
        draw(ax, x_np)
        ax.set_title(f"t={t:.2f}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved {save_path}")
    plt.close()
    return save_path


def visualize_moons(x, solver, time_grid, num_plots=10, save_path="outputs/moons_generation.png"):
    def draw(ax, x_np):
        ax.scatter(x_np[:, 0], x_np[:, 1], s=20, alpha=0.6)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(True, alpha=0.3)

    snapshots = solve_with_snapshots(x, solver, time_grid, num_plots)
    return _save_snapshot_row(snapshots, draw, save_path)


def to_display(x_np: np.ndarray, value_range: tuple[float, float]) -> np.ndarray:
    """Map images from the training value range to [0, 1] and clip. No rescaling by content."""
    low, high = value_range
    return np.clip((x_np - low) / (high - low), 0, 1)


def visualize_mnist(
    x,
    solver,
    time_grid,
    num_plots=10,
    save_path="outputs/mnist_generation.png",
    value_range=(0.0, 1.0),
):
    def draw(ax, x_np):
        img = to_display(x_np[0].reshape((28, 28)), value_range)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    snapshots = solve_with_snapshots(x, solver, time_grid, num_plots)
    return _save_snapshot_row(snapshots, draw, save_path)


def visualize_cifar10(
    x,
    solver,
    time_grid,
    num_plots=10,
    save_path="outputs/cifar10_generation.png",
    value_range=(0.0, 1.0),
):
    """Plot the first sample of x of shape (batch_size, 3, 32, 32) at intermediate times."""

    def draw(ax, x_np):
        ax.imshow(to_display(x_np[0].transpose(1, 2, 0), value_range))
        ax.axis("off")

    snapshots = solve_with_snapshots(x, solver, time_grid, num_plots)
    return _save_snapshot_row(snapshots, draw, save_path)


def unpickle(file):
    with open(file, "rb") as fo:
        dct = pickle.load(fo, encoding="bytes")
    return dct


def preprocess_time(t: Tensor, rhs_prev: Tensor):
    """Repeat the scalar time t for each sample in the batch, shape (batch_size, 1)."""
    t = t.reshape((1, 1))
    return t.repeat(rhs_prev.shape[0], 1)
