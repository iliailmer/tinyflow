"""Shared helpers for the ODE-solver/schedule research scripts in this directory."""

import matplotlib.pyplot as plt
import numpy as np
from tinygrad.nn.state import safe_load


def detect_time_embed_dim(model_path: str, in_channels: int) -> int:
    """Detect time_embed_dim from saved weights by inspecting enc1.conv.weight shape."""
    state = safe_load(model_path)
    key = "enc1.conv.weight"
    if key in state:
        return int(state[key].shape[1]) - in_channels
    return 64


def schedule_grid(N: int, p: float) -> np.ndarray:
    """Time grid t_k = 1 - (1 - k/N)^p; p=1 is uniform, p>1 clusters steps near t=1."""
    k = np.arange(N + 1) / N
    return 1.0 - (1.0 - k) ** p


def normalize_for_plot(x_np: np.ndarray) -> np.ndarray:
    out = (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-8)
    return np.clip(out, 0, 1)


def make_grid(x_np: np.ndarray, grid: int = 3) -> np.ndarray:
    h, w = x_np.shape[-2:]
    canvas = np.ones((grid * h + (grid - 1), grid * w + (grid - 1)))
    for i in range(grid):
        for j in range(grid):
            idx = i * grid + j
            canvas[i * (h + 1) : i * (h + 1) + h, j * (w + 1) : j * (w + 1) + w] = x_np[idx, 0]
    return canvas


def save_grid_figure(x_np: np.ndarray, title: str, path: str, figsize=(4, 4.2)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(make_grid(normalize_for_plot(x_np)), cmap="gray")
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
