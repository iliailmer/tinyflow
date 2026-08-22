"""Shared helpers for the ODE-solver/schedule research scripts in this directory."""

import matplotlib.pyplot as plt
import numpy as np
from tinygrad.nn.state import safe_load
from tinygrad.tensor import Tensor as T


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


def per_sample_norm_mean(x: T) -> float:
    """Mean over the batch of the per-sample L2 norm."""
    flat = x.reshape(x.shape[0], -1)
    return ((flat * flat).sum(axis=-1) + 1e-20).sqrt().mean().numpy().item()


def normalize_per_sample(x: T) -> T:
    """Normalizes each sample in the batch to unit L2 norm."""
    flat = x.reshape(x.shape[0], -1)
    mag = ((flat * flat).sum(axis=-1) + 1e-20).sqrt()
    view_shape = (x.shape[0],) + (1,) * (len(x.shape) - 1)
    return x / mag.reshape(view_shape)


def compute_fid_vs_real(
    generated_np: np.ndarray, dataset_name: str, n_samples: int, weights_dir: str = "weights"
) -> float | None:
    """FID of generated images against real images freshly sampled from `dataset_name`."""
    from tinyflow.dataloader import CIFAR10Loader, FashionMNISTLoader, MNISTLoader
    from tinyflow.metrics import calculate_fid, get_feature_extractor

    classifier = get_feature_extractor(dataset_name, weights_dir=weights_dir)
    if not classifier._weights_loaded:
        print(f"Warning: FID classifier weights not found for {dataset_name}, skipping FID.")
        return None

    if dataset_name == "mnist":
        dataloader = MNISTLoader(
            path="dataset/mnist/trainset/trainingSet/*/*.jpg",
            batch_size=64,
            shuffle=True,
            flatten=False,
        )
    elif dataset_name == "fashion_mnist":
        dataloader = FashionMNISTLoader(
            path="dataset/fashion_mnist", batch_size=64, shuffle=True, flatten=False, train=True
        )
    elif dataset_name == "cifar10":
        dataloader = CIFAR10Loader(
            path="dataset/cifar10/cifar-10-batches-py", batch_size=64, shuffle=True, train=True
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    real_images = []
    collected = 0
    for batch_images, _ in dataloader:
        real_images.append(batch_images)
        collected += len(batch_images)
        if collected >= n_samples:
            break
    real_images = np.concatenate(real_images, axis=0)[:n_samples]

    return calculate_fid(real_images, generated_np, feature_extractor=classifier, batch_size=64)
