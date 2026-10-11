"""
Unified generation entry point (Hydra configuration).

Loads a trained flow-matching model and integrates the learned velocity field
from noise (t=0) to data (t=1). Supports the 2D moons dataset and image
datasets (MNIST, Fashion MNIST, CIFAR-10).

Note: the scheduler/path used during training is not needed here - the model
has already learned the velocity field.

Usage:
    # Image datasets (default config base: MNIST)
    uv run scripts/generate.py generation.model_path=model_mnist_unet_linear.safetensors
    uv run scripts/generate.py dataset=fashion_mnist generation.model_path=model_fashion_mnist_unet_linear.safetensors
    uv run scripts/generate.py dataset=cifar10 generation.model_path=model_cifar10_unet_linear.safetensors

    # Animated GIF
    uv run scripts/generate.py generation.model_path=model.safetensors --animated

    # Override grid size and steps
    uv run scripts/generate.py generation.model_path=model.safetensors generation.grid_size=3 generation.num_steps=50

    # Moons dataset - switch the base config with --config-name
    uv run scripts/generate.py --config-name=generate_moons_config generation.model_path=model_moons_neural_network_linear.safetensors
    uv run scripts/generate.py --config-name=generate_moons_config generation.model_path=model.safetensors --animated
"""

import os
import sys

import hydra
import matplotlib.pyplot as plt
import numpy as np
from _common import create_solver, detect_time_embed_dim
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tinygrad.helpers import TRAINING
from tinygrad.tensor import Tensor as T
from tqdm import tqdm

from tinyflow.nn import NeuralNetwork, UNetTinygrad
from tinyflow.solver import uniform_time_grid
from tinyflow.trainer import BaseTrainer
from tinyflow.utils import (
    preprocess_time_cifar,
    preprocess_time_mnist,
    preprocess_time_moons,
    solve_with_snapshots,
)

plt.style.use("ggplot")


def create_model(cfg: DictConfig):
    """Create model from config, sized to match the checkpoint being loaded."""
    model_type = cfg.model.type
    if model_type == "neural_network":
        return NeuralNetwork(
            in_dim=cfg.model.input_dim,
            time_embed_dim=cfg.model.time_embed_dim,
            out_dim=cfg.model.output_dim,
        )

    dataset_type = cfg.dataset.get("type", cfg.dataset.name)
    model_path = cfg.generation.model_path
    if model_type == "unet":
        if "mnist" in dataset_type:
            time_embed_dim = detect_time_embed_dim(model_path, in_channels=1)
            return UNetTinygrad(time_embed_dim=time_embed_dim)
        if "cifar" in dataset_type:
            time_embed_dim = detect_time_embed_dim(model_path, in_channels=3)
            return UNetTinygrad(3, 3, time_embed_dim=time_embed_dim)
    raise ValueError(f"Unknown model type: {model_type}")


# --------------------------------------------------------------------------
# Image datasets (MNIST, Fashion MNIST, CIFAR-10)
# --------------------------------------------------------------------------


def get_dataset_config(cfg: DictConfig):
    """Get dataset configuration."""
    dataset_type = cfg.dataset.get("type", cfg.dataset.name)

    if dataset_type in ["mnist", "fashion_mnist"]:
        return {
            "shape": (1, 28, 28),
            "preprocess_hook": preprocess_time_mnist,
            "is_color": False,
            "cmap": "gray",
        }
    elif dataset_type == "cifar10":
        return {
            "shape": (3, 32, 32),
            "preprocess_hook": preprocess_time_cifar,
            "is_color": True,
            "cmap": None,
        }
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def generate_image_grid(cfg: DictConfig, model, solver, dataset_config):
    """Generate a static grid of samples."""
    grid_size = cfg.generation.grid_size
    num_steps = cfg.generation.num_steps
    dataset_name = cfg.dataset.get("type", cfg.dataset.name)

    print(f"Generating {grid_size}x{grid_size} grid ({grid_size * grid_size} samples)...")

    # Generate samples
    TRAINING.value = 0
    batch_size = grid_size * grid_size
    shape = (batch_size,) + dataset_config["shape"]
    x = solver.solve(T.randn(*shape), uniform_time_grid(num_steps))

    # Convert to numpy and normalize
    x_np = x.numpy()
    x_np = (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-8)
    x_np = np.clip(x_np, 0, 1)

    # Plot grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    if grid_size == 1:
        axes = np.array([[axes]])
    elif len(axes.shape) == 1:
        axes = axes.reshape(-1, 1)

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            ax = axes[i, j]

            if dataset_config["is_color"]:
                img = x_np[idx].transpose(1, 2, 0)
                ax.imshow(img)
            else:
                img = x_np[idx, 0]
                ax.imshow(img, cmap=dataset_config["cmap"])

            ax.axis("off")

    plt.tight_layout()

    # Save
    os.makedirs(cfg.generation.output_dir, exist_ok=True)
    output_file = os.path.join(
        cfg.generation.output_dir, f"generated_{dataset_name}_{grid_size}x{grid_size}.png"
    )
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_file}")

    if cfg.generation.get("show_plot", False):
        plt.show()
    else:
        plt.close()


def generate_image_animation(cfg: DictConfig, model, solver, dataset_config):
    """Generate an animated GIF of the generation process."""
    grid_size = cfg.generation.grid_size
    num_steps = cfg.generation.num_steps
    num_frames = cfg.generation.num_frames
    fps = cfg.generation.fps
    dataset_name = cfg.dataset.get("type", cfg.dataset.name)

    print(f"Generating {grid_size}x{grid_size} animated grid with {num_frames} frames...")

    # Generate samples and capture frames
    TRAINING.value = 0
    batch_size = grid_size * grid_size
    shape = (batch_size,) + dataset_config["shape"]
    capture_steps = set(np.linspace(1, num_steps, min(num_frames, num_steps)).round().astype(int))
    frames = []

    def capture_frame(i, t, x_t):
        if i + 1 not in capture_steps:
            return
        x_np = x_t.numpy()
        x_normalized = (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-8)
        x_normalized = np.clip(x_normalized, 0, 1)

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
        fig.suptitle(f"t = {t:.2f}", fontsize=16, y=0.98)

        if grid_size == 1:
            axes = np.array([[axes]])
        elif len(axes.shape) == 1:
            axes = axes.reshape(-1, 1)

        for row in range(grid_size):
            for col in range(grid_size):
                idx = row * grid_size + col
                ax = axes[row, col]

                if dataset_config["is_color"]:
                    ax.imshow(x_normalized[idx].transpose(1, 2, 0))
                else:
                    ax.imshow(x_normalized[idx, 0], cmap=dataset_config["cmap"])

                ax.axis("off")

        plt.tight_layout()

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(Image.fromarray(frame))
        plt.close(fig)

    solver.solve(T.randn(*shape), uniform_time_grid(num_steps), on_step=capture_frame)

    # Save as GIF
    os.makedirs(cfg.generation.output_dir, exist_ok=True)
    output_file = os.path.join(
        cfg.generation.output_dir,
        f"generation_{dataset_name}_{grid_size}x{grid_size}.gif",
    )

    # Add final frame multiple times to pause at the end
    frames.extend([frames[-1]] * fps)

    frames[0].save(
        output_file,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // fps,
        loop=0,
    )

    print(f"Animation saved to {output_file}")
    print(f"  Frames: {len(frames)}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {len(frames) / fps:.1f} seconds")


def run_image_generation(cfg: DictConfig, animated: bool):
    """Load an image-dataset model and generate a grid or animation."""
    model_path = cfg.generation.model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"\nLoading model from {model_path}...")
    model = create_model(cfg)
    BaseTrainer.load_model(model, model_path)

    # Note: Scheduler/path not needed at generation time - the model has already
    # learned the velocity field during training
    dataset_config = get_dataset_config(cfg)
    solver = create_solver(cfg, model, dataset_config["preprocess_hook"])

    if animated:
        generate_image_animation(cfg, model, solver, dataset_config)
    else:
        generate_image_grid(cfg, model, solver, dataset_config)


# --------------------------------------------------------------------------
# Moons (2D toy dataset)
# --------------------------------------------------------------------------


def generate_moons_grid(cfg: DictConfig, solver):
    """Generate static visualization with multiple time snapshots."""
    n_samples = cfg.generation.n_samples
    num_steps = cfg.generation.num_steps
    num_plots = cfg.generation.num_plots

    print(f"Generating {n_samples} samples with {num_plots} snapshots...")

    # Initialize samples
    TRAINING.value = 0
    x = T.randn(n_samples, 2)
    snapshots = solve_with_snapshots(x, solver, uniform_time_grid(num_steps), num_plots)

    _, ax = plt.subplots(
        1, len(snapshots), figsize=(30, 4), sharex=True, sharey=True, squeeze=False
    )
    for i, (t, x_np) in enumerate(snapshots):
        ax[0, i].scatter(x_np[:, 0], x_np[:, 1], s=20, alpha=0.6)
        ax[0, i].set_title(f"t={t:.2f}")
        ax[0, i].set_xlim(-3, 3)
        ax[0, i].set_ylim(-3, 3)
        ax[0, i].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    os.makedirs(cfg.generation.output_dir, exist_ok=True)
    output_file = os.path.join(cfg.generation.output_dir, "moons_generation.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_file}")

    if cfg.generation.get("show_plot", False):
        plt.show()
    else:
        plt.close()


def generate_moons_animation(cfg: DictConfig, solver):
    """Generate an animated GIF of the generation process."""
    n_samples = cfg.generation.n_samples
    num_steps = cfg.generation.num_steps
    num_plots = cfg.generation.num_plots
    fps = cfg.generation.fps

    print(f"Generating {n_samples} samples with {num_plots} frames...")

    # Initialize samples
    TRAINING.value = 0
    x = T.randn(n_samples, 2)
    snapshots = solve_with_snapshots(x, solver, uniform_time_grid(num_steps), num_plots)

    # Create frames
    frames = []
    for t, x_np in tqdm(snapshots, desc="Creating frames"):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(x_np[:, 0], x_np[:, 1], s=20, alpha=0.6)
        ax.set_title(f"t = {t:.2f}", fontsize=16)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # Convert plot to image
        fig.canvas.draw()
        # Use buffer_rgba() for compatibility across backends
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)
        # Convert RGBA to RGB
        frame = frame[:, :, :3]
        frames.append(Image.fromarray(frame))
        plt.close(fig)

    # Save as GIF
    os.makedirs(cfg.generation.output_dir, exist_ok=True)
    output_file = os.path.join(cfg.generation.output_dir, "moons_generation.gif")

    # Add final frame multiple times to pause at the end
    frames.extend([frames[-1]] * fps)

    frames[0].save(
        output_file,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // fps,
        loop=0,
    )

    print(f"Animation saved to {output_file}")
    print(f"  Frames: {len(frames)}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {len(frames) / fps:.1f} seconds")


def run_moons_generation(cfg: DictConfig, animated: bool):
    """Load a moons model and generate a static plot or animation."""
    model_path = cfg.generation.model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"\nLoading model from {model_path}...")
    model = create_model(cfg)
    BaseTrainer.load_model(model, model_path)

    # Note: Scheduler/path not needed at generation time - the model has already
    # learned the velocity field during training
    solver = create_solver(cfg, model, preprocess_time_moons)

    if animated:
        generate_moons_animation(cfg, solver)
    else:
        generate_moons_grid(cfg, solver)


def main_impl(cfg: DictConfig):
    """Resolve config, then dispatch to the moons or image generation path."""
    animated = cfg.generation.get("animated", False)

    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print(f"\nAnimated: {animated}")

    # Set random seed
    if cfg.get("seed"):
        T.manual_seed(cfg.seed)

    dataset_type = cfg.dataset.get("type", cfg.dataset.name)
    if dataset_type == "moons":
        run_moons_generation(cfg, animated)
    else:
        run_image_generation(cfg, animated)


@hydra.main(version_base=None, config_path="../configs", config_name="generate_config")
def main(cfg: DictConfig):
    """Hydra entry point. Use --config-name=generate_moons_config for the moons dataset."""
    main_impl(cfg)


if __name__ == "__main__":
    # Check for --animated flag and remove it before Hydra processes args
    animated = "--animated" in sys.argv
    if animated:
        sys.argv.remove("--animated")

    # Pass animated flag through config override
    if animated:
        sys.argv.append("generation.animated=true")

    main()
