"""
Generate samples from a trained flow matching model for image datasets (MNIST, Fashion MNIST, CIFAR-10).

Usage:
    # Generate static grid for MNIST
    uv run examples/generate_images.py generation.model_path=model_mnist_unet_linear.safetensors

    # Generate for Fashion MNIST
    uv run examples/generate_images.py dataset=fashion_mnist generation.model_path=model_fashion_mnist_unet_linear.safetensors

    # Generate for CIFAR-10
    uv run examples/generate_images.py dataset=cifar10 generation.model_path=model_cifar10_unet_linear.safetensors

    # Generate with animation
    uv run examples/generate_images.py generation.model_path=model_mnist_unet_linear.safetensors --animated

    # Override grid size and steps
    uv run examples/generate_images.py generation.model_path=model.safetensors generation.grid_size=3 generation.num_steps=50
"""

import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tinygrad import TinyJit
from tinygrad.tensor import Tensor as T
from tqdm import tqdm

from tinyflow.nn import UNetTinygrad
from tinyflow.solver import Heun
from tinyflow.trainer import BaseTrainer
from tinyflow.utils import preprocess_time_cifar, preprocess_time_mnist

plt.style.use("ggplot")


def create_model(cfg: DictConfig):
    """Create model from config."""
    model_type = cfg.model.type
    dataset_type = cfg.dataset.get("type", cfg.dataset.name)

    if model_type == "unet":
        if "mnist" in dataset_type:
            return UNetTinygrad()
        if "cifar" in dataset_type:
            return UNetTinygrad(3, 3)
    raise ValueError(f"Unknown model type: {model_type}")


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


def generate_static_grid(cfg: DictConfig, model, solver, dataset_config):
    """Generate a static grid of samples."""
    grid_size = cfg.generation.grid_size
    num_steps = cfg.generation.num_steps
    dataset_name = cfg.dataset.get("type", cfg.dataset.name)

    print(f"Generating {grid_size}x{grid_size} grid ({grid_size * grid_size} samples)...")

    # Generate samples
    T.training = False
    batch_size = grid_size * grid_size
    shape = (batch_size,) + dataset_config["shape"]
    x = T.randn(*shape)
    h_step = 1.0 / num_steps

    # JIT compile the solver step for better performance
    @TinyJit
    def jit_step(h, t, x):
        return solver.sample(h, t, x)

    # Solve ODE from t=0 to t=1
    for step in tqdm(range(num_steps), desc="Generating"):
        t = (T.zeros(1) + step * h_step).contiguous()
        x = jit_step(h_step, t, x)

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


def generate_animation(cfg: DictConfig, model, solver, dataset_config):
    """Generate an animated GIF of the generation process."""
    grid_size = cfg.generation.grid_size
    num_steps = cfg.generation.num_steps
    num_frames = cfg.generation.num_frames
    fps = cfg.generation.fps
    dataset_name = cfg.dataset.get("type", cfg.dataset.name)

    print(f"Generating {grid_size}x{grid_size} animated grid with {num_frames} frames...")

    # Generate samples and capture frames
    T.training = False
    batch_size = grid_size * grid_size
    shape = (batch_size,) + dataset_config["shape"]
    x = T.randn(*shape)
    h_step = 1.0 / num_steps

    # Determine which steps to capture
    capture_steps = np.linspace(0, num_steps - 1, num_frames, dtype=int)
    frames = []

    # JIT compile the solver step for better performance
    @TinyJit
    def jit_step(h, t, x):
        return solver.sample(h, t, x)

    # Solve ODE and capture frames
    for step in tqdm(range(num_steps), desc="Generating animation"):
        t = (T.zeros(1) + step * h_step).contiguous()
        x = jit_step(h_step, t, x)

        # Capture frame at specified steps
        if step in capture_steps:
            x_np = x.numpy()
            x_normalized = (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-8)
            x_normalized = np.clip(x_normalized, 0, 1)

            # Create grid image
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
            fig.suptitle(f"t = {step / num_steps:.2f}", fontsize=16, y=0.98)

            if grid_size == 1:
                axes = np.array([[axes]])
            elif len(axes.shape) == 1:
                axes = axes.reshape(-1, 1)

            for i in range(grid_size):
                for j in range(grid_size):
                    idx = i * grid_size + j
                    ax = axes[i, j]

                    if dataset_config["is_color"]:
                        img = x_normalized[idx].transpose(1, 2, 0)
                        ax.imshow(img)
                    else:
                        img = x_normalized[idx, 0]
                        ax.imshow(img, cmap=dataset_config["cmap"])

                    ax.axis("off")

            plt.tight_layout()

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


def main_impl(cfg: DictConfig):
    """Main generation function."""
    animated = cfg.generation.get("animated", False)

    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print(f"\nAnimated: {animated}")

    # Set random seed
    if cfg.get("seed"):
        T.manual_seed(cfg.seed)

    # Load model
    model_path = cfg.generation.model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"\nLoading model from {model_path}...")
    model = create_model(cfg)
    BaseTrainer.load_model(model, model_path)

    # Get dataset configuration
    # Note: Scheduler/path not needed at generation time - the model has already
    # learned the velocity field during training
    dataset_config = get_dataset_config(cfg)

    # Create solver
    solver = Heun(model, preprocess_hook=dataset_config["preprocess_hook"])

    # Generate samples
    if animated:
        generate_animation(cfg, model, solver, dataset_config)
    else:
        generate_static_grid(cfg, model, solver, dataset_config)


@hydra.main(version_base=None, config_path="../configs", config_name="generate_config")
def main(cfg: DictConfig):
    """Hydra entry point."""
    main_impl(cfg)


if __name__ == "__main__":
    import sys

    # Check for --animated flag and remove it before Hydra processes args
    animated = "--animated" in sys.argv
    if animated:
        sys.argv.remove("--animated")

    # Pass animated flag through config override
    if animated:
        sys.argv.append("generation.animated=true")

    main()
