"""
Generate samples from a trained flow matching model.

Usage:
    uv run examples/generate_samples.py --model model.safetensors --dataset mnist --grid-size 5
    uv run examples/generate_samples.py --model model.safetensors --dataset fashion_mnist --grid-size 4
    uv run examples/generate_samples.py --model model.safetensors --dataset cifar10 --grid-size 3
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from tinygrad.tensor import Tensor as T

from tinyflow.nn import UNetTinygrad
from tinyflow.solver import RK4
from tinyflow.utils import preprocess_time_cifar, preprocess_time_mnist


def generate_grid(
    model_path: str,
    dataset: str = "mnist",
    grid_size: int = 5,
    num_steps: int = 100,
    seed: int = 42,
):
    """Generate a grid of samples from a trained model."""

    # Set seed
    T.manual_seed(seed)

    # Load model
    print(f"Loading model from {model_path}...")
    model = UNetTinygrad()
    from tinyflow.trainer import BaseTrainer

    BaseTrainer.load_model(model, model_path)

    # Configure for dataset
    if dataset in ["mnist", "fashion_mnist"]:
        shape = (grid_size * grid_size, 1, 28, 28)
        preprocess_hook = preprocess_time_mnist
        is_color = False
    elif dataset == "cifar10":
        shape = (grid_size * grid_size, 3, 32, 32)
        preprocess_hook = preprocess_time_cifar
        is_color = True
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Create solver
    solver = RK4(model, preprocess_hook=preprocess_hook)

    # Generate samples
    print(f"Generating {grid_size}×{grid_size} = {grid_size * grid_size} samples...")
    T.training = False

    x = T.randn(*shape)
    h_step = 1.0 / num_steps

    # Solve ODE from t=0 to t=1
    for step in range(num_steps):
        t = T.zeros(1) + step * h_step
        x = solver.sample(h_step, t, x)

    # Convert to numpy and normalize
    x_np = x.numpy()

    # Normalize for display
    x_np = (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-8)
    x_np = np.clip(x_np, 0, 1)

    # Plot grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            ax = axes[i, j] if grid_size > 1 else axes

            if is_color:
                # CIFAR-10: (C, H, W) -> (H, W, C)
                img = x_np[idx].transpose(1, 2, 0)
                ax.imshow(img)
            else:
                # MNIST/Fashion MNIST: (1, H, W) -> (H, W)
                img = x_np[idx, 0]
                ax.imshow(img, cmap="gray")

            ax.axis("off")

    plt.tight_layout()

    # Save
    output_file = f"generated_{dataset}_{grid_size}x{grid_size}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Saved to {output_file}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Generate samples from trained flow matching model"
    )
    parser.add_argument(
        "--model", type=str, default="model.safetensors", help="Path to model weights"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist", "cifar10"],
        help="Dataset type",
    )
    parser.add_argument("--grid-size", type=int, default=5, help="Grid size (n×n)")
    parser.add_argument("--num-steps", type=int, default=100, help="Number of ODE steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    generate_grid(
        model_path=args.model,
        dataset=args.dataset,
        grid_size=args.grid_size,
        num_steps=args.num_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
