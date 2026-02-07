"""
Generate an animated GIF showing the flow matching generation process.

Usage:
    uv run examples/generate_animation.py --model model.safetensors --dataset mnist --grid-size 3
    uv run examples/generate_animation.py --model model.safetensors --dataset fashion_mnist --grid-size 4 --fps 10
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tinygrad import TinyJit
from tinygrad.tensor import Tensor as T
from tqdm import tqdm

from tinyflow.nn import UNetTinygrad
from tinyflow.solver import RK4
from tinyflow.utils import preprocess_time_cifar, preprocess_time_mnist


def generate_animation(
    model_path: str,
    dataset: str = "mnist",
    grid_size: int = 3,
    num_steps: int = 100,
    num_frames: int = 50,
    fps: int = 10,
    seed: int = 42,
):
    """Generate an animated GIF of the generation process."""

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

    # Generate samples and capture frames
    print(f"Generating {grid_size}×{grid_size} samples with {num_frames} frames...")
    T.training = False

    x = T.randn(*shape)
    h_step = 1.0 / num_steps

    # Determine which steps to capture
    capture_steps = np.linspace(0, num_steps - 1, num_frames, dtype=int)
    snapshots = []  # (step_index, numpy_array) pairs

    # Capture the initial noise as the first frame
    snapshots.append((0, x.numpy().copy()))

    # JIT compile the solver step for better performance
    @TinyJit
    def jit_step(h, t, x):
        return solver.sample(h, t, x)

    # Solve ODE and capture intermediate states
    for step in tqdm(range(num_steps), desc="Generating"):
        t = (T.zeros(1) + step * h_step).contiguous()
        x = jit_step(h_step, t, x)

        if step in capture_steps:
            # Ensure tensor is realized before capturing to avoid memory accumulation
            x.realize()
            snapshots.append((step + 1, x.numpy().copy()))

    # Compute global min/max across all snapshots for consistent normalization
    global_min = min(s.min() for _, s in snapshots)
    global_max = max(s.max() for _, s in snapshots)

    # Render frames with consistent normalization
    frames = []
    for step_idx, x_np in snapshots:
        x_normalized = (x_np - global_min) / (global_max - global_min + 1e-8)
        x_normalized = np.clip(x_normalized, 0, 1)

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
        fig.suptitle(f"t = {step_idx / num_steps:.2f}", fontsize=16, y=0.98)

        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                ax = axes[i, j] if grid_size > 1 else axes

                if is_color:
                    img = x_normalized[idx].transpose(1, 2, 0)
                    ax.imshow(img)
                else:
                    img = x_normalized[idx, 0]
                    ax.imshow(img, cmap="gray")

                ax.axis("off")

        plt.tight_layout()

        fig.canvas.draw()
        data_rgba = np.asarray(fig.canvas.buffer_rgba())
        frame = data_rgba[..., :3]
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(Image.fromarray(frame))
        plt.close(fig)

    # Save as GIF
    output_file = f"generation_{dataset}_{grid_size}x{grid_size}.gif"
    print(f"Saving animation to {output_file}...")

    # Add final frame multiple times to pause at the end
    frames.extend([frames[-1]] * fps)

    frames[0].save(
        output_file,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // fps,  # milliseconds per frame
        loop=0,
    )

    print(f"✓ Animation saved to {output_file}")
    print(f"  Frames: {len(frames)}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {len(frames) / fps:.1f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Generate animation of flow matching generation")
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
    parser.add_argument("--grid-size", type=int, default=3, help="Grid size (n×n)")
    parser.add_argument("--num-steps", type=int, default=100, help="Number of ODE steps")
    parser.add_argument("--num-frames", type=int, default=50, help="Number of frames in animation")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    generate_animation(
        model_path=args.model,
        dataset=args.dataset,
        grid_size=args.grid_size,
        num_steps=args.num_steps,
        num_frames=args.num_frames,
        fps=args.fps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
