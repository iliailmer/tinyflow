"""
Generate an animated GIF showing the flow matching generation process.

Usage:
    uv run examples/generate_animation.py --model model.safetensors --dataset mnist --grid-size 3
    uv run examples/generate_animation.py --model model.safetensors --dataset fashion_mnist --grid-size 4 --fps 10
    uv run examples/generate_animation.py --model model.safetensors --dataset mnist --per-frame-norm  # Show noise structure
    uv run examples/generate_animation.py --model model.safetensors --dataset mnist --show-distribution  # Show distribution evolution
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
    per_frame_norm: bool = False,
    show_distribution: bool = False,
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

    # Compute normalization parameters
    if per_frame_norm:
        print("Using per-frame normalization (shows noise structure)")
        global_min = global_max = None
    else:
        print("Using global normalization (consistent brightness)")
        global_min = min(s.min() for _, s in snapshots)
        global_max = max(s.max() for _, s in snapshots)

    # Store raw snapshots for distribution visualization
    if show_distribution:
        initial_raw = snapshots[0][1]  # Raw initial noise
        final_raw = snapshots[-1][1]  # Raw final images

    # Render frames
    frames = []
    for step_idx, x_np in snapshots:
        if per_frame_norm:
            # Normalize each frame independently
            frame_min = x_np.min()
            frame_max = x_np.max()
            x_normalized = (x_np - frame_min) / (frame_max - frame_min + 1e-8)
        else:
            # Normalize all frames to same global scale
            x_normalized = (x_np - global_min) / (global_max - global_min + 1e-8)

        x_normalized = np.clip(x_normalized, 0, 1)

        # Create figure with optional distribution panel
        if show_distribution:
            fig = plt.figure(figsize=(12, 8))
            # Image grid takes left 2/3
            gs = fig.add_gridspec(grid_size, grid_size + 1, width_ratios=[1] * grid_size + [0.5])
            axes = [[fig.add_subplot(gs[i, j]) for j in range(grid_size)] for i in range(grid_size)]
            # Distribution plot takes right 1/3
            ax_dist = fig.add_subplot(gs[:, -1])
        else:
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
            if grid_size == 1:
                axes = [[axes]]

        fig.suptitle(f"t = {step_idx / num_steps:.2f}", fontsize=16, y=0.98)

        # Render image grid
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                ax = axes[i][j]

                if is_color:
                    img = x_normalized[idx].transpose(1, 2, 0)
                    ax.imshow(img)
                else:
                    img = x_normalized[idx, 0]
                    ax.imshow(img, cmap="gray")

                ax.axis("off")

        # Add distribution visualization
        if show_distribution:
            ax_dist.clear()

            # Plot histograms of raw pixel values (before normalization)
            initial_flat = initial_raw.flatten()
            final_flat = final_raw.flatten()
            current_flat = x_np.flatten()

            # Compute common bins for consistent visualization
            all_values = np.concatenate([initial_flat, final_flat, current_flat])
            bins = np.linspace(all_values.min(), all_values.max(), 50)

            # Plot reference distributions with transparency
            ax_dist.hist(initial_flat, bins=bins, alpha=0.3, color='blue',
                        label=f't=0.00 (noise)', density=True)
            ax_dist.hist(final_flat, bins=bins, alpha=0.3, color='green',
                        label=f't=1.00 (data)', density=True)

            # Plot current distribution prominently
            ax_dist.hist(current_flat, bins=bins, alpha=0.7, color='red',
                        label=f't={step_idx / num_steps:.2f}', density=True)

            ax_dist.set_xlabel('Pixel Value')
            ax_dist.set_ylabel('Density')
            ax_dist.set_title('Distribution Evolution')
            ax_dist.legend(fontsize=8)
            ax_dist.grid(True, alpha=0.3)

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
    parser.add_argument(
        "--per-frame-norm",
        action="store_true",
        help="Use per-frame normalization to show noise structure (vs global normalization)",
    )
    parser.add_argument(
        "--show-distribution",
        action="store_true",
        help="Add histogram showing evolution of pixel value distribution",
    )

    args = parser.parse_args()

    generate_animation(
        model_path=args.model,
        dataset=args.dataset,
        grid_size=args.grid_size,
        num_steps=args.num_steps,
        num_frames=args.num_frames,
        fps=args.fps,
        seed=args.seed,
        per_frame_norm=args.per_frame_norm,
        show_distribution=args.show_distribution,
    )


if __name__ == "__main__":
    main()
