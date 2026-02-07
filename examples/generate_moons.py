"""
Generate samples from a trained flow matching model for the 2D moons dataset.

Usage:
    # Generate static visualization
    uv run examples/generate_moons.py generation.model_path=model_moons_neural_network_linear.safetensors

    # Generate with animation
    uv run examples/generate_moons.py generation.model_path=model_moons_neural_network_linear.safetensors --animated

    # Override number of samples and steps
    uv run examples/generate_moons.py generation.model_path=model.safetensors generation.n_samples=200 generation.num_steps=200
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

from tinyflow.nn import NeuralNetwork
from tinyflow.solver import DDIM, Euler, Heun, MidpointSolver, RK4
from tinyflow.trainer import BaseTrainer
from tinyflow.utils import preprocess_time_moons

plt.style.use("ggplot")


def create_solver(cfg: DictConfig, model, preprocess_hook):
    """Create ODE solver from config."""
    solver_type = cfg.solver.type
    if solver_type == "euler":
        return Euler(model, preprocess_hook=preprocess_hook)
    if solver_type == "heun":
        return Heun(model, preprocess_hook=preprocess_hook)
    if solver_type == "midpoint":
        return MidpointSolver(model, preprocess_hook=preprocess_hook)
    if solver_type == "rk4":
        return RK4(model, preprocess_hook=preprocess_hook)
    if solver_type == "ddim":
        eta = cfg.solver.get("eta", 0.0)
        return DDIM(model, preprocess_hook=preprocess_hook, eta=eta)
    raise ValueError(f"Unknown solver type: {solver_type}")


def create_model(cfg: DictConfig):
    """Create model from config."""
    model_type = cfg.model.type
    if model_type == "neural_network":
        return NeuralNetwork(
            in_dim=cfg.model.input_dim,
            time_embed_dim=cfg.model.time_embed_dim,
            out_dim=cfg.model.output_dim,
        )
    raise ValueError(f"Unknown model type: {model_type}")


def generate_static_visualization(cfg: DictConfig, solver):
    """Generate static visualization with multiple time snapshots."""
    n_samples = cfg.generation.n_samples
    num_steps = cfg.generation.num_steps
    step_size = cfg.generation.step_size
    num_plots = cfg.generation.num_plots

    print(f"Generating {n_samples} samples with {num_plots} snapshots...")

    # Initialize samples
    T.training = False
    x = T.randn(n_samples, 2)
    time_grid = T.linspace(0, 1, num_steps)
    sample_every = time_grid.shape[0] // num_plots

    # Store snapshots to visualize
    snapshots = []
    snapshot_times = []

    # JIT compile the solver step for better performance
    @TinyJit
    def jit_step(step_size, t, x):
        return solver.sample(step_size, t, x)

    # Generate all samples first
    for idx in tqdm(range(int(time_grid.shape[0])), desc="Generating samples"):
        t = time_grid[idx].contiguous()
        x = jit_step(step_size, t, x)

        # Store reference for visualization
        if (idx + 1) % sample_every == 0:
            snapshots.append(x)
            snapshot_times.append(t)

    # Visualize all snapshots
    _, ax = plt.subplots(1, num_plots, figsize=(30, 4), sharex=True, sharey=True)
    for i, (snapshot, t) in enumerate(zip(snapshots, snapshot_times, strict=False)):
        x_np = snapshot.numpy()
        ax[i].scatter(x_np[:, 0], x_np[:, 1], s=20, alpha=0.6)
        ax[i].set_title(f"t={t.numpy():.2f}")
        ax[i].set_xlim(-3, 3)
        ax[i].set_ylim(-3, 3)
        ax[i].grid(True, alpha=0.3)

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


def generate_animation(cfg: DictConfig, solver):
    """Generate an animated GIF of the generation process."""
    n_samples = cfg.generation.n_samples
    num_steps = cfg.generation.num_steps
    step_size = cfg.generation.step_size
    num_plots = cfg.generation.num_plots
    fps = cfg.generation.fps

    print(f"Generating {n_samples} samples with {num_plots} frames...")

    # Initialize samples
    T.training = False
    x = T.randn(n_samples, 2)
    time_grid = T.linspace(0, 1, num_steps)
    sample_every = time_grid.shape[0] // num_plots

    # Store snapshots to visualize
    snapshots = []
    snapshot_times = []

    # JIT compile the solver step for better performance
    @TinyJit
    def jit_step(step_size, t, x):
        return solver.sample(step_size, t, x)

    # Generate all samples first
    for idx in tqdm(range(int(time_grid.shape[0])), desc="Generating animation"):
        t = time_grid[idx].contiguous()
        x = jit_step(step_size, t, x)

        # Store reference for visualization
        if (idx + 1) % sample_every == 0:
            snapshots.append(x)
            snapshot_times.append(t)

    # Create frames
    frames = []
    for snapshot, t in tqdm(
        zip(snapshots, snapshot_times, strict=False), desc="Creating frames", total=len(snapshots)
    ):
        x_np = snapshot.numpy()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(x_np[:, 0], x_np[:, 1], s=20, alpha=0.6)
        ax.set_title(f"t = {t.numpy():.2f}", fontsize=16)
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

    # Create solver
    # Note: Scheduler/path not needed at generation time - the model has already
    # learned the velocity field during training
    solver = create_solver(cfg, model, preprocess_time_moons)

    # Generate samples
    if animated:
        generate_animation(cfg, solver)
    else:
        generate_static_visualization(cfg, solver)


@hydra.main(version_base=None, config_path="../configs", config_name="generate_moons_config")
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
