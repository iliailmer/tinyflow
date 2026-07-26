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

    # Animated GIF, class predictions, FID
    uv run scripts/generate.py generation.model_path=model.safetensors --animated
    uv run scripts/generate.py generation.model_path=model.safetensors generation.show_predictions=true generation.compute_fid=true

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
from tinygrad import TinyJit
from tinygrad.tensor import Tensor as T
from tqdm import tqdm

from tinyflow.metrics import get_feature_extractor
from tinyflow.nn import NeuralNetwork, UNetTinygrad
from tinyflow.trainer import BaseTrainer
from tinyflow.utils import preprocess_time_cifar, preprocess_time_mnist, preprocess_time_moons

plt.style.use("ggplot")

# Class names for image datasets
CLASS_NAMES = {
    "mnist": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "fashion_mnist": [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ],
    "cifar10": [
        "Airplane",
        "Automobile",
        "Bird",
        "Cat",
        "Deer",
        "Dog",
        "Frog",
        "Horse",
        "Ship",
        "Truck",
    ],
}


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


def compute_fid_for_generated(
    generated_images: np.ndarray, dataset_name: str, cfg: DictConfig
) -> float | None:
    """
    Compute FID score between generated images and real dataset images.

    Args:
        generated_images: Generated images, shape (n, channels, H, W), normalized to [0, 1]
        dataset_name: Dataset name (mnist, fashion_mnist, cifar10)
        cfg: Configuration object

    Returns:
        FID score or None if not available
    """
    try:
        from tinyflow.dataloader import CIFAR10Loader, FashionMNISTLoader, MNISTLoader
        from tinyflow.metrics import calculate_fid

        # Load classifier for FID
        classifier = get_feature_extractor(dataset_name)

        if not classifier._weights_loaded:
            print(f"Warning: FID classifier weights not found for {dataset_name}")
            print(f"Run: uv run scripts/train_fid_classifier.py --dataset {dataset_name}")
            return None

        # Load real images from dataset
        num_samples = len(generated_images)
        print(f"Loading {num_samples} real images from {dataset_name}...")

        if dataset_name == "mnist":
            dataloader = MNISTLoader(
                path="dataset/mnist/trainset/trainingSet/*/*.jpg",
                batch_size=64,
                shuffle=True,
                flatten=False,
            )
        elif dataset_name == "fashion_mnist":
            dataloader = FashionMNISTLoader(
                path="dataset/fashion_mnist",
                batch_size=64,
                shuffle=True,
                flatten=False,
                train=True,
            )
        elif dataset_name == "cifar10":
            dataloader = CIFAR10Loader(
                path="dataset/cifar10/cifar-10-batches-py",
                batch_size=64,
                shuffle=True,
                train=True,
            )
        else:
            print(f"Unknown dataset: {dataset_name}")
            return None

        # Collect real images
        real_images = []
        for batch_images, _ in dataloader:
            real_images.append(batch_images)
            if len(np.concatenate(real_images, axis=0)) >= num_samples:
                break

        real_images = np.concatenate(real_images, axis=0)[:num_samples]

        # Compute FID
        fid_score = calculate_fid(
            real_images, generated_images, feature_extractor=classifier, batch_size=64
        )

        return fid_score

    except Exception as e:
        print(f"Error computing FID: {e}")
        import traceback

        traceback.print_exc()
        return None


def classify_images(images_np: np.ndarray, dataset_name: str):
    """
    Classify generated images and return predictions.

    Args:
        images_np: Generated images, shape (n, channels, H, W), normalized to [0, 1]
        dataset_name: Dataset name (mnist, fashion_mnist, cifar10)

    Returns:
        List of tuples (class_idx, class_name, confidence) or None if classifier not available
    """
    try:
        # Load classifier
        classifier = get_feature_extractor(dataset_name)

        # Check if weights are loaded
        if not classifier._weights_loaded:
            print(f"Warning: Classifier weights not found for {dataset_name}")
            print(f"Run: uv run scripts/train_fid_classifier.py --dataset {dataset_name}")
            return None

        # Run inference
        with T.train(False):
            images_tensor = T(images_np)
            logits = classifier.model(images_tensor)
            probs = logits.softmax(axis=-1)

            # Get predictions
            class_indices = probs.numpy().argmax(axis=-1)
            confidences = probs.numpy().max(axis=-1)

        # Get class names
        class_names = CLASS_NAMES.get(dataset_name, [str(i) for i in range(10)])

        results = []
        for idx, conf in zip(class_indices, confidences, strict=False):
            results.append((int(idx), class_names[idx], float(conf)))

        return results

    except Exception as e:
        print(f"Error classifying images: {e}")
        return None


def generate_image_grid(cfg: DictConfig, model, solver, dataset_config):
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

    # Ensure final result is realized before conversion
    x.realize()

    # Convert to numpy and normalize
    x_np = x.numpy()
    x_np = (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-8)
    x_np = np.clip(x_np, 0, 1)

    # Compute FID if enabled
    if cfg.generation.get("compute_fid", False):
        print("\nComputing FID score...")
        fid_score = compute_fid_for_generated(x_np, dataset_name, cfg)
        if fid_score is not None:
            print(f"FID Score: {fid_score:.4f}")

    # Classify generated images if enabled
    predictions = None
    if cfg.generation.get("show_predictions", False):
        print("Classifying generated images...")
        predictions = classify_images(x_np, dataset_name)

    # Plot grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    if grid_size == 1:
        axes = np.array([[axes]])
    elif len(axes.shape) == 1:
        axes = axes.reshape(-1, 1)

    # Add FID score as figure title if computed
    if cfg.generation.get("compute_fid", False) and fid_score is not None:
        fig.suptitle(f"FID Score: {fid_score:.2f}", fontsize=16, fontweight="bold", y=0.98)

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

            # Add prediction label if available
            if predictions is not None:
                class_idx, class_name, confidence = predictions[idx]
                label = f"{class_name}\n{confidence:.1%}"
                ax.set_title(label, fontsize=12, pad=4, fontweight="bold")

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
            # Ensure tensor is realized before capturing to avoid memory accumulation
            x.realize()
            x_np = x.numpy()
            x_normalized = (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-8)
            x_normalized = np.clip(x_normalized, 0, 1)

            # Classify if at final step and predictions enabled
            predictions = None
            if cfg.generation.get("show_predictions", False) and step == capture_steps[-1]:
                predictions = classify_images(x_normalized, dataset_name)

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

                    # Add prediction label if available (only on final frame)
                    if predictions is not None:
                        class_idx, class_name, confidence = predictions[idx]
                        label = f"{class_name}\n{confidence:.1%}"
                        ax.set_title(label, fontsize=6, pad=2)

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


def generate_moons_animation(cfg: DictConfig, solver):
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
