# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Overview

This is a flow matching implementation using the `tinygrad` library instead of PyTorch. Flow matching is a generative modeling technique that learns to transform noise distributions into data distributions through continuous-time flows. The project implements various schedulers, paths, solvers, and neural network architectures for flow matching.

## Development Commands

### Environment Setup

This project uses `uv` for dependency management. Install dependencies:

```bash
uv sync
```

### Running Examples

#### With Hydra Configuration (Recommended)

```bash
# Moons dataset with default config
uv run examples/train_moons_hydra.py

# MNIST training with default config
uv run examples/train_mnist_hydra.py

# Run with specific experiment configuration
uv run examples/train_moons_hydra.py +experiment=quick_test

# Override specific parameters
uv run examples/train_moons_hydra.py scheduler=cosine optimizer.lr=0.01

# Compare multiple schedulers (multirun)
uv run examples/train_moons_hydra.py -m scheduler=linear,cosine,polynomial

# Fashion MNIST training
uv run examples/train_mnist_hydra.py dataset=fashion_mnist

# CIFAR-10 training
uv run examples/train_mnist_hydra.py dataset=cifar10 model=unet
```

#### Training FID Classifier (Optional)

Before using FID metrics or class predictions, train a classifier for your dataset:

```bash
# Train classifier for MNIST
uv run scripts/train_fid_classifier.py --dataset mnist --epochs 10

# Train for Fashion MNIST (with validation set)
uv run scripts/train_fid_classifier.py --dataset fashion_mnist --epochs 10

# Train for CIFAR-10 (with validation set)
uv run scripts/train_fid_classifier.py --dataset cifar10 --epochs 10
```

Trained weights are saved to `weights/fid_classifier_{dataset_name}.safetensors`.

#### Generating Samples

After training a model, generate samples using the dedicated generation scripts:

```bash
# Generate static grid for MNIST
uv run examples/generate_images.py generation.model_path=model_mnist_unet_linear.safetensors

# Generate for Fashion MNIST
uv run examples/generate_images.py dataset=fashion_mnist generation.model_path=model_fashion_mnist_unet_linear.safetensors

# Generate for CIFAR-10
uv run examples/generate_images.py dataset=cifar10 generation.model_path=model_cifar10_unet_linear.safetensors

# Generate animated GIF for MNIST
uv run examples/generate_images.py generation.model_path=model_mnist_unet_linear.safetensors --animated

# Generate with class predictions (requires trained FID classifier)
uv run examples/generate_images.py generation.model_path=model.safetensors generation.show_predictions=true

# Generate with FID computation (requires trained FID classifier)
uv run examples/generate_images.py generation.model_path=model.safetensors generation.compute_fid=true

# Generate with both predictions and FID
uv run examples/generate_images.py generation.model_path=model.safetensors generation.show_predictions=true generation.compute_fid=true

# Override grid size and steps
uv run examples/generate_images.py generation.model_path=model.safetensors generation.grid_size=3 generation.num_steps=50

# Generate samples for moons dataset
uv run examples/generate_moons.py generation.model_path=model_moons_neural_network_linear.safetensors

# Generate animated moons visualization
uv run examples/generate_moons.py generation.model_path=model_moons_neural_network_linear.safetensors --animated

# Customize moons generation
uv run examples/generate_moons.py generation.model_path=model.safetensors generation.n_samples=200 generation.num_steps=100
```

Generated outputs:
- **Images**: Static grids saved as PNG in `outputs/generated/`
- **Animations**: GIF files showing the generation process over time
- **Moons**: 2D scatter plots showing the flow from noise to data distribution
- **Class Predictions**: When `show_predictions=true`, each generated image displays its predicted class and confidence
- **FID Scores**: When `compute_fid=true`, FID score is computed against real dataset images

**Important Notes:**
- **Schedulers**: The scheduler (LinearScheduler, CosineScheduler, etc.) is **only used during training** to define the interpolation path. At generation time, the scheduler is **not needed** because the model has already learned the velocity field. The generation scripts only require the trained model weights - they integrate the learned velocity field using the ODE solver.
- **FID Classifier**: Class predictions and FID computation require a trained FID classifier. Train one using the commands above before enabling these features.

#### Legacy Examples (argparse)

```bash
# More advanced moons training with configurable schedulers
uv run examples/train_moons.py --scheduler linear --epochs 5000 --learning-rate 0.001

# MNIST training
uv run examples/train_mnist.py
```

### Testing

```bash
# Run all tests
uv run pytest tests/ -v --tb=short --color=yes

# Run specific test file
uv run pytest tests/test_paths.py -v

# Run specific test
uv run pytest tests/test_paths.py::test_name -v
```

### Linting and Formatting

```bash
# Run ruff linting
uv run ruff check tinyflow/ tests/ examples/

# Auto-fix linting issues
uv run ruff check tinyflow/ tests/ examples/ --fix

# Format code
uv run ruff format tinyflow/ tests/ examples/

# Check formatting without making changes
uv run ruff format --check tinyflow/ tests/ examples/

# Type checking
uv run pyright tinyflow/
```

### Pre-commit Hooks

The project uses pre-commit hooks for code quality. Install hooks:

```bash
uv run pre-commit install
```

### MLflow Experiment Tracking

View experiment results and metrics:

```bash
# Start MLflow UI
uv run mlflow ui

# View in browser at http://localhost:5000
```

MLflow automatically logs:

- Training metrics (loss, epoch)
- Hyperparameters (model, scheduler, optimizer settings)
- Artifacts (loss curves, generated samples)
- Tags (model type, scheduler, dataset)

## Architecture

### Core Components

1. **Paths** (`tinyflow/path/path.py`)
   - Define interpolation paths between noise and data distributions
   - `AffinePath`: General affine interpolation using schedulers
   - `OptimalTransportPath`: Optimal transport-based interpolation
   - Paths compute both `x_t` (interpolated state) and `dx_t` (velocity) at time `t`

2. **Schedulers** (`tinyflow/path/scheduler.py`)
   - Control the interpolation schedule via `alpha_t` and `sigma_t` functions
   - Requirements: `alpha_0 = sigma_1 = 0`, `alpha_1 = sigma_0 = 1`, `alpha_t` increases, `sigma_t` decreases
   - Available: `LinearScheduler`, `PolynomialScheduler`, `CosineScheduler`, `LinearVarPresScheduler`
   - Each provides `alpha_t`, `alpha_t_dot`, `sigma_t`, `sigma_t_dot` methods

3. **Neural Networks** (`tinyflow/nn.py`)
   - All models inherit from `BaseNeuralNetwork`
   - Models take `(x, t)` as input where `x` is the state and `t` is the time
   - `MLPNetwork`: Flexible MLP with configurable hidden layers and activation functions
   - `UNetTinygrad`: U-Net architecture for image generation (MNIST, CIFAR-10)
   - Helper functions: `create_mlp_simple()`, `create_mlp_deep()` for quick model creation

4. **ODE Solvers** (`tinyflow/solver/`)
   - Base class: `ODESolver` in `solver.py`
   - Implementations: `Euler`, `RK4`, `Midpoint`
   - Solvers integrate the learned velocity field to generate samples
   - All solvers support a `preprocess_hook` for time/state preprocessing

5. **Trainer** (`tinyflow/trainer.py`)
   - `BaseTrainer`: Abstract base class for training loops with MLflow integration
   - `MNISTTrainer`: Specialized trainer for MNIST dataset
   - `FashionMNISTTrainer`: Specialized trainer for Fashion MNIST dataset
   - `CIFAR10Trainer`: Specialized trainer for CIFAR-10 dataset
   - Handles training loop, data iteration, loss tracking, and visualization
   - Integrates directly with MLflow for experiment tracking

6. **Dataloader** (`tinyflow/dataloader.py`)
   - `BaseDataloader`: Abstract base class for all dataloaders
   - `MNISTLoader`: Custom MNIST dataloader that reads from disk
   - `FashionMNISTLoader`: Fashion MNIST dataloader (inherits from MNISTLoader)
   - `CIFAR10Loader`: Custom CIFAR-10 dataloader with in-memory caching
   - Supports batching, shuffling, flattening, and normalization options

### Training Flow

1. Sample data point `x_1` from target distribution
2. Sample noise `x_0` from Gaussian
3. Sample random time `t ∈ [0, 1]`
4. Compute interpolated state `x_t` and target velocity `dx_t` using the path
5. Predict velocity with neural network: `v_θ(x_t, t)`
6. Minimize MSE loss: `||v_θ(x_t, t) - dx_t||²`
7. After training, generate samples by solving ODE from `t=0` to `t=1`

### Key Mathematical Concepts

- **Conditional Flow Matching**: The model learns `u_t(x_t|x_1)` which is the time derivative of the interpolation path
- **Linear Path** (simplest): `x_t = t·x_1 + (1-t)·x_0`, so `dx_t = x_1 - x_0`
- **Affine Path**: `x_t = α_t·x_1 + σ_t·x_0` where `α_t` and `σ_t` are scheduler functions
- At inference, integrate the learned velocity field backwards from noise to data

### File Organization

```
tinyflow/
├── __init__.py
├── nn.py              # Neural network architectures
├── losses.py          # Loss functions (MSE)
├── trainer.py         # Training loops with MLflow integration
├── dataloader.py      # Data loading utilities
├── utils.py           # Visualization and preprocessing
├── path/
│   ├── path.py        # Path implementations
│   └── scheduler.py   # Scheduler implementations
├── solver/
│   ├── solver.py      # Base ODE solver
│   ├── euler.py       # Euler method
│   ├── rk4.py         # 4th order Runge-Kutta
│   └── midpoint.py    # Midpoint method
└── nn_utils/
    ├── conv.py        # Convolutional building blocks
    └── time_embedding.py  # Time embedding layers

configs/                # Hydra configuration files
├── config.yaml         # Main configuration
├── model/              # Model architectures
│   ├── neural_network.yaml
│   └── unet.yaml
├── scheduler/          # Scheduler types
│   ├── linear.yaml
│   ├── cosine.yaml
│   ├── polynomial.yaml
│   └── lvp.yaml
├── optimizer/          # Optimizer settings
│   └── adam.yaml
├── dataset/            # Dataset configurations
│   ├── mnist.yaml
│   ├── cifar10.yaml
│   └── moons.yaml
├── training/           # Training parameters
│   ├── mnist_default.yaml
│   └── moons_default.yaml
└── experiment/         # Pre-configured experiments
    └── quick_test.yaml

examples/
├── train_moons_hydra.py    # Moons training with Hydra
├── train_mnist_hydra.py    # MNIST training with Hydra (also works for CIFAR-10)
├── generate_images.py      # Generate samples for image datasets (MNIST, Fashion MNIST, CIFAR-10)
├── generate_moons.py       # Generate samples for moons dataset
├── train_moons.py          # Legacy moons training
├── train_mnist.py          # Legacy MNIST training
├── generate_samples.py     # Legacy generation script (deprecated - use generate_images.py)
└── generate_animation.py   # Legacy animation script (deprecated - use --animated flag)
```

## Experiment Management

This project uses **Hydra** for configuration management and **MLflow** for experiment tracking.

### Configuration with Hydra

- Configurations are organized in the `configs/` directory
- Use composable configs to mix and match components
- Override parameters from command line
- See `docs/MLFLOW_HYDRA_GUIDE.md` for detailed usage

### Experiment Tracking with MLflow

- Automatic logging of all hyperparameters
- Real-time metric tracking (loss, epoch)
- Artifact storage (plots, visualizations)
- Run comparison and visualization
- MLflow integration is built directly into the `BaseTrainer` class
- See `docs/MLFLOW_HYDRA_GUIDE.md` for detailed usage

## Supported Datasets

1. **Moons**: 2D toy dataset for quick testing and visualization
2. **MNIST**: Handwritten digits (28x28 grayscale images)
3. **Fashion MNIST**: Fashion items (28x28 grayscale images, 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
4. **CIFAR-10**: Natural images (32x32 RGB images, 10 classes)

**Dataset Setup:**

- MNIST: Place images in `dataset/mnist/trainset/trainingSet/*/*.jpg`
- Fashion MNIST: Place images in `dataset/fashion_mnist/train/*/*.jpg`
- CIFAR-10: Place batch files in `dataset/cifar10/cifar-10-batches-py/` (uses UNet model for best results)

## References

The implementation is based on:

- [Flow Matching for Generative Modeling](https://arxiv.org/pdf/2210.02747)
- [Flow Matching Guide and Code](https://arxiv.org/pdf/2412.06264)
- [Hydra Documentation](https://hydra.cc/)
- [MLflow Documentation](https://mlflow.org/)
