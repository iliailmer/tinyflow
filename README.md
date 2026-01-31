# Flow Matching using Tinygrad

[![Tests](https://github.com/iliailmer/tinyflow/actions/workflows/tests.yml/badge.svg)](https://github.com/iliailmer/tinyflow/actions/workflows/tests.yml)
[![Lint](https://github.com/iliailmer/tinyflow/actions/workflows/lint.yml/badge.svg)](https://github.com/iliailmer/flow_matching_tinygrad/actions/workflows/lint.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive implementation of **Flow Matching** algorithms using the `tinygrad` library. This project prioritizes pedagogical clarity and extensibility, making it ideal for learning, research prototyping, and understanding flow-based generative models.

## What is Flow Matching?

Flow matching is a modern approach to generative modeling that learns to transform noise distributions into data distributions through continuous-time flows. Unlike diffusion models, flow matching directly learns the velocity field of the transformation, often leading to:

- Faster sampling (fewer integration steps needed)
- Simpler training (direct regression instead of score matching)
- Better mathematical clarity (continuous normalizing flows)

## Features

### Core Capabilities ✅

- **Multiple Schedulers:** Linear, Cosine, Polynomial, Linear Variance Preserving (LVP)
- **ODE Solvers:** Euler, Midpoint, RK4, Heun, DDIM
- **Architectures:** MLP networks, UNet for images
- **Datasets:** MNIST, Fashion MNIST, CIFAR-10, 2D toy datasets (moons)
- **Time Embeddings:** Sinusoidal/Fourier embeddings for temporal conditioning
- **Metrics:** FID (Fréchet Inception Distance) with trained classifiers
- **Experiment Tracking:** Hydra configuration + MLflow logging
- **Generation Tools:** Class predictions, FID computation, animated generation

### Evaluation & Visualization

- **FID Metric:** Measure generation quality quantitatively
- **Classifier Predictions:** Show predicted class and confidence on generated images
- **Animated Generation:** Visualize the flow from noise to data
- **MLflow Integration:** Track experiments, hyperparameters, and metrics

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/iliailmer/flow_matching_tinygrad
cd flow_matching_tinygrad

# Install dependencies (recommended: use uv)
uv sync
```

### Training Your First Model

```bash
# Train on 2D moons dataset (fast, for testing)
uv run examples/train_moons_hydra.py

# Train on MNIST
uv run examples/train_mnist_hydra.py

# Train on Fashion MNIST
uv run examples/train_mnist_hydra.py dataset=fashion_mnist

# Train on CIFAR-10
uv run examples/train_mnist_hydra.py dataset=cifar10
```

### Generating Samples

```bash
# Generate static grid for MNIST
uv run examples/generate_images.py generation.model_path=model_mnist_unet_linear.safetensors

# Generate with class predictions and FID score
uv run examples/generate_images.py \
    generation.model_path=model_mnist_unet_linear.safetensors \
    generation.show_predictions=true \
    generation.compute_fid=true

# Generate animated GIF showing the flow process
uv run examples/generate_images.py \
    generation.model_path=model_mnist_unet_linear.safetensors \
    --animated

# Generate for 2D moons dataset
uv run examples/generate_moons.py generation.model_path=model_moons_neural_network_linear.safetensors
```

### Training FID Classifier (for evaluation)

```bash
# Train classifier for MNIST
uv run scripts/train_fid_classifier.py --dataset mnist --epochs 10

# Train classifier for Fashion MNIST
uv run scripts/train_fid_classifier.py --dataset fashion_mnist --epochs 10

# Train classifier for CIFAR-10
uv run scripts/train_fid_classifier.py --dataset cifar10 --epochs 10
```

Generated outputs are saved in `outputs/generated/` directory.

## Project Structure

```
tinyflow/
├── nn.py                      # Neural network architectures (UNet, MLP)
├── losses.py                  # Loss functions
├── trainer.py                 # Training loops with MLflow integration
├── dataloader.py              # Dataset loaders (MNIST, Fashion MNIST, CIFAR-10)
├── metrics.py                 # FID metric and classifiers
├── utils.py                   # Visualization utilities
├── path/
│   ├── path.py                # Flow matching paths (Affine, OT)
│   └── scheduler.py           # Schedulers (Linear, Cosine, Polynomial, LVP)
├── solver/
│   ├── solver.py              # Base ODE solver
│   ├── euler.py               # Euler method
│   ├── rk4.py                 # 4th order Runge-Kutta
│   ├── midpoint.py            # Midpoint method
│   ├── heun.py                # Heun's method
│   └── ddim.py                # DDIM-style deterministic sampling
└── nn_utils/
    ├── conv.py                # Convolutional building blocks
    └── time_embedding.py      # Time embedding layers

configs/                       # Hydra configuration files
├── config.yaml                # Main configuration
├── model/                     # Model architectures
├── scheduler/                 # Scheduler types
├── optimizer/                 # Optimizer settings
├── dataset/                   # Dataset configurations
├── training/                  # Training parameters
└── generation/                # Generation settings

examples/
├── train_mnist_hydra.py       # MNIST/CIFAR-10 training
├── train_moons_hydra.py       # 2D moons training
├── generate_images.py         # Image generation with predictions/FID
└── generate_moons.py          # 2D moons generation

scripts/
└── train_fid_classifier.py    # Train FID classifier with validation

docs/
├── MLFLOW_HYDRA_GUIDE.md      # Experiment tracking guide
├── TRAINING_OPTIMIZATION.md   # Performance optimization
├── ODE_SOLVER_IMPROVEMENTS.md # Solver improvements
└── CIFAR10_IMPLEMENTATION.md  # CIFAR-10 setup guide
```

## Configuration with Hydra

This project uses Hydra for flexible configuration management:

```bash
# Override specific parameters
uv run examples/train_mnist_hydra.py scheduler=cosine optimizer.lr=0.001

# Compare multiple schedulers (multirun)
uv run examples/train_mnist_hydra.py -m scheduler=linear,cosine,polynomial

# Use experiment configs
uv run examples/train_moons_hydra.py +experiment=quick_test
```

See `docs/MLFLOW_HYDRA_GUIDE.md` for detailed usage.

## Experiment Tracking with MLflow

All experiments are automatically logged to MLflow:

```bash
# Start MLflow UI
uv run mlflow ui

# View at http://localhost:5000
```

MLflow tracks:
- Hyperparameters (model, scheduler, optimizer settings)
- Metrics (loss, FID score)
- Artifacts (generated images, loss curves)
- Tags (model type, dataset)

## Development Roadmap

This project is under active development. See [ROADMAP.md](ROADMAP.md) for:
- Planned features (EMA, conditional generation, advanced samplers)
- Implementation priorities
- Research directions (discrete flows, manifold flows)
- Community building efforts

**High Priority Next Steps:**
- Exponential Moving Average (EMA) for better quality
- Class-conditional generation
- Comprehensive documentation and tutorials
- Pre-trained models

## Documentation

- **[ROADMAP.md](ROADMAP.md)** - Development roadmap and priorities
- **[CLAUDE.md](CLAUDE.md)** - Instructions for Claude Code assistant
- **[docs/MLFLOW_HYDRA_GUIDE.md](docs/MLFLOW_HYDRA_GUIDE.md)** - Experiment tracking guide
- **[docs/TRAINING_OPTIMIZATION.md](docs/TRAINING_OPTIMIZATION.md)** - Performance optimization tips
- **[docs/ODE_SOLVER_IMPROVEMENTS.md](docs/ODE_SOLVER_IMPROVEMENTS.md)** - Advanced solver implementations
- **[docs/CIFAR10_IMPLEMENTATION.md](docs/CIFAR10_IMPLEMENTATION.md)** - CIFAR-10 setup guide

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_paths.py -v

# Run with coverage
uv run pytest tests/ --cov=tinyflow --cov-report=html
```

## Contributing

Contributions welcome! This project prioritizes:
- Educational clarity over performance
- Clean, readable code
- Comprehensive documentation
- Extensibility for research

See [ROADMAP.md](ROADMAP.md) for planned features and priorities.

## Why Tinygrad?

- **Lightweight:** Minimal dependencies, easier to understand
- **Educational:** Simpler codebase than PyTorch/JAX
- **Transparent:** Mathematical operations more visible
- **Research-friendly:** Easy to modify for novel experiments
- **Growing ecosystem:** Help build the tinygrad ML community

## References

### Core Papers
- [Flow Matching for Generative Modeling](https://arxiv.org/pdf/2210.02747) - Lipman et al., 2022
- [Flow Matching Guide and Code](https://arxiv.org/pdf/2412.06264) - Pooladian et al., 2024

### Frameworks
- [Tinygrad](https://github.com/tinygrad/tinygrad) - Minimalist ML framework
- [Hydra](https://hydra.cc/) - Configuration management
- [MLflow](https://mlflow.org/) - Experiment tracking

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

This project builds on the flow matching literature and the tinygrad ecosystem. Special thanks to the authors of the flow matching papers and the tinygrad community.
