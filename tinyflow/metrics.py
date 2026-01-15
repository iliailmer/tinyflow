"""
Evaluation metrics for generative models.

Implements FID (Fréchet Inception Distance) for assessing generation quality.
"""

import os

import numpy as np
from loguru import logger
from scipy import linalg
from tinygrad import nn
from tinygrad.nn.state import load_state_dict, safe_load
from tinygrad.tensor import Tensor as T


def calculate_activation_statistics(features: T) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean and covariance of activation features.

    Uses tinygrad for computation, returns numpy arrays for scipy compatibility.

    Args:
        features: Activation features tensor, shape (n_samples, n_features)

    Returns:
        mu: Mean vector (n_features,)
        sigma: Covariance matrix (n_features, n_features)
    """
    # Compute in tinygrad
    mu = features.mean(axis=0)
    centered = features - mu
    n = features.shape[0]
    cov = (centered.T @ centered) / (n - 1)

    # Return as numpy for scipy.linalg.sqrtm
    return mu.numpy(), cov.numpy()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    """
    Calculate the Fréchet distance between two multivariate Gaussians.

    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))

    Args:
        mu1, mu2: Mean vectors (d,)
        sigma1, sigma2: Covariance matrices (d, d)

    Returns:
        Fréchet distance (scalar)
    """
    diff = mu1 - mu2

    # Matrix square root using scipy (accurate and fast)
    try:
        covmean = linalg.sqrtm(sigma1 @ sigma2)
    except linalg.LinAlgError:
        covmean = np.zeros_like(sigma1)

    # Handle numerical complex values
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            logger.warning(f"Imaginary component {np.max(np.abs(covmean.imag)):.4f} in covariance")
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


# =============================================================================
# Feature Extractors
# =============================================================================


class SimpleFeatureExtractor:
    """Simple CNN feature extractor (128-dim features)."""

    def __init__(self, input_channels=1):
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

    def __call__(self, x: T) -> T:
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.conv3(x).relu()
        return x.mean(axis=(2, 3))  # Global average pooling

    def extract_features(self, images) -> T:
        """Extract features from images. Returns tinygrad Tensor."""
        if isinstance(images, np.ndarray):
            images = T(images)
        with T.train(False):
            return self(images)


class LeNetMNIST:
    """LeNet classifier for MNIST/Fashion MNIST (256-dim features)."""

    def __init__(self):
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 10)

    def parameters(self):
        return [
            self.conv1.weight,
            self.conv1.bias,
            self.conv2.weight,
            self.conv2.bias,
            self.fc1.weight,
            self.fc1.bias,
            self.fc2.weight,
            self.fc2.bias,
        ]

    def features(self, x: T) -> T:
        x = self.conv1(x).relu().max_pool2d(kernel_size=2)
        x = self.conv2(x).relu().max_pool2d(kernel_size=2)
        return self.fc1(x.flatten(1)).relu()

    def __call__(self, x: T) -> T:
        return self.fc2(self.features(x))


class LeNetCIFAR10:
    """CNN classifier for CIFAR-10 (512-dim features)."""

    def __init__(self):
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 10)

    def parameters(self):
        return [
            self.conv1.weight,
            self.conv1.bias,
            self.conv2.weight,
            self.conv2.bias,
            self.conv3.weight,
            self.conv3.bias,
            self.fc1.weight,
            self.fc1.bias,
            self.fc2.weight,
            self.fc2.bias,
        ]

    def features(self, x: T) -> T:
        x = self.conv1(x).relu().max_pool2d(kernel_size=2)
        x = self.conv2(x).relu().max_pool2d(kernel_size=2)
        x = self.conv3(x).relu().max_pool2d(kernel_size=2)
        return self.fc1(x.flatten(1)).relu()

    def __call__(self, x: T) -> T:
        return self.fc2(self.features(x))


class ClassifierFeatureExtractor:
    """Feature extractor using a trained classifier."""

    def __init__(self, dataset_name: str, weights_dir: str = "weights"):
        self.dataset_name = dataset_name

        if dataset_name in ("mnist", "fashion_mnist"):
            self.model = LeNetMNIST()
            self.feature_dim = 256
        elif dataset_name == "cifar10":
            self.model = LeNetCIFAR10()
            self.feature_dim = 512
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        weights_path = os.path.join(weights_dir, f"fid_classifier_{dataset_name}.safetensors")
        if os.path.exists(weights_path):
            load_state_dict(self.model, safe_load(weights_path))
            self._weights_loaded = True
            logger.info(f"Loaded FID classifier weights from {weights_path}")
        else:
            self._weights_loaded = False
            logger.warning(f"FID weights not found at {weights_path}")

    def extract_features(self, images, batch_size: int = 64) -> T:
        """Extract features from images. Returns tinygrad Tensor."""
        if isinstance(images, np.ndarray):
            images_np = images
        else:
            images_np = images.numpy()

        features_list = []
        with T.train(False):
            for i in range(0, len(images_np), batch_size):
                batch = T(images_np[i : i + batch_size])
                features_list.append(self.model.features(batch))

        return T.cat(*features_list, dim=0)


def get_feature_extractor(dataset_name: str, weights_dir: str = "weights"):
    """Factory function to get a feature extractor for a dataset."""
    return ClassifierFeatureExtractor(dataset_name, weights_dir)


# =============================================================================
# FID Calculation
# =============================================================================


def calculate_fid(real_images, generated_images, feature_extractor=None, batch_size: int = 64):
    """
    Calculate FID score between real and generated images.

    Args:
        real_images: Real images, shape (n, channels, H, W)
        generated_images: Generated images, shape (n, channels, H, W)
        feature_extractor: Optional feature extractor (default: SimpleFeatureExtractor)
        batch_size: Batch size for feature extraction

    Returns:
        FID score (lower is better)
    """
    if feature_extractor is None:
        n_channels = real_images.shape[1] if len(real_images.shape) == 4 else 1
        feature_extractor = SimpleFeatureExtractor(input_channels=n_channels)

    def extract_batched(images) -> T:
        features_list = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            features_list.append(feature_extractor.extract_features(batch))
        return T.cat(*features_list, dim=0)

    logger.info("Extracting features from real images...")
    real_features = extract_batched(real_images)

    logger.info("Extracting features from generated images...")
    gen_features = extract_batched(generated_images)

    # Statistics computed in tinygrad, returned as numpy for scipy
    mu_real, sigma_real = calculate_activation_statistics(real_features)
    mu_gen, sigma_gen = calculate_activation_statistics(gen_features)

    fid = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    logger.info(f"FID score: {fid:.4f}")
    return fid


# Aliases for backwards compatibility
calculate_fid_simple = calculate_fid
calculate_fid_batched = calculate_fid
