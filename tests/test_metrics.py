"""Tests for evaluation metrics."""

import numpy as np
import pytest
from tinygrad.tensor import Tensor as T

from tinyflow.metrics import (
    ClassifierFeatureExtractor,
    LeNetCIFAR10,
    LeNetMNIST,
    SimpleFeatureExtractor,
    calculate_activation_statistics,
    calculate_fid,
    calculate_frechet_distance,
    get_feature_extractor,
)


class TestFrechetDistance:
    def test_identical_distributions(self):
        """FID should be 0 for identical distributions."""
        mu = np.array([1.0, 2.0, 3.0])
        sigma = np.eye(3)

        fid = calculate_frechet_distance(mu, sigma, mu, sigma)
        assert abs(fid) < 1e-6

    def test_different_means(self):
        """FID should increase with mean difference."""
        mu1 = np.array([0.0, 0.0])
        mu2 = np.array([1.0, 0.0])
        sigma = np.eye(2)

        fid = calculate_frechet_distance(mu1, sigma, mu2, sigma)
        assert fid > 0

    def test_different_covariances(self):
        """FID should be non-zero for different covariances."""
        mu = np.array([0.0, 0.0])
        sigma1 = np.eye(2)
        sigma2 = np.eye(2) * 2

        fid = calculate_frechet_distance(mu, sigma1, mu, sigma2)
        assert fid > 0


class TestActivationStatistics:
    def test_statistics_shape(self):
        """Test activation statistics calculation."""
        features = T.randn(100, 64)

        mu, sigma = calculate_activation_statistics(features)

        assert mu.shape == (64,)
        assert sigma.shape == (64, 64)

    def test_statistics_values(self):
        """Test statistics are reasonable."""
        features = T.randn(1000, 10)

        mu, sigma = calculate_activation_statistics(features)

        # Mean should be close to 0
        assert np.allclose(mu, 0, atol=0.15)

        # Diagonal of covariance should be close to 1
        assert np.allclose(np.diag(sigma), 1, atol=0.25)


class TestSimpleFeatureExtractor:
    def test_output_shape(self):
        """Test feature extractor output shape."""
        extractor = SimpleFeatureExtractor(input_channels=1)

        x = T.randn(8, 1, 28, 28)
        features = extractor(x)

        assert features.shape == (8, 128)

    def test_extract_features(self):
        """Test feature extraction from images."""
        extractor = SimpleFeatureExtractor(input_channels=1)

        images = np.random.randn(16, 1, 28, 28).astype(np.float32)
        features = extractor.extract_features(images)

        assert features.shape == (16, 128)
        assert isinstance(features, T)

    def test_multi_channel(self):
        """Test feature extractor with RGB images."""
        extractor = SimpleFeatureExtractor(input_channels=3)

        x = T.randn(4, 3, 32, 32)
        features = extractor(x)

        assert features.shape == (4, 128)


class TestFIDCalculation:
    def test_fid_identical_images(self):
        """FID should be near 0 for identical image sets."""
        images = np.random.randn(50, 1, 28, 28).astype(np.float32)

        fid = calculate_fid(images, images)

        # Should be very close to 0
        assert fid < 1.0

    def test_fid_different_images(self):
        """FID should be positive for different image sets."""
        real_images = np.random.randn(50, 1, 28, 28).astype(np.float32)
        gen_images = np.random.randn(50, 1, 28, 28).astype(np.float32) + 1.0

        fid = calculate_fid(real_images, gen_images)

        assert fid > 0

    def test_fid_batched(self):
        """Test batched FID calculation."""
        real_images = np.random.randn(100, 1, 28, 28).astype(np.float32)
        gen_images = np.random.randn(100, 1, 28, 28).astype(np.float32)

        fid = calculate_fid(real_images, gen_images, batch_size=32)

        assert fid >= 0
        assert not np.isnan(fid)

    def test_fid_rgb_images(self):
        """Test FID with RGB images (CIFAR-10 style)."""
        real_images = np.random.randn(50, 3, 32, 32).astype(np.float32)
        gen_images = np.random.randn(50, 3, 32, 32).astype(np.float32)

        fid = calculate_fid(real_images, gen_images)

        assert fid >= 0
        assert not np.isnan(fid)


class TestLeNetMNIST:
    def test_forward_pass(self):
        """Test forward pass returns correct shape."""
        model = LeNetMNIST()
        x = T.randn(8, 1, 28, 28)
        logits = model(x)
        assert logits.shape == (8, 10)

    def test_features(self):
        """Test feature extraction returns correct shape."""
        model = LeNetMNIST()
        x = T.randn(8, 1, 28, 28)
        features = model.features(x)
        assert features.shape == (8, 256)

    def test_parameters(self):
        """Test model has correct number of parameters."""
        model = LeNetMNIST()
        params = model.parameters()
        # 2 conv layers (weight + bias each) + 2 fc layers (weight + bias each) = 8
        assert len(params) == 8


class TestLeNetCIFAR10:
    def test_forward_pass(self):
        """Test forward pass returns correct shape."""
        model = LeNetCIFAR10()
        x = T.randn(4, 3, 32, 32)
        logits = model(x)
        assert logits.shape == (4, 10)

    def test_features(self):
        """Test feature extraction returns correct shape."""
        model = LeNetCIFAR10()
        x = T.randn(4, 3, 32, 32)
        features = model.features(x)
        assert features.shape == (4, 512)

    def test_parameters(self):
        """Test model has correct number of parameters."""
        model = LeNetCIFAR10()
        params = model.parameters()
        # 3 conv layers (weight + bias each) + 2 fc layers (weight + bias each) = 10
        assert len(params) == 10


class TestClassifierFeatureExtractor:
    def test_mnist_extractor_no_weights(self):
        """Test MNIST extractor initializes without weights."""
        extractor = ClassifierFeatureExtractor("mnist", weights_dir="/nonexistent")
        assert extractor.feature_dim == 256
        assert not extractor._weights_loaded

    def test_fashion_mnist_extractor_no_weights(self):
        """Test Fashion MNIST extractor initializes without weights."""
        extractor = ClassifierFeatureExtractor("fashion_mnist", weights_dir="/nonexistent")
        assert extractor.feature_dim == 256
        assert not extractor._weights_loaded

    def test_cifar10_extractor_no_weights(self):
        """Test CIFAR-10 extractor initializes without weights."""
        extractor = ClassifierFeatureExtractor("cifar10", weights_dir="/nonexistent")
        assert extractor.feature_dim == 512
        assert not extractor._weights_loaded

    def test_unknown_dataset_raises(self):
        """Test unknown dataset raises ValueError."""
        with pytest.raises(ValueError):
            ClassifierFeatureExtractor("unknown_dataset")

    def test_extract_features_mnist_shape(self):
        """Test feature extraction for MNIST-shaped images."""
        extractor = ClassifierFeatureExtractor("mnist", weights_dir="/nonexistent")
        images = np.random.randn(16, 1, 28, 28).astype(np.float32)
        features = extractor.extract_features(images)
        assert features.shape == (16, 256)

    def test_extract_features_cifar_shape(self):
        """Test feature extraction for CIFAR-shaped images."""
        extractor = ClassifierFeatureExtractor("cifar10", weights_dir="/nonexistent")
        images = np.random.randn(8, 3, 32, 32).astype(np.float32)
        features = extractor.extract_features(images)
        assert features.shape == (8, 512)


class TestGetFeatureExtractor:
    def test_factory_mnist(self):
        """Test factory returns correct extractor for MNIST."""
        extractor = get_feature_extractor("mnist", weights_dir="/nonexistent")
        assert isinstance(extractor, ClassifierFeatureExtractor)
        assert extractor.feature_dim == 256

    def test_factory_cifar10(self):
        """Test factory returns correct extractor for CIFAR-10."""
        extractor = get_feature_extractor("cifar10", weights_dir="/nonexistent")
        assert isinstance(extractor, ClassifierFeatureExtractor)
        assert extractor.feature_dim == 512


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
