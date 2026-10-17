"""Sample quality metrics on precomputed features."""

import numpy as np


def frechet_distance(feats1: np.ndarray, feats2: np.ndarray) -> float:
    """Frechet distance between Gaussians fitted to two feature sets of shape (N, D): FID."""
    mu1, mu2 = feats1.mean(0), feats2.mean(0)
    sigma1, sigma2 = np.cov(feats1, rowvar=False), np.cov(feats2, rowvar=False)
    # trace(sqrtm(sigma1 @ sigma2)) is the sum of the square roots of its eigenvalues,
    # which are real and >= 0 up to rounding
    eigenvalues = np.real(np.linalg.eigvals(sigma1 @ sigma2)).clip(min=0)
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.sqrt(eigenvalues).sum())
