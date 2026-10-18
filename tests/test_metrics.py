import numpy as np
import pytest

from tinyflow.metrics import frechet_distance


def test_frechet_distance_matches_closed_form_for_independent_features():
    """For diagonal covariances: |mu1 - mu2|^2 + sum((s1 - s2)^2) with standard deviations s."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, size=(200_000, 3))
    b = rng.normal([1.0, 0.0, 0.0], [1.0, 2.0, 3.0], size=(200_000, 3))

    assert frechet_distance(a, a) == pytest.approx(0, abs=1e-8)
    assert frechet_distance(a, b) == pytest.approx(1 + 1 + 4, rel=0.02)
