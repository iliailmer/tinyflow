import numpy as np
import pytest
from tinygrad.tensor import Tensor as T

from tinyflow.time_sampler import BaseTimeSampler, LogitNormalSampler, UniformTimeSampler


class TestBaseTimeSampler:
    def test_sample_not_implemented(self):
        """Base class sample() must be overridden by subclasses"""
        sampler = BaseTimeSampler()
        with pytest.raises(NotImplementedError):
            sampler.sample(4, 1)


class TestUniformTimeSampler:
    def test_default_output_shape(self):
        """Sampled tensor shape matches requested shape"""
        sampler = UniformTimeSampler()
        t = sampler.sample(16, 1)

        assert t.shape == (16, 1)

    def test_default_bounds(self):
        """Default low=0.0, high=1.0 bounds all samples in [0, 1)"""
        T.manual_seed(0)
        sampler = UniformTimeSampler()
        t = sampler.sample(1000, 1).numpy()

        assert t.min() >= 0.0
        assert t.max() < 1.0

    def test_custom_bounds(self):
        """Custom low/high bounds are respected"""
        T.manual_seed(0)
        sampler = UniformTimeSampler(low=0.2, high=0.7)
        t = sampler.sample(1000, 1).numpy()

        assert t.min() >= 0.2
        assert t.max() < 0.7

    def test_legacy_clamp_range(self):
        """low=0.0, high=0.99 reproduces the historical `T.rand(...) * 0.99` clamp"""
        T.manual_seed(0)
        sampler = UniformTimeSampler(low=0.0, high=0.99)
        t = sampler.sample(1000, 1).numpy()

        assert t.min() >= 0.0
        assert t.max() < 0.99


class TestLogitNormalSampler:
    def test_output_shape(self):
        """Sampled tensor shape matches requested shape"""
        sampler = LogitNormalSampler()
        t = sampler.sample(16, 1)

        assert t.shape == (16, 1)

    def test_output_in_unit_interval(self):
        """sigmoid(.) squashes all samples into the open interval (0, 1)"""
        T.manual_seed(0)
        sampler = LogitNormalSampler(mean=0.0, stddev=1.0)
        t = sampler.sample(1000, 1).numpy()

        assert t.min() > 0.0
        assert t.max() < 1.0

    def test_default_mean_is_centered(self):
        """m=0.0 is the median of the underlying normal, so sigmoid(0)=0.5 is
        the median of the sampled distribution"""
        T.manual_seed(0)
        sampler = LogitNormalSampler(mean=0.0, stddev=1.0)
        t = sampler.sample(5000, 1).numpy()

        median = np.median(t)
        assert abs(median - 0.5) < 0.05

    def test_shifted_mean_skews_distribution(self):
        """A positive m shifts probability mass toward t=1"""
        T.manual_seed(0)
        low_m_sampler = LogitNormalSampler(mean=-2.0, stddev=1.0)
        high_m_sampler = LogitNormalSampler(mean=2.0, stddev=1.0)

        t_low = low_m_sampler.sample(2000, 1).numpy()
        t_high = high_m_sampler.sample(2000, 1).numpy()

        assert t_low.mean() < 0.5 < t_high.mean()

    def test_small_scale_concentrates_near_median(self):
        """A small s concentrates samples tightly around sigmoid(m)"""
        T.manual_seed(0)
        sampler = LogitNormalSampler(mean=0.0, stddev=0.01)
        t = sampler.sample(1000, 1).numpy()

        assert abs(t.mean() - 0.5) < 0.05
        assert t.std() < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
