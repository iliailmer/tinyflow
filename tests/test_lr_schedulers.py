"""Minimal tests for learning rate schedulers."""

import pytest
from tinygrad.nn.optim import Adam
from tinygrad.tensor import Tensor

from tinyflow.nn_utils.lr_scheduler import (
    CosineAnnealingLR,
    NullLRScheduler,
    StepLRScheduler,
    WarmupScheduler,
)


@pytest.fixture
def optimizer():
    """Create a simple optimizer for testing."""
    params = [Tensor.ones(10, 10)]
    return Adam(params, lr=0.1)


def test_null_scheduler_keeps_constant_lr(optimizer):
    """Test that NullLRScheduler maintains constant learning rate."""
    scheduler = NullLRScheduler(optimizer)
    initial_lr = scheduler.get_lr()

    for i in range(10):
        scheduler.step(i)
        assert scheduler.get_lr() == pytest.approx(initial_lr)


def test_step_scheduler_decays(optimizer):
    """Test that StepLRScheduler decays learning rate."""
    scheduler = StepLRScheduler(optimizer, gamma=0.5, step_size=10)
    initial_lr = scheduler.get_lr()

    # Before step_size, LR should be unchanged
    scheduler.step(5)
    assert scheduler.get_lr() == pytest.approx(initial_lr)

    # At step_size, LR should decay
    scheduler.step(10)
    assert scheduler.get_lr() == pytest.approx(initial_lr * 0.5)

    # At 2 * step_size, LR should decay again
    scheduler.step(20)
    assert scheduler.get_lr() == pytest.approx(initial_lr * 0.25)


def test_cosine_annealing_varies(optimizer):
    """Test that CosineAnnealingLR varies learning rate."""
    scheduler = CosineAnnealingLR(optimizer, t_max=100, eta_min=0.01, warm=False)
    initial_lr = scheduler.get_lr()

    # At t=0, should be at max
    scheduler.step(0)
    assert scheduler.get_lr() == pytest.approx(initial_lr, rel=0.01)

    # At t=t_max/2, should be at minimum
    scheduler.step(50)
    assert scheduler.get_lr() < initial_lr
    assert scheduler.get_lr() >= 0.01

    # At t=t_max, should be back near minimum (eta_min)
    scheduler.step(100)
    assert scheduler.get_lr() == pytest.approx(0.01, abs=0.01)


def test_cosine_warm_restarts(optimizer):
    """Test that warm restarts reset the cosine cycle."""
    scheduler = CosineAnnealingLR(optimizer, t_max=10, warm=True)

    # First cycle
    scheduler.step(0)
    lr_at_0 = scheduler.get_lr()

    # After one full cycle, should restart
    scheduler.step(10)
    lr_at_10 = scheduler.get_lr()

    assert lr_at_0 == pytest.approx(lr_at_10, rel=0.01)


def test_warmup_scheduler_linear_ramp(optimizer):
    """Test that WarmupScheduler does linear warmup."""
    base_scheduler = StepLRScheduler(optimizer, gamma=0.1, step_size=1000)
    scheduler = WarmupScheduler(
        optimizer, base_scheduler=base_scheduler, warmup_steps=10, warmup_start_lr=0.0
    )

    # At step 0, should be at warmup_start_lr
    scheduler.step(0)
    assert scheduler.get_lr() == pytest.approx(0.0, abs=1e-6)

    # At step 5 (midpoint), should be halfway to base_lr
    scheduler.step(5)
    assert scheduler.get_lr() == pytest.approx(0.05, rel=0.01)

    # At step 10 (end of warmup), should delegate to base scheduler
    scheduler.step(10)
    # Should be at base_lr since base scheduler hasn't decayed yet
    assert scheduler.get_lr() == pytest.approx(0.1, rel=0.01)


def test_state_dict_save_load(optimizer):
    """Test that scheduler state can be saved and loaded."""
    scheduler = CosineAnnealingLR(optimizer, t_max=100, eta_min=0.01)
    scheduler.step(50)

    state = scheduler.state_dict()
    assert "base_lr" in state
    assert "t_max" in state

    # Create new scheduler and load state
    new_optimizer = Adam([Tensor.ones(10, 10)], lr=0.5)
    new_scheduler = CosineAnnealingLR(new_optimizer, t_max=10)
    new_scheduler.load_state_dict(state)

    assert new_scheduler.base_lr == pytest.approx(scheduler.base_lr)
    assert new_scheduler.t_max == scheduler.t_max
