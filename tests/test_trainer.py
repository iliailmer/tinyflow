import numpy as np
import pytest
from tinygrad import nn
from tinygrad.helpers import Context
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters

from tinyflow.losses import mse
from tinyflow.nn import BaseNeuralNetwork
from tinyflow.nn_utils.lr_scheduler import StepLRScheduler
from tinyflow.path import AffinePath
from tinyflow.path.scheduler import LinearScheduler
from tinyflow.time_sampler import UniformTimeSampler
from tinyflow.trainer import BaseTrainer


class TinyModel(BaseNeuralNetwork):
    def __init__(self):
        self.linear = nn.Linear(5, 4)

    def __call__(self, x, t):
        return self.linear(x.flatten(1).cat(t, dim=1)).reshape(x.shape)


class Batches:
    """10 samples in batches of 4, so the last batch is smaller."""

    def __init__(self):
        self.images = np.random.rand(10, 1, 2, 2).astype(np.float32)

    def __len__(self):
        return 3

    def __iter__(self):
        return iter([(self.images[i : i + 4], None) for i in range(0, 10, 4)])


def make_trainer(lr_scheduler_gamma=None, **kwargs):
    model = TinyModel()
    optim = Adam(get_parameters(model), lr=0.01)
    lr_scheduler = None
    if lr_scheduler_gamma is not None:
        lr_scheduler = StepLRScheduler(optim, gamma=lr_scheduler_gamma, step_size=1)
    return BaseTrainer(
        model=model,
        dataloader=Batches(),
        optim=optim,
        loss_fn=mse,
        path=AffinePath(LinearScheduler()),
        time_sampler=UniformTimeSampler(),
        lr_scheduler=lr_scheduler,
        **kwargs,
    )


@pytest.mark.parametrize("accumulation_steps", [1, 2])
def test_epoch_updates_the_model(accumulation_steps):
    trainer = make_trainer(gradient_accumulation_steps=accumulation_steps)
    before = trainer.model.linear.weight.numpy().copy()

    with Context(TRAINING=1):
        losses = [trainer.epoch(i) for i in range(4)]

    assert np.isfinite(losses).all()
    assert len(trainer._losses) == 12
    assert not np.allclose(before, trainer.model.linear.weight.numpy())


def test_lr_scheduler_changes_reach_the_compiled_step():
    """With the learning rate near 0 after the first steps, later epochs change nothing."""
    trainer = make_trainer(lr_scheduler_gamma=1e-9)

    with Context(TRAINING=1):
        trainer.epoch(0)
        after_first_epoch = trainer.model.linear.weight.numpy().copy()
        trainer.epoch(1)
        trainer.epoch(2)

    assert np.allclose(after_first_epoch, trainer.model.linear.weight.numpy())
