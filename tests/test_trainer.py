import numpy as np
import pytest
from tinygrad import nn
from tinygrad.helpers import Context
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters, safe_load
from tinygrad.tensor import Tensor as T

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


def test_ema_weights_are_swapped_in_and_restored():
    trainer = make_trainer(ema_decay=0.9)
    weight = trainer.model.linear.weight
    initial = weight.numpy().copy()

    with Context(TRAINING=1):
        for i in range(3):
            trainer.epoch(i)
    raw = weight.numpy().copy()

    with trainer.ema_weights():
        ema = weight.numpy().copy()

    assert not np.allclose(ema, raw)
    assert not np.allclose(ema, initial)
    # The average is behind the raw weights, between the start and the current value
    assert np.abs(ema - raw).max() < np.abs(initial - raw).max()
    assert np.allclose(weight.numpy(), raw)


def test_checkpoint_during_training_saves_both_weights_and_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr("tinyflow.trainer.mlflow.log_metrics", lambda *args, **kwargs: None)

    def run(checkpoint_after: int | None):
        T.manual_seed(0)
        np.random.seed(0)
        trainer = make_trainer(ema_decay=0.9)
        trainer.num_epochs = 4

        def on_epoch_end(epoch_idx):
            if epoch_idx == checkpoint_after:
                trainer.save_weights(str(tmp_path / "ckpt.safetensors"))
                with trainer.ema_weights():
                    expected["ema"] = trainer.model.linear.weight.numpy().copy()
                expected["raw"] = trainer.model.linear.weight.numpy().copy()

        trainer.train(on_epoch_end=on_epoch_end)
        return trainer.model.linear.weight.numpy()

    expected = {}
    with_checkpoint = run(checkpoint_after=1)
    without_checkpoint = run(checkpoint_after=None)

    assert np.allclose(with_checkpoint, without_checkpoint)
    ema = safe_load(str(tmp_path / "ckpt.safetensors"))["linear.weight"].numpy()
    raw = safe_load(str(tmp_path / "ckpt_raw.safetensors"))["linear.weight"].numpy()
    assert np.allclose(ema, expected["ema"])
    assert np.allclose(raw, expected["raw"])
    assert not np.allclose(ema, raw)
