"""
Profile one training step: wall time, GPU time, FLOPs, memory traffic, device memory, top kernels.

Wall time much larger than GPU time means the step is limited by Python and scheduling;
about equal means the kernels are the limit.

Usage:
    uv run experiments/profile_step.py --model-config unet --dataset cifar10
    BEAM=2 uv run experiments/profile_step.py --model-config unet_large --batch-size 128
"""

import argparse
import contextlib
import io
import re
import statistics
import time

from omegaconf import OmegaConf
from tinygrad.helpers import Context, GlobalCounters
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor as T

from tinyflow.losses import mse
from tinyflow.nn import UNet
from tinyflow.path import AffinePath
from tinyflow.path.scheduler import LinearScheduler
from tinyflow.time_sampler import UniformTimeSampler
from tinyflow.trainer import BaseTrainer

IMAGE_SHAPES = {"mnist": (1, 28), "cifar10": (3, 32)}
ANSI = re.compile(r"\x1b\[[0-9;]*m")
KERNEL_LINE = re.compile(r"\*\*\*\s+\S+\s+\d+\s+(\S+).*?tm\s+([\d.]+)(us|ms|s)\s*/")
UNIT = {"us": 1e-6, "ms": 1e-3, "s": 1.0}


def make_trainer(args):
    m = OmegaConf.load(f"configs/model/{args.model_config}.yaml").model
    channels, size = IMAGE_SHAPES[args.dataset]
    model = UNet(
        in_channels=channels,
        image_size=size,
        channels=list(m.channels),
        blocks_per_level=m.blocks_per_level,
        attention_resolutions=list(m.attention_resolutions),
        time_embed_dim=m.time_embed_dim,
    )
    trainer = BaseTrainer(
        model=model,
        dataloader=[],  # pyright: ignore
        optim=Adam(get_parameters(model), lr=1e-4),
        loss_fn=mse,
        path=AffinePath(LinearScheduler()),
        time_sampler=UniformTimeSampler(),
    )
    n_params = sum(p.numel() for p in get_parameters(model))
    return trainer, (args.batch_size, channels, size, size), n_params


def inputs(trainer, shape):
    x = T.rand(*shape).realize()
    t = trainer.time_sampler.sample(shape[0], 1).realize()
    return x, t, T.randn(*shape).realize()


def gpu_pass(step, trainer, shape):
    """Run one step with per-kernel GPU timing; return GPU seconds, FLOPs, bytes, DEBUG=2 output."""
    batch = inputs(trainer, shape)
    GlobalCounters.reset()
    out = io.StringIO()
    with Context(DEBUG=2), contextlib.redirect_stdout(out):
        step(*batch).item()
    return (
        GlobalCounters.time_sum_s,
        GlobalCounters.global_ops,
        GlobalCounters.global_mem,
        out.getvalue(),
    )


def top_kernels(debug_output: str, n: int):
    totals: dict[str, list[float]] = {}
    for line in ANSI.sub("", debug_output).splitlines():
        if match := KERNEL_LINE.search(line):
            name, value, unit = match.groups()
            totals.setdefault(name, []).append(float(value) * UNIT[unit])
    rows = sorted(((sum(v), len(v), k) for k, v in totals.items()), reverse=True)
    return rows[:n], sum(r[0] for r in rows)


def main():
    parser = argparse.ArgumentParser(description="Profile one training step.")
    parser.add_argument("--model-config", default="unet")
    parser.add_argument("--dataset", default="cifar10", choices=list(IMAGE_SHAPES))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    trainer, shape, n_params = make_trainer(args)
    with Context(TRAINING=1):
        for _ in range(3):  # JIT capture
            trainer._jit_train_step(*inputs(trainer, shape)).item()
        walls = []
        for _ in range(args.steps):
            batch = inputs(trainer, shape)
            start = time.perf_counter()
            trainer._jit_train_step(*batch).item()
            walls.append(time.perf_counter() - start)
        gpu_s, ops, mem, _ = gpu_pass(trainer._jit_train_step, trainer, shape)
        _, _, _, kernel_output = gpu_pass(trainer._train_step, trainer, shape)

    wall = statistics.median(walls)
    print(f"{args.model_config} on {args.dataset}: {n_params / 1e6:.1f}M params, batch {shape[0]}")
    print(f"  wall time / step   {wall * 1e3:8.1f} ms  ({shape[0] / wall:.0f} images/s)")
    print(
        f"  GPU time / step    {gpu_s * 1e3:8.1f} ms  (host share {max(0, 1 - gpu_s / wall):.0%})"
    )
    print(f"  compute            {ops / 1e9:8.1f} GFLOP -> {ops / gpu_s / 1e12:.2f} TFLOPS")
    print(f"  memory traffic     {mem / 1e9:8.2f} GB    -> {mem / gpu_s / 1e9:.0f} GB/s")
    print(f"  device memory      {GlobalCounters.mem_used / 1e9:8.2f} GB allocated")

    rows, total = top_kernels(kernel_output, args.top)
    print(f"\nTop kernels of one step without JIT ({total * 1e3:.1f} ms in all kernels):")
    for seconds, count, name in rows:
        print(f"  {seconds * 1e3:7.2f} ms {seconds / total:5.1%}  x{count:<3d} {name}")


if __name__ == "__main__":
    main()
