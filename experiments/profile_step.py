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
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import Context, GlobalCounters
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor as T

from tinyflow.losses import mse
from tinyflow.nn import UNet
from tinyflow.nn_utils.attention import SelfAttention
from tinyflow.path import AffinePath
from tinyflow.path.scheduler import LinearScheduler
from tinyflow.time_sampler import UniformTimeSampler
from tinyflow.trainer import BaseTrainer

IMAGE_SHAPES = {"mnist": (1, 28), "cifar10": (3, 32)}
ANSI = re.compile(r"\x1b\[[0-9;]*m")
KERNEL_LINE = re.compile(
    r"\*\*\*\s+\S+\s+\d+\s+(\S+).*?tm\s+([\d.]+)(us|ms|s)\s*/.*?\(\s*([\d.]+) (GFLOPS|TFLOPS)\s+([\d.]+)\|"
)
UNIT = {"us": 1e-6, "ms": 1e-3, "s": 1.0}


def make_trainer(args):
    m = OmegaConf.load(f"configs/model/{args.model_config}.yaml").model
    m = OmegaConf.merge(m, OmegaConf.from_dotlist(args.set))
    channels, size = IMAGE_SHAPES[args.dataset]
    model = UNet(
        in_channels=channels,
        image_size=size,
        channels=list(m.channels),
        blocks_per_level=m.blocks_per_level,
        attention_resolutions=list(m.attention_resolutions),
        time_embed_dim=m.time_embed_dim,
        dropout=m.get("dropout", 0.0),
    )
    trainer = BaseTrainer(
        model=model,
        dataloader=[],  # pyright: ignore
        optim=Adam(get_parameters(model), lr=1e-4),
        loss_fn=mse,
        path=AffinePath(LinearScheduler()),
        time_sampler=UniformTimeSampler(),
        grad_clip=args.grad_clip,
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
    """Kernels by total GPU time, with GFLOPS and GB/s, and the time share of each kernel kind."""
    totals: dict[str, list] = {}
    for line in ANSI.sub("", debug_output).splitlines():
        if match := KERNEL_LINE.search(line):
            name, value, unit, flops, flops_unit, membw = match.groups()
            seconds = float(value) * UNIT[unit]
            gflops = float(flops) * (1000 if flops_unit == "TFLOPS" else 1)
            row = totals.setdefault(name, [0.0, 0, gflops, float(membw)])
            row[0] += seconds
            row[1] += 1
    rows = sorted(((v[0], v[1], v[2], v[3], k) for k, v in totals.items()), reverse=True)
    kinds: dict[str, float] = {}
    for seconds, _, _, _, name in rows:
        kinds[name.split("_")[0]] = kinds.get(name.split("_")[0], 0.0) + seconds
    return rows[:n], sum(r[0] for r in rows), kinds


class Recorder:
    """Stands in for a UNet block and records the block's input shape."""

    def __init__(self, name, block, records):
        self.name, self.block, self.records = name, block, records

    def __call__(self, h, t_emb=None):
        self.records.append(
            (self.name, self.block, h.shape, None if t_emb is None else t_emb.shape)
        )
        return self.block(h) if isinstance(self.block, SelfAttention) else self.block(h, t_emb)


def block_inputs(model: UNet, shape) -> list:
    """(name, block, input shape, time vector shape) of each UNet block, in call order."""
    records = []
    groups = [(f"enc{i}", blocks) for i, blocks in enumerate(model.encoder)]
    groups += [("mid", model.bottleneck)]
    groups += [(f"dec{i}", blocks) for i, blocks in enumerate(model.decoder)]
    originals = [list(blocks) for _, blocks in groups]
    for prefix, blocks in groups:
        blocks[:] = [Recorder(f"{prefix}.{j}", b, records) for j, b in enumerate(blocks)]
    model(T.randn(*shape), T.rand(shape[0], 1))
    for (_, blocks), original in zip(groups, originals, strict=True):
        blocks[:] = original
    return records


def time_block(block, x_shape, t_shape, repeats: int):
    """Median seconds and FLOPs of forward + backward of one block (input and weight gradients)."""
    params = get_parameters(block)
    attention = isinstance(block, SelfAttention)

    @TinyJit
    def forward_backward(x, t_emb):
        out = block(x) if attention else block(x, t_emb)
        loss = out.square().mean()
        return loss.realize(*loss.gradient(x, *params))

    x = T.randn(*x_shape).realize()
    t_emb = T.randn(*t_shape).realize() if t_shape else T.zeros(1).realize()
    for _ in range(3):
        forward_backward(x, t_emb)
    times = []
    for _ in range(repeats):
        Device[Device.DEFAULT].synchronize()
        start = time.perf_counter()
        forward_backward(x, t_emb)
        Device[Device.DEFAULT].synchronize()
        times.append(time.perf_counter() - start)
    GlobalCounters.reset()
    forward_backward(x, t_emb)
    return statistics.median(times), GlobalCounters.global_ops


def describe(block) -> str:
    if isinstance(block, SelfAttention):
        return f"SelfAttention {block.channels}"
    return f"{type(block).__name__} {block.in_channels}->{block.out_channels}"


def print_layers(model, shape, repeats: int):
    rows = []
    for name, block, x_shape, t_shape in block_inputs(model, shape):
        seconds, ops = time_block(block, x_shape, t_shape, repeats)
        n_params = sum(p.numel() for p in get_parameters(block))
        rows.append((name, describe(block), x_shape[-1], n_params, seconds, ops))
    total = sum(r[4] for r in rows)
    print(f"\nBlocks, forward + backward each on its own ({total * 1e3:.1f} ms in all):")
    print(
        f"  {'block':7s} {'type':28s} {'size':>4s} {'params':>7s} {'ms':>7s} {'share':>6s} {'TFLOPS':>6s}"
    )
    for name, kind, size, n_params, seconds, ops in rows:
        print(
            f"  {name:7s} {kind:28s} {size:4d} {n_params / 1e6:6.2f}M {seconds * 1e3:7.2f}"
            f" {seconds / total:6.1%} {ops / seconds / 1e12:6.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Profile one training step.")
    parser.add_argument("--model-config", default="unet")
    parser.add_argument("--dataset", default="cifar10", choices=list(IMAGE_SHAPES))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--layers", action="store_true", help="also time each UNet block")
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument(
        "--set", nargs="*", default=[], help="model config overrides, e.g. attention_resolutions=[]"
    )
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
    name = " ".join([args.model_config, *args.set])
    print(f"{name} on {args.dataset}: {n_params / 1e6:.1f}M params, batch {shape[0]}")
    print(f"  wall time / step   {wall * 1e3:8.1f} ms  ({shape[0] / wall:.0f} images/s)")
    print(
        f"  GPU time / step    {gpu_s * 1e3:8.1f} ms  (host share {max(0, 1 - gpu_s / wall):.0%})"
    )
    print(f"  compute            {ops / 1e9:8.1f} GFLOP -> {ops / gpu_s / 1e12:.2f} TFLOPS")
    print(f"  memory traffic     {mem / 1e9:8.2f} GB    -> {mem / gpu_s / 1e9:.0f} GB/s")
    print(f"  device memory      {GlobalCounters.mem_used / 1e9:8.2f} GB allocated")

    rows, total, kinds = top_kernels(kernel_output, args.top)
    print(f"\nTop kernels of one step without JIT ({total * 1e3:.1f} ms in all kernels):")
    shares = ", ".join(
        f"{k}: {v / total:.0%}" for k, v in sorted(kinds.items(), key=lambda x: -x[1])
    )
    print(f"  by kernel kind (r = reduce, E = elementwise): {shares}")
    for seconds, count, gflops, membw, name in rows:
        print(
            f"  {seconds * 1e3:7.2f} ms {seconds / total:5.1%}  x{count:<3d}"
            f" {gflops:6.0f} GFLOPS {membw:5.0f} GB/s  {name}"
        )

    if args.layers:
        with Context(TRAINING=1):
            print_layers(trainer.model, shape, args.steps)


if __name__ == "__main__":
    main()
