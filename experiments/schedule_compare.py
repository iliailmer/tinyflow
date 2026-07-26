"""
Compare uniform vs back-loaded time schedules on a trained MNIST model.

Both schedules sample the same Euler ODE; only the time grid differs:
  - uniform:     t_k = k / N
  - back-loaded: t_k = 1 - (1 - k/N)^p   (clusters steps near t=1)

Same starting noise (fixed seed) for every (schedule, N) combo so any
visual difference is attributable to the schedule, not the seed.

Each (schedule, N) is one MLflow run; artifact is a 9-sample grid.
A combined side-by-side figure is logged to a parent run.

Usage:
    uv run experiments/schedule_compare.py
    uv run experiments/schedule_compare.py --steps 10 20 50 --p 2.5
"""

import argparse
import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from _common import (
    detect_time_embed_dim,
    make_grid,
    normalize_for_plot,
    save_grid_figure,
    schedule_grid,
)
from tinygrad.tensor import Tensor as T
from tqdm import tqdm

from tinyflow.nn import UNetTinygrad
from tinyflow.trainer import BaseTrainer
from tinyflow.utils import preprocess_time_mnist

plt.style.use("ggplot")

mlflow.set_tracking_uri("sqlite:///mlflow.db")


def euler_integrate(model, x0, ts: np.ndarray, preprocess):
    x = x0
    for i in tqdm(range(len(ts) - 1), desc=f"integrate N={len(ts) - 1}", leave=False):
        t = (T.zeros(1) + float(ts[i])).contiguous()
        h = float(ts[i + 1] - ts[i])
        v = model(x, preprocess(t, x))
        x = (x + h * v).realize()
    return x


def make_combined_figure(results, p: float, path: str):
    Ns = sorted({N for (_, N) in results})
    fig, axes = plt.subplots(len(Ns), 2, figsize=(8, 4 * len(Ns)))
    if len(Ns) == 1:
        axes = axes[None, :]
    for row, N in enumerate(Ns):
        for col, (label, sched_p) in enumerate([("uniform", 1.0), (f"back-loaded p={p}", p)]):
            x_np = results[(sched_p, N)]
            axes[row, col].imshow(make_grid(normalize_for_plot(x_np)), cmap="gray")
            axes[row, col].set_title(f"{label}, N={N} (NFE={N})", fontsize=11)
            axes[row, col].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="model_mnist_unet_linear.safetensors")
    parser.add_argument("--steps", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument(
        "--p",
        type=float,
        default=2.5,
        help="back-loading exponent; 1=uniform, >1 clusters near t=1",
    )
    parser.add_argument("--n-samples", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment", default="schedule_compare")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    in_ch = 1
    time_embed_dim = detect_time_embed_dim(args.model_path, in_ch)
    model = UNetTinygrad(in_ch, in_ch, time_embed_dim=time_embed_dim)
    BaseTrainer.load_model(model, args.model_path)
    T.training = False

    mlflow.set_experiment(args.experiment)

    schedules = [("uniform", 1.0), ("back-loaded", args.p)]
    results = {}

    with mlflow.start_run(run_name=f"compare_p{args.p}") as parent:
        mlflow.set_tags({"kind": "schedule_compare", "dataset": "mnist"})
        mlflow.log_params(
            {
                "model_path": args.model_path,
                "p": args.p,
                "n_samples": args.n_samples,
                "seed": args.seed,
                "steps": ",".join(str(s) for s in args.steps),
            }
        )

        for N in args.steps:
            for label, sched_p in schedules:
                T.manual_seed(args.seed)
                x0 = T.randn(args.n_samples, 1, 28, 28).realize()

                ts = schedule_grid(N, sched_p)
                with mlflow.start_run(run_name=f"{label}_N{N}", nested=True) as child:
                    mlflow.set_tags({"schedule": label, "kind": "schedule_compare"})
                    mlflow.log_params({"schedule": label, "p": sched_p, "N": N, "NFE": N})

                    x_final = euler_integrate(model, x0, ts, preprocess_time_mnist)
                    x_np = x_final.numpy()
                    results[(sched_p, N)] = x_np

                    grid_path = f"/tmp/grid_{label}_N{N}.png"
                    save_grid_figure(x_np, f"{label}, N={N}", grid_path, figsize=(4, 4.4))
                    mlflow.log_artifact(grid_path)
                    print(f"logged child run {child.info.run_id}: {label} N={N}")

        combined_path = "/tmp/schedule_compare_mnist.png"
        make_combined_figure(results, args.p, combined_path)
        mlflow.log_artifact(combined_path)
        print(f"parent run: {parent.info.run_id}")
        print(f"combined figure: {combined_path}")


if __name__ == "__main__":
    main()
