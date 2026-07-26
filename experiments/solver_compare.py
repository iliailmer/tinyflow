"""
Compare ODE solvers and time schedules at fixed NFE budgets on MNIST.

Each cell uses the same starting noise (fixed seed) so visual differences
trace to the solver/schedule, not the seed.

NFE accounting (model forward calls):
    Euler  : 1 eval/step  -> N = NFE
    Heun   : 2 evals/step -> N = NFE / 2
    RK4    : 4 evals/step -> N = NFE / 4

Configs at each NFE budget:
    Euler uniform              t_k = k/N
    Euler back-loaded (p=2.5)  t_k = 1 - (1 - k/N)^p
    Heun uniform
    RK4 uniform

Usage:
    uv run experiments/solver_compare.py
    uv run experiments/solver_compare.py --nfes 8 12 20 40 --p 2.5
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

from tinyflow.nn import UNetTinygrad
from tinyflow.solver import RK4, Euler, Heun
from tinyflow.trainer import BaseTrainer
from tinyflow.utils import preprocess_time_mnist

plt.style.use("ggplot")
mlflow.set_tracking_uri("sqlite:///mlflow.db")


CONFIGS = [
    ("euler_uniform", "Euler uniform", "euler", 1.0, 1),
    ("euler_backloaded", "Euler back-loaded", "euler", None, 1),
    ("heun_uniform", "Heun uniform", "heun", 1.0, 2),
    ("rk4_uniform", "RK4 uniform", "rk4", 1.0, 4),
]


def make_solver(name: str, model):
    if name == "euler":
        return Euler(model, preprocess_hook=preprocess_time_mnist)
    if name == "heun":
        return Heun(model, preprocess_hook=preprocess_time_mnist)
    if name == "rk4":
        return RK4(model, preprocess_hook=preprocess_time_mnist)
    raise ValueError(name)


def integrate(solver, x0, ts: np.ndarray):
    x = x0
    for i in range(len(ts) - 1):
        t = (T.zeros(1) + float(ts[i])).contiguous()
        h = float(ts[i + 1] - ts[i])
        x = solver.sample(h, t, x).realize()
    return x


def make_combined_figure(results, nfes, path: str):
    n_rows = len(nfes)
    n_cols = len(CONFIGS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    for r, nfe in enumerate(nfes):
        for c, (key, label, _, _, _) in enumerate(CONFIGS):
            ax = axes[r, c]
            x_np = results.get((key, nfe))
            if x_np is None:
                ax.text(
                    0.5,
                    0.5,
                    "skipped\n(N<1)",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="gray",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.imshow(make_grid(normalize_for_plot(x_np)), cmap="gray")
                ax.axis("off")
            ax.set_title(f"{label}\nNFE={nfe}", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="model_mnist_unet_linear.safetensors")
    parser.add_argument("--nfes", type=int, nargs="+", default=[8, 12, 20, 40])
    parser.add_argument("--p", type=float, default=2.5)
    parser.add_argument("--n-samples", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment", default="solver_compare")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    in_ch = 1
    time_embed_dim = detect_time_embed_dim(args.model_path, in_ch)
    model = UNetTinygrad(in_ch, in_ch, time_embed_dim=time_embed_dim)
    BaseTrainer.load_model(model, args.model_path)
    T.training = False

    mlflow.set_experiment(args.experiment)
    results = {}

    with mlflow.start_run(run_name=f"compare_p{args.p}") as parent:
        mlflow.set_tags({"kind": "solver_compare", "dataset": "mnist"})
        mlflow.log_params(
            {
                "model_path": args.model_path,
                "p": args.p,
                "n_samples": args.n_samples,
                "seed": args.seed,
                "nfes": ",".join(str(s) for s in args.nfes),
            }
        )

        for nfe in args.nfes:
            for key, label, solver_name, default_p, evals_per_step in CONFIGS:
                p_val = default_p if default_p is not None else args.p
                N = nfe // evals_per_step
                if N < 1:
                    print(f"  skip {key} NFE={nfe}: would need N={N}")
                    continue

                T.manual_seed(args.seed)
                x0 = T.randn(args.n_samples, 1, 28, 28).realize()
                solver = make_solver(solver_name, model)
                ts = schedule_grid(N, p_val)

                with mlflow.start_run(run_name=f"{key}_NFE{nfe}", nested=True) as child:
                    mlflow.set_tags({"config": key, "kind": "solver_compare"})
                    mlflow.log_params(
                        {
                            "config": key,
                            "solver": solver_name,
                            "p": p_val,
                            "N": N,
                            "NFE": nfe,
                        }
                    )

                    x_final = integrate(solver, x0, ts)
                    x_np = x_final.numpy()
                    results[(key, nfe)] = x_np

                    grid_path = f"/tmp/grid_{key}_NFE{nfe}.png"
                    save_grid_figure(x_np, f"{label} NFE={nfe}", grid_path)
                    mlflow.log_artifact(grid_path)

                    print(f"  {key} NFE={nfe} N={N} -> run {child.info.run_id}")

        combined_path = "/tmp/solver_compare_mnist.png"
        make_combined_figure(results, args.nfes, combined_path)
        mlflow.log_artifact(combined_path)
        print(f"parent run: {parent.info.run_id}")
        print(f"combined: {combined_path}")


if __name__ == "__main__":
    main()
