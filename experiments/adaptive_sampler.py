"""
Experiment 1 (HOMOTOPY_EXPERIMENTS.md): Condition-Number-Adaptive Sampler.

Replaces fixed-step Euler integration with curvature-aware step sizing:

    kappa ~= ||v_theta(x + eps*u, t) - v_theta(x, t)|| / eps      (u: random unit dir)
    dt    <- dt_base / (1 + beta * kappa)

Compares, at a matched NFE budget, three configs on MNIST:
    reference   fixed uniform Euler at --reference-nfe (gold standard, e.g. 50)
    fixed       fixed uniform Euler at the tested budget (control)
    adaptive    curvature-adaptive Euler at the tested budget

Usage:
    uv run experiments/adaptive_sampler.py
    uv run experiments/adaptive_sampler.py --nfe-budgets 10 20 --beta 5.0
"""

import argparse
import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from _common import (
    compute_fid_vs_real,
    detect_time_embed_dim,
    normalize_per_sample,
    per_sample_norm_mean,
    save_grid_figure,
    schedule_grid,
)
from tinygrad.tensor import Tensor as T

from tinyflow.nn import UNetTinygrad
from tinyflow.trainer import BaseTrainer
from tinyflow.utils import preprocess_time_mnist

plt.style.use("ggplot")
mlflow.set_tracking_uri("sqlite:///mlflow.db")


def integrate_fixed(model, preprocess, x0, N: int):
    x = x0
    ts = schedule_grid(N, 1.0)
    nfe = 0
    for i in range(len(ts) - 1):
        t = (T.zeros(1) + float(ts[i])).contiguous()
        t_in = preprocess(t, x)
        v = model(x, t_in).realize()
        nfe += 1
        h = float(ts[i + 1] - ts[i])
        x = (x + h * v).realize()
    return x, {"steps": N, "nfe": nfe}


def integrate_adaptive(
    model,
    preprocess,
    x0,
    nfe_target: int,
    beta: float,
    eps_x: float,
    kappa_every: int,
    max_steps: int = 500,
):
    """Runs to completion (t reaches 1), unlike a fixed-N loop. `nfe_target`
    only sets the nominal dt_base (1/nfe_target); curvature can make the
    *actual* NFE spent come out higher or lower than that nominal target —
    that actual count is what should be compared against a fixed baseline."""
    x = x0
    t_val = 0.0
    nfe = 0
    steps = 0
    kappa = 0.0
    dt_base = 1.0 / nfe_target
    log = {"t": [], "dt": [], "kappa": []}

    while t_val < 1.0 - 1e-9 and steps < max_steps:
        t = (T.zeros(1) + t_val).contiguous()
        t_in = preprocess(t, x)
        v = model(x, t_in).realize()
        nfe += 1

        if steps % kappa_every == 0:
            u_hat = normalize_per_sample(T.randn(*x.shape).realize())
            v_pert = model(x + eps_x * u_hat, t_in).realize()
            nfe += 1
            kappa = per_sample_norm_mean((v_pert - v) * (1.0 / eps_x))

        dt = dt_base / (1.0 + beta * kappa)
        dt = min(dt, 1.0 - t_val)
        x = (x + dt * v).realize()
        t_val += dt
        steps += 1
        log["t"].append(t_val)
        log["dt"].append(dt)
        log["kappa"].append(kappa)

    return x, {"steps": steps, "nfe": nfe, **log}


def make_dt_plot(log: dict, budget: int) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(log["t"], log["dt"], marker="o", markersize=3)
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("dt")
    axes[0].set_title(f"Adaptive step size (budget NFE={budget})")

    axes[1].plot(log["t"], log["kappa"], marker="o", markersize=3, color="C1")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("kappa (curvature estimate)")
    axes[1].set_title("Curvature along trajectory")

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="weights/model_mnist_unet_linear.safetensors")
    parser.add_argument(
        "--nfe-budgets",
        type=int,
        nargs="+",
        default=[10, 20],
        help="Nominal NFE targets (sets dt_base=1/target); adaptive runs to "
        "completion so actual NFE spent may differ, and the fixed baseline "
        "is matched to that actual value post-hoc.",
    )
    parser.add_argument("--reference-nfe", type=int, default=50)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--kappa-every", type=int, default=4)
    parser.add_argument("--eps-x", type=float, default=1e-2)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment", default="homotopy_experiments")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    in_ch = 1
    time_embed_dim = detect_time_embed_dim(args.model_path, in_ch)
    model = UNetTinygrad(in_ch, in_ch, time_embed_dim=time_embed_dim)
    BaseTrainer.load_model(model, args.model_path)
    T.training = False

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name="1_adaptive_sampler") as parent:
        mlflow.set_tags({"exp": "1_adaptive_sampler", "dataset": "mnist"})
        mlflow.log_params(
            {
                "model_path": args.model_path,
                "reference_nfe": args.reference_nfe,
                "beta": args.beta,
                "kappa_every": args.kappa_every,
                "eps_x": args.eps_x,
                "n_samples": args.n_samples,
                "seed": args.seed,
                "nfe_budgets": ",".join(str(b) for b in args.nfe_budgets),
            }
        )

        # gold-standard reference
        T.manual_seed(args.seed)
        x0_ref = T.randn(args.n_samples, 1, 28, 28).realize()
        with mlflow.start_run(run_name=f"reference_NFE{args.reference_nfe}", nested=True):
            x_ref, info = integrate_fixed(model, preprocess_time_mnist, x0_ref, args.reference_nfe)
            x_ref_np = x_ref.numpy()
            fid_ref = compute_fid_vs_real(x_ref_np, "mnist", args.n_samples)
            mlflow.log_params({"config": "reference", **info})
            if fid_ref is not None:
                mlflow.log_metric("fid", fid_ref)
            grid_path = "/tmp/adaptive_sampler_reference.png"
            save_grid_figure(x_ref_np, f"Reference NFE={args.reference_nfe}", grid_path)
            mlflow.log_artifact(grid_path)
            print(f"reference NFE={args.reference_nfe}: fid={fid_ref}")

        for target in args.nfe_budgets:
            # adaptive runs to completion; the target only sets the nominal
            # dt_base, actual NFE spent is discovered here
            T.manual_seed(args.seed)
            x0 = T.randn(args.n_samples, 1, 28, 28).realize()

            with mlflow.start_run(run_name=f"adaptive_target{target}", nested=True):
                x_adapt, info = integrate_adaptive(
                    model,
                    preprocess_time_mnist,
                    x0,
                    target,
                    args.beta,
                    args.eps_x,
                    args.kappa_every,
                )
                actual_nfe = info["nfe"]
                x_adapt_np = x_adapt.numpy()
                fid_adapt = compute_fid_vs_real(x_adapt_np, "mnist", args.n_samples)
                mlflow.log_params(
                    {
                        "config": "adaptive",
                        "nfe_target": target,
                        "steps": info["steps"],
                        "nfe": actual_nfe,
                        "beta": args.beta,
                        "kappa_every": args.kappa_every,
                    }
                )
                if fid_adapt is not None:
                    mlflow.log_metric("fid", fid_adapt)
                mlflow.log_metric("mean_dt", float(np.mean(info["dt"])))
                mlflow.log_metric("mean_kappa", float(np.mean(info["kappa"])))

                grid_path = f"/tmp/adaptive_sampler_adaptive_target{target}.png"
                save_grid_figure(
                    x_adapt_np, f"Adaptive target={target} (actual NFE={actual_nfe})", grid_path
                )
                mlflow.log_artifact(grid_path)

                dt_fig = make_dt_plot(info, target)
                mlflow.log_figure(dt_fig, f"dt_profile_target{target}.png")
                plt.close(dt_fig)

                print(
                    f"adaptive target={target} -> actual steps={info['steps']}, "
                    f"actual nfe={actual_nfe}: fid={fid_adapt}"
                )

            # fixed baseline at the SAME actual NFE the adaptive run spent
            T.manual_seed(args.seed)
            x0 = T.randn(args.n_samples, 1, 28, 28).realize()

            with mlflow.start_run(run_name=f"fixed_NFE{actual_nfe}", nested=True):
                x_fixed, info = integrate_fixed(model, preprocess_time_mnist, x0, actual_nfe)
                x_fixed_np = x_fixed.numpy()
                fid_fixed = compute_fid_vs_real(x_fixed_np, "mnist", args.n_samples)
                mlflow.log_params({"config": "fixed", "matched_to_target": target, **info})
                if fid_fixed is not None:
                    mlflow.log_metric("fid", fid_fixed)
                grid_path = f"/tmp/adaptive_sampler_fixed_NFE{actual_nfe}.png"
                save_grid_figure(x_fixed_np, f"Fixed NFE={actual_nfe}", grid_path)
                mlflow.log_artifact(grid_path)
                print(f"fixed NFE={actual_nfe} (matched to target={target}): fid={fid_fixed}")

        print(f"parent run: {parent.info.run_id}")
        print("view with: uv run mlflow ui  (tracking uri: sqlite:///mlflow.db)")


if __name__ == "__main__":
    main()
