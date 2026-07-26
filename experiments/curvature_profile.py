"""
Curvature profile for trained flow-matching models. Logs to MLflow.

At each Euler step, finite-difference-estimate:
  - ||v||              velocity magnitude
  - ||d_t v||          partial wrt time
  - ||J_x v · v_hat||  directional derivative along streamline
  - ||J_x v · eta||    Hutchinson-style random-direction proxy
  - lte_proxy = ||d_t v|| + ||J_x v · v_hat|| * ||v||  (Euler local truncation / h^2)

Usage:
    uv run experiments/curvature_profile.py --dataset moons
    uv run experiments/curvature_profile.py --dataset mnist --num-steps 50
"""

import argparse
import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from _common import detect_time_embed_dim, schedule_grid
from tinygrad.tensor import Tensor as T
from tqdm import tqdm

from tinyflow.nn import NeuralNetwork, UNetTinygrad
from tinyflow.trainer import BaseTrainer
from tinyflow.utils import preprocess_time_cifar, preprocess_time_mnist, preprocess_time_moons

plt.style.use("ggplot")

mlflow.set_tracking_uri("sqlite:///mlflow.db")


DATASET_CONFIGS = {
    "moons": {
        "shape": (2,),
        "kind": "mlp",
        "preprocess": preprocess_time_moons,
        "default_path": "model_moons_neural_network_linear.safetensors",
        "default_batch": 256,
    },
    "mnist": {
        "shape": (1, 28, 28),
        "kind": "unet",
        "in_channels": 1,
        "preprocess": preprocess_time_mnist,
        "default_path": "model_mnist_unet_linear.safetensors",
        "default_batch": 32,
    },
    "fashion_mnist": {
        "shape": (1, 28, 28),
        "kind": "unet",
        "in_channels": 1,
        "preprocess": preprocess_time_mnist,
        "default_path": "model_fashion_mnist_unet_linear.safetensors",
        "default_batch": 32,
    },
    "cifar10": {
        "shape": (3, 32, 32),
        "kind": "unet",
        "in_channels": 3,
        "preprocess": preprocess_time_cifar,
        "default_path": "model_cifar10_unet_linear.safetensors",
        "default_batch": 16,
    },
}


def build_model(dataset: str, model_path: str):
    cfg = DATASET_CONFIGS[dataset]
    if cfg["kind"] == "mlp":
        return NeuralNetwork(in_dim=2, time_embed_dim=64, out_dim=2)
    in_ch = cfg["in_channels"]
    return UNetTinygrad(in_ch, in_ch, time_embed_dim=detect_time_embed_dim(model_path, in_ch))


def per_sample_norm_mean(x: T) -> float:
    flat = x.reshape(x.shape[0], -1)
    return ((flat * flat).sum(axis=-1) + 1e-20).sqrt().mean().numpy().item()


def normalize_per_sample(x: T) -> T:
    flat = x.reshape(x.shape[0], -1)
    mag = ((flat * flat).sum(axis=-1) + 1e-20).sqrt()
    view_shape = (x.shape[0],) + (1,) * (len(x.shape) - 1)
    return x / mag.reshape(view_shape)


def profile(
    dataset: str, model_path: str, num_steps: int, batch_size: int, eps_x: float, eps_t: float
):
    cfg = DATASET_CONFIGS[dataset]
    preprocess = cfg["preprocess"]

    model = build_model(dataset, model_path)
    BaseTrainer.load_model(model, model_path)

    T.training = False
    shape = tuple([batch_size] + list(cfg["shape"]))
    x = T.randn(*shape).realize()
    h = 1.0 / num_steps

    records = []
    for step in tqdm(range(num_steps), desc="profiling"):
        t_val = step * h
        t = (T.zeros(1) + t_val).contiguous()
        t_in = preprocess(t, x)

        v = model(x, t_in).realize()
        v_norm = per_sample_norm_mean(v)

        t_plus_val = min(t_val + eps_t, 1.0)
        t_plus = preprocess((T.zeros(1) + t_plus_val).contiguous(), x)
        v_tplus = model(x, t_plus).realize()
        dvdt_norm = per_sample_norm_mean((v_tplus - v) * (1.0 / max(t_plus_val - t_val, 1e-12)))

        v_hat = normalize_per_sample(v)
        v_at_xv = model(x + eps_x * v_hat, t_in).realize()
        jvv_norm = per_sample_norm_mean((v_at_xv - v) * (1.0 / eps_x))

        eta_hat = normalize_per_sample(T.randn(*shape).realize())
        v_at_eta = model(x + eps_x * eta_hat, t_in).realize()
        jeta_norm = per_sample_norm_mean((v_at_eta - v) * (1.0 / eps_x))

        lte_proxy = dvdt_norm + jvv_norm * v_norm
        record = {
            "t": t_val,
            "v": v_norm,
            "dvdt": dvdt_norm,
            "Jvv": jvv_norm,
            "Jeta": jeta_norm,
            "lte_proxy": lte_proxy,
        }
        mlflow.log_metrics(record, step=step)
        records.append(record)

        x = (x + h * v).realize()

    return records


def make_basic_plot(records, dataset: str):
    ts = [r["t"] for r in records]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(ts, [r["v"] for r in records], label="‖v‖", linewidth=2)
    ax.plot(ts, [r["dvdt"] for r in records], label="‖∂_t v‖", linewidth=2)
    ax.plot(ts, [r["Jvv"] for r in records], label="‖J_x v · v̂‖", linewidth=2)
    ax.plot(ts, [r["Jeta"] for r in records], label="‖J_x v · η̂‖", linewidth=2)
    ax.set_xlabel("t")
    ax.set_ylabel("magnitude")
    ax.set_title(f"Curvature profile — {dataset}")
    ax.legend()

    ax = axes[1]
    ax.plot(ts, [r["lte_proxy"] for r in records], label="LTE proxy", linewidth=2, color="C3")
    ax.set_xlabel("t")
    ax.set_ylabel("LTE proxy / h²")
    ax.set_title("Per-step error growth proxy")
    ax.legend()

    plt.tight_layout()
    return fig


def make_story_plot(records, dataset: str, N: int, p: float):
    ts = np.array([r["t"] for r in records])
    v = np.array([r["v"] for r in records])
    dvdt = np.array([r["dvdt"] for r in records])
    Jvv = np.array([r["Jvv"] for r in records])
    lte = np.array([r["lte_proxy"] for r in records])

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    ax = axes[0]
    ax.plot(ts, v, label="‖v‖  (sample velocity)", linewidth=2.2)
    ax.plot(ts, dvdt, label="‖∂_t v‖", linewidth=2.2)
    ax.plot(ts, Jvv, label="‖J_x v · v̂‖", linewidth=2.2)
    ax.plot(ts, lte, label="LTE proxy", linewidth=2.5, color="C3", linestyle="--")
    ax.set_yscale("log")
    ax.set_ylabel("magnitude (log scale)")
    ax.set_title(f"Magnitude profile along Euler trajectory — {dataset}")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    above = np.where(lte > 2.0 * lte[0])[0]
    if above.size > 0:
        t_thr = ts[above[0]]
        ax.axvspan(t_thr, 1.0, color="orange", alpha=0.12)
        ax.text(
            (t_thr + 1.0) / 2,
            ax.get_ylim()[1] * 0.4,
            f"LTE proxy > 2× start\n(t > {t_thr:.2f})",
            ha="center",
            fontsize=9,
            color="C1",
        )

    ax = axes[1]
    uniform = schedule_grid(N, 1.0)
    backloaded = schedule_grid(N, p)
    ax.scatter(uniform, np.full_like(uniform, 1.0), marker="|", s=400, color="C0")
    ax.scatter(backloaded, np.full_like(backloaded, 0.0), marker="|", s=400, color="C2")
    ax.text(-0.04, 1.0, "uniform", ha="right", va="center", fontsize=10)
    ax.text(-0.04, 0.0, f"back-loaded\n(p={p})", ha="right", va="center", fontsize=10)
    ax.set_yticks([])
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlim(-0.10, 1.02)
    ax.set_xlabel("t")
    ax.set_title(f"Step allocation comparison (N={N})")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS), required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eps-x", type=float, default=1e-2)
    parser.add_argument("--eps-t", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--schedule-N",
        type=int,
        default=20,
        help="N for the step-allocation strip in the story plot",
    )
    parser.add_argument(
        "--schedule-p",
        type=float,
        default=2.5,
        help="back-loading exponent for the comparison strip",
    )
    parser.add_argument("--experiment", default="curvature_profiling")
    args = parser.parse_args()

    T.manual_seed(args.seed)

    cfg = DATASET_CONFIGS[args.dataset]
    model_path = args.model_path or cfg["default_path"]
    batch_size = args.batch_size or cfg["default_batch"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"{args.dataset}_N{args.num_steps}"):
        mlflow.set_tags({"dataset": args.dataset, "kind": "curvature"})
        mlflow.log_params(
            {
                "dataset": args.dataset,
                "model_path": model_path,
                "num_steps": args.num_steps,
                "batch_size": batch_size,
                "eps_x": args.eps_x,
                "eps_t": args.eps_t,
                "seed": args.seed,
            }
        )

        records = profile(
            args.dataset, model_path, args.num_steps, batch_size, args.eps_x, args.eps_t
        )

        basic = make_basic_plot(records, args.dataset)
        mlflow.log_figure(basic, f"curvature_{args.dataset}.png")
        plt.close(basic)

        story = make_story_plot(records, args.dataset, args.schedule_N, args.schedule_p)
        mlflow.log_figure(story, f"story_{args.dataset}.png")
        plt.close(story)

        lte = np.array([r["lte_proxy"] for r in records])
        ratio = float(lte.max() / max(lte.min(), 1e-12))
        mlflow.log_metrics(
            {
                "summary_lte_max_over_min": ratio,
                "summary_lte_max": float(lte.max()),
                "summary_lte_min": float(lte.min()),
            }
        )
        print(f"done. ratio max/min(lte) = {ratio:.2f}")


if __name__ == "__main__":
    main()
