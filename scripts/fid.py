"""
FID and KID of generated images against the training images of the dataset.

Features are Inception-v3 in the clean-fid "legacy_tensorflow" mode, the setting of most
published CIFAR-10 numbers. The two steps run in two processes, so tinygrad and PyTorch
never use the GPU at the same time:

    # 1. sample with tinygrad, save uint8 images
    uv run scripts/fid.py sample --weights weights/cifar10_unet_50ep.safetensors \
        --n 10000 --out outputs/fid/unet_50ep.npz

    # 2. FID and KID with PyTorch
    uv run --group eval scripts/fid.py score outputs/fid/unet_50ep.npz

The first score call per dataset computes the reference features of the training set and
caches them in dataset/fid_ref/. `score --real-test` scores the CIFAR-10 test set instead
of samples, as a check of the setup.

FID needs about 50k samples for numbers comparable with papers; with fewer samples it is
biased upwards. KID is unbiased and suits 5k-10k samples. Compare runs at the same n.
"""

import argparse
import json
import os

import numpy as np

MODE = "legacy_tensorflow"
REF_DIR = "dataset/fid_ref"


def sample(args):
    from _common import IMAGE_SHAPES, create_model, create_solver
    from omegaconf import OmegaConf
    from tinygrad.helpers import Context
    from tinygrad.tensor import Tensor as T

    from tinyflow.solver import uniform_time_grid
    from tinyflow.trainer import BaseTrainer
    from tinyflow.utils import to_display

    model_cfg = OmegaConf.load(f"configs/model/{args.model_config}.yaml").model
    cfg = OmegaConf.create(
        {"model": model_cfg, "dataset": {"name": args.dataset}, "solver": {"type": args.solver}}
    )
    model = create_model(cfg)
    BaseTrainer.load_model(model, args.weights)
    solver = create_solver(cfg, model)
    channels, size = IMAGE_SHAPES[args.dataset]
    value_range = args.value_range or OmegaConf.load(
        f"configs/dataset/{args.dataset}.yaml"
    ).dataset.get("value_range", [0.0, 1.0])
    value_range = (float(value_range[0]), float(value_range[1]))
    args.value_range = list(value_range)

    T.manual_seed(args.seed)
    batches = []
    with Context(TRAINING=0):
        for start in range(0, args.n, args.batch_size):
            n = min(args.batch_size, args.n - start)
            x = solver.solve(T.randn(n, channels, size, size), uniform_time_grid(args.steps))
            images = to_display(x.numpy(), value_range)
            batches.append((images * 255).round().astype(np.uint8))
            print(f"{start + n}/{args.n}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, images=np.concatenate(batches), meta=json.dumps(vars(args)))
    print(f"Saved {args.n} samples to {args.out}")


def load_real(dataset: str, train: bool) -> np.ndarray:
    """Real images as uint8 of shape (N, C, H, W)."""
    if dataset == "cifar10":
        from tinyflow.dataloader import CIFAR10Loader

        images = CIFAR10Loader(train=train, shuffle=False, normalize=False).images
        return images.astype(np.uint8)
    if dataset == "mnist":
        from tinyflow.dataloader import MNISTLoader

        if not train:
            raise ValueError("No MNIST test split on disk")
        return (MNISTLoader(shuffle=False).images * 255).round().astype(np.uint8)
    raise ValueError(f"Unknown dataset: {dataset}")


def features(images: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    import torch
    from cleanfid.features import build_feature_extractor
    from tqdm import tqdm

    model = build_feature_extractor(MODE, torch.device(device), use_dataparallel=False)
    if images.shape[1] == 1:
        images = images.repeat(3, axis=1)
    feats = []
    with torch.no_grad():
        for start in tqdm(range(0, len(images), batch_size), desc="Inception features"):
            batch = torch.from_numpy(images[start : start + batch_size]).float().to(device)
            feats.append(model(batch).cpu().numpy())
    return np.concatenate(feats)


def reference(dataset: str, batch_size: int, device: str) -> np.ndarray:
    path = os.path.join(REF_DIR, f"{dataset}_train_{MODE}.npy")
    if not os.path.exists(path):
        os.makedirs(REF_DIR, exist_ok=True)
        np.save(path, features(load_real(dataset, train=True), batch_size, device))
    return np.load(path)


def score(args):
    import torch
    from cleanfid.fid import kernel_distance

    from tinyflow.metrics import frechet_distance

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    if args.real_test:
        dataset, images, meta = args.dataset, load_real(args.dataset, train=False), {}
        out = os.path.join("outputs", "fid", f"{dataset}_real_test_scores.json")
    else:
        data = np.load(args.samples)
        images, meta = data["images"], json.loads(str(data["meta"]))
        dataset = meta["dataset"]
        out = os.path.splitext(args.samples)[0] + "_scores.json"

    ref = reference(dataset, args.batch_size, device)
    feats = features(images, args.batch_size, device)
    np.random.seed(0)
    result = {
        "fid": frechet_distance(feats, ref),
        "kid_x1000": 1000 * kernel_distance(ref, feats),
        "n": len(images),
        "dataset": dataset,
        "mode": MODE,
        "sample_args": meta,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ["fid", "kid_x1000", "n"]}))
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="FID and KID of generated images.")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sample", help="generate samples with tinygrad")
    s.add_argument("--weights", required=True)
    s.add_argument("--dataset", default="cifar10", choices=["cifar10", "mnist"])
    s.add_argument("--model-config", default="unet", help="configs/model/<name>.yaml")
    s.add_argument(
        "--value-range",
        type=float,
        nargs=2,
        default=None,
        help="value range of the training images; default from configs/dataset/<dataset>.yaml",
    )
    s.add_argument("--solver", default="heun", choices=["euler", "heun", "midpoint", "rk4"])
    s.add_argument("--steps", type=int, default=50)
    s.add_argument("--n", type=int, default=10_000)
    s.add_argument("--batch-size", type=int, default=500)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)

    c = sub.add_parser("score", help="FID and KID with PyTorch (eval group)")
    c.add_argument("samples", nargs="?", help=".npz from the sample command")
    c.add_argument("--real-test", action="store_true", help="score the real test set instead")
    c.add_argument("--dataset", default="cifar10", help="dataset for --real-test")
    c.add_argument("--batch-size", type=int, default=200)
    c.add_argument("--device", default=None, help="torch device; default mps if available")

    args = parser.parse_args()
    if args.command == "sample":
        sample(args)
    elif args.real_test == bool(args.samples):
        parser.error("score needs a samples file or --real-test, not both")
    else:
        score(args)


if __name__ == "__main__":
    main()
