#!/usr/bin/env python3
"""
Download datasets (Fashion MNIST, CIFAR-10) to the datasets/ directory.

Usage:
    python scripts/download_datasets.py --dataset fashion_mnist
    python scripts/download_datasets.py --dataset cifar10
    python scripts/download_datasets.py --all
"""

import argparse
import gzip
import os
import pickle
import shutil
import tarfile
import urllib.request
from pathlib import Path

DATASETS_DIR = Path("dataset")

# Fashion MNIST URLs (from Zalando's GitHub)
FASHION_MNIST_URLS = {
    "train-images-idx3-ubyte.gz": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
}

# CIFAR-10 URL
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def download_file(url: str, dest: Path, desc: str = "") -> None:
    """Download a file with progress indicator."""
    print(f"Downloading {desc or url}...")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            print(f"\r  Progress: {percent:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook)
    print()


def download_fashion_mnist() -> None:
    """Download and extract Fashion MNIST dataset."""
    dest_dir = DATASETS_DIR / "fashion_mnist"
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("Downloading Fashion MNIST dataset...")
    print("=" * 50)

    for filename, url in FASHION_MNIST_URLS.items():
        gz_path = dest_dir / filename
        final_path = dest_dir / filename.replace(".gz", "")

        if final_path.exists():
            print(f"  {final_path.name} already exists, skipping.")
            continue

        download_file(url, gz_path, filename)

        # Extract gzip file
        print(f"  Extracting {filename}...")
        with gzip.open(gz_path, "rb") as f_in:
            with open(final_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove the .gz file
        gz_path.unlink()

    print(f"\nFashion MNIST downloaded to: {dest_dir}")
    print("Files:")
    for f in sorted(dest_dir.iterdir()):
        print(f"  - {f.name}")


def download_cifar10() -> None:
    """Download and extract CIFAR-10 dataset."""
    dest_dir = DATASETS_DIR / "cifar10"
    dest_dir.mkdir(parents=True, exist_ok=True)

    final_dir = dest_dir / "cifar-10-batches-py"

    print("=" * 50)
    print("Downloading CIFAR-10 dataset...")
    print("=" * 50)

    if final_dir.exists() and any(final_dir.iterdir()):
        print(f"  CIFAR-10 already exists at {final_dir}, skipping.")
        return

    tar_path = dest_dir / "cifar-10-python.tar.gz"
    download_file(CIFAR10_URL, tar_path, "cifar-10-python.tar.gz")

    # Extract tar.gz file
    print("  Extracting cifar-10-python.tar.gz...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)

    # Remove the tar.gz file
    tar_path.unlink()

    print(f"\nCIFAR-10 downloaded to: {final_dir}")
    print("Files:")
    for f in sorted(final_dir.iterdir()):
        print(f"  - {f.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for tinyflow training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_datasets.py --dataset fashion_mnist
  python scripts/download_datasets.py --dataset cifar10
  python scripts/download_datasets.py --all
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["fashion_mnist", "cifar10"],
        help="Dataset to download",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available datasets",
    )

    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.print_help()
        return

    if args.all or args.dataset == "fashion_mnist":
        download_fashion_mnist()

    if args.all or args.dataset == "cifar10":
        download_cifar10()

    print("\nDone!")


if __name__ == "__main__":
    main()
