#!/usr/bin/env python3
"""
Train simple CNN classifiers for FID feature extraction.

Usage:
    uv run scripts/train_fid_classifier.py --dataset mnist
    uv run scripts/train_fid_classifier.py --dataset fashion_mnist
    uv run scripts/train_fid_classifier.py --dataset cifar10
"""

import argparse
import os

import numpy as np
from loguru import logger
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_state_dict, safe_save
from tinygrad.tensor import Tensor as T
from tqdm import tqdm

from tinyflow.dataloader import CIFAR10Loader, FashionMNISTLoader, MNISTLoader
from tinyflow.metrics import LeNetCIFAR10, LeNetMNIST


def train_classifier(
    model,
    dataloader,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
):
    """Train a classifier with cross-entropy loss."""
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_images, batch_labels in pbar:
            # Convert to tensors
            x = T(batch_images)
            y = T(batch_labels.astype(np.int32))

            # Forward pass
            optimizer.zero_grad()
            logits = model(x)

            # Cross-entropy loss
            loss = logits.sparse_categorical_crossentropy(y)
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            predictions = logits.argmax(axis=1).numpy()
            correct += (predictions == batch_labels).sum()
            total += len(batch_labels)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.1f}%")

        epoch_loss = total_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        logger.info(f"Epoch {epoch + 1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.1f}%")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train FID classifier")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["mnist", "fashion_mnist", "cifar10"],
        help="Dataset to train on",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--output-dir", type=str, default="weights", help="Output directory for weights"
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset and create model
    if args.dataset == "mnist":
        logger.info("Loading MNIST dataset...")
        dataloader = MNISTLoader(
            path="dataset/mnist/trainset/trainingSet/*/*.jpg",
            batch_size=args.batch_size,
            shuffle=True,
            flatten=False,
        )
        model = LeNetMNIST()
        output_path = os.path.join(args.output_dir, "fid_classifier_mnist.safetensors")

    elif args.dataset == "fashion_mnist":
        logger.info("Loading Fashion MNIST dataset...")
        dataloader = FashionMNISTLoader(
            path="dataset/fashion_mnist",
            batch_size=args.batch_size,
            shuffle=True,
            flatten=False,
            train=True,
        )
        model = LeNetMNIST()
        output_path = os.path.join(args.output_dir, "fid_classifier_fashion_mnist.safetensors")

    elif args.dataset == "cifar10":
        logger.info("Loading CIFAR-10 dataset...")
        dataloader = CIFAR10Loader(
            path="dataset/cifar10/cifar-10-batches-py",
            batch_size=args.batch_size,
            shuffle=True,
            train=True,
        )
        model = LeNetCIFAR10()
        output_path = os.path.join(args.output_dir, "fid_classifier_cifar10.safetensors")

    # Train the classifier
    logger.info(f"Training {args.dataset} classifier for {args.epochs} epochs...")
    with T.train(True):
        model = train_classifier(
            model,
            dataloader,
            num_epochs=args.epochs,
            learning_rate=args.lr,
        )

    # Save weights
    safe_save(get_state_dict(model), output_path)
    logger.info(f"Saved classifier weights to: {output_path}")


if __name__ == "__main__":
    main()
