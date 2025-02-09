import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from tqdm.auto import tqdm

from tinyflow.nn import Tensor


def visualize(x, solver, time_grid, h_step, num_plots=10, dataset="moons"):
    i = 0
    _, ax = plt.subplots(1, num_plots, figsize=(30, 4), sharex=True, sharey=True)
    sample_every = time_grid.shape[0] // num_plots
    if dataset == "moons":
        for idx in tqdm(range(int(time_grid.shape[0]))):
            t = time_grid[idx]
            if (idx + 1) % sample_every == 0:
                ax[i].scatter(x.numpy()[:, 0], x.numpy()[:, 1], s=5)
                ax[i].set_title(f"Time: t={t.numpy():.2f}")
                i += 1
            x = solver.sample(h_step, t, x)
    if dataset == "mnist":
        for idx in tqdm(range(int(time_grid.shape[0]))):
            t = time_grid[idx]
            if (idx + 1) % sample_every == 0:
                ax[i].imshow(x.numpy()[0, :].reshape((8, 8)), cmap="gray")
                ax[i].axis("off")
                i += 1
            x = solver.sample(h_step, t, x)
    if dataset == "cifar":
        for idx in tqdm(range(int(time_grid.shape[0]))):
            t = time_grid[idx]
            x_vis = (x - x.min()) / (x.max() - x.min())
            if (idx + 1) % sample_every == 0:
                ax[i].imshow(
                    x_vis.numpy()[0, :].reshape((3, 32, 32)).transpose(1, 2, 0)
                )
                ax[i].axis("off")
                i += 1
            x = solver.sample(h_step, t, x)
    plt.tight_layout()
    plt.show()


def unpickle(file):
    with open(file, "rb") as fo:
        dct = pickle.load(fo, encoding="bytes")
    return dct


def mnist():
    return load_digits(return_X_y=True)[0]


def cifar10(path="dataset/cifar10/cifar-10-batches-py/"):
    files = [os.path.join(path, x) for x in os.listdir(path) if "data_batch_" in x]
    dataset = []
    for file in files:
        dct = unpickle(file)
        # filenames = dct[b"filenames"]
        # labels = dct[b"labels"]
        dataset.append(dct[b"data"])

    # data = np.concatenate(dataset, axis=1)
    return np.vstack(dataset)


def preprocess_time_moons(t: Tensor, rhs_prev: Tensor):
    t = t.reshape((1, 1))
    return t.repeat(rhs_prev.shape[0], 1)


def preprocess_time_mnist(t: Tensor, rhs_prev: Tensor):
    t = t.reshape((1, 1))
    return t.repeat(rhs_prev.shape[0], 1)


def preprocess_time_cifar(t: Tensor, rhs_prev: Tensor):
    t = t.reshape((1, 1, 1, 1)).expand((1, 1, 32, 32))
    return t.repeat(rhs_prev.shape[0], 1, 1, 1)
