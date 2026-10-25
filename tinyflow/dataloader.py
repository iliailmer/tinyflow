import os
import struct
from glob import glob

import numpy as np
from skimage.io import imread
from tqdm import tqdm


def read_idx_images(filename):
    """Read images from MNIST/Fashion MNIST idx3-ubyte format."""
    with open(filename, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Invalid magic number: {magic}"

        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
        return images


def read_idx_labels(filename):
    """Read labels from MNIST/Fashion MNIST idx1-ubyte format."""
    with open(filename, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Invalid magic number: {magic}"

        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels


class BaseDataloader:
    """
    Iterates over (images, labels) batches of a dataset held in memory.

    With shuffle=True the order changes on every pass. With drop_last=True the
    last batch is dropped when it is smaller than batch_size, so every batch
    has the same shape and len() is the number of batches a pass returns.

    images must be in [0, 1]. value_range=(low, high) maps them linearly to
    [low, high]; (-1, 1) centres the data on the mean of the noise.

    With hflip=True each image of a batch is mirrored left-right with probability 1/2.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        value_range: tuple[float, float] = (0.0, 1.0),
        hflip: bool = False,
    ) -> None:
        if len(images) != len(labels):
            raise ValueError(f"Got {len(images)} images and {len(labels)} labels")
        low, high = value_range
        if (low, high) != (0.0, 1.0):
            images = low + (high - low) * images
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.hflip = hflip
        self.index = 0
        self._order = np.arange(len(images))

    def __len__(self) -> int:
        full_batches, rest = divmod(len(self.images), self.batch_size)
        return full_batches if self.drop_last or rest == 0 else full_batches + 1

    def __iter__(self):
        self.index = 0
        if self.shuffle:
            self._order = np.random.permutation(len(self.images))
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        if self.index >= len(self) * self.batch_size:
            raise StopIteration
        batch = self._order[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        images = self.images[batch]
        if self.hflip:
            flip = np.random.rand(len(images)) < 0.5
            images = np.where(flip[:, None, None, None], images[..., ::-1], images)
        return images, self.labels[batch]


class MNISTLoader(BaseDataloader):
    """MNIST dataloader for a directory tree of JPG files, one directory per digit."""

    def __init__(
        self,
        path: str = "dataset/mnist/trainset/trainingSet/*/*.jpg",
        batch_size: int = 10,
        shuffle: bool = True,
        flatten: bool = True,
        drop_last: bool = True,
        value_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.path = path
        self.flatten = flatten
        files = sorted(glob(path))
        if not files:
            raise FileNotFoundError(f"No MNIST images found at {path}")

        images = np.stack(
            [imread(f) for f in tqdm(files, desc="Loading MNIST", leave=False)]
        ).astype(np.float32)
        images = images / 255.0
        images = images.reshape(len(files), -1) if flatten else images[:, None, :, :]
        labels = np.array([int(os.path.basename(os.path.dirname(f))) for f in files])
        super().__init__(images, labels, batch_size, shuffle, drop_last, value_range)


class FashionMNISTLoader(BaseDataloader):
    """
    Fashion MNIST dataloader for idx-ubyte format.
    Fashion MNIST contains 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat,
    Sandal, Shirt, Sneaker, Bag, Ankle boot.
    """

    def __init__(
        self,
        path: str = "dataset/fashion_mnist",
        batch_size: int = 32,
        shuffle: bool = True,
        flatten: bool = False,
        train: bool = True,
        drop_last: bool = True,
        value_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.path = path
        self.flatten = flatten
        self.train = train

        prefix = "train" if train else "t10k"
        images = read_idx_images(os.path.join(path, f"{prefix}-images-idx3-ubyte"))
        labels = read_idx_labels(os.path.join(path, f"{prefix}-labels-idx1-ubyte"))

        images = images.astype(np.float32) / 255.0
        images = images.reshape(-1, 28 * 28) if flatten else images.reshape(-1, 1, 28, 28)
        super().__init__(images, labels, batch_size, shuffle, drop_last, value_range)


class CIFAR10Loader(BaseDataloader):
    """
    CIFAR-10 dataloader.

    The CIFAR-10 dataset is stored as pickle files with structure:
    {
        'data': numpy array of shape (10000, 3072),  # 32x32x3 flattened
        'labels': list of 10000 labels (0-9),
        'batch_label': string identifier
    }
    """

    def __init__(
        self,
        path: str = "dataset/cifar10/cifar-10-batches-py",
        batch_size: int = 128,
        shuffle: bool = True,
        train: bool = True,
        normalize: bool = True,
        drop_last: bool = True,
        value_range: tuple[float, float] = (0.0, 1.0),
        hflip: bool = False,
    ):
        from tinyflow.utils import unpickle

        self.path = path
        self.train = train
        self.normalize = normalize

        names = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
        all_images = []
        all_labels = []
        for name in tqdm(names, desc="Loading CIFAR-10", leave=False):
            data_dict = unpickle(os.path.join(path, name))
            all_images.append(data_dict[b"data"].reshape(-1, 3, 32, 32).astype(np.float32))
            all_labels.extend(data_dict[b"labels"])

        images = np.concatenate(all_images, axis=0)
        if normalize:
            images = images / 255.0
        elif tuple(value_range) != (0.0, 1.0):
            raise ValueError("value_range needs normalize=True")
        super().__init__(
            images, np.array(all_labels), batch_size, shuffle, drop_last, value_range, hflip
        )
