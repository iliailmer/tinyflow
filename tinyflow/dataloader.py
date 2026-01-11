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
    def __init__(self) -> None:
        pass

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError


class MNISTLoader(BaseDataloader):
    def __init__(
        self,
        path: str = "dataset/mnist/trainset/trainingSet/*/*.jpg",
        batch_size: int = 10,
        shuffle: bool = True,
        flatten: bool = True,
    ) -> None:
        super().__init__()
        self.path = path
        self.mnist_files = glob(path)
        self.shuffle = shuffle
        self.batch_size = batch_size
        if shuffle:
            np.random.shuffle(self.mnist_files)
        self.flatten = flatten
        self.index = 0

    def __len__(self):
        return len(self.mnist_files) // self.batch_size

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.mnist_files):
            raise StopIteration
        batch_files = self.mnist_files[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        images = []
        labels = []

        for filepath in batch_files:
            img = imread(filepath).astype(np.float32) / 255.0  # normalize
            if self.flatten:
                images.append(img.ravel())  # flatten
            else:
                images.append(img.reshape((1, img.shape[0], img.shape[1])))
            label = int(os.path.basename(os.path.dirname(filepath)))
            labels.append(label)

        return np.stack(images), np.array(labels)


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
    ) -> None:
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.flatten = flatten
        self.train = train
        self.index = 0

        # Load images and labels from idx-ubyte format
        if train:
            images_file = os.path.join(path, "train-images-idx3-ubyte")
            labels_file = os.path.join(path, "train-labels-idx1-ubyte")
        else:
            images_file = os.path.join(path, "t10k-images-idx3-ubyte")
            labels_file = os.path.join(path, "t10k-labels-idx1-ubyte")

        print(f"Loading Fashion MNIST from {images_file}...")
        self.images = read_idx_images(images_file).astype(np.float32) / 255.0
        self.labels = read_idx_labels(labels_file)

        # Reshape if not flattening
        if not flatten:
            self.images = self.images.reshape(-1, 1, 28, 28)
        else:
            self.images = self.images.reshape(-1, 28 * 28)

        # Shuffle if requested
        if shuffle:
            indices = np.random.permutation(len(self.images))
            self.images = self.images[indices]
            self.labels = self.labels[indices]

    def __len__(self):
        return len(self.images) // self.batch_size

    def __iter__(self):
        self.index = 0
        if self.shuffle:
            indices = np.random.permutation(len(self.images))
            self.images = self.images[indices]
            self.labels = self.labels[indices]
        return self

    def __next__(self):
        if self.index >= len(self.images):
            raise StopIteration

        batch_images = self.images[self.index : self.index + self.batch_size]
        batch_labels = self.labels[self.index : self.index + self.batch_size]
        self.index += self.batch_size

        return batch_images, batch_labels


class CIFAR10Loader(BaseDataloader):
    """
    CIFAR-10 dataloader with in-memory caching for fast training.

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
        cache: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train = train
        self.cache = cache
        self.normalize = normalize
        self.index = 0

        if cache:
            self._cached_data = self._load_all_images()
        else:
            self.batch_files = self._get_batch_files()

    def _get_batch_files(self):
        """Get list of batch files to load."""
        if self.train:
            return [os.path.join(self.path, f"data_batch_{i}") for i in range(1, 6)]
        else:
            return [os.path.join(self.path, "test_batch")]

    def _load_all_images(self):
        """Load all CIFAR-10 images into memory."""
        from tinyflow.utils import unpickle

        all_images = []
        all_labels = []

        batch_files = self._get_batch_files()
        for batch_file in tqdm(batch_files, desc="Loading CIFAR-10"):
            data_dict = unpickle(batch_file)

            images = data_dict[b"data"]
            labels = data_dict[b"labels"]
            images = images.reshape(-1, 3, 32, 32)

            images = images.astype(np.float32)
            if self.normalize:
                images = images / 255.0  # Normalize to [0, 1]

            all_images.append(images)
            all_labels.extend(labels)

        all_images = np.concatenate(all_images, axis=0)
        all_labels = np.array(all_labels)

        if self.shuffle:
            indices = np.random.permutation(len(all_images))
            all_images = all_images[indices]
            all_labels = all_labels[indices]

        return all_images, all_labels

    def __len__(self):
        if self.cache:
            images, _ = self._cached_data
            return len(images) // self.batch_size
        else:
            return (len(self.batch_files) * 10000) // self.batch_size

    def __iter__(self):
        self.index = 0
        if self.cache and self.shuffle:
            images, labels = self._cached_data
            indices = np.random.permutation(len(images))
            self._cached_data = (images[indices], labels[indices])
        return self

    def __next__(self):
        if not self.cache:
            raise NotImplementedError("Non-cached mode not implemented yet")

        images, labels = self._cached_data
        if self.index >= len(images):
            raise StopIteration

        batch_images = images[self.index : self.index + self.batch_size]
        batch_labels = labels[self.index : self.index + self.batch_size]
        self.index += self.batch_size

        return batch_images, batch_labels
