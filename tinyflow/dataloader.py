import os
from glob import glob

import numpy as np
from skimage.io import imread
from tqdm import tqdm


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

        # Load dataset
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

            # Extract images and labels
            images = data_dict[b"data"]  # Shape: (10000, 3072)
            labels = data_dict[b"labels"]  # List of 10000 labels

            # Reshape to (N, 3, 32, 32) - RGB format
            images = images.reshape(-1, 3, 32, 32)

            # Convert to float32 and normalize
            images = images.astype(np.float32)
            if self.normalize:
                images = images / 255.0  # Normalize to [0, 1]

            all_images.append(images)
            all_labels.extend(labels)

        # Stack all batches
        all_images = np.concatenate(all_images, axis=0)
        all_labels = np.array(all_labels)

        # Shuffle if requested
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
            # Each batch file has 10000 images
            return (len(self.batch_files) * 10000) // self.batch_size

    def __iter__(self):
        self.index = 0
        if self.cache and self.shuffle:
            # Reshuffle cached data
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

        # Get batch
        batch_images = images[self.index : self.index + self.batch_size]
        batch_labels = labels[self.index : self.index + self.batch_size]
        self.index += self.batch_size

        return batch_images, batch_labels
