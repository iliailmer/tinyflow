from glob import glob

import numpy as np
from skimage.io import imread
import os


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
