import pickle
import struct

import numpy as np
import pytest
from skimage.io import imsave

from tinyflow.dataloader import BaseDataloader, CIFAR10Loader, FashionMNISTLoader, MNISTLoader


def make_loader(n=10, batch_size=4, **kwargs):
    images = np.arange(n, dtype=np.float32).reshape(n, 1)
    return BaseDataloader(images, np.arange(n), batch_size, **kwargs)


class TestBaseDataloader:
    def test_drop_last_gives_equal_batches(self):
        loader = make_loader(n=10, batch_size=4, shuffle=False)
        batches = list(loader)
        assert len(loader) == len(batches) == 2
        assert all(images.shape == (4, 1) for images, _ in batches)

    def test_keep_last_batch(self):
        loader = make_loader(n=10, batch_size=4, shuffle=False, drop_last=False)
        batches = list(loader)
        assert len(loader) == len(batches) == 3
        assert batches[-1][0].shape == (2, 1)

    def test_hflip_mirrors_about_half_of_the_images(self):
        images = np.random.rand(64, 3, 4, 4)
        loader = BaseDataloader(images, np.arange(64), batch_size=64, shuffle=False, hflip=True)
        batch, labels = next(iter(loader))
        flipped = np.all(batch == images[..., ::-1], axis=(1, 2, 3))
        kept = np.all(batch == images, axis=(1, 2, 3))
        assert np.all(flipped | kept)
        assert 10 < flipped.sum() < 54
        assert np.array_equal(labels, np.arange(64))

    def test_exact_multiple(self):
        assert len(make_loader(n=8, batch_size=4, drop_last=False)) == 2
        assert len(make_loader(n=8, batch_size=4, drop_last=True)) == 2

    def test_no_shuffle_keeps_order(self):
        loader = make_loader(n=8, batch_size=4, shuffle=False)
        labels = np.concatenate([labels for _, labels in loader])
        assert labels.tolist() == list(range(8))

    def test_shuffle_changes_order_each_pass(self):
        np.random.seed(0)
        loader = make_loader(n=64, batch_size=8)
        first = np.concatenate([labels for _, labels in loader])
        second = np.concatenate([labels for _, labels in loader])
        assert sorted(first.tolist()) == sorted(second.tolist()) == list(range(64))
        assert first.tolist() != second.tolist()

    def test_shuffle_keeps_images_and_labels_together(self):
        for images, labels in make_loader(n=64, batch_size=8):
            assert images[:, 0].tolist() == labels.tolist()

    def test_same_seed_gives_same_order(self):
        orders = []
        for _ in range(2):
            np.random.seed(7)
            orders.append(np.concatenate([labels for _, labels in make_loader(n=32)]).tolist())
        assert orders[0] == orders[1]

    def test_value_range_maps_zero_one_linearly(self):
        images = np.array([[0.0], [0.5], [1.0]], dtype=np.float32)
        loader = BaseDataloader(images, np.arange(3), 3, shuffle=False, value_range=(-1.0, 1.0))
        batch, _ = next(iter(loader))
        assert batch[:, 0].tolist() == [-1.0, 0.0, 1.0]

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError):
            BaseDataloader(np.zeros((3, 1)), np.zeros(2), batch_size=1)


class TestMNISTLoader:
    @pytest.fixture
    def mnist_dir(self, tmp_path):
        for digit in range(3):
            (tmp_path / str(digit)).mkdir()
            for i in range(4):
                image = np.full((28, 28), digit * 100, dtype=np.uint8)
                imsave(tmp_path / str(digit) / f"{i}.jpg", image, check_contrast=False)
        return str(tmp_path / "*" / "*.jpg")

    def test_shapes_and_range(self, mnist_dir):
        loader = MNISTLoader(path=mnist_dir, batch_size=5, flatten=False)
        images, labels = next(iter(loader))
        assert images.shape == (5, 1, 28, 28)
        assert images.dtype == np.float32
        assert labels.shape == (5,)
        assert 0.0 <= images.min() and images.max() <= 1.0
        assert len(loader) == 2

    def test_flatten(self, mnist_dir):
        images, _ = next(iter(MNISTLoader(path=mnist_dir, batch_size=5, flatten=True)))
        assert images.shape == (5, 784)

    def test_labels_match_images(self, mnist_dir):
        for images, labels in MNISTLoader(path=mnist_dir, batch_size=4, flatten=False):
            intensity = images.mean(axis=(1, 2, 3)) * 255
            assert np.round(intensity / 100).astype(int).tolist() == labels.tolist()

    def test_missing_files(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MNISTLoader(path=str(tmp_path / "*" / "*.jpg"))


def test_fashion_mnist_loader(tmp_path):
    images = np.random.randint(0, 256, size=(6, 28, 28), dtype=np.uint8)
    labels = np.arange(6, dtype=np.uint8)
    with open(tmp_path / "train-images-idx3-ubyte", "wb") as f:
        f.write(struct.pack(">IIII", 2051, 6, 28, 28) + images.tobytes())
    with open(tmp_path / "train-labels-idx1-ubyte", "wb") as f:
        f.write(struct.pack(">II", 2049, 6) + labels.tobytes())

    loader = FashionMNISTLoader(path=str(tmp_path), batch_size=4, shuffle=False)
    batch_images, batch_labels = next(iter(loader))
    assert len(loader) == 1
    assert batch_images.shape == (4, 1, 28, 28)
    assert np.allclose(batch_images[:, 0], images[:4] / 255.0)
    assert batch_labels.tolist() == [0, 1, 2, 3]


def test_cifar10_loader(tmp_path):
    data = np.random.randint(0, 256, size=(6, 3072), dtype=np.uint8)
    with open(tmp_path / "test_batch", "wb") as f:
        pickle.dump({b"data": data, b"labels": list(range(6))}, f)

    loader = CIFAR10Loader(path=str(tmp_path), batch_size=3, shuffle=False, train=False)
    batch_images, batch_labels = next(iter(loader))
    assert len(loader) == 2
    assert batch_images.shape == (3, 3, 32, 32)
    assert batch_images.dtype == np.float32
    assert np.allclose(batch_images, data[:3].reshape(3, 3, 32, 32) / 255.0)
    assert batch_labels.tolist() == [0, 1, 2]
