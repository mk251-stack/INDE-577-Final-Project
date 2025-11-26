# src/rice_ml/data/mnist.py

from __future__ import annotations

import os
import struct
from typing import Tuple

import numpy as np

# Human-readable class names for Fashion-MNIST
FASHION_MNIST_CLASSES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


def _load_idx_images(path: str) -> np.ndarray:
    """
    Load images from an IDX file (e.g., Fashion-MNIST images).

    Returns an array of shape (n_samples, rows, cols) with dtype uint8.
    """
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad magic number in image file {path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def _load_idx_labels(path: str) -> np.ndarray:
    """
    Load labels from an IDX file (e.g., Fashion-MNIST labels).

    Returns an array of shape (n_samples,) with dtype uint8.
    """
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad magic number in label file {path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def load_fashion_mnist_raw(
    data_dir: str,
    flatten: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Fashion-MNIST from a directory containing the four raw IDX files.

    Expected filenames in `data_dir`:
        - train-images-idx3-ubyte
        - train-labels-idx1-ubyte
        - t10k-images-idx3-ubyte
        - t10k-labels-idx1-ubyte

    Parameters
    ----------
    data_dir : str
        Directory containing the raw IDX files.

    flatten : bool, default=True
        If True, returns images as flat vectors (n_samples, 784).
        If False, returns images as (n_samples, 28, 28).

    normalize : bool, default=True
        If True, converts pixel values to float32 in [0, 1].

    Returns
    -------
    X_train, y_train, X_test, y_test : arrays
        Train and test sets. Labels are returned as 1D arrays of ints.
    """
    paths = {
        "train_images": os.path.join(data_dir, "train-images-idx3-ubyte"),
        "train_labels": os.path.join(data_dir, "train-labels-idx1-ubyte"),
        "test_images": os.path.join(data_dir, "t10k-images-idx3-ubyte"),
        "test_labels": os.path.join(data_dir, "t10k-labels-idx1-ubyte"),
    }

    for name, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file for {name}: {p}")

    train_images = _load_idx_images(paths["train_images"])
    train_labels = _load_idx_labels(paths["train_labels"])
    test_images = _load_idx_images(paths["test_images"])
    test_labels = _load_idx_labels(paths["test_labels"])

    if normalize:
        train_images = train_images.astype("float32") / 255.0
        test_images = test_images.astype("float32") / 255.0

    if flatten:
        X_train = train_images.reshape(len(train_images), -1)
        X_test = test_images.reshape(len(test_images), -1)
    else:
        X_train = train_images
        X_test = test_images

    y_train = train_labels.astype(int)
    y_test = test_labels.astype(int)
    return X_train, y_train, X_test, y_test
