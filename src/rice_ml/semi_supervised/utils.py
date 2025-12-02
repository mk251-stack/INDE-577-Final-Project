# src/rice_ml/semi_supervised/utils.py

from __future__ import annotations

from typing import Tuple

import numpy as np


def make_semi_supervised_labels(
    y: np.ndarray,
    n_labeled_per_class: int,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a semi-supervised label vector where only a fixed number of
    examples per class are labeled and the rest are marked as -1.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Ground-truth class labels (e.g., Fashion-MNIST digits 0–9).

    n_labeled_per_class : int
        Number of labeled examples to keep for each class.

    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    y_semi : ndarray of shape (n_samples,)
        Semi-supervised labels. Labeled samples keep their class; unlabeled
        samples are set to -1.

    labeled_mask : ndarray of bool, shape (n_samples,)
        True for labeled samples, False otherwise.

    unlabeled_mask : ndarray of bool, shape (n_samples,)
        True for unlabeled samples, False otherwise.
    """
    rng = np.random.RandomState(random_state)
    y = np.asarray(y).astype(int)

    y_semi = np.full_like(y, -1)
    labeled_mask = np.zeros_like(y, dtype=bool)

    classes = np.unique(y)
    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        k = min(n_labeled_per_class, len(idx))
        chosen = rng.choice(idx, size=k, replace=False)
        y_semi[chosen] = c
        labeled_mask[chosen] = True

    unlabeled_mask = ~labeled_mask
    return y_semi, labeled_mask, unlabeled_mask
