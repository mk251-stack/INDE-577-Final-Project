"""
Community Detection via Unsupervised Label Propagation (LPA).

This module implements an UNSUPERVISED graph-based community detection algorithm:
- No labels are provided or clamped.
- Each node starts with a unique label.
- Labels iteratively update to match the most frequent neighbor label
  (weighted by edge strength).

The output is an integer community label per node.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _validate_adjacency(A: np.ndarray) -> np.ndarray:
    """
    Validate and sanitize an adjacency matrix.

    Parameters
    ----------
    A : np.ndarray
        Candidate adjacency matrix.

    Returns
    -------
    np.ndarray
        Validated adjacency matrix cast to float.

    Raises
    ------
    ValueError
        If the matrix is not square or contains negative values.
    """
    A = np.asarray(A)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square (n x n) adjacency matrix.")

    if np.any(A < 0):
        raise ValueError("A must be nonnegative.")

    # Zero diagonal is typical but not enforced
    return A.astype(float)


def label_propagation_communities(
    A: np.ndarray,
    max_iter: int = 200,
    seed: Optional[int] = 42,
    shuffle: bool = True,
) -> np.ndarray:
    """
    Perform unsupervised label propagation for community detection.

    Parameters
    ----------
    A : np.ndarray
        Square (n, n) adjacency matrix representing a similarity graph.
        Edge weights must be nonnegative; larger values indicate stronger ties.
    max_iter : int, default 200
        Maximum number of full passes over all nodes.
    seed : int or None, default 42
        Random seed used for update order shuffling and tie-breaking.
    shuffle : bool, default True
        Whether to randomize node update order at each iteration.

    Returns
    -------
    np.ndarray
        One-dimensional array of length n containing integer community
        labels for each node, relabeled to the range 0..k-1.
    """
    A = _validate_adjacency(A)
    n = A.shape[0]

    # Initialize with a unique label per node
    labels = np.arange(n, dtype=int)
    rng = np.random.default_rng(seed)

    for _ in range(max_iter):
        changed = False

        order = np.arange(n)
        if shuffle:
            rng.shuffle(order)

        for i in order:
            # Neighbors are nodes with positive edge weight
            neighbors = np.flatnonzero(A[i] > 0)
            if neighbors.size == 0:
                continue

            neighbor_labels = labels[neighbors]
            neighbor_weights = A[i, neighbors]

            # Weighted vote: sum edge weights per label
            unique_labels = np.unique(neighbor_labels)
            scores = np.zeros(unique_labels.shape[0], dtype=float)

            for idx, lab in enumerate(unique_labels):
                scores[idx] = neighbor_weights[neighbor_labels == lab].sum()

            best_score = scores.max()
            best_labels = unique_labels[scores == best_score]

            # Random tie-breaking among best labels
            new_label = int(rng.choice(best_labels))

            if new_label != labels[i]:
                labels[i] = new_label
                changed = True

        if not changed:
            break

    # Relabel communities to contiguous integers 0..k-1
    _, relabeled = np.unique(labels, return_inverse=True)
    return relabeled

