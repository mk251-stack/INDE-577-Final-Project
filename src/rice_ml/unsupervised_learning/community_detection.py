"""
Community Detection via Unsupervised Label Propagation (LPA).

This is an UNSUPERVISED algorithm:
- No labels are provided or clamped.
- Each node starts with a unique label.
- Labels iteratively update to match the most common neighbor label.

Output: an integer community label per node.
"""

from __future__ import annotations

import numpy as np


def _validate_adjacency(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square (n x n) adjacency matrix.")
    if np.any(A < 0):
        raise ValueError("A must be nonnegative.")
    # Zero diagonal is typical, but don't force it.
    return A.astype(float)


def label_propagation_communities(
    A: np.ndarray,
    max_iter: int = 200,
    seed: int | None = 42,
    shuffle: bool = True,
) -> np.ndarray:
    """
    Unsupervised Label Propagation for community detection.

    Args:
        A: (n, n) adjacency matrix (weighted or unweighted). Larger = stronger tie.
        max_iter: maximum number of passes over nodes.
        seed: RNG seed used only for update order shuffling / tie-breaking.
        shuffle: whether to randomize update order each iteration (recommended).

    Returns:
        labels: (n,) integer community label for each node.
    """
    A = _validate_adjacency(A)
    n = A.shape[0]

    # Start with a unique label per node
    labels = np.arange(n, dtype=int)

    rng = np.random.default_rng(seed)

    for _ in range(max_iter):
        changed = False

        order = np.arange(n)
        if shuffle:
            rng.shuffle(order)

        for i in order:
            # neighbors are nodes with positive edge weight
            neighbors = np.flatnonzero(A[i] > 0)
            if neighbors.size == 0:
                continue

            neighbor_labels = labels[neighbors]
            neighbor_weights = A[i, neighbors]

            # Weighted vote: sum weights per label
            unique = np.unique(neighbor_labels)
            scores = np.zeros(unique.shape[0], dtype=float)
            for idx, lab in enumerate(unique):
                scores[idx] = neighbor_weights[neighbor_labels == lab].sum()

            best_score = scores.max()
            best_labels = unique[scores == best_score]

            # Tie-break randomly among best labels
            new_label = int(rng.choice(best_labels))

            if new_label != labels[i]:
                labels[i] = new_label
                changed = True

        if not changed:
            break

    # Relabel communities to 0..k-1 for neatness
    _, relabeled = np.unique(labels, return_inverse=True)
    return relabeled
