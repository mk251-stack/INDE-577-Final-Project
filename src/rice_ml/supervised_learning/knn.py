"""
k-Nearest Neighbors (NumPy-only).

This module provides simple, dependency-free KNN models suitable for teaching
and lightweight usage. It supports:
- Classification (`KNNClassifier`) with `predict` and `predict_proba`
- Regression (`KNNRegressor`) with `predict`
- Distance metrics: 'euclidean', 'manhattan'
- Weighting: 'uniform' or 'distance'
- Convenience methods: `kneighbors`, `score`

Robust error handling and NumPy-style docstrings make this module easy to test
with doctest/pytest.

Examples
--------
Basic classification:

>>> import numpy as np
>>> from rice_ml.supervised_learning.knn import KNNClassifier
>>> X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
>>> y = np.array([0, 0, 1, 1])
>>> clf = KNNClassifier(n_neighbors=3, metric="euclidean", weights="uniform").fit(X, y)
>>> clf.predict([[0.1, 0.1]]).tolist()
[0]
>>> np.round(clf.predict_proba([[0.1, 0.1]]), 2).tolist()
[[0.67, 0.33]]

Basic regression:

>>> from rice_ml.supervised_learning.knn import KNNRegressor
>>> X = np.array([[0],[1],[2],[3]], dtype=float)
>>> y = np.array([0.0, 1.0, 1.5, 3.0])
>>> reg = KNNRegressor(n_neighbors=2, weights="distance").fit(X, y)
>>> round(reg.predict([[1.5]])[0], 4)
np.float64(1.25)
"""

from __future__ import annotations
from typing import Literal, Optional, Tuple, Union, Sequence

import numpy as np

__all__ = [
    'KNNClassifier',
    'KNNRegressor',
]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


# ----------------------------- Helpers & Validation -----------------------------

def _ensure_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    """Ensure X is a 2D numeric ndarray of dtype float."""
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(float, copy=False)
        except (TypeError, ValueError) as e:
            raise TypeError(f"All elements of {name} must be numeric.") from e
    else:
        arr = arr.astype(float, copy=False)
    return arr


def _ensure_1d(y, name: str = "y") -> np.ndarray:
    """Ensure y is a 1D array (labels may be any dtype for classifier; numeric for regressor)."""
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


def _rng_from_seed(seed: Optional[int]) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("random_state must be an integer or None.")
    return np.random.default_rng(int(seed))


def _validate_common_params(
    n_neighbors: int,
    metric: Literal["euclidean", "manhattan"],
    weights: Literal["uniform", "distance"],
) -> None:
    if not isinstance(n_neighbors, (int, np.integer)) or n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer.")
    if metric not in ("euclidean", "manhattan"):
        raise ValueError("metric must be 'euclidean' or 'manhattan'.")
    if weights not in ("uniform", "distance"):
        raise ValueError("weights must be 'uniform' or 'distance'.")


def _pairwise_distances(XA: np.ndarray, XB: np.ndarray, metric: str) -> np.ndarray:
    """
    Compute pairwise distances between rows of XA and XB.

    Parameters
    ----------
    XA, XB : ndarray, shape (n_a, d), (n_b, d)
        Input matrices.
    metric : {"euclidean", "manhattan"}
        Distance metric.

    Returns
    -------
    D : ndarray, shape (n_a, n_b)
        Distances.

    Notes
    -----
    - Uses vectorized NumPy operations (no Python loops).
    """
    if metric == "euclidean":
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b -> then sqrt
        aa = np.sum(XA * XA, axis=1, keepdims=True)       # (n_a, 1)
        bb = np.sum(XB * XB, axis=1, keepdims=True).T     # (1, n_b)
        # numerical stability: distances can't be negative
        D2 = np.maximum(aa + bb - 2.0 * XA @ XB.T, 0.0)
        return np.sqrt(D2, dtype=float)
    elif metric == "manhattan":
        # expand dims for broadcasting: (n_a, 1, d) - (1, n_b, d)
        diff = XA[:, None, :] - XB[None, :, :]
        return np.sum(np.abs(diff), axis=2, dtype=float)
    else:
        # Checked earlier
        raise ValueError("Unsupported metric.")


def _neighbors(X_train: np.ndarray, X_query: np.ndarray, n_neighbors: int, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (distances, indices) of the n_neighbors nearest neighbors in the training set for each query.
    Distances/indices are sorted per query row.

    Returns
    -------
    distances : ndarray, shape (n_query, n_neighbors)
    indices   : ndarray, shape (n_query, n_neighbors)
    """
    D = _pairwise_distances(X_query, X_train, metric)  # (nq, n_train)
    if n_neighbors > X_train.shape[0]:
        raise ValueError(f"n_neighbors={n_neighbors} cannot exceed number of training samples={X_train.shape[0]}.")
    # argsort along columns to get nearest neighbors
    idx = np.argpartition(D, kth=n_neighbors - 1, axis=1)[:, :n_neighbors]
    # sort those neighbors by distance
    row_indices = np.arange(D.shape[0])[:, None]
    dsel = D[row_indices, idx]
    order = np.argsort(dsel, axis=1)
    idx_sorted = idx[row_indices, order]
    d_sorted = dsel[row_indices, order]
    return d_sorted, idx_sorted


def _weights_from_distances(dist: np.ndarray, scheme: str, eps: float = 1e-12) -> np.ndarray:
    """
    Compute neighbor weights from distances.

    - uniform: all 1s
    - distance: 1 / d, but if any d==0 for a query, give weight 1 to d==0 neighbors and 0 to others.

    Parameters
    ----------
    dist : ndarray, shape (n_query, k)
        Neighbor distances per query.
    scheme : {"uniform", "distance"}
        Weighting method.
    eps : float, default=1e-12
        For numerical stability when inverting distances.

    Returns
    -------
    w : ndarray, shape (n_query, k)
        Non-negative weights (not normalized).
    """
    if scheme == "uniform":
        return np.ones_like(dist, dtype=float)

    # distance weighting
    zero_mask = (dist <= eps)
    w = np.empty_like(dist, dtype=float)
    # If any exact duplicate neighbors exist, restrict weights to them
    any_zero = zero_mask.any(axis=1)
    if np.any(any_zero):
        w[any_zero] = zero_mask[any_zero].astype(float)
    # Otherwise, inverse distance
    if np.any(~any_zero):
        w[~any_zero] = 1.0 / np.maximum(dist[~any_zero], eps)
    return w


# ---------------------------------- Base ----------------------------------

class _KNNBase:
    """Shared functionality for KNN models."""

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        metric: Literal["euclidean", "manhattan"] = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform",
    ) -> None:
        _validate_common_params(n_neighbors, metric, weights)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.weights = weights
        self._X: Optional[np.ndarray] = None  # fitted features
        self._y: Optional[np.ndarray] = None  # fitted targets/labels

    # ---------------- API ----------------

    def fit(self, X: ArrayLike, y: ArrayLike):
        """
        Fit the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training features.
        y : array_like, shape (n_samples,)
            Training targets (regression) or labels (classification).

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If shapes are invalid.
        TypeError
            If X is not numeric.
        """
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d(y, "y")
        if len(y_arr) != X_arr.shape[0]:
            raise ValueError(f"X and y length mismatch: len(y)={len(y_arr)} vs X.shape[0]={X_arr.shape[0]}")
        if self.n_neighbors > X_arr.shape[0]:
            raise ValueError("n_neighbors cannot exceed the number of training samples.")
        self._X = X_arr
        self._y = y_arr
        return self

    def _check_is_fitted(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._X is None or self._y is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")
        return self._X, self._y

    def kneighbors(self, X: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the nearest neighbors of the provided samples.

        Parameters
        ----------
        X : array_like, shape (n_query, n_features)
            Query samples.

        Returns
        -------
        distances : ndarray, shape (n_query, n_neighbors)
            Distances to neighbors.
        indices : ndarray, shape (n_query, n_neighbors)
            Indices of neighbors in the training set.

        Raises
        ------
        RuntimeError
            If called before fit.
        ValueError, TypeError
            On invalid X.
        """
        X_train, _ = self._check_is_fitted()
        Xq = _ensure_2d_float(X, "X")
        if Xq.shape[1] != X_train.shape[1]:
            raise ValueError(f"X has {Xq.shape[1]} features, expected {X_train.shape[1]}.")
        return _neighbors(X_train, Xq, self.n_neighbors, self.metric)


# -------------------------------- Classifier --------------------------------

class KNNClassifier(_KNNBase):
    """
    k-Nearest Neighbors classifier.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use.
    metric : {"euclidean", "manhattan"}, default="euclidean"
        Distance metric.
    weights : {"uniform", "distance"}, default="uniform"
        Weighting scheme.

    Notes
    -----
    - `predict_proba` returns class probabilities ordered by sorted class labels.
    - When `weights="distance"` and a query has any zero-distance neighbors,
      only those neighbors are used (uniformly) to avoid division-by-zero.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = KNNClassifier(n_neighbors=3, metric="euclidean", weights="uniform").fit(X, y)
    >>> clf.predict([[0.1, 0.1]]).tolist()
    [0]
    >>> np.round(clf.predict_proba([[0.1, 0.1]]), 2).tolist()
    [[0.67, 0.33]]
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        metric: Literal["euclidean", "manhattan"] = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform",
    ) -> None:
        super().__init__(n_neighbors=n_neighbors, metric=metric, weights=weights)
        self.classes_: Optional[np.ndarray] = None  # learned label set (sorted)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNClassifier":
        super().fit(X, y)
        # Establish class order (sorted unique)
        self.classes_ = np.unique(self._y)
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array_like, shape (n_query, n_features)
            Query samples.

        Returns
        -------
        proba : ndarray, shape (n_query, n_classes)
            Probabilities per class (ordered as `classes_`).

        Raises
        ------
        RuntimeError
            If called before fit.
        """
        X_train, y_train = self._check_is_fitted()
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted.")
        Xq = _ensure_2d_float(X, "X")
        if Xq.shape[1] != X_train.shape[1]:
            raise ValueError(f"X has {Xq.shape[1]} features, expected {X_train.shape[1]}.")

        dist, idx = _neighbors(X_train, Xq, self.n_neighbors, self.metric)
        w = _weights_from_distances(dist, self.weights)

        # Vote per class
        n_query = Xq.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_query, n_classes), dtype=float)

        # map neighbor labels to class indices
        # y_train arbitrary dtype; use searchsorted on sorted classes_
        # (classes_ is sorted because of np.unique)
        for i in range(n_query):
            neigh_labels = y_train[idx[i]]
            class_ids = np.searchsorted(self.classes_, neigh_labels)
            # sum weights per class
            # (vectorized bincount on per-row basis)
            counts = np.bincount(class_ids, weights=w[i], minlength=n_classes)
            total = counts.sum()
            if total == 0:
                # all weights zero (shouldn't happen), fallback uniform
                proba[i] = 1.0 / n_classes
            else:
                proba[i] = counts / total
        return proba

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict the most probable class.

        Returns
        -------
        y_pred : ndarray, shape (n_query,)
            Predicted labels (same dtype as `classes_`).

        Examples
        --------
        >>> import numpy as np
        >>> X = np.array([[0,0],[1,1],[0,1],[1,0]], dtype=float)
        >>> y = np.array(["A","B","A","B"], dtype=object)
        >>> clf = KNNClassifier(n_neighbors=3).fit(X, y)
        >>> clf.predict([[0.1, 0.2]]).tolist()
        ['A']
        """
        proba = self.predict_proba(X)
        best = np.argmax(proba, axis=1)
        return self.classes_[best]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Classification accuracy on (X, y).

        Returns
        -------
        float
            Fraction of correct predictions.
        """
        y_true = _ensure_1d(y, "y")
        y_pred = self.predict(X)
        if len(y_true) != len(y_pred):
            raise ValueError("X and y lengths do not match.")
        return float(np.mean(y_true == y_pred))


# -------------------------------- Regressor --------------------------------

class KNNRegressor(_KNNBase):
    """
    k-Nearest Neighbors regressor.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use.
    metric : {"euclidean", "manhattan"}, default="euclidean"
        Distance metric.
    weights : {"uniform", "distance"}, default="uniform"
        Weighting scheme.

    Notes
    -----
    - With `weights="distance"`, if a query has any zero-distance neighbors,
      only those neighbors are used (uniformly) to avoid division-by-zero.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0],[1],[2],[3]], dtype=float)
    >>> y = np.array([0.0, 1.0, 1.5, 3.0])
    >>> reg = KNNRegressor(n_neighbors=2, weights="distance").fit(X, y)
    >>> round(float(reg.predict([[1.5]])[0]), 4)
    1.25
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNRegressor":
        X_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d(y, "y")
        # require numeric targets
        if not np.issubdtype(y_arr.dtype, np.number):
            try:
                y_arr = y_arr.astype(float, copy=False)
            except (TypeError, ValueError) as e:
                raise TypeError("Regression target values must be numeric.") from e
        if len(y_arr) != X_arr.shape[0]:
            raise ValueError(f"X and y length mismatch: len(y)={len(y_arr)} vs X.shape[0]={X_arr.shape[0]}")
        if self.n_neighbors > X_arr.shape[0]:
            raise ValueError("n_neighbors cannot exceed the number of training samples.")
        self._X = X_arr
        self._y = y_arr.astype(float, copy=False)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict regression targets.

        Parameters
        ----------
        X : array_like, shape (n_query, n_features)
            Query samples.

        Returns
        -------
        y_pred : ndarray, shape (n_query,)
            Predicted values (float).

        Raises
        ------
        RuntimeError
            If called before fit.
        """
        X_train, y_train = self._check_is_fitted()
        Xq = _ensure_2d_float(X, "X")
        if Xq.shape[1] != X_train.shape[1]:
            raise ValueError(f"X has {Xq.shape[1]} features, expected {X_train.shape[1]}.")

        dist, idx = _neighbors(X_train, Xq, self.n_neighbors, self.metric)
        w = _weights_from_distances(dist, self.weights)
        # Weighted average per query
        y_neighbors = y_train[idx]  # (nq, k)
        wsum = np.sum(w, axis=1)
        # avoid divide-by-zero: if all weights zero, fallback to simple mean
        with np.errstate(divide="ignore", invalid="ignore"):
            y_pred = np.divide(np.sum(w * y_neighbors, axis=1), wsum, where=wsum != 0)
        fallback = (wsum == 0)
        if np.any(fallback):
            y_pred[fallback] = np.mean(y_neighbors[fallback], axis=1)
        return y_pred.astype(float, copy=False)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        R^2 score on (X, y).

        Returns
        -------
        float
            Coefficient of determination (R^2).

        Raises
        ------
        ValueError
            - If y is constant and you're not scoring on the exact training inputs.
            - If y is constant and predictions are not perfect.
        """
        X_train, _ = self._check_is_fitted()
        Xq = _ensure_2d_float(X, "X")
        y_true = np.asarray(_ensure_1d(y, "y"), dtype=float)

        if Xq.shape[0] != y_true.shape[0]:
            raise ValueError("X and y lengths do not match.")

        y_pred = self.predict(Xq)

        ss_res = np.sum((y_true - y_pred) ** 2)
        y_mean = np.mean(y_true)
        ss_tot = np.sum((y_true - y_mean) ** 2)

        if ss_tot == 0:
            # y_true is constant.
            # Allow a well-defined perfect score ONLY when evaluating exactly on the
            # training inputs and predictions are perfect. Otherwise raise.
            if np.array_equal(Xq, X_train) and ss_res == 0:
                return 1.0
            raise ValueError(
                "R^2 is undefined when y_true is constant unless scoring on the "
                "training inputs with a perfect fit."
            )

        return float(1.0 - ss_res / ss_tot)
