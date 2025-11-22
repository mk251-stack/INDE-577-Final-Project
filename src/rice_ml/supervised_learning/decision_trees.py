"""
Decision tree classifier implementation for the rice_ml package.

This module provides a simple, user-friendly API for a CART-style decision
tree classifier using the Gini impurity. It is implemented from scratch
using NumPy only (no scikit-learn dependency) so that students can read
and understand the core ideas.

Example
-------
>>> import numpy as np
>>> from rice_ml.supervised_learning.decision_tree import DecisionTreeClassifier
>>>
>>> X = np.array([[0, 0],
...               [0, 1],
...               [1, 0],
...               [1, 1]])
>>> y = np.array([0, 0, 1, 1])
>>>
>>> tree = DecisionTreeClassifier(max_depth=2, random_state=42)
>>> tree.fit(X, y)
>>> tree.predict(X)
array([0, 0, 1, 1])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class _TreeNode:
    """Internal node representation for the decision tree.

    Parameters
    ----------
    feature_index : int | None
        Index of the feature used to split at this node.
        If None, the node is a leaf.
    threshold : float | None
        Threshold value for the split: x[feature_index] <= threshold goes left.
        If None, the node is a leaf.
    left : _TreeNode | None
        Left child node.
    right : _TreeNode | None
        Right child node.
    proba : np.ndarray | None
        Class probability distribution at this node (for leaves).
        Shape (n_classes,).
    """
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_TreeNode"] = None
    right: Optional["_TreeNode"] = None
    proba: Optional[np.ndarray] = None

    def is_leaf(self) -> bool:
        return self.feature_index is None


class DecisionTreeClassifier:
    """Decision tree classifier using the CART algorithm and Gini impurity.

    This implementation supports basic hyperparameters that mirror the
    high-level API of popular libraries, but is intentionally compact for
    teaching purposes.

    Parameters
    ----------
    max_depth : int, optional
        Maximum depth of the tree. If None, the tree is expanded
        until all leaves are pure or contain fewer than
        ``min_samples_split`` samples.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    max_features : int or float or None, optional
        Number of features to consider when looking for the best split.
        If int, consider exactly that many features.
        If float in (0, 1], use that fraction of the total number of features.
        If None, use all features.
    random_state : int or None, optional
        Seed for the random number generator used when subsampling features.

    Attributes
    ----------
    n_classes_ : int
        Number of classes.
    n_features_ : int
        Number of features in the input data.
    tree_ : _TreeNode
        Root node of the grown decision tree.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[float | int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.n_classes_: Optional[int] = None
        self.n_features_: Optional[int] = None
        self.tree_: Optional[_TreeNode] = None
        self._rng: Optional[np.random.Generator] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """Fit the decision tree classifier on the given training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target class labels. Must be integer-encoded from 0, 1, ..., K-1.

        Returns
        -------
        self : DecisionTreeClassifier
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of class labels.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.n_features_ = X.shape[1]
        classes = np.unique(y)
        # Require integer-encoded classes for simplicity
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("y must contain integer-encoded class labels (0, 1, 2, ...).")
        self.n_classes_ = int(classes.max() + 1)

        self._rng = np.random.default_rng(self.random_state)

        self.tree_ = self._grow_tree(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for the given samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for the given samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes_)
            Predicted class probabilities.
        """
        if self.tree_ is None or self.n_classes_ is None:
            raise RuntimeError("The tree has not been fitted yet. Call `fit` first.")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")

        n_samples = X.shape[0]
        proba = np.zeros((n_samples, self.n_classes_), dtype=float)

        for i in range(n_samples):
            node = self._traverse_tree(X[i], self.tree_)
            proba[i] = node.proba

        return proba

    # ------------------------------------------------------------------
    # Internal tree growing logic
    # ------------------------------------------------------------------
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        n_samples, n_features = X.shape
        num_labels = len(np.unique(y))

        # Compute class distribution at this node
        proba = self._class_proba(y)

        # Stopping criteria: pure node, max depth, or too few samples
        if (
            num_labels == 1
            or (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
        ):
            return _TreeNode(proba=proba)

        # Try to find the best split
        feat_idx, threshold, (left_mask, right_mask) = self._best_split(X, y)

        # If no valid split was found, make this a leaf
        if feat_idx is None:
            return _TreeNode(proba=proba)

        # Recursively grow children
        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return _TreeNode(
            feature_index=feat_idx,
            threshold=threshold,
            left=left_child,
            right=right_child,
            proba=proba,
        )

    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], Tuple[np.ndarray, np.ndarray]]:
        """Find the best feature and threshold to split on using Gini impurity."""
        n_samples, n_features = X.shape
        if n_samples < 2 * self.min_samples_leaf:
            return None, None, (np.array([]), np.array([]))

        # Determine which features to consider
        if self.max_features is None:
            feature_indices = np.arange(n_features)
        elif isinstance(self.max_features, int):
            if self.max_features <= 0 or self.max_features > n_features:
                raise ValueError("max_features int must be in [1, n_features].")
            feature_indices = self._rng.choice(n_features, self.max_features, replace=False)
        elif isinstance(self.max_features, float):
            if not (0.0 < self.max_features <= 1.0):
                raise ValueError("max_features float must be in (0, 1].")
            k = max(1, int(self.max_features * n_features))
            feature_indices = self._rng.choice(n_features, k, replace=False)
        else:
            raise ValueError("max_features must be None, int, or float.")

        best_gini = 1.0
        best_feat = None
        best_thresh = None
        best_left_mask = np.array([], dtype=bool)
        best_right_mask = np.array([], dtype=bool)

        for feat in feature_indices:
            x_column = X[:, feat]
            thresholds = np.unique(x_column)
            if thresholds.size == 1:
                # All values are identical; no split
                continue

            for thresh in thresholds:
                left_mask = x_column <= thresh
                right_mask = ~left_mask

                n_left = left_mask.sum()
                n_right = right_mask.sum()

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])

                gini_split = (n_left * gini_left + n_right * gini_right) / n_samples

                if gini_split < best_gini:
                    best_gini = gini_split
                    best_feat = feat
                    best_thresh = float(thresh)
                    best_left_mask = left_mask
                    best_right_mask = right_mask

        if best_feat is None:
            return None, None, (np.array([]), np.array([]))

        return best_feat, best_thresh, (best_left_mask, best_right_mask)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _gini(self, y: np.ndarray) -> float:
        """Compute Gini impurity for a label vector."""
        if y.size == 0:
            return 0.0
        counts = np.bincount(y, minlength=self.n_classes_)
        proba = counts / counts.sum()
        return 1.0 - np.sum(proba ** 2)

    def _class_proba(self, y: np.ndarray) -> np.ndarray:
        """Compute class probability distribution for labels y."""
        counts = np.bincount(y, minlength=self.n_classes_)
        total = counts.sum()
        if total == 0:
            # Should not happen in normal training, but guard against division by zero
            return np.full(self.n_classes_, 1.0 / self.n_classes_)
        return counts / total

    def _traverse_tree(self, x: np.ndarray, node: _TreeNode) -> _TreeNode:
        """Traverse the tree for a single sample x until a leaf node is reached."""
        while not node.is_leaf():
            assert node.feature_index is not None and node.threshold is not None
            if x[node.feature_index] <= node.threshold:
                assert node.left is not None
                node = node.left
            else:
                assert node.right is not None
                node = node.right
        return node
