"""
Semi-supervised learning via Label Propagation.

This module implements a simple graph-based label propagation algorithm
for binary or multiclass classification.

Key ideas
---------
- We are given:
    X : feature matrix for ALL points (labeled + unlabeled)
    y : label vector where:
        - labeled points have class labels (e.g. 0, 1, 2, ...)
        - unlabeled points are marked with -1

- We build a similarity graph over all samples using an RBF (Gaussian)
  kernel over the feature space.

- We then iteratively propagate labels across the graph:

    Y_{t+1} = alpha * S @ Y_t + (1 - alpha) * Y0

  where:
    - S is the row-normalized similarity matrix
    - Y0 contains one-hot labels for labeled points and zeros for unlabeled
    - Y_t is the current soft label distribution at iteration t

- After convergence, each point is assigned the class with highest
  probability in its row of Y.

Notes
-----
- This implementation is *transductive*: the graph is built on the
  training samples given to `fit`.
- For new points X_new, we approximate an inductive prediction by
  computing similarities between X_new and the training points and
  propagating labels in one step.

Typical usage
-------------
>>> from rice_ml.semi_supervised.label_propagation import LabelPropagation
>>> model = LabelPropagation(gamma=0.1, alpha=0.99, max_iter=1000)
>>> model.fit(X_all, y_with_minus_ones)
>>> y_pred = model.predict(X_all)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


@dataclass
class LabelPropagation:
    """
    Graph-based semi-supervised classifier via label propagation.

    Parameters
    ----------
    gamma : float, default=1.0
        RBF kernel width parameter. Larger gamma makes the kernel decay
        faster with distance (i.e., more local neighborhoods).

    alpha : float, default=0.99
        Clamping factor in [0, 1). Controls how strongly we keep the
        original labeled information. Values very close to 1.0 allow
        more propagation.

    max_iter : int, default=1000
        Maximum number of propagation iterations.

    tol : float, default=1e-4
        Convergence tolerance on the maximum change in label
        distributions between iterations.

    n_neighbors : Optional[int], default=None
        Reserved for future use (e.g., KNN graph). Currently ignored;
        the implementation uses a fully connected RBF kernel graph.

    verbose : bool, default=False
        If True, prints basic convergence information.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        Unique class labels (excluding -1).

    label_distributions_ : np.ndarray of shape (n_samples, n_classes)
        Final soft label distributions after propagation.

    X_fit_ : np.ndarray of shape (n_samples, n_features)
        Training data used to build the graph.

    y_fit_ : np.ndarray of shape (n_samples,)
        Original labels passed to fit (with -1 for unlabeled).

    converged_ : bool
        Whether the algorithm reached convergence before `max_iter`.

    n_iter_ : int
        Number of iterations performed.
    """

    gamma: float = 1.0
    alpha: float = 0.99
    max_iter: int = 1000
    tol: float = 1e-4
    n_neighbors: Optional[int] = None
    verbose: bool = False

    # Learned attributes (set during fit)
    classes_: Optional[np.ndarray] = None
    label_distributions_: Optional[np.ndarray] = None
    X_fit_: Optional[np.ndarray] = None
    y_fit_: Optional[np.ndarray] = None
    converged_: bool = False
    n_iter_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LabelPropagation":
        """
        Fit the label propagation model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix containing ALL points (labeled + unlabeled).

        y : array-like of shape (n_samples,)
            Labels for samples. Unlabeled samples MUST be marked as -1.
            Labeled samples carry their class label (e.g. 0, 1, 2, ...).

        Returns
        -------
        self : LabelPropagation
            Fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.X_fit_ = X
        self.y_fit_ = y

        # Identify labeled vs unlabeled points
        labeled_mask = y != -1
        unlabeled_mask = ~labeled_mask

        if not np.any(labeled_mask):
            raise ValueError("At least one point must be labeled (y != -1).")

        # Classes only from labeled points
        labeled_y = y[labeled_mask]
        self.classes_ = np.unique(labeled_y)
        n_samples = X.shape[0]
        n_classes = self.classes_.shape[0]

        # Map labels to {0, ..., n_classes-1}
        label_to_index = {label: idx for idx, label in enumerate(self.classes_)}

        # Initialize Y0 (one-hot for labeled, zeros for unlabeled)
        Y0 = np.zeros((n_samples, n_classes), dtype=float)
        for i, is_labeled in enumerate(labeled_mask):
            if is_labeled:
                class_idx = label_to_index[y[i]]
                Y0[i, class_idx] = 1.0

        # Build similarity matrix (RBF kernel) over all points
        W = rbf_kernel(X, X, gamma=self.gamma)
        # Zero out self-similarity to avoid dominance of diagonal
        np.fill_diagonal(W, 0.0)

        # Row-normalize to get stochastic matrix S
        row_sums = W.sum(axis=1, keepdims=True)
        # Avoid division by zero: rows with sum 0 stay all zeros
        S = np.zeros_like(W)
        nonzero_rows = row_sums[:, 0] > 0
        S[nonzero_rows] = W[nonzero_rows] / row_sums[nonzero_rows]

        # Iterative propagation
        Y = Y0.copy()
        self.converged_ = False
        self.n_iter_ = 0

        for it in range(1, self.max_iter + 1):
            Y_new = self.alpha * (S @ Y) + (1.0 - self.alpha) * Y0
            delta = np.max(np.abs(Y_new - Y))

            Y = Y_new
            self.n_iter_ = it

            if self.verbose and it % 10 == 0:
                print(f"[LabelPropagation] Iter {it}, max Δ={delta:.6f}")

            if delta < self.tol:
                self.converged_ = True
                if self.verbose:
                    print(f"[LabelPropagation] Converged in {it} iterations.")
                break

        if self.verbose and not self.converged_:
            print(f"[LabelPropagation] Reached max_iter={self.max_iter} "
                  f"with max Δ={delta:.6f}")

        self.label_distributions_ = Y
        return self

    # ------------------------------------------------------------------
    # Internal helper for new points
    # ------------------------------------------------------------------
    def _predict_proba_new(self, X_new: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for new samples X_new.

        We approximate inductive label propagation by:
        - computing similarities between X_new and the fitted samples
        - normalizing each row so similarities sum to 1
        - multiplying by the label distributions learned on the training graph
        """
        if self.X_fit_ is None or self.label_distributions_ is None:
            raise RuntimeError("LabelPropagation must be fitted before predicting.")

        X_new = np.asarray(X_new, dtype=float)

        # Similarities to training points
        W_new = rbf_kernel(X_new, self.X_fit_, gamma=self.gamma)
        row_sums = W_new.sum(axis=1, keepdims=True)
        S_new = np.zeros_like(W_new)
        nonzero_rows = row_sums[:, 0] > 0
        S_new[nonzero_rows] = W_new[nonzero_rows] / row_sums[nonzero_rows]

        # Propagate label distributions in one step
        Y_new = S_new @ self.label_distributions_
        # Normalize row-wise for safety (to sum to 1)
        row_sums_Y = Y_new.sum(axis=1, keepdims=True)
        nonzero_rows_Y = row_sums_Y[:, 0] > 0
        Y_new[nonzero_rows_Y] = Y_new[nonzero_rows_Y] / row_sums_Y[nonzero_rows_Y]

        return Y_new

    # ------------------------------------------------------------------
    # Public prediction API
    # ------------------------------------------------------------------
    def predict_proba(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict class probabilities for given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or None, default=None
            Data for which to compute probabilities.
            - If None, returns probabilities for the fitted samples (training set).
            - If not None, approximates inductive propagation using the
              method described in `_predict_proba_new`.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        if self.label_distributions_ is None:
            raise RuntimeError("LabelPropagation must be fitted before predicting.")

        if X is None:
            return self.label_distributions_.copy()
        else:
            return self._predict_proba_new(X)

    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict hard class labels for given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or None, default=None
            Data for which to predict labels.
            - If None, returns predictions for the fitted samples.
            - If not None, approximates inductive prediction for new samples.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        if self.classes_ is None:
            raise RuntimeError("LabelPropagation must be fitted before predicting.")

        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        return self.classes_[class_indices]

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Convenience method: fit the model and return predictions
        for the training data.

        Equivalent to:

            model.fit(X, y)
            return model.predict()

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
        """
        self.fit(X, y)
        return self.predict()
