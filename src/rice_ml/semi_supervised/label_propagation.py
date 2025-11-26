"""
Semi-supervised learning via Label Propagation.

This module implements a k-NN graph–based label propagation algorithm
for binary or multiclass classification.

Key ideas
---------
- We are given:
    X : feature matrix for ALL points (labeled + unlabeled)
    y : label vector where:
        - labeled points have class labels (e.g. 0, 1, 2, ...),
        - unlabeled points are marked with -1.

- We build a sparse k-nearest-neighbor similarity graph over all samples.
  Edge weights are RBF-style similarities:

      w_ij = exp(-gamma * ||x_i - x_j||^2)

  If `gamma` is None, a global scale sigma^2 is estimated from neighbor
  distances and we use:

      w_ij = exp(- ||x_i - x_j||^2 / sigma^2).

- We iteratively propagate labels across the graph:

      F_{t+1} = alpha * W @ F_t + (1 - alpha) * Y0

  where:
    - W is the row-normalized similarity matrix (stochastic),
    - Y0 contains one-hot labels for labeled points and zeros for unlabeled,
    - F_t is the current soft label distribution at iteration t.

- After convergence, each point is assigned the class with highest
  probability in its row of F.

Notes
-----
- This implementation is *transductive*: the graph is built on the
  training samples given to `fit`.
- For new points X_new, we perform an approximate inductive prediction
  using k-NN in the original feature space and a soft-label weighted vote.

Typical usage
-------------
>>> from rice_ml.semi_supervised.label_propagation import LabelPropagation
>>> lp = LabelPropagation(n_neighbors=10, gamma=None, alpha=0.99)
>>> lp.fit(X_all, y_with_minus_ones)
>>> y_pred_train = lp.predict()          # transductive
>>> y_pred_test  = lp.predict(X_test)    # inductive
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import time

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


@dataclass
class LabelPropagation:
    """
    k-NN graph–based Label Propagation for semi-supervised learning.

    This implementation follows the classical graph-based label propagation
    algorithm but uses a sparse k-nearest-neighbor (k-NN) graph instead of a
    fully dense similarity matrix. This avoids O(n^2) memory and makes the
    algorithm usable on datasets such as MNIST or Fashion-MNIST.

    Parameters
    ----------
    n_neighbors : int, default=10
        Number of nearest neighbors used to build the sparse similarity graph.
        Each sample connects to its `n_neighbors` nearest neighbors.

    k_neighbors : int, optional
        Backwards-compatible alias for `n_neighbors`. If provided, overrides
        the value of `n_neighbors`. Accepting this argument ensures examples
        using `k_neighbors=` still work.

    alpha : float, default=0.99
        Clamping (regularization) factor controlling how strongly the original
        labeled data influence the propagation. `alpha → 1` means labels are
        propagated with little clamping; smaller alpha increases supervision.

    max_iter : int, default=200
        Maximum number of propagation iterations.

    tol : float, default=1e-4
        Convergence tolerance. If the Frobenius norm of the change in the
        soft label matrix falls below `tol`, propagation stops.

    gamma : float or None, default=None
        RBF kernel width for neighbor weights.

        - If `gamma` is not None:
              w_ij = exp(-gamma * dist_ij^2)
        - If `gamma` is None (default):
              sigma^2 is estimated from the neighbor distances and we use
              w_ij = exp(-dist_ij^2 / sigma^2).

    algorithm : {"auto", "ball_tree", "kd_tree", "brute"}, default="auto"
        Algorithm used by scikit-learn's `NearestNeighbors`.

    n_jobs : int, default=-1
        Number of parallel jobs used by `NearestNeighbors`. -1 means "all cores".

    verbose : bool, default=False
        If True, prints progress during graph building and propagation.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Sorted array of unique class labels encountered among the labeled data.

    F_ : ndarray of shape (n_samples, n_classes)
        The soft label matrix optimized during propagation.

    W_ : csr_matrix of shape (n_samples, n_samples)
        Row-normalized sparse similarity graph (k-NN based).

    sigma2_ : float
        Estimated global squared length scale used when `gamma=None`.

    converged_ : bool
        Whether the algorithm converged within `max_iter`.

    n_iter_ : int
        Number of iterations executed during propagation.

    timing_ : dict
        Dictionary with timing information for `"graph_build"`, `"propagation"`,
        `"fit_total"`, and `"predict"`.
    """

    n_neighbors: int = 10
    k_neighbors: Optional[int] = None
    alpha: float = 0.99
    max_iter: int = 200
    tol: float = 1e-4
    gamma: Optional[float] = None
    algorithm: str = "auto"
    n_jobs: int = -1
    verbose: bool = False

    classes_: np.ndarray = field(init=False, default=None)
    F_: np.ndarray = field(init=False, default=None)
    W_: csr_matrix = field(init=False, default=None)
    sigma2_: float = field(init=False, default=1.0)
    converged_: bool = field(init=False, default=False)
    n_iter_: int = field(init=False, default=0)

    timing_: Dict[str, float] = field(init=False, default_factory=dict)

    # Internal cached data
    _X_fit: np.ndarray = field(init=False, default=None)
    _inductive_nbrs: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        """
        Post-initialization hook.

        Ensures `k_neighbors` can override `n_neighbors` if provided.
        """
        if self.k_neighbors is not None:
            self.n_neighbors = self.k_neighbors

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def _build_knn_graph(self, X: np.ndarray) -> csr_matrix:
        """
        Build a sparse k-NN similarity graph.

        Steps
        -----
        1. Fit a k-NN model on X using scikit-learn's `NearestNeighbors`.
        2. Query each point's `n_neighbors + 1` nearest neighbors
           (including itself).
        3. Compute RBF-style similarity weights for neighbor edges:
               - If gamma is not None:
                     w_ij = exp(-gamma * dist_ij^2)
               - Else:
                     sigma^2 = mean(dist_ij^2)
                     w_ij = exp(-dist_ij^2 / sigma^2)
        4. Construct a CSR matrix W from these edges.
        5. Row-normalize W so that each row sums to 1.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        W : csr_matrix of shape (n_samples, n_samples)
            Row-normalized sparse similarity matrix.
        """
        nbrs = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,
            algorithm=self.algorithm,
            n_jobs=self.n_jobs,
        ).fit(X)

        distances, indices = nbrs.kneighbors(X)

        # Exclude self-distance at [:, 0]
        d2 = distances[:, 1:] ** 2

        if self.gamma is None:
            # Automatic scale
            sigma2 = float(np.mean(d2) + 1e-9)
            self.sigma2_ = sigma2
            weights = np.exp(-d2 / sigma2)
        else:
            self.sigma2_ = float(np.mean(d2) + 1e-9)  # stored for reference
            weights = np.exp(-self.gamma * d2)

        rows = np.repeat(np.arange(X.shape[0]), self.n_neighbors)
        cols = indices[:, 1:].reshape(-1)
        vals = weights.reshape(-1)

        W = csr_matrix((vals, (rows, cols)), shape=(X.shape[0], X.shape[0]))

        # Row-normalize to make W stochastic
        row_sums = np.array(W.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        W = W.multiply(1.0 / row_sums[:, None])

        return W

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LabelPropagation":
        """
        Fit the Label Propagation model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix (all data: labeled and unlabeled).

        y : ndarray of shape (n_samples,)
            Label vector. Use `-1` to mark unlabeled points.

        Returns
        -------
        self : LabelPropagation
            Fitted estimator.
        """
        t0_total = time.perf_counter()
        self.timing_.clear()

        X = np.asarray(X)
        y = np.asarray(y)

        self._X_fit = X  # store for inductive predictions
        n_samples = X.shape[0]

        # Extract classes from labeled data only
        self.classes_ = np.unique(y[y >= 0])

        # Initialize soft label matrix
        F = np.zeros((n_samples, len(self.classes_)), dtype=float)
        labeled_mask = y >= 0

        for idx, c in enumerate(self.classes_):
            F[labeled_mask & (y == c), idx] = 1.0

        self.F_ = F.copy()

        # Build k-NN graph
        if self.verbose:
            print(f"[LP] Building sparse k-NN graph (k={self.n_neighbors})...")
        t0_graph = time.perf_counter()
        self.W_ = self._build_knn_graph(X)
        t1_graph = time.perf_counter()
        self.timing_["graph_build"] = t1_graph - t0_graph

        # Cache a k-NN model for inductive predictions
        self._inductive_nbrs = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm=self.algorithm,
            n_jobs=self.n_jobs,
        ).fit(X)

        # Propagation loop
        if self.verbose:
            print("[LP] Starting propagation...")

        t0_prop = time.perf_counter()
        self.converged_ = False

        for it in range(self.max_iter):
            F_new = self.alpha * (self.W_ @ self.F_) + (1 - self.alpha) * F
            delta = float(np.linalg.norm(F_new - self.F_))

            self.F_ = F_new

            if self.verbose:
                print(f"  iter {it:03d}, delta={delta:.6f}")

            if delta < self.tol:
                self.converged_ = True
                break

        self.n_iter_ = it + 1
        t1_prop = time.perf_counter()
        self.timing_["propagation"] = t1_prop - t0_prop

        t1_total = time.perf_counter()
        self.timing_["fit_total"] = t1_total - t0_total

        if self.verbose:
            print(
                f"[LP] Finished in {self.n_iter_} iterations "
                f"(converged={self.converged_})."
            )

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict class labels.

        Two modes exist:

        **Transductive mode** (`X=None`)
            Returns predictions for the training data used in `.fit()`.
            This corresponds to selecting, for each row of the label matrix F,
            the class with maximum score.

        **Inductive mode** (`X` provided)
            Returns predictions for new data points.

            Because label propagation is transductive by nature, inductive
            predictions are approximated via a k-NN classifier with *soft*
            weights:

                - neighbors are found in the original feature space,
                - neighbor distances are converted to weights using the same
                  RBF rule as in graph construction,
                - soft labels are averaged using these weights,
                - argmax over classes gives the final predicted label.

        Parameters
        ----------
        X : ndarray of shape (n_samples_new, n_features), optional
            New unseen data for which to compute predictions.
            If None, returns transductive predictions.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        # ---------- Transductive ----------
        if X is None:
            idx = np.argmax(self.F_, axis=1)
            return self.classes_[idx]

        # ---------- Inductive (soft kNN over training set) ----------
        if self._inductive_nbrs is None:
            # Safety fallback (should not happen if fit was called)
            self._inductive_nbrs = NearestNeighbors(
                n_neighbors=self.n_neighbors,
                algorithm=self.algorithm,
                n_jobs=self.n_jobs,
            ).fit(self._X_fit)

        t0_pred = time.perf_counter()

        X = np.asarray(X)
        distances, indices = self._inductive_nbrs.kneighbors(X)

        d2 = distances ** 2
        if self.gamma is None:
            # Use the same global sigma^2 as during graph construction
            weights = np.exp(-d2 / self.sigma2_)
        else:
            weights = np.exp(-self.gamma * d2)

        # Normalize weights per sample
        weights_sum = weights.sum(axis=1, keepdims=True)
        weights_sum[weights_sum == 0] = 1.0
        weights = weights / weights_sum

        # Soft-label vote: shape (n_samples_new, n_classes)
        F_neighbors = self.F_[indices]            # (n_samples_new, k, n_classes)
        soft_scores = np.einsum("nk,nkc->nc", weights, F_neighbors)

        y_idx = np.argmax(soft_scores, axis=1)
        y_pred = self.classes_[y_idx]

        t1_pred = time.perf_counter()
        self.timing_["predict"] = t1_pred - t0_pred

        return y_pred
