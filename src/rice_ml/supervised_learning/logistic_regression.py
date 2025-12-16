"""Logistic Regression implemented with gradient descent for binary targets.

This module provides a lightweight, from-scratch implementation of logistic
regression suitable for educational use and small tabular datasets. The model
performs batch gradient descent on the binary cross-entropy objective and
supports an optional intercept term. Targets must be binary ({0, 1} or
{-1, 1}); {-1, 1} labels are normalized internally to {0, 1}.

The solver stops early when the improvement in loss falls below a tolerance.
If the maximum number of iterations is reached without meeting the tolerance,
the estimator raises a UserWarning to signal potential non-convergence.
"""

import numpy as np
import warnings

class LogisticRegression:
    """
    Logistic Regression classifier implemented from scratch using Gradient Descent.

    Parameters
    ----------
    lr : float, optional (default=0.01)
        Learning rate for gradient descent.
    num_iter : int, optional (default=10000)
        Number of iterations used during gradient descent.
    fit_intercept : bool, optional (default=True)
        Whether to include an intercept term.
    tol : float, optional (default=1e-4)
        Minimum improvement in loss required between iterations to continue
        training.
    verbose : bool, optional (default=False)
        If True, prints the loss every 1000 iterations.
    """

    def __init__(self, lr=0.01, num_iter=10000, fit_intercept=True, verbose=False, tol=1e-4):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.tol = tol
        self.theta = None
        self.n_iter_ = 0
        self.converged_ = False

    # ------------------------------
    # Utility Functions
    # ------------------------------
    def _add_intercept(self, X):
        """Adds intercept column of 1s to the dataset."""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def _sigmoid(self, z):
        """Numerically stable sigmoid."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _loss(self, h, y):
        """Binary cross-entropy loss."""
        h = np.clip(h, 1e-10, 1 - 1e-10)  # avoid log(0)
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    # ------------------------------
    # Main Model Functions
    # ------------------------------
    def fit(self, X, y):
        """
        Fit logistic regression using gradient descent.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if y.ndim != 1:
            raise ValueError("y must be a 1D array of binary labels.")

        unique_labels = set(np.unique(y))
        if unique_labels == {-1, 1}:
            warnings.warn(
                "Converting labels from {-1, 1} to {0, 1} for logistic regression.",
                UserWarning,
            )
            y = (y + 1) / 2
        elif unique_labels != {0, 1}:
            warnings.warn(
                "LogisticRegression only supports binary targets {0, 1} or {-1, 1}.",
                UserWarning,
            )
            raise ValueError("Invalid labels: expected binary targets {0, 1} or {-1, 1}.")

        if self.fit_intercept:
            X = self._add_intercept(X)

        # Initialize weights
        n_features = X.shape[1]
        self.theta = np.zeros(n_features)
        self.converged_ = False

        # Gradient descent
        prev_loss = np.inf
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self._sigmoid(z)

            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            self.n_iter_ = i + 1

            loss = self._loss(h, y)
            if abs(prev_loss - loss) < self.tol:
                self.converged_ = True
                break
            prev_loss = loss

            if self.verbose and i % 1000 == 0:
                print(f"Iteration {i} - Loss: {loss:.6f}")

        if not self.converged_:
            warnings.warn(
                "LogisticRegression did not converge within the specified iterations; "
                "consider increasing num_iter, lowering lr, or relaxing tol.",
                UserWarning,
            )

        return self

    def predict_proba(self, X):
        """
        Predict probability estimates for samples in X.
        """
        X = np.array(X)
        if self.fit_intercept:
            X = self._add_intercept(X)

        return self._sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.5):
        """
        Predict binary labels for samples in X.
        """
        return (self.predict_proba(X) >= threshold).astype(int)
