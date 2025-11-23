"""
Scaling and normalization utilities.

This module provides a NumPy-based StandardScaler
(similar to scikit-learn), with explicit validation logic.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

__all__ = ["StandardScaler"]


def _to_2d_float_array(X, name: str) -> np.ndarray:
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"Input '{name}' must be a 2D array.")
    try:
        return arr.astype(float)
    except Exception:
        raise TypeError(f"All elements of '{name}' must be numeric.")


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.

    Parameters
    ----------
    with_mean : bool, default=True
        Whether to center the data before scaling.
    with_std : bool, default=True
        Whether to scale to unit variance.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X) -> "StandardScaler":
        """
        Compute mean and std for scaling.

        Returns
        -------
        self : StandardScaler
        """
        X_arr = _to_2d_float_array(X, "X")

        self.mean_ = X_arr.mean(axis=0) if self.with_mean else np.zeros(X_arr.shape[1])
        self.std_ = X_arr.std(axis=0, ddof=0) if self.with_std else np.ones(X_arr.shape[1])

        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)  # avoid division by zero

        return self

    def transform(self, X):
        """
        Scale features using previously computed mean/std.
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet.")

        X_arr = _to_2d_float_array(X, "X")

        return (X_arr - self.mean_) / self.std_

    def fit_transform(self, X):
        """
        Fit and transform.
        """
        self.fit(X)
        return self.transform(X)
