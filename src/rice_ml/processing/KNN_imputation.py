"""
KNN-based imputation module.
Provides a simple, NumPy-only implementation of KNN imputation for missing values.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

__all__ = ["KNNImputer", "knn_impute"]

def _to_2d_float_array(X, name: str) -> np.ndarray:
    """Convert input to a 2D float NumPy array with validation."""
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"Input '{name}' must be a 2D array.")
    try:
        arr = arr.astype(float)
    except Exception:
        raise TypeError(f"All elements of '{name}' must be numeric or NaN.")
    return arr

class KNNImputer:
    """Simple K-Nearest Neighbors imputation (NumPy-only)."""

    def __init__(self, n_neighbors: int = 5):
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be >= 1.")
        self.n_neighbors = n_neighbors

    def fit_transform(self, X):
        """Fit and impute missing values."""
        X_arr = _to_2d_float_array(X, "X")
        X_out = X_arr.copy()
        n_samples, n_features = X_arr.shape

        for i in range(n_samples):
            row = X_arr[i]
            if not np.isnan(row).any():
                continue  # nothing to impute

            # Identify missing positions
            missing_idx = np.where(np.isnan(row))[0]

            # Compute distances to all other rows
            distances = []
            for j in range(n_samples):
                if i == j:
                    continue
                other = X_arr[j]
                mask = ~np.isnan(row) & ~np.isnan(other)
                if mask.sum() == 0:
                    continue
                dist = np.linalg.norm(row[mask] - other[mask])
                distances.append((dist, j))

            if not distances:
                raise ValueError(
                    f"Cannot impute: no valid neighbors found for row {i}."
                )

            # Select K nearest neighbors
            distances.sort(key=lambda x: x[0])
            neighbors = [idx for _, idx in distances[: self.n_neighbors]]

            for col in missing_idx:
                vals = [X_arr[j, col] for j in neighbors if not np.isnan(X_arr[j, col])]
                if not vals:
                    vals = X_arr[:, col][~np.isnan(X_arr[:, col])]
                X_out[i, col] = np.mean(vals)

        return X_out

def knn_impute(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """Impute numeric columns in a pandas DataFrame using KNNImputer."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("knn_impute expects a pandas DataFrame")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return df.copy()

    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(df[num_cols].values)
    out = df.copy()
    out[num_cols] = pd.DataFrame(imputed_array, columns=num_cols, index=df.index)
    return out
