"""
DBSCAN clustering utilities for numerical datasets.

This module provides reusable helper functions for:
- Loading datasets
- Selecting numeric features
- Feature scaling
- k-distance computation (eps selection)
- Running DBSCAN
- Attaching cluster labels
- Visualization helpers

Designed for educational and exploratory clustering workflows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


# ------------------------------------------------------------------
# Data Loading & Preprocessing
# ------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a dataset from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    return pd.read_csv(path)


def select_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select numeric columns and drop rows with missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only numeric columns.
    """
    return df.select_dtypes(include=["number"]).dropna()


def scale_features(df: pd.DataFrame) -> np.ndarray:
    """
    Standardize numeric features to zero mean and unit variance.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing numeric features only.

    Returns
    -------
    np.ndarray
        Scaled feature matrix.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(df.values)


# ------------------------------------------------------------------
# DBSCAN Utilities
# ------------------------------------------------------------------

def compute_k_distance(data: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Compute sorted k-distances for DBSCAN eps selection.

    Parameters
    ----------
    data : np.ndarray
        Scaled feature matrix.
    k : int, default 5
        Number of neighbors.

    Returns
    -------
    np.ndarray
        Sorted array of k-distances.

    Raises
    ------
    ValueError
        If k < 1.
    TypeError
        If data is not a numpy array.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy array")
    if k < 1:
        raise ValueError("k must be >= 1")

    nbrs = NearestNeighbors(n_neighbors=k).fit(data)
    distances, _ = nbrs.kneighbors(data)

    return np.sort(distances[:, -1])


def run_dbscan(data: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    Run DBSCAN clustering.

    Parameters
    ----------
    data : np.ndarray
        Scaled feature matrix.
    eps : float
        Neighborhood radius.
    min_samples : int
        Minimum samples for a core point.

    Returns
    -------
    np.ndarray
        Cluster labels (noise labeled as -1).

    Raises
    ------
    TypeError
        If data is not a numpy array.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy array")

    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(data)


def add_cluster_labels(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Attach cluster labels to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame.
    labels : np.ndarray
        Cluster labels.

    Returns
    -------
    pd.DataFrame
        New DataFrame with 'cluster' column.
    """
    df_out = df.copy()
    df_out["cluster"] = labels
    return df_out


# ------------------------------------------------------------------
# Visualization Helpers (Notebook Use)
# ------------------------------------------------------------------

def plot_k_distance(distances: np.ndarray) -> None:
    """
    Plot sorted k-distance curve to guide eps selection.

    Parameters
    ----------
    distances : np.ndarray
        Sorted k-distance values.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(distances)
    plt.xlabel("Points (sorted)")
    plt.ylabel("k-distance")
    plt.title("k-distance Plot for DBSCAN eps Selection")
    plt.grid(True)
    plt.show()


def plot_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    x_col: str,
    y_col: str
) -> None:
    """
    Plot DBSCAN clusters in 2D (typically PCA output).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 2D coordinates.
    labels : np.ndarray
        Cluster labels.
    x_col : str
        X-axis column name.
    y_col : str
        Y-axis column name.
    """
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        df[x_col],
        df[y_col],
        c=labels,
        cmap="viridis",
        s=40
    )
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("DBSCAN Clusters")
    plt.colorbar(scatter, label="Cluster Label")
    plt.grid(True)
    plt.show()
