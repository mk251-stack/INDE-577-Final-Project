"""
DBSCAN Analysis Utility Functions
---------------------------------
This module contains helper functions commonly used when performing DBSCAN
clustering on numerical datasets such as energy.csv or BostonHousing.

Functions included:
- load_dataset(path)
- select_numeric(df)
- scale_features(df)
- compute_k_distance(df, k)
- run_dbscan(df, eps, min_samples)
- add_cluster_labels(df, labels)
- plot_k_distance(distances)
- plot_clusters(df, labels, x_col, y_col)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


def load_dataset(path: str) -> pd.DataFrame:
    """Load a dataset from a CSV file."""
    return pd.read_csv(path)


def select_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Return only numeric columns from the dataset."""
    return df.select_dtypes(include=["number"]).dropna()


def scale_features(df: pd.DataFrame) -> np.ndarray:
    """Scale features using StandardScaler and return numpy array."""
    scaler = StandardScaler()
    return scaler.fit_transform(df)


def compute_k_distance(data: np.ndarray, k: int = 5) -> np.ndarray:
    """Compute the k-distance for each point (used for selecting eps)."""
    nbrs = NearestNeighbors(n_neighbors=k).fit(data)
    distances, _ = nbrs.kneighbors(data)
    k_distances = np.sort(distances[:, -1])
    return k_distances


def run_dbscan(data: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Run DBSCAN clustering and return cluster labels."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data)
    return labels


def add_cluster_labels(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Add cluster labels to the dataframe and return it."""
    df = df.copy()
    df["cluster"] = labels
    return df


def plot_k_distance(distances: np.ndarray):
    """Plot the sorted k-distance curve to help choose eps."""
    plt.figure(figsize=(8,4))
    plt.plot(distances)
    plt.title("k-distance Plot")
    plt.xlabel("Points Sorted by Distance")
    plt.ylabel("k-distance")
    plt.grid(True)
    plt.show()


def plot_clusters(df: pd.DataFrame, labels: np.ndarray, x_col: str, y_col: str):
    """Plot DBSCAN clusters using two selected numeric columns."""
    plt.figure(figsize=(7,5))
    scatter = plt.scatter(df[x_col], df[y_col], c=labels, cmap="viridis", s=40)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("DBSCAN Clusters")
    plt.colorbar(scatter, label="Cluster Label")
    plt.grid(True)
    plt.show()
