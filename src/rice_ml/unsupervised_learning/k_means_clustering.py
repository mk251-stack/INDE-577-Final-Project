"""
K-Means clustering utilities for the Census Income dataset.

This module provides reusable helper functions for:
  - Loading and cleaning the dataset
  - Encoding categorical variables
  - Scaling features
  - Running the elbow method
  - Fitting a K-Means model
  - Reducing dimensionality with PCA
  - Attaching cluster labels and summarizing clusters
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------
# Data Loading & Preprocessing
# ------------------------------------------------------------------

def clean_census_data(df: pd.DataFrame, selected_columns: Sequence[str]) -> pd.DataFrame:
    """
    Subset to selected columns, replace '?' with NaN, and drop rows with missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame.
    selected_columns : Sequence[str]
        Columns to keep for clustering.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df_sub = df.loc[:, selected_columns].copy()
    df_sub.replace("?", np.nan, inplace=True)
    df_sub.dropna(inplace=True)
    df_sub.reset_index(drop=True, inplace=True)
    return df_sub


def encode_features(
    df: pd.DataFrame,
    categorical_cols: Sequence[str],
    drop_first: bool = True
) -> pd.DataFrame:
    """
    One-hot encode categorical columns using pandas.get_dummies.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame.
    categorical_cols : Sequence[str]
        Names of categorical columns to encode.
    drop_first : bool, default True
        Whether to drop the first level of each categorical variable
        to avoid multicollinearity.

    Returns
    -------
    pd.DataFrame
        DataFrame with numerical and one-hot encoded categorical features.
    """
    return pd.get_dummies(df, columns=list(categorical_cols), drop_first=drop_first)


def scale_features(X: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardize features to have mean 0 and variance 1.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    X_scaled : np.ndarray
        Scaled feature matrix.
    scaler : StandardScaler
        Fitted scaler instance (can be reused for inverse transform or new data).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    return X_scaled, scaler


# ------------------------------------------------------------------
# K-Means & Elbow Method
# ------------------------------------------------------------------

def compute_elbow_inertia(
    X_scaled: np.ndarray,
    k_values: Sequence[int],
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
) -> List[Tuple[int, float]]:
    """
    Compute inertia values for a range of cluster counts (K) for the elbow method.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    k_values : Sequence[int]
        Sequence of K values to evaluate.
    random_state : int, default 42
        Random state for KMeans reproducibility.
    n_init : int, default 10
        Number of time the k-means algorithm will be run with different centroid seeds.
    max_iter : int, default 300
        Maximum number of iterations of the k-means algorithm.

    Returns
    -------
    List[Tuple[int, float]]
        List of (K, inertia) pairs.
    """
    results = []
    for k in k_values:
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
        )
        model.fit(X_scaled)
        results.append((k, float(model.inertia_)))
    return results


def fit_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
) -> KMeans:
    """
    Fit a K-Means clustering model.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    n_clusters : int
        Number of clusters (K).
    random_state : int, default 42
        Random state for reproducibility.
    n_init : int, default 10
        Number of time the k-means algorithm will be run with different centroid seeds.
    max_iter : int, default 300
        Maximum number of iterations of the k-means algorithm.

    Returns
    -------
    KMeans
        Fitted KMeans model.
    """
    if isinstance(X_scaled, pd.DataFrame):
        X_scaled = X_scaled.values
    elif not isinstance(X_scaled, np.ndarray):
        raise TypeError("X_scaled must be a numpy array or pandas DataFrame")    

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )
    model.fit(X_scaled)
    return model


# ------------------------------------------------------------------
# Dimensionality Reduction & Post-processing
# ------------------------------------------------------------------

def run_pca(X_scaled: np.ndarray, n_components: int = 2) -> Tuple[PCA, np.ndarray]:
    """
    Run PCA on scaled features for visualization.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    n_components : int, default 2
        Number of principal components.

    Returns
    -------
    pca : PCA
        Fitted PCA object.
    X_pca : np.ndarray
        Transformed data of shape (n_samples, n_components).
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca


def attach_clusters(
    df: pd.DataFrame,
    labels: Sequence[int],
    label_name: str = "cluster"
) -> pd.DataFrame:
    """
    Attach cluster labels to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Original or cleaned DataFrame with one row per observation.
    labels : Sequence[int]
        Cluster labels (e.g., model.labels_).
    label_name : str, default "cluster"
        Name of the new cluster column.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional cluster label column.
    """
    df_with_clusters = df.copy()
    df_with_clusters[label_name] = labels
    return df_with_clusters


def summarize_clusters(
    df_with_clusters: pd.DataFrame,
    cluster_col: str,
    numeric_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Compute summary statistics for each cluster on selected numeric columns.

    Parameters
    ----------
    df_with_clusters : pd.DataFrame
        DataFrame that includes a cluster column.
    cluster_col : str
        Name of the cluster label column.
    numeric_cols : Sequence[str]
        List of numeric column names to summarize.

    Returns
    -------
    pd.DataFrame
        Multi-index DataFrame with statistics (mean, median, count) for each cluster.
    """
    grouped = df_with_clusters.groupby(cluster_col)[list(numeric_cols)]
    summary = grouped.agg(["mean", "median", "count"])
    return summary
