"""
K-Means clustering utilities for the Census Income dataset.

This module provides reusable helper functions for:
  - Loading and cleaning the dataset
  - Encoding categorical variables
  - Scaling features
  - Running the elbow method
  - Fitting a K-Means model
  - Reducing dimensionality with PCA (for visualization)
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
    """
    return pd.get_dummies(df, columns=list(categorical_cols), drop_first=drop_first)


def scale_features(X: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardize features to mean 0 and variance 1.
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
    """
    if isinstance(X_scaled, pd.DataFrame):
        X_scaled = X_scaled.values
    elif not isinstance(X_scaled, np.ndarray):
        raise TypeError("X_scaled must be a numpy array or pandas DataFrame")

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
    Run PCA on scaled features for visualization and exploratory analysis.
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
    """
    if len(df) != len(labels):
        raise ValueError("Length of labels must match number of rows in df")

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
    """
    grouped = df_with_clusters.groupby(cluster_col)[list(numeric_cols)]
    summary = grouped.agg(["mean", "median", "count"])
    return summary
