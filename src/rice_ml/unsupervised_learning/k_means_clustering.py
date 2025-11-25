"""
 K-Means Clustering Implementation and Helper Functions.

This module provides a structured pipeline for K-Means clustering analysis,
including essential steps for data preprocessing (handling missing values,
One-Hot Encoding, standardization), model fitting, and optimal K determination.

The functions are designed to be imported and orchestrated within a
Jupyter Notebook, where visualization functions will be defined or called
separately for reproducible data science workflows.
"""
from __future__ import annotations
from typing import Tuple

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

__all__ = [
    "load_and_clean_data",
    "preprocess_for_kmeans",
    "find_optimal_k",
    "perform_kmeans_clustering",
]

# ---------------------------------------------------------------------
# Data Preparation and Preprocessing
# ---------------------------------------------------------------------

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Loads the census income data and performs initial cleaning steps.

    1. Replaces ' ?' (implicit missing values) with NaN.
    2. Drops columns that are redundant for clustering ('education', 'fnlwgt', 'income').
    3. Imputes missing categorical values using the mode.

    Parameters
    ----------
    filepath : str
        Path to the census_income.csv file.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame ready for feature engineering.
    """
    # Load the data and replace explicit missing markers ' ?' with NaN
    df = pd.read_csv(filepath)
    df = df.replace(' ?', np.nan)

    # 1. Drop redundant columns for clustering
    df = df.drop(columns=['education', 'fnlwgt', 'income'], errors='ignore')

    # 2. Impute missing categorical data using the mode
    # Missing values are typically in 'workclass', 'occupation', 'native_country'
    for col in ['workclass', 'occupation', 'native_country']:
        if col in df.columns and df[col].isnull().any():
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)

    print(f"Initial shape after cleaning: {df.shape}")
    print(f"Missing values remaining: {df.isnull().sum().sum()}")
    return df

def preprocess_for_kmeans(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Performs feature engineering and scaling necessary for K-Means.

    1. Groups 'native_country' into two categories (United-States/Other).
    2. Applies One-Hot Encoding (OHE) to all categorical features.
    3. Applies StandardScaler (Standardization) to the final numerical matrix.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, StandardScaler]
        - df_scaled: The final scaled data used for clustering.
        - df_encoded: The one-hot encoded but unscaled data (for profiling).
        - scaler: The fitted StandardScaler object.
    """
    df_processed = df.copy()

    # 1. Address high cardinality in 'native_country'
    df_processed['native_country_simple'] = np.where(
        df_processed['native_country'] == ' United-States',
        'United-States',
        'Other'
    )
    df_processed = df_processed.drop(columns=['native_country'])

    # 2. Identify categorical columns
    categorical_cols = df_processed.select_dtypes(include='object').columns

    # 3. One-Hot Encoding (OHE) for all categorical features
    df_encoded = pd.get_dummies(
        df_processed,
        columns=categorical_cols,
        drop_first=True # Drop one category to avoid multicollinearity
    )
    print(f"Shape after One-Hot Encoding: {df_encoded.shape}")

    # 4. Feature Scaling (StandardScaler is mandatory for distance-based K-Means)
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_encoded),
        columns=df_encoded.columns,
        index=df_encoded.index
    )

    return df_scaled, df_encoded, scaler

# ---------------------------------------------------------------------
# K-Means Modeling and Optimal K Determination
# ---------------------------------------------------------------------

def find_optimal_k(df_scaled: pd.DataFrame, max_k: int = 15) -> pd.DataFrame:
    """
    Calculates Inertia (WCSS) and Silhouette Score for a range of K values
    to help determine the optimal number of clusters.

    Parameters
    ----------
    df_scaled : pd.DataFrame
        The scaled data for clustering.
    max_k : int, default=15
        Maximum K value to test.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing K, Inertia, and Silhouette Score for K=2 to max_k.
    """
    results = []

    for k in range(2, max_k + 1):
        try:
            # Use fixed random_state and n_init for reproducibility and stability
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(df_scaled)
            inertia = kmeans.inertia_

            # Calculate Silhouette Score
            silhouette_avg = silhouette_score(df_scaled, kmeans.labels_)

            results.append({
                'K': k,
                'Inertia': inertia,
                'Silhouette Score': silhouette_avg
            })
            print(f"Calculated metrics for K={k}")
        except Exception as e:
            print(f"Error for K={k}: {e}")
            break

    return pd.DataFrame(results)

def perform_kmeans_clustering(df_scaled: pd.DataFrame, df_original: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Fits the final K-Means model with the chosen K and assigns cluster labels
    back to the original (pre-scaled, one-hot encoded) DataFrame.

    Parameters
    ----------
    df_scaled : pd.DataFrame
        The scaled data used for fitting.
    df_original : pd.DataFrame
        The one-hot encoded (but unscaled) data used for profiling.
    k : int
        The chosen optimal number of clusters.

    Returns
    -------
    pd.DataFrame
        The original encoded DataFrame with the 'Cluster_Label' column added.

    Raises
    ------
    ValueError
        If the number of clusters (k) is less than 2.
    """
    if k < 2:
        raise ValueError("The number of clusters (k) must be 2 or greater for K-Means.")

    print(f"Fitting K-Means model with K={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)

    # Assign cluster labels back to the DataFrame used for analysis
    df_clustered = df_original.copy()
    df_clustered['Cluster_Label'] = kmeans.labels_

    print("Clustering complete. Labels added.")
    return df_clustered