"""
PCA utilities for dimensionality reduction and analysis on energy datasets.
"""

from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------
# Data Loading & Preprocessing
# ------------------------------------------------------------------

def load_energy_data(path: str) -> pd.DataFrame:
    """
    Load energy dataset from a CSV file.

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


def select_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only numerical columns for PCA.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only numeric columns.
    """
    return df.select_dtypes(include=["float64", "int64"])


def scale_energy_features(X: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardize numeric features to mean 0 and variance 1.

    Parameters
    ----------
    X : pd.DataFrame
        Numeric feature subset.

    Returns
    -------
    np.ndarray
        Scaled feature matrix.
    StandardScaler
        Fitted scaler instance for later reuse.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    return X_scaled, scaler


# ------------------------------------------------------------------
# PCA
# ------------------------------------------------------------------

def run_energy_pca(
    X_scaled: np.ndarray,
    n_components: int = 2,
    random_state: int = 42
) -> Tuple[PCA, np.ndarray]:
    """
    Fit and transform data using PCA.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    n_components : int, default 2
        Number of PCA dimensions to reduce to.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    PCA
        Fitted PCA model.
    np.ndarray
        Reduced data matrix of shape (n_samples, n_components)
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca


def get_pca_variance(pca: PCA) -> List[Tuple[str, float]]:
    """
    Return explained variance ratio of PCA components.

    Parameters
    ----------
    pca : PCA
        Fitted PCA model.

    Returns
    -------
    List of (component_name, variance_ratio) pairs
    """
    return [(f"PC{i+1}", var) for i, var in enumerate(pca.explained_variance_ratio_)]


# ------------------------------------------------------------------
# Post Processing
# ------------------------------------------------------------------

def create_pca_dataframe(X_pca: np.ndarray) -> pd.DataFrame:
    """
    Convert PCA output matrix into a DataFrame with PC labels.

    Parameters
    ----------
    X_pca : np.ndarray
        PCA-reduced feature matrix.

    Returns
    -------
    pd.DataFrame
        DataFrame with labeled principal components.
    """
    n = X_pca.shape[1]
    cols = [f"PC{i+1}" for i in range(n)]
    return pd.DataFrame(X_pca, columns=cols)


def save_reduced_data(df_pca: pd.DataFrame, output_path: str) -> None:
    """
    Save PCA-reduced DataFrame to CSV for reuse.

    Parameters
    ----------
    df_pca : pd.DataFrame
        Reduced PCA feature DataFrame.
    output_path : str
        Path to save the CSV.
    """
    df_pca.to_csv(output_path, index=False)
