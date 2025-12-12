import numpy as np
import pandas as pd
import pytest
from rice_ml.unsupervised_learning.dimensionality_reduction import (
    load_energy_data,
    select_numeric_features,
    scale_energy_features,
    run_energy_pca,
    create_pca_dataframe
)


# ----------------------------
# PCA — Basic functionality
# ----------------------------

def test_pca_runs_and_returns_components():
    X = np.array([
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [4.0, 5.0, 6.0],
        [4.1, 5.1, 6.1]
    ])

    # Simulate scaling
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)

    pca, X_pca = run_energy_pca(X_scaled, n_components=2)

    assert X_pca.shape[1] == 2
    assert hasattr(pca, "components_")
    assert pca.components_.shape == (2, 3)


def test_pca_explained_variance_is_valid():
    X = np.random.rand(50, 5)
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)

    pca, X_pca = run_energy_pca(X_scaled, 3)

    evr = pca.explained_variance_ratio_
    assert len(evr) == 3
    assert np.all(evr >= 0)
    assert np.sum(evr) <= 1.0


def test_pca_dataframe_output_format():
    X = np.random.rand(20, 4)
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)

    pca, X_pca = run_energy_pca(X_scaled, 3)
    df_pca = create_pca_dataframe(X_pca)

    assert isinstance(df_pca, pd.DataFrame)
    assert df_pca.shape == (20, 3)
    assert all(col in df_pca.columns for col in ["PC1", "PC2", "PC3"])


# ----------------------------
# PCA — Error handling
# ----------------------------

def test_pca_rejects_non_array_input():
    with pytest.raises(Exception):
        run_energy_pca("not-an-array", 2)


def test_pca_requires_numeric_values():
    df = pd.DataFrame({
        "A": [1, 2, "bad"],
        "B": [3, 4, 5]
    })

    # Should drop column A and keep only numeric column B
    numeric_df = select_numeric_features(df)
    assert list(numeric_df.columns) == ["B"]

    # Scaling should work on remaining numeric data
    X_scaled, _ = scale_energy_features(numeric_df)
    assert X_scaled.shape == (3, 1)

# ----------------------------
# PCA — Behavior tests
# ----------------------------

def test_pca_captures_separation_in_variance():
    # Two distinct groups in 5D space
    group1 = np.random.normal(0, 0.1, (50, 5))
    group2 = np.random.normal(5, 0.1, (50, 5))
    X = np.vstack([group1, group2])

    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)

    pca, X_pca = run_energy_pca(X_scaled, 1)

    # PC1 should strongly separate the two clusters
    pc1_values = X_pca[:, 0]
    diff = np.abs(np.mean(pc1_values[:50]) - np.mean(pc1_values[50:]))

    assert diff > 3.0  # large separation in projected 1D space
