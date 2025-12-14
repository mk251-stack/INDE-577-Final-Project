import numpy as np
import pandas as pd
import pytest

from rice_ml.unsupervised_learning.dimensionality_reduction import (
    load_energy_data,
    select_numeric_features,
    scale_energy_features,
    run_energy_pca,
    create_pca_dataframe,
    get_pca_variance,
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

    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)

    pca, X_pca = run_energy_pca(X_scaled, n_components=2)

    assert X_pca.shape == (4, 2)
    assert hasattr(pca, "components_")
    assert pca.components_.shape == (2, 3)


def test_pca_explained_variance_is_valid():
    X = np.random.rand(50, 5)

    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)

    pca, _ = run_energy_pca(X_scaled, 3)

    evr = pca.explained_variance_ratio_
    assert len(evr) == 3
    assert np.all(evr >= 0)
    assert np.sum(evr) <= 1.0


def test_pca_dataframe_output_format():
    X = np.random.rand(20, 4)

    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)

    _, X_pca = run_energy_pca(X_scaled, 3)
    df_pca = create_pca_dataframe(X_pca)

    assert isinstance(df_pca, pd.DataFrame)
    assert df_pca.shape == (20, 3)
    assert list(df_pca.columns) == ["PC1", "PC2", "PC3"]


# ----------------------------
# PCA — Error handling
# ----------------------------

def test_pca_rejects_non_array_input():
    with pytest.raises(ValueError):
        run_energy_pca("not-an-array", 2)


def test_pca_requires_numeric_values():
    df = pd.DataFrame({
        "A": [1, 2, "bad"],
        "B": [3, 4, 5]
    })

    numeric_df = select_numeric_features(df)
    assert list(numeric_df.columns) == ["B"]

    X_scaled, _ = scale_energy_features(numeric_df)
    assert X_scaled.shape == (3, 1)


# ----------------------------
# PCA — Behavioral tests
# ----------------------------

def test_pca_captures_separation_in_variance():
    rng = np.random.default_rng(42)

    group1 = rng.normal(0, 0.1, (50, 5))
    group2 = rng.normal(5, 0.1, (50, 5))
    X = np.vstack([group1, group2])

    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)

    _, X_pca = run_energy_pca(X_scaled, 1)

    pc1 = X_pca[:, 0]
    diff = abs(pc1[:50].mean() - pc1[50:].mean())

    assert diff > 3.0


def test_pca_is_deterministic_given_seed():
    X = np.random.rand(30, 4)

    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)

    _, X1 = run_energy_pca(X_scaled, 2, random_state=42)
    _, X2 = run_energy_pca(X_scaled, 2, random_state=42)

    assert np.allclose(X1, X2)


# ----------------------------
# PCA — Utility function tests
# ----------------------------

def test_get_pca_variance_matches_explained_variance():
    X = np.random.rand(20, 3)

    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)

    pca, _ = run_energy_pca(X_scaled, 2)
    variance = get_pca_variance(pca)

    assert len(variance) == 2
    assert variance[0][0] == "PC1"
    assert variance[1][0] == "PC2"
    assert np.isclose(variance[0][1], pca.explained_variance_ratio_[0])
    assert np.isclose(variance[1][1], pca.explained_variance_ratio_[1])
