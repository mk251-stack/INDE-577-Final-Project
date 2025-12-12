import numpy as np
import pandas as pd
import pytest

from rice_ml.unsupervised_learning.dbscan import (
    load_dataset,
    select_numeric,
    scale_features,
    compute_k_distance,
    run_dbscan,
    add_cluster_labels,
)

# --------------------------------------------------
# Fixtures
# --------------------------------------------------

@pytest.fixture
def sample_dataframe():
    """Create a small mixed-type dataframe for testing."""
    return pd.DataFrame({
        "numeric_1": [1.0, 2.0, 3.0, 4.0],
        "numeric_2": [10.0, 20.0, 30.0, 40.0],
        "category": ["a", "b", "c", "d"]
    })


@pytest.fixture
def numeric_dataframe(sample_dataframe):
    """Return only numeric columns."""
    return select_numeric(sample_dataframe)


# --------------------------------------------------
# Tests
# --------------------------------------------------

def test_select_numeric_returns_only_numeric_columns(sample_dataframe):
    numeric_df = select_numeric(sample_dataframe)

    assert isinstance(numeric_df, pd.DataFrame)
    assert numeric_df.shape[1] == 2
    assert "category" not in numeric_df.columns


def test_scale_features_output_shape(numeric_dataframe):
    scaled = scale_features(numeric_dataframe)

    assert isinstance(scaled, np.ndarray)
    assert scaled.shape == numeric_dataframe.shape


def test_scale_features_mean_close_to_zero(numeric_dataframe):
    scaled = scale_features(numeric_dataframe)

    # Mean of scaled features should be ~0
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-7)


def test_compute_k_distance_returns_sorted_array(numeric_dataframe):
    scaled = scale_features(numeric_dataframe)
    distances = compute_k_distance(scaled, k=2)

    assert isinstance(distances, np.ndarray)
    assert distances.ndim == 1
    assert np.all(distances[:-1] <= distances[1:])


def test_run_dbscan_returns_labels(numeric_dataframe):
    scaled = scale_features(numeric_dataframe)

    labels = run_dbscan(
        data=scaled,
        eps=1.5,
        min_samples=2
    )

    assert isinstance(labels, np.ndarray)
    assert len(labels) == len(numeric_dataframe)


def test_run_dbscan_noise_label_exists():
    """DBSCAN should assign -1 when eps is very small."""
    data = np.array([
        [0.0, 0.0],
        [0.01, 0.01],
        [10.0, 10.0]
    ])

    labels = run_dbscan(data, eps=0.05, min_samples=2)

    assert -1 in labels


def test_add_cluster_labels_adds_column(numeric_dataframe):
    scaled = scale_features(numeric_dataframe)
    labels = run_dbscan(scaled, eps=1.5, min_samples=2)

    clustered_df = add_cluster_labels(numeric_dataframe, labels)

    assert "cluster" in clustered_df.columns
    assert len(clustered_df["cluster"]) == len(numeric_dataframe)


def test_add_cluster_labels_does_not_modify_original_df(numeric_dataframe):
    scaled = scale_features(numeric_dataframe)
    labels = run_dbscan(scaled, eps=1.5, min_samples=2)

    _ = add_cluster_labels(numeric_dataframe, labels)

    # Original dataframe should not have 'cluster'
    assert "cluster" not in numeric_dataframe.columns
