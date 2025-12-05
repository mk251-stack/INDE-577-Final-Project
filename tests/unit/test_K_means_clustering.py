import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from rice_ml.unsupervised_learning.k_means_clustering import (
    load_census_data,
    clean_census_data,
    encode_features,
    scale_features,
    fit_kmeans,
    summarize_clusters
)

# Synthetic scaled sample for clustering utilities
def test_kmeans_fit_clusters():
    X = np.array([
        [0.2, 0.6],
        [0.21, 0.59],
        [0.8, 0.1],
        [0.79, 0.12],
        [0.5, 0.5]
    ])
    k = 2
    model = fit_kmeans(X, k)
    assert hasattr(model, "cluster_centers_")
    assert model.cluster_centers_.shape[0] == k
    assert model.labels_.shape == (X.shape[0],)

def test_kmeans_prediction_returns_valid_cluster():
    X = np.random.rand(50, 5)
    model = fit_kmeans(X, 3)
    test_sample = X[0].reshape(1, -1)
    cluster = model.predict(test_sample)[0]
    assert isinstance(cluster, (np.integer, int))
    assert 0 <= cluster < 3

def test_kmeans_summarize_requires_numeric_list():
    df = pd.DataFrame({
        "age": [22, 25, 30, 28],
        "education_num": [10, 13, 14, 12],
        "hours_per_week": [40, 50, 60, 45],
        "cluster": [0, 0, 1, 1],
        "income": [0, 1, 1, 0]
    })
    summary = summarize_clusters(df, "cluster", ["age", "education_num"])
    assert ("mean" in summary.columns.get_level_values(1))
    assert summary.shape[0] == 2

def test_kmeans_rejects_bad_input_format():
    with pytest.raises(TypeError):
        fit_kmeans("not-an-array", 3)

def test_kmeans_captures_separation_on_clear_clusters():
    X = np.array([
        [0.1, 0.1],
        [0.15, 0.12],
        [0.12, 0.08],
        [10.0, 10.0],
        [10.5, 9.8],
        [9.9, 10.2]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
    clusters = model.labels_
    separated_cluster = clusters[:3]
    assert len(set(separated_cluster)) == 1
