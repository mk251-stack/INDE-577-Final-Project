import numpy as np
from rice_ml.processing.scaling import StandardScaler

def test_standard_scaler_fit_transform():
    X = np.array([[1, 2], [3, 4], [5, 6]])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Mean of scaled data should be approx 0
    assert np.allclose(X_scaled.mean(axis=0), 0.0)

    # Std of scaled data should be approx 1
    assert np.allclose(X_scaled.std(axis=0), 1.0)

def test_standard_scaler_inverse_behavior():
    X = np.array([[10, 20], [30, 40]])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Transforming again should not change mean/std
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-7)


def test_standard_scaler_recovers_original_values():
    X = np.array([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reconstruct the original values using stored mean and std
    X_recovered = X_scaled * scaler.std_ + scaler.mean_

    assert np.allclose(X_recovered, X)
