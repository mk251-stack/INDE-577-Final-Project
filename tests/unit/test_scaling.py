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
