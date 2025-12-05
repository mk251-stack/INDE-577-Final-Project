import numpy as np
import pytest

from rice_ml.supervised_learning.logistic_regression import LogisticRegression


def test_logistic_regression_fits_linearly_separable_data():
    # Simple 1D dataset: negative values -> class 0, positive values -> class 1
    X = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegression(lr=0.1, num_iter=5000)
    model.fit(X, y)

    probs = model.predict_proba(X)
    preds = model.predict(X)

    # Model should confidently separate negative and positive values
    assert np.all(probs[:3] < 0.1)
    assert np.all(probs[3:] > 0.9)
    assert np.array_equal(preds, y)


def test_logistic_regression_predict_proba_bounds_and_shape():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 0, 1])

    model = LogisticRegression(lr=0.1, num_iter=2000)
    model.fit(X, y)

    probs = model.predict_proba(X)

    assert probs.shape == (3,)
    assert np.all(probs > 0)
    assert np.all(probs < 1)


def test_logistic_regression_rejects_non_1d_targets():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([[0], [1], [0]])  # 2D labels should be rejected

    model = LogisticRegression()

    with pytest.raises(ValueError):
        model.fit(X, y)