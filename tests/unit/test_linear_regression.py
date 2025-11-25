import numpy as np
import pytest
from rice_ml.supervised_learning.linear_regression import LinearRegression

def test_linear_regression_fit_and_predict():
    # X has no collinearity
    X = np.array([
        [1, 2],
        [2, 0],
        [3, 4],
        [4, 1]
    ])
    
    # Create a synthetic linear relationship:
    # y = 3*x1 - 2*x2 + 5
    y = 3*X[:,0] - 2*X[:,1] + 5

    model = LinearRegression().fit(X, y)

    # Check coefficients are close to expected
    assert np.allclose(model.intercept_, 5.0, atol=1e-6)
    assert np.allclose(model.coef_[0], 3.0, atol=1e-6)
    assert np.allclose(model.coef_[1], -2.0, atol=1e-6)

def test_linear_regression_predict_shape():
    # Non-collinear features
    X = np.array([
        [1, 5],
        [2, 1],
        [4, 3],
        [5, 2],
        [3, 7],
    ])

    # Simple target (any values)
    y = np.array([10, 20, 30, 25, 15])

    model = LinearRegression().fit(X, y)
    preds = model.predict(X)

    # Check correct output shape
    assert preds.shape == (5,)


def test_linear_regression_requires_fit_before_predict():
    model = LinearRegression()
    X = np.array([[1.0, 2.0]])

    with pytest.raises(ValueError):
        model.predict(X)


def test_linear_regression_rejects_singular_design_matrix():
    # Perfect collinearity: second column equals first column
    X = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
    ])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(np.linalg.LinAlgError):
        LinearRegression().fit(X, y)


def test_linear_regression_feature_mismatch_raises():
    X_train = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [2.0, 1.0],
    ])
    y_train = np.array([1.0, 1.5, 2.5, 3.0])

    model = LinearRegression().fit(X_train, y_train)

    X_bad = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        model.predict(X_bad)


def test_linear_regression_rejects_non_numeric_input():
    X = np.array([["a", "b"], ["c", "d"]])
    y = np.array([1.0, 2.0])

    with pytest.raises(TypeError):
        LinearRegression().fit(X, y)


def test_linear_regression_mismatched_lengths_raise_value_error():
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    y = np.array([1.0])

    with pytest.raises(ValueError):
        LinearRegression().fit(X, y)


def test_linear_regression_predict_requires_2d_input():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    y = np.array([1.0, 2.0, 2.5, 3.0])

    model = LinearRegression().fit(X, y)

    with pytest.raises(ValueError):
        model.predict(np.array([1.0, 2.0, 3.0]))


def test_linear_regression_score_matches_manual_r2():
    X = np.array([
        [1.0, 0.5],
        [2.0, 1.0],
        [3.0, 1.5],
        [4.0, 1.2],
        [5.0, 2.0],
    ])
    y = np.array([2.0, 4.1, 6.2, 7.9, 10.1])

    model = LinearRegression().fit(X, y)
    manual_r2 = 1 - np.sum((y - model.predict(X)) ** 2) / np.sum((y - np.mean(y)) ** 2)

    assert np.isclose(model.score(X, y), manual_r2)
    assert model.r2_ == pytest.approx(manual_r2)
    assert model.y_pred_.shape == y.shape
