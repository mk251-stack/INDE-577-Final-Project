import numpy as np
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
        [4, 3]
    ])
    
    # Simple target (any values)
    y = np.array([10, 20, 30])

    model = LinearRegression().fit(X, y)
    preds = model.predict(X)

    # Check correct output shape
    assert preds.shape == (3,)
