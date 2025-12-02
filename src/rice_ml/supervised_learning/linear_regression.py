"""
Linear Regression (Ordinary Least Squares) model implementation.

This module provides a from-scratch OLS linear regression model with full
statistical outputs, including coefficient estimates, residual analysis,
standard errors, t-statistics, p-values, and confidence intervals.

The implementation is designed for educational clarity and transparency.
NumPy is used for all matrix operations.

Example
-------
>>> import numpy as np
>>> from rice_ml.supervised_learning.linear_regression import LinearRegression
>>> X = np.array([[1, 2], [2, 0], [3, 4]])
>>> y = np.array([3, 1, 7])
>>> model = LinearRegression().fit(X, y)
>>> model.coef_
array([...])
>>> model.predict(X)
array([...])
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

__all__ = ["LinearRegression"]


# ---------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------

def _to_2d_float_array(X, name: str) -> np.ndarray:
    """
    Convert input to a 2D float NumPy array with consistent validation.

    Parameters
    ----------
    X : array_like
        Input feature matrix.
    name : str
        Name used in error messages ("X").

    Returns
    -------
    np.ndarray
        2D float array.

    Raises
    ------
    ValueError
        If input is not 2D.
    TypeError
        If non-numeric values are detected.
    """
    arr = np.asarray(X)

    if arr.ndim != 2:
        raise ValueError(f"Input '{name}' must be a 2D array; got {arr.ndim}D.")

    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"All elements of '{name}' must be numeric.")

    try:
        arr = arr.astype(float, copy=False)
    except Exception:
        raise TypeError(f"All elements of '{name}' must be convertible to float.")

    return arr


def _to_1d_float_array(y, name: str) -> np.ndarray:
    """
    Convert input to a 1D float NumPy array (target vector).
    """
    arr = np.asarray(y)

    if arr.ndim != 1:
        raise ValueError(f"Input '{name}' must be 1-dimensional.")

    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"All elements of '{name}' must be numeric.")

    try:
        arr = arr.astype(float, copy=False)
    except Exception:
        raise TypeError(f"All elements of '{name}' must be convertible to float.")

    return arr


def _validate_X_y(X, y) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and align feature matrix X and target vector y.
    """
    X_arr = _to_2d_float_array(X, "X")
    y_arr = _to_1d_float_array(y, "y")

    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples: "
            f"X has {X_arr.shape[0]} rows, y has {y_arr.shape[0]}."
        )

    return X_arr, y_arr


# ---------------------------------------------------------------------
# Linear Regression Model
# ---------------------------------------------------------------------

class LinearRegression:
    """
    Ordinary Least Squares (OLS) Linear Regression.

    This class estimates regression parameters using the closed-form
    normal equation:

        beta = (X^T X)^(-1) X^T y

    where X includes an intercept term automatically.

    Attributes
    ----------
    coef_ : np.ndarray
        Estimated coefficient vector (excluding intercept).
    intercept_ : float
        Estimated intercept term.
    residuals_ : np.ndarray
        Vector of residuals (y - ŷ).
    y_pred_ : np.ndarray
        Fitted values.
    r2_ : float
        Coefficient of determination.
    adj_r2_ : float
        Adjusted R².
    stderr_ : np.ndarray
        Standard errors of coefficient estimates.
    tstats_ : np.ndarray
        t-statistics for coefficients.
    pvalues_ : np.ndarray
        Two-sided p-values for coefficient significance.
    cov_matrix_ : np.ndarray
        Variance-covariance matrix of β estimates.
    """

    def __init__(self) -> None:
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.residuals_: Optional[np.ndarray] = None
        self.y_pred_: Optional[np.ndarray] = None
        self.r2_: Optional[float] = None
        self.adj_r2_: Optional[float] = None
        self.stderr_: Optional[np.ndarray] = None
        self.tstats_: Optional[np.ndarray] = None
        self.pvalues_: Optional[np.ndarray] = None
        self.cov_matrix_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Fit Model
    # ------------------------------------------------------------------
    def fit(self, X, y) -> "LinearRegression":
        """
        Fit the OLS regression model.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Input feature matrix.
        y : array_like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : LinearRegression
            Fitted model.
        """
        X_arr, y_arr = _validate_X_y(X, y)

        # Add intercept column of 1s
        X_design = np.column_stack([np.ones(X_arr.shape[0]), X_arr])

        # Compute beta = (X^T X)^(-1) X^T y
        XtX = X_design.T @ X_design

        # Numerical stability check
        if np.linalg.cond(XtX) > 1e12:
            raise np.linalg.LinAlgError(
                "X^T X is nearly singular; model may be unstable."
            )

        beta = np.linalg.solve(XtX, X_design.T @ y_arr)

        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]

        # Predictions
        y_pred = X_design @ beta
        self.y_pred_ = y_pred

        # Residuals
        residuals = y_arr - y_pred
        self.residuals_ = residuals

        # -----------------------------
        # Statistics
        # -----------------------------
        n, p = X_arr.shape
        df = n - (p + 1)  # degrees of freedom
        if df <= 0:
            raise ValueError(
                "Degrees of freedom must be positive; provide more samples "
                "or reduce the number of features."
            )
        
        sse = np.sum(residuals**2)
        sst = np.sum((y_arr - np.mean(y_arr))**2)
        ssr = sst - sse

        self.r2_ = 1 - (sse / sst)
        self.adj_r2_ = 1 - ((1 - self.r2_) * (n - 1) / df)

        # Variance of residuals
        sigma2 = sse / df

        # Var(beta) = sigma^2 * (X^T X)^(-1)
        cov = sigma2 * np.linalg.solve(XtX, np.eye(XtX.shape[0]))
        self.cov_matrix_ = cov

        # Standard errors exclude intercept? No ? include all.
        stderr = np.sqrt(np.diag(cov))
        self.stderr_ = stderr

        # t-statistics (avoid divide-by-zero warnings when a std err is zero)
        safe_stderr = np.where(stderr == 0, np.inf, stderr)
        tstats = beta / safe_stderr
        self.tstats_ = tstats

        # Two-sided p-values (normal approx for simplicity)
        # Could replace with t-distribution if desired.
        from scipy.stats import t
        self.pvalues_ = 2 * (1 - t.cdf(np.abs(tstats), df=df))

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X) -> np.ndarray:
        """
        Predict target values using the fitted model.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted values.

        Raises
        ------
        ValueError
            If model has not been fitted.
        """
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Model has not been fitted yet.")

        X_arr = _to_2d_float_array(X, "X")

        if X_arr.shape[1] != len(self.coef_):
            raise ValueError(
                f"Input has {X_arr.shape[1]} features, "
                f"but model expects {len(self.coef_)}."
            )

        return self.intercept_ + X_arr @ self.coef_

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score(self, X, y) -> float:
        """
        Compute R² (coefficient of determination).

        Parameters
        ----------
        X : array_like
            Feature matrix.
        y : array_like
            True target values.

        Returns
        -------
        float
            R² score.
        """
        y_arr = _to_1d_float_array(y, "y")
        y_pred = self.predict(X)
        sse = np.sum((y_arr - y_pred)**2)
        sst = np.sum((y_arr - np.mean(y_arr))**2)
        return 1 - sse / sst
