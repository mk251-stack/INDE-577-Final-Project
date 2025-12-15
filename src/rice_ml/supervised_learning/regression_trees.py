import pandas as pd
from dataclasses import dataclass
from typing import Optional

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

"""
Regression Tree utilities for supervised learning on tabular regression datasets.

This module provides a thin, explicit wrapper around scikit learn's
DecisionTreeRegressor. It follows a clear workflow of data loading,
validation, train test splitting, model fitting, evaluation, and
hyperparameter exploration.

The design favors interpretability, reproducibility, and alignment
with notebook based experimentation rather than automation or abstraction.
"""

@dataclass
class RegressionTreeConfig:
    """
    Configuration container for RegressionTree.

    This dataclass defines all tunable parameters used for data splitting
    and tree construction. It allows consistent reuse of hyperparameters
    across experiments and notebooks.

    Attributes
    test_size
        Proportion of the dataset used for the test split.
    random_state
        Seed for reproducibility of splits and model training.
    max_depth
        Maximum depth of the regression tree. Controls model complexity.
    min_samples_split
        Minimum number of samples required to split an internal node.
    min_samples_leaf
        Minimum number of samples required at a leaf node.
    """
        
    test_size: float = 0.33
    random_state: int = 42
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1


def load_boston_csv(csv_path: str, target_col: str = "medv"):
    """
    Load a regression dataset from CSV and split into features and target.

    This function reads a CSV file, separates the target column from the
    feature matrix, validates the inputs, and returns clean numeric data
    suitable for regression tree training.

    Parameters
    csv_path
        Path to the CSV file.
    target_col
        Name of the target column.

    Returns
    X
        Feature matrix as a pandas DataFrame.
    y
        Target values as a pandas Series.
    df
        Original loaded DataFrame.

    Raises
    ValueError
        If the target column is missing or validation fails.
    """

    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    _validate_xy(X, y)
    return X, y, df


def _validate_xy(X: pd.DataFrame, y: pd.Series):
    """
    Validate feature matrix and target vector for regression modeling.

    This function enforces strict assumptions required by the regression
    tree implementation. It ensures that inputs are numeric, non empty,
    aligned in length, and free of missing values.

    Parameters
    X
        Feature matrix.
    y
        Target vector.

    Raises
    TypeError
        If X or y are of incorrect types.
    ValueError
        If data is empty, misaligned, contains missing values,
        or includes non numeric features.
    """
    
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series")
    if len(X) == 0:
        raise ValueError("X is empty")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")
    if X.isnull().any().any():
        raise ValueError("X contains missing values")
    if y.isnull().any():
        raise ValueError("y contains missing values")
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            raise ValueError(f"Non numeric feature column: {c}")
    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError("y must be numeric")


class RegressionTree:
    """
    Wrapper class for DecisionTreeRegressor with explicit workflow control.

    This class encapsulates data splitting, training, prediction, and
    evaluation while preserving transparency of model behavior.
    Feature names are tracked to ensure consistent column alignment
    during prediction.
    """    
    def __init__(self, config: Optional[RegressionTreeConfig] = None):
        self.config = config or RegressionTreeConfig()
        self.model = DecisionTreeRegressor(
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
        )
        self.feature_names: Optional[list[str]] = None

    def split(self, X: pd.DataFrame, y: pd.Series):
        _validate_xy(X, y)
        return train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            shuffle=True,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        _validate_xy(X_train, y_train)
        self.feature_names = list(X_train.columns)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if self.feature_names is not None:
            X = X[self.feature_names]
        return self.model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        _validate_xy(X_test, y_test)
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {"mse": float(mse), "mae": float(mae), "r2": float(r2)}


def tune_max_depth(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    depths=range(1, 21),
    random_state: int = 42,
):
    """
    Evaluate regression tree performance across multiple tree depths.

    This function trains separate regression trees for each candidate
    maximum depth and records error metrics on the test set.
    It is used to identify the depth that best balances bias and variance.

    Parameters
    X_train
        Training feature matrix.
    y_train
        Training target values.
    X_test
        Test feature matrix.
    y_test
        Test target values.
    depths
        Iterable of max depth values to evaluate.
    random_state
        Seed for reproducibility.

    Returns
    pandas DataFrame
        Table containing max_depth, mse, mae, and r2 for each depth,
        sorted by mean squared error.
    """    
    _validate_xy(X_train, y_train)
    _validate_xy(X_test, y_test)

    rows = []
    for d in depths:
        m = DecisionTreeRegressor(max_depth=int(d), random_state=random_state)
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        rows.append(
            {
                "max_depth": int(d),
                "mse": float(mean_squared_error(y_test, pred)),
                "mae": float(mean_absolute_error(y_test, pred)),
                "r2": float(r2_score(y_test, pred)),
            }
        )

    return pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)
