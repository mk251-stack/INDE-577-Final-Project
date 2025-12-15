import pandas as pd
from dataclasses import dataclass
from typing import Optional

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


@dataclass
class RegressionTreeConfig:
    """
    Configuration container for the RegressionTree model.

    Parameters
    ----------
    test_size : float, default=0.33
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random seed for reproducibility.
    max_depth : int or None, default=None
        Maximum depth of the regression tree. If None, the tree is grown fully.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    """
    test_size: float = 0.33
    random_state: int = 42
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1


def load_boston_csv(csv_path: str, target_col: str = "medv"):
    """
    Load the Boston Housing dataset from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    target_col : str, default="medv"
        Name of the target column.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target vector.
    df : pandas.DataFrame
        Full dataset including target column.

    Raises
    ------
    ValueError
        If the target column is not present in the dataset.
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

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target vector.

    Raises
    ------
    TypeError
        If X is not a DataFrame or y is not a Series.
    ValueError
        If inputs are empty, misaligned, contain missing values,
        or include non-numeric features.
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
            raise ValueError(f"Non-numeric feature column: {c}")
    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError("y must be numeric")


class RegressionTree:
    """
    Wrapper around scikit-learn's DecisionTreeRegressor with
    validation and evaluation utilities.

    Parameters
    ----------
    config : RegressionTreeConfig or None, default=None
        Configuration object controlling tree hyperparameters
        and train/test split behavior.
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
        """
        Split the dataset into training and testing sets.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.
        y : pandas.Series
            Target vector.

        Returns
        -------
        X_train, X_test, y_train, y_test
            Stratified train/test split.
        """
        _validate_xy(X, y)
        return train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            shuffle=True,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fit the regression tree model.

        Parameters
        ----------
        X_train : pandas.DataFrame
            Training feature matrix.
        y_train : pandas.Series
            Training target vector.

        Returns
        -------
        self : RegressionTree
            Fitted model.
        """
        _validate_xy(X_train, y_train)
        self.feature_names = list(X_train.columns)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame):
        """
        Generate predictions for new data.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.

        Returns
        -------
        numpy.ndarray
            Predicted target values.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if self.feature_names is not None:
            X = X[self.feature_names]
        return self.model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Evaluate model performance on a test set.

        Parameters
        ----------
        X_test : pandas.DataFrame
            Test feature matrix.
        y_test : pandas.Series
            True target values.

        Returns
        -------
        dict
            Dictionary containing MSE, MAE, and R² metrics.
        """
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
    Tune the maximum depth of a regression tree by evaluating
    test-set performance across multiple depths.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training feature matrix.
    y_train : pandas.Series
        Training target vector.
    X_test : pandas.DataFrame
        Test feature matrix.
    y_test : pandas.Series
        Test target vector.
    depths : iterable of int, default=range(1, 21)
        Candidate max_depth values to evaluate.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Table of max_depth values and corresponding MSE, MAE, and R²,
        sorted by increasing MSE.
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
