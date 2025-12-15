import pandas as pd
from dataclasses import dataclass
from typing import Optional

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


@dataclass
class RegressionTreeConfig:
    test_size: float = 0.33
    random_state: int = 42
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1


def load_boston_csv(csv_path: str, target_col: str = "medv"):
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    _validate_xy(X, y)
    return X, y, df


def _validate_xy(X: pd.DataFrame, y: pd.Series):
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
