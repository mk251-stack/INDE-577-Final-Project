import numpy as np
import pandas as pd
import pytest

from rice_ml.supervised_learning.regression_trees import (
    RegressionTree,
    RegressionTreeConfig,
    _validate_xy,
    tune_max_depth,
)

"""
Why these tests exist

These checks focus on real world failure modes:
1) bad data reaches training because validation is missing or wrong
2) split fit predict evaluate does not work end to end in a notebook workflow
3) evaluation output is missing keys or returns non numeric values
4) predictions change when dataframe columns are reordered
5) tuning output is malformed or not sorted by the metric it claims to optimize

Passing means the Regression Tree code is reliable enough to trust inside notebooks and future extensions.
"""


def _make_toy_regression(n=300, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    noise = rng.normal(0, 0.5, n)

    y = 3.0 * x1 - 2.0 * x2 + 0.5 * x3 + noise

    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    y = pd.Series(y, name="target")
    return X, y


def test_regtree_validate_xy_accepts_clean_numeric():
    """
    Rationale
    Validation is the first safety gate.
    Passing means clean numeric X and numeric y are accepted.
    """
    X, y = _make_toy_regression()
    _validate_xy(X, y)


def test_regtree_validate_xy_rejects_missing_values():
    """
    Rationale
    The implementation forbids missing values.
    Passing means NaNs raise instead of silently training.
    """
    X, y = _make_toy_regression()

    X2 = X.copy()
    X2.loc[0, "x1"] = np.nan
    with pytest.raises(ValueError):
        _validate_xy(X2, y)

    y2 = y.copy()
    y2.iloc[0] = np.nan
    with pytest.raises(ValueError):
        _validate_xy(X, y2)


def test_regtree_validate_xy_rejects_non_numeric_feature():
    """
    Rationale
    Feature columns must be numeric.
    Passing means object columns are rejected loudly.
    """
    X, y = _make_toy_regression()
    X2 = X.copy()
    X2["bad"] = "oops"
    with pytest.raises(ValueError):
        _validate_xy(X2, y)


def test_regtree_split_fit_predict_evaluate_end_to_end():
    """
    Rationale
    End to end notebook flow smoke test.
    Passing means the main API works together and returns sane metrics.
    """
    X, y = _make_toy_regression()
    rt = RegressionTree(RegressionTreeConfig(test_size=0.33, random_state=7, max_depth=5))

    X_train, X_test, y_train, y_test = rt.split(X, y)
    rt.fit(X_train, y_train)

    preds = rt.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X_test)

    out = rt.evaluate(X_test, y_test)
    assert isinstance(out, dict)
    assert set(out.keys()) == {"mse", "mae", "r2"}
    assert all(isinstance(out[k], float) for k in out)
    assert out["mse"] >= 0.0
    assert out["mae"] >= 0.0
    assert np.isfinite(out["r2"])


def test_regtree_predictions_stable_under_column_reorder():
    """
    Rationale
    Pandas columns get reordered often.
    Your predict method reorders to training feature_names.
    Passing means reordering does not change predictions.
    """
    X, y = _make_toy_regression()
    rt = RegressionTree(RegressionTreeConfig(test_size=0.33, random_state=7, max_depth=5))

    X_train, X_test, y_train, y_test = rt.split(X, y)
    rt.fit(X_train, y_train)

    p1 = rt.predict(X_test)
    X_reordered = X_test[["x3", "x1", "x2"]]
    p2 = rt.predict(X_reordered)

    assert np.allclose(p1, p2)


def test_regtree_predict_rejects_non_dataframe():
    """
    Rationale
    Predict requires a DataFrame.
    Passing means wrong input types raise clearly.
    """
    X, y = _make_toy_regression()
    rt = RegressionTree(RegressionTreeConfig(max_depth=3))

    X_train, X_test, y_train, y_test = rt.split(X, y)
    rt.fit(X_train, y_train)

    with pytest.raises(TypeError):
        rt.predict(X_test.to_numpy())


def test_regtree_tune_max_depth_schema_and_sorted_by_mse():
    """
    Rationale
    Tuning output is used directly in the notebook.
    Passing means correct columns, correct types, and sorted by mse ascending.
    """
    X, y = _make_toy_regression()
    rt = RegressionTree(RegressionTreeConfig(test_size=0.33, random_state=11))

    X_train, X_test, y_train, y_test = rt.split(X, y)

    depths = range(1, 11)
    tuning = tune_max_depth(X_train, y_train, X_test, y_test, depths=depths, random_state=11)

    assert isinstance(tuning, pd.DataFrame)
    assert list(tuning.columns) == ["max_depth", "mse", "mae", "r2"]
    assert tuning["max_depth"].isin(list(depths)).all()

    mse_vals = tuning["mse"].to_numpy()
    assert np.all(np.diff(mse_vals) >= 0.0)


def test_regtree_split_reproducible_with_same_random_state():
    """
    Rationale
    Reproducibility matters for grading and debugging.
    Passing means same random_state yields identical split indices.
    """
    X, y = _make_toy_regression()

    rt1 = RegressionTree(RegressionTreeConfig(test_size=0.33, random_state=99))
    rt2 = RegressionTree(RegressionTreeConfig(test_size=0.33, random_state=99))

    Xtr1, Xte1, ytr1, yte1 = rt1.split(X, y)
    Xtr2, Xte2, ytr2, yte2 = rt2.split(X, y)

    assert Xtr1.index.equals(Xtr2.index)
    assert Xte1.index.equals(Xte2.index)
    assert ytr1.index.equals(ytr2.index)
    assert yte1.index.equals(yte2.index)
