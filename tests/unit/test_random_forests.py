import numpy as np
import pandas as pd
import pytest

from rice_ml.supervised_learning.random_forests import (
    RandomForestConfig,
    train_random_forest,
    evaluate_random_forest,
    get_feature_importances,
)

"""
Why these tests exist

These checks focus on real world failure modes:
1) pipeline wiring breaks and training or evaluation silently fails
2) model performs no better than a naive majority class baseline
3) evaluation outputs have incorrect shapes or missing keys
4) results are not reproducible across runs with the same configuration
5) feature importances are unusable, inconsistent, or misleading
6) invalid input schemas such as reordered feature columns do not fail loudly

Passing means the Random Forest code is reliable enough to trust inside notebooks and future extensions, and that obvious misuse is caught early instead of producing silently incorrect results.
"""


def _make_toy_classification(n=400, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)

    score = 1.4 * x1 + 1.0 * x2 + 0.2 * x3 + rng.normal(0, 0.7, n)
    y = np.where(score > 0.7, ">50K", "<=50K")

    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
        }
    )
    y = pd.Series(y, name="income")
    return X, y


def _majority_baseline_accuracy(y):
    vc = pd.Series(y).value_counts()
    return (vc.max() / vc.sum()) if len(vc) else 0.0


def test_rf_end_to_end_train_and_evaluate():
    """
    Rationale
    End to end smoke test for training plus evaluation.
    Passing means the core API works together and returns the expected output keys and basic types.
    """
    X, y = _make_toy_classification()
    config = RandomForestConfig(n_estimators=80, max_depth=8, random_state=11, n_jobs=1)

    model = train_random_forest(X, y, config)
    out = evaluate_random_forest(model, X, y)

    assert isinstance(out, dict)
    assert "accuracy" in out
    assert "classification_report" in out
    assert "confusion_matrix" in out
    assert 0.0 <= out["accuracy"] <= 1.0
    assert isinstance(out["confusion_matrix"], np.ndarray)


def test_rf_beats_majority_class_baseline():
    """
    Rationale
    A model that cannot beat a dumb majority class predictor is not useful.
    Passing means the Random Forest is learning signal beyond class imbalance.
    """
    X, y = _make_toy_classification()
    baseline = _majority_baseline_accuracy(y)

    config = RandomForestConfig(n_estimators=120, max_depth=10, random_state=7, n_jobs=1)
    model = train_random_forest(X, y, config)
    out = evaluate_random_forest(model, X, y)

    assert out["accuracy"] >= baseline + 0.02


def test_rf_confusion_matrix_shape_binary():
    """
    Rationale
    Ensures the confusion matrix matches a binary classification problem.
    Passing means evaluation output structure is consistent and easy to consume in notebooks.
    """
    X, y = _make_toy_classification()
    config = RandomForestConfig(n_estimators=60, max_depth=6, random_state=3, n_jobs=1)

    model = train_random_forest(X, y, config)
    out = evaluate_random_forest(model, X, y)

    cm = out["confusion_matrix"]
    assert cm.shape == (2, 2)
    assert cm.sum() == len(y)


def test_rf_reproducible_with_same_random_state():
    """
    Rationale
    Reproducibility matters for debugging and grading.
    Passing means same data and same config produce the same predictions.
    """
    X, y = _make_toy_classification()
    config = RandomForestConfig(n_estimators=100, max_depth=9, random_state=123, n_jobs=1)

    m1 = train_random_forest(X, y, config)
    m2 = train_random_forest(X, y, config)

    p1 = m1.predict(X)
    p2 = m2.predict(X)

    assert np.array_equal(p1, p2)


def test_rf_feature_importances_are_sane():
    """
    Rationale
    Feature importance is part of the deliverable, so it must be consistent and interpretable.
    Passing means we get a dataframe with the right features, non negative importances, and a valid total weight.
    """
    X, y = _make_toy_classification()
    config = RandomForestConfig(n_estimators=120, max_depth=10, random_state=5, n_jobs=1)

    model = train_random_forest(X, y, config)
    imps = get_feature_importances(model, X.columns.tolist())

    assert isinstance(imps, pd.DataFrame)
    assert list(imps.columns) == ["feature", "importance"]
    assert set(imps["feature"].tolist()) == set(X.columns.tolist())
    assert (imps["importance"] >= 0).all()

    total = float(imps["importance"].sum())
    assert 0.99 <= total <= 1.01


def test_rf_raises_error_on_column_reorder():
    """
    Rationale
    In real workflows, feature columns may accidentally be reordered.
    Scikit-learn enforces that the feature names and their order must match what was seen during fit.
    Passing this test means the model fails loudly instead of producing silently incorrect predictions.
    """
    X, y = _make_toy_classification()
    config = RandomForestConfig(n_estimators=120, max_depth=10, random_state=9, n_jobs=1)

    model = train_random_forest(X, y, config)

    X_reordered = X[["x3", "x1", "x2"]]

    with pytest.raises(ValueError):
        model.predict(X_reordered)



def test_rf_input_validation_errors():
    """
    Rationale
    Bad configs should fail loudly instead of silently producing nonsense.
    Passing means invalid hyperparameters trigger an exception.
    """
    X, y = _make_toy_classification()

    with pytest.raises(Exception):
        train_random_forest(X, y, RandomForestConfig(n_estimators=0))

    with pytest.raises(Exception):
        train_random_forest(X, y, RandomForestConfig(min_samples_split=1))
