import numpy as np
import pandas as pd
import pytest

"""
Rationale:

These tests are written to validate the K Nearest Neighbors implementation from an applied ML perspective, not just a math correctness perspective.
The goal is to catch silent failure modes that still produce outputs but are not trustworthy, for example data leakage, broken preprocessing, unstable behavior across splits, or a model that does not beat a trivial baseline.
Each test is therefore a sanity check that the pipeline is learning real signal, behaving consistently, and responding to the KNN knobs that should matter in practice.

"""

from rice_ml.supervised_learning.k_nearest_neighbors import (
    build_knn_pipeline,
    train_knn_model,
    evaluate_knn_model,
)


def _make_toy_df(n=300, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    sector = np.where(x1 > 0.8, "tech", np.where(x1 < -0.8, "health", "other"))
    edu = np.where(x2 > 0.6, "masters", np.where(x2 < -0.6, "hs", "bachelors"))
    score = 1.3 * x1 + 0.9 * x2 + rng.normal(0, 0.5, n)
    income = np.where(score > 0.65, ">50K", "<=50K")
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "sector": sector,
            "education": edu,
            "income": income,
        }
    )


def _majority_baseline_accuracy(y):
    vc = pd.Series(y).value_counts()
    return (vc.max() / vc.sum()) if len(vc) else 0.0


def test_knn_end_to_end_train_and_evaluate():
    """
    Rationale
    This is an end to end smoke test for the full training and evaluation flow.
    Passing means the pipeline can split data, fit the preprocessing steps, train the model, and return evaluation outputs with the expected structure.
    This catches wiring issues such as broken imports, mis ordered pipeline steps, or evaluation code that returns inconsistent keys.
    """

    df = _make_toy_df()
    model, X_train, X_test, y_train, y_test, cat_cols, num_cols = train_knn_model(
        df, target_col="income", test_size=0.25, random_state=7, n_neighbors=5
    )
    out = evaluate_knn_model(model, X_test, y_test, print_report=False)
    assert isinstance(out, dict)
    assert "accuracy" in out and "confusion_matrix" in out and "classification_report" in out
    assert 0.0 <= out["accuracy"] <= 1.0
    assert len(cat_cols) > 0 and len(num_cols) > 0


def test_knn_beats_majority_class_baseline():
    """
    Rationale
    A classification model should beat a trivial predictor that always outputs the majority class.
    Passing means the model is learning signal from features and is not just exploiting class imbalance or producing near random output.
    This is the simplest sanity check that the model is doing better than doing almost nothing.
    """

    df = _make_toy_df()
    model, X_train, X_test, y_train, y_test, *_ = train_knn_model(
        df, target_col="income", test_size=0.25, random_state=7, n_neighbors=7
    )
    out = evaluate_knn_model(model, X_test, y_test, print_report=False)
    baseline = _majority_baseline_accuracy(y_train)
    assert out["accuracy"] >= baseline + 0.02


def test_knn_confusion_matrix_shape_binary():
    """
    Rationale
    For a binary classification problem the confusion matrix should be 2 by 2.
    Passing means evaluation is consistent with the intended task and class handling is not silently collapsing or expanding labels.
    This catches issues where labels are malformed, evaluation is mis configured, or the returned object is not in the expected format.
    """

    df = _make_toy_df()
    model, X_train, X_test, y_train, y_test, *_ = train_knn_model(
        df, target_col="income", test_size=0.25, random_state=7, n_neighbors=5
    )
    out = evaluate_knn_model(model, X_test, y_test, print_report=False)
    cm = out["confusion_matrix"]
    assert isinstance(cm, np.ndarray)
    assert cm.shape == (2, 2)


def test_knn_handles_unseen_category_in_test_set():
    """
    Rationale
    Real data often contains categorical values in the test set that were not present in training.
    The pipeline uses OneHotEncoder with handle_unknown set to ignore, so prediction should not crash in this situation.
    Passing means the preprocessing step is configured correctly and the model remains usable under realistic category drift.
    """
    df = _make_toy_df()
    model, X_train, X_test, y_train, y_test, cat_cols, num_cols = train_knn_model(
        df, target_col="income", test_size=0.25, random_state=7, n_neighbors=5
    )
    X_test2 = X_test.copy()
    if "sector" in X_test2.columns:
        X_test2.loc[X_test2.index[:3], "sector"] = "never_seen_before"
    preds = model.predict(X_test2)
    assert len(preds) == len(X_test2)


def test_knn_reproducible_with_same_random_state():
    """
    Rationale
    Using the same random_state should produce the same train test split and therefore identical predictions for the same configuration.
    Passing means the experiment is reproducible, which is important for debugging, grading, and comparing models fairly.
    This also catches accidental sources of randomness in the training or data splitting process.
    """
    df = _make_toy_df()
    m1, Xtr1, Xte1, ytr1, yte1, *_ = train_knn_model(
        df, target_col="income", test_size=0.25, random_state=123, n_neighbors=5
    )
    m2, Xtr2, Xte2, ytr2, yte2, *_ = train_knn_model(
        df, target_col="income", test_size=0.25, random_state=123, n_neighbors=5
    )
    p1 = m1.predict(Xte1)
    p2 = m2.predict(Xte2)
    assert np.array_equal(p1, p2)
    assert np.array_equal(np.array(yte1), np.array(yte2))


def test_knn_input_validation_errors():
    """
    Rationale
    The training function should fail loudly for invalid configurations rather than producing silent incorrect behavior.
    Passing means obvious misuse is caught early, such as a missing target column or an invalid neighbor count.
    This protects the pipeline from misleading results and makes error handling predictable.
    """
    df = _make_toy_df()
    with pytest.raises(Exception):
        train_knn_model(df, target_col="not_a_real_target", test_size=0.25, random_state=7, n_neighbors=5)
    with pytest.raises(Exception):
        train_knn_model(df, target_col="income", test_size=0.25, random_state=7, n_neighbors=0)
