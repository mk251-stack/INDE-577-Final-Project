import numpy as np
import pandas as pd
import pytest

from rice_ml.supervised_learning.decision_tree import (
    DecisionTreeClassifier,
    load_census_dataset,
    prepare_train_test,
    train_decision_tree_classifier,
    evaluate_classifier,
)

"""
Why these tests exist

These checks focus on real world failure modes:
1) end to end wiring breaks between load split train evaluate
2) model performs no better than a naive baseline
3) evaluation outputs have wrong shapes or inconsistent counts
4) results are not reproducible across runs
5) wrapper validation does not fail loudly for bad inputs
6) census loader returns clean binary labels and usable feature matrix

Passing means the Decision Tree component is reliable enough to trust inside notebooks and future extensions.
"""


def _make_toy_binary(n=300, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    score = 1.2 * x1 + 0.8 * x2 + rng.normal(0, 0.6, n)
    y = (score > 0.5).astype(int)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, y


def _majority_baseline_accuracy(y):
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    return counts.max() / counts.sum()


def test_dt_end_to_end_toy_train_and_evaluate():
    """
    Rationale
    This is an end to end smoke test for the main training plus evaluation flow.
    Passing means splitting works, the model trains, and evaluate_classifier returns outputs in the expected structure.
    """
    X, y = _make_toy_binary()
    X_train, X_test, y_train, y_test = prepare_train_test(X, y, test_size=0.25, random_state=7)
    model = train_decision_tree_classifier(X_train, y_train, max_depth=6, random_state=7)

    acc, cm, report, y_pred = evaluate_classifier(model, X_test, y_test)

    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
    assert isinstance(cm, np.ndarray)
    assert cm.shape == (2, 2)
    assert isinstance(report, str)
    assert len(y_pred) == len(y_test)


def test_dt_beats_majority_baseline_on_toy_data():
    """
    Rationale
    If the model cannot beat a dumb majority class predictor, it is not learning anything useful.
    Passing means the tree is extracting signal beyond class imbalance.
    """
    X, y = _make_toy_binary(seed=11)
    X_train, X_test, y_train, y_test = prepare_train_test(X, y, test_size=0.25, random_state=11)
    baseline = _majority_baseline_accuracy(y_train)

    model = train_decision_tree_classifier(X_train, y_train, max_depth=6, random_state=11)
    acc, _, _, _ = evaluate_classifier(model, X_test, y_test)

    assert acc >= baseline + 0.02


def test_dt_confusion_matrix_counts_match_test_size():
    """
    Rationale
    Confusion matrix issues are common when labels are mis encoded or prediction lengths mismatch.
    Passing means the confusion matrix is consistent with the number of test samples.
    """
    X, y = _make_toy_binary(seed=3)
    X_train, X_test, y_train, y_test = prepare_train_test(X, y, test_size=0.25, random_state=3)
    model = train_decision_tree_classifier(X_train, y_train, max_depth=5, random_state=3)

    _, cm, _, y_pred = evaluate_classifier(model, X_test, y_test)

    assert cm.sum() == len(y_test)
    assert len(y_pred) == len(y_test)


def test_dt_reproducible_with_same_random_state():
    """
    Rationale
    Reproducibility matters for debugging and grading.
    Passing means training twice with the same random_state produces identical predictions on the same data.
    """
    X, y = _make_toy_binary(seed=21)
    X_train, X_test, y_train, y_test = prepare_train_test(X, y, test_size=0.25, random_state=21)

    m1 = train_decision_tree_classifier(X_train, y_train, max_depth=6, random_state=21)
    m2 = train_decision_tree_classifier(X_train, y_train, max_depth=6, random_state=21)

    p1 = m1.predict(X_test)
    p2 = m2.predict(X_test)

    assert np.array_equal(p1, p2)


def test_dt_wrapper_validation_and_fit_checks():
    """
    Rationale
    My wrapper exists to enforce strict behaviour expected by tests and notebooks.
    Passing means invalid usage fails loudly instead of silently producing nonsense.
    """
    X, y = _make_toy_binary(seed=5)

    clf = DecisionTreeClassifier(max_depth=3, random_state=5)

    with pytest.raises(RuntimeError):
        clf.predict(X)

    with pytest.raises(ValueError):
        clf.fit(X.values, y.astype(float))

    with pytest.raises(ValueError):
        clf.fit(X.values.reshape(-1), y)


def test_dt_load_census_dataset_contract():
    """
    Rationale
    The notebook depends on load_census_dataset producing a usable feature matrix and binary labels.
    Passing means y is binary, lengths match, and X has columns after encoding.
    """
    X, y = load_census_dataset(target_col="income")

    assert hasattr(X, "shape") and hasattr(y, "shape")
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] > 0
    uniq = set(np.unique(np.asarray(y)))
    assert uniq.issubset({0, 1})

