import numpy as np
import pytest

from rice_ml.supervised_learning import DecisionTreeClassifier


def test_decision_tree_basic_predictions_and_proba():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 1, 1])

    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)

    preds = clf.predict(X)
    assert preds.tolist() == [0, 0, 1, 1]

    proba = clf.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba.argmax(axis=1) == preds).all()


def test_decision_tree_validation_errors():
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0, 1])

    clf = DecisionTreeClassifier(max_depth=1)

    with pytest.raises(RuntimeError):
        clf.predict([[0, 0]])

    with pytest.raises(ValueError):
        clf.fit(X, np.array([[0], [1]]))  # wrong shape

    with pytest.raises(ValueError):
        clf.fit(X, np.array([0.5, 1.5]))  # non-integer labels
