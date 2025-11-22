import numpy as np
import pytest

from rice_ml import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    log_loss,
    mse,
    rmse,
    mae,
    r2_score,
)


# -------------------- Classification: binary --------------------

def test_binary_basic_metrics():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    assert accuracy_score(y_true, y_pred) == 0.75
    assert precision_score(y_true, y_pred, average="binary") == 1.0  # only one positive predicted and it's correct
    assert recall_score(y_true, y_pred, average="binary") == 0.5
    assert f1_score(y_true, y_pred, average="binary") == 2 * 1.0 * 0.5 / (1.0 + 0.5)

    cm = confusion_matrix(y_true, y_pred)
    assert cm.tolist() == [[2, 0], [1, 1]]


def test_roc_auc_and_log_loss_binary():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    assert round(roc_auc_score(y_true, scores), 2) == 0.75

    probs = np.array([0.1, 0.9])  # prob of class 1
    y = np.array([0, 1])
    ll = log_loss(y, probs)
    assert np.isclose(ll, -np.log(0.9))


def test_log_loss_multiclass_one_hot():
    y_true = np.array([0, 1, 2])
    probs = np.eye(3)
    assert log_loss(y_true, probs) == 0.0


def test_binary_metric_errors():
    with pytest.raises(ValueError):
        precision_score([0, 1, 2], [0, 1, 2], average="binary")  # not binary classes
    with pytest.raises(ValueError):
        roc_auc_score([0, 0, 0], [0.1, 0.2, 0.3])  # only one class present
    with pytest.raises(ValueError):
        log_loss([0, 1], np.array([[0.6, 0.4], [0.6, 0.5]]))  # rows not summing to 1 OK (we renorm), but probs invalid?
    # invalid probabilities: out of range
    with pytest.raises(ValueError):
        log_loss([0, 1], np.array([1.2, 0.5]))


# -------------------- Classification: multiclass --------------------

def test_multiclass_macro_micro():
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 2, 2, 1])

    # per-class:
    # class 0: P=1, R=1, F1=1
    # class 1: P=0, R=0, F1=0
    # class 2: P=0.5, R=0.5, F1=0.5
    assert accuracy_score(y_true, y_pred) == 0.5
    assert precision_score(y_true, y_pred, average="macro") == 0.5
    assert recall_score(y_true, y_pred, average="macro") == 0.5
    assert f1_score(y_true, y_pred, average="macro") == 0.5

    # micro = accuracy for single-label multiclass
    assert precision_score(y_true, y_pred, average="micro") == 0.5
    assert recall_score(y_true, y_pred, average="micro") == 0.5
    assert f1_score(y_true, y_pred, average="micro") == 0.5

    cm = confusion_matrix(y_true, y_pred)
    assert cm.shape == (3, 3)


def test_confusion_with_custom_labels_ignores_unknown():
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 3, 2, 1])  # 3 not in labels below -> ignored
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    # the prediction "3" for a true "1" is ignored -> no column accumulates it
    assert cm.tolist() == [[1, 0, 0],
                           [0, 0, 0],
                           [0, 1, 1]]


# --------------------------- Regression ---------------------------

def test_regression_metrics():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])

    assert mse(y_true, y_pred) == 0.375
    assert round(rmse(y_true, y_pred), 6) == 0.612372
    assert mae(y_true, y_pred) == 0.5
    assert round(r2_score(y_true, y_pred), 6) == 0.948608


def test_regression_shape_type_errors():
    with pytest.raises(ValueError):
        mse([1, 2], [1])
    with pytest.raises(TypeError):
        mae(["a", "b"], [1, 2])
    with pytest.raises(ValueError):
        r2_score([1, 1, 1], [1, 2, 3])  # constant true: returns -inf if mismatch, but here ok to compute
