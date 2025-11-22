"""
Post-training evaluation metrics for classification and regression (NumPy-only).

This module provides common metrics with robust validation and NumPy-style
docstrings suitable for doctest and pytest. Functions are intentionally
scikit-learn-like but rely only on NumPy.

Classification
--------------
accuracy_score
precision_score
recall_score
f1_score
confusion_matrix
roc_auc_score          (binary only)
log_loss               (binary or multiclass)

Regression
----------
mse
rmse
mae
r2_score
"""

from __future__ import annotations
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    'accuracy_score',
    'precision_score',
    'recall_score',
    'f1_score',
    'confusion_matrix',
    'roc_auc_score',
    'log_loss',
    'mse',
    'rmse',
    'mae',
    'r2_score',
]

ArrayLike = Union[np.ndarray, Sequence]
NumArrayLike = Union[np.ndarray, Sequence[float], Sequence[int]]


# ------------------------- Validation helpers -------------------------

def _ensure_1d(y: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


def _ensure_1d_numeric(y: NumArrayLike, name: str) -> np.ndarray:
    arr = _ensure_1d(y, name)
    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(float, copy=False)
        except (TypeError, ValueError) as e:
            raise TypeError(f"All elements of {name} must be numeric.") from e
    else:
        arr = arr.astype(float, copy=False)
    return arr


def _validate_pair(y_true: ArrayLike, y_pred: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    yt = _ensure_1d(y_true, "y_true")
    yp = _ensure_1d(y_pred, "y_pred")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"y_true and y_pred must have same length; got {yt.shape[0]} vs {yp.shape[0]}.")
    return yt, yp


def _validate_probs(y_true: ArrayLike, y_prob: ArrayLike) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Validate probabilities for log_loss/roc_auc.

    Supports:
    - Binary: y_prob shape (n,) or (n, 2) (prob of class 1 or [p0, p1]).
    - Multiclass: y_prob shape (n, K).

    Returns
    -------
    y_true_labels : np.ndarray (1D)
    probs : np.ndarray (2D: (n, K))
    K : int
    """
    yt = _ensure_1d(y_true, "y_true")
    probs = np.asarray(y_prob)
    if probs.ndim == 1:
        probs = probs.astype(float)
        if probs.shape[0] != yt.shape[0]:
            raise ValueError("For binary, y_prob (n,) must match length of y_true.")
        # Interpret as prob of positive class (class == positive label)
        probs = np.stack([1.0 - probs, probs], axis=1)
        K = 2
    elif probs.ndim == 2:
        if probs.shape[0] != yt.shape[0]:
            raise ValueError("y_prob must have same first dimension as y_true.")
        probs = probs.astype(float)
        K = probs.shape[1]
    else:
        raise ValueError("y_prob must be 1D or 2D array.")

    if np.any(probs < 0) or np.any(probs > 1) or np.any(~np.isfinite(probs)):
        raise ValueError("Probabilities must be finite and in [0, 1].")

    return yt, probs, K


# ---------------------------- Classification ----------------------------

def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Classification accuracy.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
        True labels.
    y_pred : array_like, shape (n_samples,)
        Predicted labels.

    Returns
    -------
    float
        Fraction of correct predictions.

    Examples
    --------
    >>> accuracy_score([0, 1, 2], [0, 2, 2])
    0.6666666666666666
    """
    yt, yp = _validate_pair(y_true, y_pred)
    return float(np.mean(yt == yp))


def _per_class_counts(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[Sequence]=None):
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    L = len(labels)
    # Map labels to indices
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    y_true_idx = np.array([label_to_idx.get(lab, -1) for lab in y_true])
    y_pred_idx = np.array([label_to_idx.get(lab, -1) for lab in y_pred])

    conf = np.zeros((L, L), dtype=int)
    for t, p in zip(y_true_idx, y_pred_idx):
        if t == -1 or p == -1:
            # Labels outside provided 'labels' are ignored (consistent with sklearn)
            continue
        conf[t, p] += 1

    tp = np.diag(conf).astype(float)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    return tp, fp, fn, labels, conf


def precision_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    average: Optional[str] = "binary",
    labels: Optional[Sequence] = None
) -> float:
    """
    Precision score.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
        True labels.
    y_pred : array_like, shape (n_samples,)
        Predicted labels.
    average : {"binary", "macro", "micro", None}, default="binary"
        - "binary": treat the positive class as the greater of the two unique labels.
        - "macro": unweighted mean of per-class precision.
        - "micro": global TP / (TP + FP).
        - None: return per-class array (aligned with `labels` or sorted unique labels).
    labels : sequence, optional
        Class label order (otherwise inferred from data).

    Returns
    -------
    float or ndarray
        Precision according to averaging.

    Examples
    --------
    >>> precision_score([0,1,1,0], [0,1,0,0], average="binary")
    1.0
    """
    yt, yp = _validate_pair(y_true, y_pred)

    if average == "binary":
        uniq = np.unique(np.concatenate([yt, yp]))
        if uniq.size != 2:
            raise ValueError("binary average requires exactly 2 classes.")
        pos_label = np.max(uniq)
        tp = np.sum((yt == pos_label) & (yp == pos_label))
        fp = np.sum((yt != pos_label) & (yp == pos_label))
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    tp, fp, fn, labels_out, _ = _per_class_counts(yt, yp, labels)

    if average == "micro":
        TP = tp.sum()
        FP = fp.sum()
        return float(TP / (TP + FP)) if (TP + FP) > 0 else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)

    if average == "macro":
        return float(np.mean(per_class))
    if average is None:
        return per_class
    raise ValueError('average must be one of {"binary", "macro", "micro", None}.')


def recall_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    average: Optional[str] = "binary",
    labels: Optional[Sequence] = None
) -> float:
    """
    Recall score.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
        True labels.
    y_pred : array_like, shape (n_samples,)
        Predicted labels.
    average : {"binary", "macro", "micro", None}, default="binary"
        Averaging strategy (see `precision_score`).
    labels : sequence, optional
        Class label order (otherwise inferred).

    Returns
    -------
    float or ndarray
        Recall according to averaging.

    Examples
    --------
    >>> recall_score([0,1,1,0], [0,1,0,0], average="binary")
    0.5
    """
    yt, yp = _validate_pair(y_true, y_pred)

    if average == "binary":
        uniq = np.unique(np.concatenate([yt, yp]))
        if uniq.size != 2:
            raise ValueError("binary average requires exactly 2 classes.")
        pos_label = np.max(uniq)
        tp = np.sum((yt == pos_label) & (yp == pos_label))
        fn = np.sum((yt == pos_label) & (yp != pos_label))
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    tp, fp, fn, labels_out, _ = _per_class_counts(yt, yp, labels)

    if average == "micro":
        TP = tp.sum()
        FN = fn.sum()
        return float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)

    if average == "macro":
        return float(np.mean(per_class))
    if average is None:
        return per_class
    raise ValueError('average must be one of {"binary", "macro", "micro", None}.')


def f1_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    average: Optional[str] = "binary",
    labels: Optional[Sequence] = None
) -> float:
    """
    F1 score (harmonic mean of precision and recall).

    Parameters
    ----------
    y_true : array_like
        True labels.
    y_pred : array_like
        Predicted labels.
    average : {"binary", "macro", "micro", None}, default="binary"
        Averaging strategy (see `precision_score`).
    labels : sequence, optional
        Class label order (otherwise inferred).

    Returns
    -------
    float or ndarray
        F1 according to averaging.

    Examples
    --------
    >>> f1_score([0,1,1,0], [0,1,0,0], average="binary")
    0.6666666666666666
    """
    yt, yp = _validate_pair(y_true, y_pred)

    if average == "binary":
        p = precision_score(yt, yp, average="binary")
        r = recall_score(yt, yp, average="binary")
        return float(0.0 if (p + r) == 0 else (2 * p * r) / (p + r))

    if average == "micro":
        p = precision_score(yt, yp, average="micro")
        r = recall_score(yt, yp, average="micro")
        return float(0.0 if (p + r) == 0 else (2 * p * r) / (p + r))

    # macro or None (per-class)
    tp, fp, fn, _, _ = _per_class_counts(yt, yp, labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        rec  = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    if average == "macro":
        return float(np.mean(f1))
    if average is None:
        return f1
    raise ValueError('average must be one of {"binary", "macro", "micro", None}.')


def confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: Optional[Sequence] = None
) -> np.ndarray:
    """
    Confusion matrix.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
        True labels.
    y_pred : array_like, shape (n_samples,)
        Predicted labels.
    labels : sequence, optional
        Class label order; otherwise inferred from data.

    Returns
    -------
    ndarray of shape (n_classes, n_classes)
        Matrix where rows = true classes, columns = predicted classes.

    Examples
    --------
    >>> cm = confusion_matrix([0,1,1,0], [0,1,0,0])
    >>> cm.tolist()
    [[2, 0], [1, 1]]
    """
    yt, yp = _validate_pair(y_true, y_pred)
    _, _, _, labels_out, cm = _per_class_counts(yt, yp, labels)
    return cm


def roc_auc_score(y_true: ArrayLike, y_scores: ArrayLike) -> float:
    """
    ROC AUC for **binary** classification.

    Parameters
    ----------
    y_true : array_like of shape (n_samples,)
        True labels containing exactly 2 classes.
    y_scores : array_like of shape (n_samples,)
        Scores for the positive class (higher = more positive).

    Returns
    -------
    float
        Area under the ROC curve.

    Raises
    ------
    ValueError
        If data is not binary or only one class present.

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> round(roc_auc_score(y_true, scores), 2)
    0.75
    """
    yt = _ensure_1d(y_true, "y_true")
    ys = _ensure_1d_numeric(y_scores, "y_scores")
    uniq = np.unique(yt)
    if uniq.size != 2:
        raise ValueError("roc_auc_score requires exactly 2 classes.")
    if np.all(yt == uniq[0]) or np.all(yt == uniq[1]):
        raise ValueError("y_true must contain at least one sample from each class.")

    # Rank-based AUC (equivalent to Mann–Whitney U / Wilcoxon)
    # Compute AUC = (sum of ranks for positive class - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    order = np.argsort(ys, kind="mergesort")  # stable
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(ys) + 1)  # 1..n

    pos = yt == uniq.max()
    n_pos = np.sum(pos)
    n_neg = len(yt) - n_pos
    sum_ranks_pos = np.sum(ranks[pos])
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def log_loss(y_true: ArrayLike, y_prob: ArrayLike, eps: float = 1e-15) -> float:
    """
    Logarithmic loss (cross-entropy) for binary or multiclass.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
        True labels.
    y_prob : array_like
        Predicted probabilities.
        - Binary: shape (n,) for P(class=1) or (n, 2) for [P0, P1].
        - Multiclass: shape (n, K) for K classes (rows sum to 1).
    eps : float, default=1e-15
        Lower clipping to avoid log(0).

    Returns
    -------
    float
        Mean negative log-likelihood.

    Raises
    ------
    ValueError
        If shapes invalid, probabilities outside [0, 1], or (for 2D) rows do not sum to 1.
    """
    yt, probs, K = _validate_probs(y_true, y_prob)
    if eps <= 0 or not np.isfinite(eps):
        raise ValueError("eps must be positive and finite.")

    # Determine label mapping to columns
    labels = np.unique(yt)
    if K == 2 and labels.size == 2:
        label_to_col = {labels.min(): 0, labels.max(): 1}
    else:
        if labels.size != K:
            labels = np.arange(K)
        label_to_col = {lab: i for i, lab in enumerate(labels)}

    # If probs is 2D, require rows to sum to 1 (within tolerance)
    if probs.ndim == 2:
        row_sums = probs.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=0.0):
            raise ValueError("Each probability row must sum to 1 within tolerance.")
    # For 1D (binary vector of P(class=1)), _validate_probs already converted to 2-col.

    # Lower-clip only (upper bound allowed at 1.0 so one-hot gives exact 0 loss)
    p = np.clip(probs, eps, 1.0)

    # Gather probabilities of the true classes
    cols = np.array([label_to_col.get(lab, None) for lab in yt])
    if np.any(cols == None):  # noqa: E711
        raise ValueError("Could not map some y_true labels to probability columns.")
    ll = -np.log(p[np.arange(len(yt)), cols.astype(int)])
    return float(np.mean(ll))


# ----------------------------- Regression -----------------------------

def mse(y_true: NumArrayLike, y_pred: NumArrayLike) -> float:
    """
    Mean squared error.

    Examples
    --------
    >>> y_true = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5, 0.0, 2, 8])
    >>> mse(y_true, y_pred)
    0.375
    """
    yt = _ensure_1d_numeric(y_true, "y_true")
    yp = _ensure_1d_numeric(y_pred, "y_pred")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have same length.")
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true: NumArrayLike, y_pred: NumArrayLike) -> float:
    """
    Root mean squared error.

    Examples
    --------
    >>> y_true = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5, 0.0, 2, 8])
    >>> round(rmse(y_true, y_pred), 6)
    0.612372
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: NumArrayLike, y_pred: NumArrayLike) -> float:
    """
    Mean absolute error.

    Examples
    --------
    >>> y_true = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5, 0.0, 2, 8])
    >>> mae(y_true, y_pred)
    0.5
    """
    yt = _ensure_1d_numeric(y_true, "y_true")
    yp = _ensure_1d_numeric(y_pred, "y_pred")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have same length.")
    return float(np.mean(np.abs(yt - yp)))


def r2_score(y_true: NumArrayLike, y_pred: NumArrayLike) -> float:
    """
    Coefficient of determination R^2.

    R^2 = 1 - SS_res / SS_tot, where SS_tot uses y_true mean.

    Raises
    ------
    ValueError
        If y_true is constant and predictions are not perfectly equal to y_true.
    """
    yt = _ensure_1d_numeric(y_true, "y_true")
    yp = _ensure_1d_numeric(y_pred, "y_pred")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have same length.")
    ss_res = np.sum((yt - yp) ** 2)
    y_mean = np.mean(yt)
    ss_tot = np.sum((yt - y_mean) ** 2)
    if ss_tot == 0:
        # Undefined per many libs; tests expect a ValueError unless perfect predictions
        if ss_res == 0:
            return 1.0
        raise ValueError("R^2 is undefined when y_true is constant and predictions are not perfect.")
    return float(1.0 - ss_res / ss_tot)

