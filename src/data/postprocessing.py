"""
Post-processing utilities for the rice_ml package.

This module provides helper functions to:
- Invert scaling on predictions
- Map encoded labels back to original categories
- Generate evaluation summaries
- Tune thresholds for binary classification
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_fscore_support


def inverse_scale_predictions(pred_scaled, scaler):
    """
    Inverse-transform scaled predictions using the fitted StandardScaler.

    Parameters
    ----------
    pred_scaled : array-like
        Predictions in scaled space.
    scaler : StandardScaler
        The scaler fitted during preprocessing.

    Returns
    -------
    numpy.ndarray
        Predictions in the original target scale.
    """
    pred_scaled = np.array(pred_scaled).reshape(-1, 1)
    return scaler.inverse_transform(pred_scaled).flatten()


def decode_labels(encoded_labels, encoder, col_index=0):
    """
    Convert encoded categorical predictions back to original categories.

    Parameters
    ----------
    encoded_labels : array-like
        Model predictions after encoding.
    encoder : OneHotEncoder or similar
        Fitted encoder from preprocessing.
    col_index : int
        For multi-column encoders, the column to decode.

    Returns
    -------
    list
        Original categorical labels.
    """
    encoded_labels = np.array(encoded_labels).reshape(-1, 1)
    return encoder.categories_[col_index][encoded_labels.flatten()].tolist()


def classification_summary(y_true, y_pred):
    """
    Generate classification metrics and a confusion matrix.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like

    Returns
    -------
    dict
        Report containing metrics and confusion matrix.
    """
    return {
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }


def compute_roc_auc(y_true, y_prob):
    """
    Compute ROC curve and AUC for binary classification.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_prob : array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        FPR, TPR, and AUC values.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc": float(auc(fpr, tpr))
    }

def find_best_threshold(y_true, y_prob, metric="f1", num_thresholds=200):
    """
    Compute the best classification threshold for a metric such as F1, precision, recall.
    Handles zero-division safely and produces a clean curve.
    """

    thresholds = np.linspace(0, 1, num_thresholds)
    metric_values = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        # Precision/Recall/F1 without warnings
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )

        if metric == "precision":
            metric_values.append(precision)
        elif metric == "recall":
            metric_values.append(recall)
        else:  # default = F1
            metric_values.append(f1)

    metric_values = np.array(metric_values)

    best_idx = np.argmax(metric_values)
    best_threshold = float(thresholds[best_idx])
    best_value = float(metric_values[best_idx])

    return {
        "best_threshold": best_threshold,
        "best_value": best_value,
        "curve": {
            "threshold": thresholds.tolist(),
            "metric": metric_values.tolist(),
        }
    }
