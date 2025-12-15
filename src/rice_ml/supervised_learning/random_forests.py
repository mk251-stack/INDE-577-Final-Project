"""
Utility functions and configuration for Random Forest classification.

This module provides a lightweight, reusable interface for training,
evaluating, and interpreting Random Forest classifiers using scikit-learn.
It is designed to support example notebooks and unit tests in the
supervised learning section of the project.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


@dataclass
class RandomForestConfig:
    """
    Configuration container for RandomForestClassifier hyperparameters.

    Attributes
    ----------
    n_estimators : int, default=200
        Number of trees in the forest.
    max_depth : int or None, default=None
        Maximum depth of each tree. If None, nodes are expanded until all
        leaves are pure or contain fewer than min_samples_split samples.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    random_state : int, default=42
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs to run. -1 uses all available cores.
    """

    n_estimators: int = 200
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    random_state: int = 42
    n_jobs: int = -1


def train_random_forest(
    X_train,
    y_train,
    config: Optional[RandomForestConfig] = None
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training feature matrix.
    y_train : array-like of shape (n_samples,)
        Training target labels.
    config : RandomForestConfig or None, default=None
        Configuration object specifying Random Forest hyperparameters.
        If None, default configuration values are used.

    Returns
    -------
    model : RandomForestClassifier
        Trained Random Forest classifier.
    """
    if config is None:
        config = RandomForestConfig()

    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
    model.fit(X_train, y_train)
    return model


def predict_random_forest(
    model: RandomForestClassifier,
    X
) -> np.ndarray:
    """
    Generate predictions from a trained Random Forest classifier.

    Parameters
    ----------
    model : RandomForestClassifier
        A fitted Random Forest model.
    X : array-like of shape (n_samples, n_features)
        Input feature matrix.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        Predicted class labels.
    """
    return model.predict(X)


def evaluate_random_forest(
    model: RandomForestClassifier,
    X_test,
    y_test
) -> Dict[str, Any]:
    """
    Evaluate a trained Random Forest classifier on test data.

    Parameters
    ----------
    model : RandomForestClassifier
        A fitted Random Forest model.
    X_test : array-like of shape (n_samples, n_features)
        Test feature matrix.
    y_test : array-like of shape (n_samples,)
        True test labels.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'accuracy' : float
            Classification accuracy on the test set.
        - 'classification_report' : str
            Text summary of precision, recall, and F1-score.
        - 'confusion_matrix' : ndarray of shape (n_classes, n_classes)
            Confusion matrix of predictions vs. true labels.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def get_feature_importances(
    model: RandomForestClassifier,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Retrieve feature importance scores from a trained Random Forest.

    Importance is computed as the normalized total reduction of impurity
    brought by each feature across all trees in the forest.

    Parameters
    ----------
    model : RandomForestClassifier
        A fitted Random Forest model.
    feature_names : list of str
        Names of input features corresponding to model inputs.

    Returns
    -------
    importances : pandas.DataFrame
        DataFrame with columns:
        - 'feature' : feature name
        - 'importance' : importance score
        Sorted in descending order of importance.
    """
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
