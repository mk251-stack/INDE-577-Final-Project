# src/rice_ml/supervised_learning/random_forests.py
"""
Random Forest classification utilities.

This module provides a lightweight interface for training, evaluating,
and interpreting Random Forest classifiers on tabular datasets.

It is designed to be used by example notebooks under
examples/Supervised_Learning/Random_Forests and follows the same
train–evaluate–analyze pattern used across other supervised learning
modules in this repository.

The implementation assumes that preprocessing has already been handled
upstream and focuses purely on model configuration, fitting, evaluation,
and feature importance extraction.
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
    Configuration container for Random Forest hyperparameters.

    This dataclass centralizes model configuration to make experiments
    reproducible and parameter choices explicit.

    Attributes
    ----------
    n_estimators : int, default=200
        Number of trees in the forest.
    max_depth : int or None, default=None
        Maximum depth of each tree. If None, trees expand until pure.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required at a leaf node.
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
    Train a Random Forest classifier using the provided configuration.

    Parameters
    ----------
    X_train : array-like or pandas.DataFrame
        Training feature matrix.
    y_train : array-like or pandas.Series
        Training target labels.
    config : RandomForestConfig or None, default=None
        Configuration object specifying model hyperparameters.
        If None, default configuration values are used.

    Returns
    -------
    model : RandomForestClassifier
        Fitted Random Forest classifier.
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
    return model.predict(X)


def evaluate_random_forest(
    model: RandomForestClassifier,
    X_test,
    y_test
) -> Dict[str, Any]:
    """
    Evaluate a trained Random Forest classifier on test data.

    Computes standard classification metrics commonly used in analysis
    notebooks, including accuracy, confusion matrix, and a detailed
    classification report.

    Parameters
    ----------
    model : RandomForestClassifier
        Trained Random Forest classifier.
    X_test : array-like or pandas.DataFrame
        Test feature matrix.
    y_test : array-like or pandas.Series
        True test labels.

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - accuracy : float
        - classification_report : str
        - confusion_matrix : ndarray
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
    Extract and rank feature importances from a trained Random Forest model.

    Feature importances are computed as the mean decrease in impurity
    across all trees in the forest.

    Parameters
    ----------
    model : RandomForestClassifier
        Trained Random Forest classifier.
    feature_names : list of str
        Names of input features corresponding to model inputs.

    Returns
    -------
    importances : pandas.DataFrame
        DataFrame sorted by descending importance with columns:
        - feature
        - importance
    """

    return (
        pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
