import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

"""
Decision Tree classifier utilities for the Census Income classification task.

Repository context
------------------
This repository is organized into three main parts:

1. src
   Reusable algorithm and data helper functions used across the project.

2. examples
   Jupyter notebooks that import from src to train, evaluate, and visualize results.

3. tests
   Unit tests that verify expected behavior, including input validation and
   correct error handling.

This module provides:
- A thin wrapper around scikit-learn's DecisionTreeClassifier with stricter
  validation behavior expected by the notebooks and unit tests.
- Helper functions to load and preprocess the Census Income dataset.
- Convenience functions to create a train test split, train a model, and
  compute standard evaluation outputs.
"""

# ---------------------------------------------------------------------
# Custom DecisionTreeClassifier wrapper for tests and notebooks
# ---------------------------------------------------------------------


class DecisionTreeClassifier(SkDecisionTreeClassifier):
    """
    Wrapper around sklearn's DecisionTreeClassifier with additional
    validation checks.

    This class enforces stricter input validation and usage patterns
    expected by unit tests and example notebooks.

    Notes
    -----
    Additional behavior enforced compared to sklearn:
    - Calling predict or predict_proba before fit raises RuntimeError
    - Input features X must be a 2D array
    - Labels y must be a 1D array of integers
    """

    def fit(self, X, y):
        """
        Fit the decision tree classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Integer class labels.

        Returns
        -------
        self : DecisionTreeClassifier
            Fitted classifier.

        Raises
        ------
        ValueError
            If X is not 2D, y is not 1D, y is not integer-valued,
            or the number of samples does not match.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("Labels y must be a 1D array")

        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("Labels y must be integers")

        if X.ndim != 2:
            raise ValueError("Input X must be a 2D array")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        return super().fit(X, y)

    def _check_fitted(self):
        """
        Check whether the classifier has been fitted.

        Raises
        ------
        RuntimeError
            If the classifier has not been fitted.
        """
        if not hasattr(self, "tree_"):
            raise RuntimeError("DecisionTreeClassifier must be fitted before prediction")

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.

        Raises
        ------
        RuntimeError
            If the classifier has not been fitted.
        ValueError
            If X is not a 2D array.
        """
        self._check_fitted()

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Input X must be a 2D array")

        return super().predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.

        Raises
        ------
        RuntimeError
            If the classifier has not been fitted.
        ValueError
            If X is not a 2D array.
        """
        self._check_fitted()

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Input X must be a 2D array")

        return super().predict_proba(X)


# ---------------------------------------------------------------------
# Helpers for the census income dataset used in notebooks
# ---------------------------------------------------------------------


def _project_root():
    """
    Compute the absolute path to the project root directory.

    Returns
    -------
    root : str
        Absolute path to the project root.
    """
    current_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(current_dir, "..", "..", ".."))


def load_census_dataset(target_col="income"):
    """
    Load and preprocess the Census Income dataset.

    The function performs the following steps:
    - Loads the CSV file from the datasets directory
    - Converts the income column to a binary label
    - Drops non-informative columns
    - One-hot encodes categorical features

    Parameters
    ----------
    target_col : str, default="income"
        Name of the target column.

    Returns
    -------
    X : pandas.DataFrame of shape (n_samples, n_features)
        Feature matrix after preprocessing.
    y : pandas.Series of shape (n_samples,)
        Binary target labels (0 or 1).
    """
    root = _project_root()
    csv_path = os.path.join(root, "datasets", "census_income.csv")

    df = pd.read_csv(csv_path)

    # binary label: 1 if income contains the symbol for >50K
    y = df[target_col].astype(str).str.contains(">").astype(int)

    # drop target and fnlwgt from features
    X = df.drop(columns=[target_col, "fnlwgt"])

    # one-hot encode categorical features
    cat_cols = X.select_dtypes(include=["object"]).columns
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X, y


def prepare_train_test(X, y, test_size=0.25, random_state=42):
    """
    Split features and labels into stratified train and test sets.

    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Feature matrix.
    y : array-like
        Target labels.
    test_size : float, default=0.25
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
        Stratified train-test split.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def train_decision_tree_classifier(
    X_train,
    y_train,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
):
    """
    Train a Decision Tree classifier with specified hyperparameters.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training features.
    y_train : array-like of shape (n_samples,)
        Training labels.
    max_depth : int or None, default=None
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required at a leaf node.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    model : DecisionTreeClassifier
        Fitted decision tree classifier.
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(model, X_test, y_test):
    """
    Evaluate a fitted classifier on a test set.
    This returns a small bundle of evaluation artifacts commonly reported in
    notebooks: overall accuracy, the confusion matrix, a text report, and the
    predicted labels.

    Parameters
    ----------
    model : DecisionTreeClassifier
        Fitted classifier.
    X_test : array-like of shape (n_samples, n_features)
        Test features.
    y_test : array-like of shape (n_samples,)
        True test labels.

    Returns
    -------
    acc : float
        Classification accuracy.
    cm : ndarray of shape (2, 2)
        Confusion matrix.
    report : str
        Text classification report (precision, recall, F1-score).
    y_pred : ndarray of shape (n_samples,)
        Predicted class labels.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return acc, cm, report, y_pred