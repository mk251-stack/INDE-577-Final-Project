"""
K-Nearest Neighbors (KNN) classification utilities.

This module provides helper functions to build, train, and evaluate a
KNN classifier using scikit-learn pipelines. Categorical variables are
one-hot encoded, numerical variables are standardized, and preprocessing
is combined with the classifier in a single pipeline to prevent data leakage.

The main entry points are:
- build_knn_pipeline
- train_knn_model
- evaluate_knn_model
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def build_knn_pipeline(cat_cols, num_cols, n_neighbors=9):
    """
    Build a preprocessing + KNN classification pipeline.

    Categorical features are one-hot encoded and numerical features are
    standardized before fitting a KNN classifier.

    Parameters
    ----------
    cat_cols : list of str
        Names of categorical feature columns.
    num_cols : list of str
        Names of numerical feature columns.
    n_neighbors : int, default=9
        Number of neighbors to use for KNN classification.

    Returns
    -------
    model : sklearn.pipeline.Pipeline
        A scikit-learn pipeline with preprocessing and KNN classifier.
    """
    preprocess = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    model = Pipeline(
        [
            ("preprocess", preprocess),
            ("knn", knn),
        ]
    )

    return model


def train_knn_model(
    df,
    target_col,
    test_size=0.2,
    random_state=42,
    n_neighbors=9,
    cat_cols=None,
    num_cols=None,
):
    """
    Train a KNN classifier on a tabular dataset.

    The dataset is split into train and test sets. If categorical or
    numerical columns are not provided, they are inferred automatically
    based on data types.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing features and target column.
    target_col : str
        Name of the target column.
    test_size : float, default=0.2
        Proportion of the dataset used for testing.
    random_state : int, default=42
        Random seed for train-test splitting.
    n_neighbors : int, default=9
        Number of neighbors to use for KNN.
    cat_cols : list of str, optional
        Categorical feature columns. If None, inferred automatically.
    num_cols : list of str, optional
        Numerical feature columns. If None, inferred automatically.

    Returns
    -------
    model : sklearn.pipeline.Pipeline
        Trained KNN pipeline.
    X_train : pandas.DataFrame
        Training feature matrix.
    X_test : pandas.DataFrame
        Test feature matrix.
    y_train : pandas.Series
        Training labels.
    y_test : pandas.Series
        Test labels.
    cat_cols : list of str
        Categorical feature columns used.
    num_cols : list of str
        Numerical feature columns used.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if cat_cols is None:
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if num_cols is None:
        num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = build_knn_pipeline(cat_cols, num_cols, n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test, cat_cols, num_cols


def evaluate_knn_model(model, X_test, y_test, print_report=True):
    """
    Evaluate a trained KNN model on a test dataset.

    Computes accuracy, confusion matrix, and classification report.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Trained KNN pipeline.
    X_test : pandas.DataFrame
        Test feature matrix.
    y_test : pandas.Series or ndarray
        True labels for the test set.
    print_report : bool, default=True
        Whether to print evaluation metrics to stdout.

    Returns
    -------
    results : dict
        Dictionary containing:
        - "accuracy" : float
        - "confusion_matrix" : ndarray of shape (2, 2)
        - "classification_report" : str
    """
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds)

    if print_report:
        print("Accuracy:", acc)
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", report)

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report,
    }
