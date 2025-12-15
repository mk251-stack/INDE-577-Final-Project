"""
k-Nearest Neighbors classification utilities.

Repository context
------------------
This repository is organized into three main parts:

1. src
   Reusable algorithm and data helper functions used across the project.

2. examples
   Jupyter notebooks that import from src to train, evaluate, and visualize results.

3. tests
   Unit tests that validate expected behavior, contracts, and edge cases.

This module provides:
- A preprocessing plus modeling pipeline for KNN classification using scikit-learn.
- A training helper that performs a train test split and fits the pipeline.
- An evaluation helper that returns common classification artifacts.

Notes
-----
KNN is distance based, so numeric feature scaling is important. This module uses
StandardScaler for numeric columns and one-hot encoding for categorical columns.
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
    Build a preprocessing plus KNN classification pipeline.

    The pipeline performs:
    - One-hot encoding for categorical columns
    - Standardization for numeric columns
    - KNN classification on the transformed feature space

    Parameters
    ----------
    cat_cols : list-like
        Names of categorical columns to be one-hot encoded.
    num_cols : list-like
        Names of numeric columns to be standardized.
    n_neighbors : int, default=9
        Number of neighbors to use for KNN.

    Returns
    -------
    model : sklearn.pipeline.Pipeline
        A fitted-ready pipeline with preprocessing and a KNN classifier.
    """

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    model = Pipeline([
        ("preprocess", preprocess),
        ("knn", knn),
    ])

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
    Train a KNN classifier on a pandas DataFrame.

    This function:
    - Splits the DataFrame into features X and target y
    - Infers categorical and numeric columns if not provided
    - Performs a train test split
    - Builds and fits the preprocessing plus KNN pipeline

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing features and target column.
    target_col : str
        Name of the target label column in df.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random seed for reproducibility of the train test split.
    n_neighbors : int, default=9
        Number of neighbors to use for KNN.
    cat_cols : list-like or None, default=None
        Explicit list of categorical columns. If None, inferred from df dtypes.
    num_cols : list-like or None, default=None
        Explicit list of numeric columns. If None, inferred from df dtypes.

    Returns
    -------
    model : sklearn.pipeline.Pipeline
        Fitted KNN pipeline.
    X_train, X_test : pandas.DataFrame
        Train and test feature splits.
    y_train, y_test : pandas.Series
        Train and test label splits.
    cat_cols, num_cols : list-like
        Column lists used for preprocessing.
    """    
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    if cat_cols is None:
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
    if num_cols is None:
        num_cols = X.select_dtypes(exclude=["object", "category"]).columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = build_knn_pipeline(cat_cols, num_cols, n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test, cat_cols, num_cols


def evaluate_knn_model(model, X_test, y_test, print_report=True):
    """
    Evaluate a trained KNN pipeline on a test set.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Trained pipeline that supports predict.
    X_test : array-like or pandas.DataFrame
        Test features.
    y_test : array-like or pandas.Series
        True test labels.
    print_report : bool, default=True
        If True, prints accuracy, confusion matrix, and classification report.

    Returns
    -------
    results : dict
        Dictionary containing:
        - accuracy : float
        - confusion_matrix : ndarray
        - classification_report : str
    """    
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds)

    if print_report:
        print("Accuracy:", acc)
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", report)

    return {"accuracy": acc, "confusion_matrix": cm, "classification_report": report}
