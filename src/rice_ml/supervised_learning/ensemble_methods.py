"""
Training and evaluation utilities for ensemble classification models.

This module provides:
- Factory functions to construct baseline and ensemble classifiers
- A unified training and evaluation routine with optional cross-validation

The models defined here are intended to be used with the preprocessing
pipeline in ``src/data/preprocessing.py`` and the analysis notebooks
under ``examples/Supervised_Learning``.
"""

from typing import Dict
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    VotingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


# ---------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------
def get_models(random_state: int = 42) -> Dict[str, object]:
    """
    Construct baseline and ensemble classification models.

    This function returns a dictionary mapping model names to initialized
    scikit-learn estimators. The models include both single learners and
    ensemble methods commonly used for tabular classification tasks.

    Parameters
    ----------
    random_state : int, default=42
        Random seed for reproducibility across all stochastic models.

    Returns
    -------
    models : dict
        Dictionary mapping model names (str) to classifier instances.
        The returned models include:
        - Logistic Regression
        - Decision Tree
        - k-Nearest Neighbors
        - Random Forest
        - AdaBoost
        - Bagging (Decision Trees)
        - Histogram Gradient Boosting
        - Soft Voting ensemble (LR + RF + HGB)
    """

    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    dt = DecisionTreeClassifier(random_state=random_state)
    knn = KNeighborsClassifier()
    rf = RandomForestClassifier(
        n_estimators=200, random_state=random_state, n_jobs=-1
    )
    ada = AdaBoostClassifier(n_estimators=100, random_state=random_state)

    try:
        bag_dt = BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        )
    except TypeError:
        # Compatibility with older scikit-learn versions
        bag_dt = BaggingClassifier(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        )

    hgb = HistGradientBoostingClassifier(
        max_iter=200, random_state=random_state
    )

    voting = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("hgb", hgb)],
        voting="soft",
        n_jobs=-1,
    )

    return {
        "Logistic": lr,
        "DecisionTree": dt,
        "KNN": knn,
        "RandomForest": rf,
        "AdaBoost": ada,
        "Bagging(DT)": bag_dt,
        "HistGradientBoosting": hgb,
        "Voting(LR+RF+HGB)": voting,
    }


# ---------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------
def train_eval(
    models,
    X,
    y,
    test_size: float = 0.25,
    random_state: int = 42,
    cv: int = 5,
):
    """
    Train and evaluate multiple classification models.

    The function performs:
    - A stratified train/test split
    - Model fitting and test-set evaluation
    - Optional stratified cross-validation using accuracy

    Cross-validation is implemented manually to ensure consistent behavior
    across scikit-learn versions and to avoid relying on estimator attributes
    such as ``predict_proba`` or ``classes_``.

    Parameters
    ----------
    models : dict
        Dictionary mapping model names to classifier instances.
    X : array-like or pandas.DataFrame of shape (n_samples, n_features)
        Feature matrix.
    y : array-like or pandas.Series of shape (n_samples,)
        Target labels.
    test_size : float, default=0.25
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random seed for reproducibility.
    cv : int, default=5
        Number of stratified folds for cross-validation.

    Returns
    -------
    results : pandas.DataFrame
        DataFrame indexed by model name with the following columns:
        - accuracy : float
            Test-set accuracy.
        - roc_auc : float or NaN
            Test-set ROC–AUC (NaN if probabilities are unavailable).
        - cv_accuracy_mean : float or NaN
            Mean cross-validated accuracy.
        - cv_accuracy_std : float or NaN
            Standard deviation of cross-validated accuracy.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    skf = StratifiedKFold(
        n_splits=cv, shuffle=True, random_state=random_state
    )

    results = []

    for name, model in models.items():
        print(f"Training {name} ...")

        # Fit and evaluate on test set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        try:
            proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
        except Exception:
            auc = np.nan

        # Manual cross-validation (accuracy only)
        try:
            cv_scores = []
            for train_idx, test_idx in skf.split(X, y):
                model_cv = clone(model)

                if hasattr(X, "iloc"):
                    X_train_cv = X.iloc[train_idx]
                    X_test_cv = X.iloc[test_idx]
                else:
                    X_train_cv = X[train_idx]
                    X_test_cv = X[test_idx]

                if hasattr(y, "iloc"):
                    y_train_cv = y.iloc[train_idx]
                    y_test_cv = y.iloc[test_idx]
                else:
                    y_train_cv = y[train_idx]
                    y_test_cv = y[test_idx]

                model_cv.fit(X_train_cv, y_train_cv)
                y_pred_cv = model_cv.predict(X_test_cv)
                cv_scores.append(
                    accuracy_score(y_test_cv, y_pred_cv)
                )

            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))
        except Exception:
            cv_mean = np.nan
            cv_std = np.nan

        results.append(
            {
                "model": name,
                "accuracy": acc,
                "roc_auc": auc,
                "cv_accuracy_mean": cv_mean,
                "cv_accuracy_std": cv_std,
            }
        )

    return pd.DataFrame(results).set_index("model")
