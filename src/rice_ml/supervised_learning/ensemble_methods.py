"""
Training and evaluation utilities for ensemble models.
Uses the preprocessing module in src/data/preprocessing.py
"""

from typing import Dict
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, BaggingClassifier,
    VotingClassifier, HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


# ---------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------
def get_models(random_state: int = 42) -> Dict[str, object]:
    """Return a dictionary of ensemble and baseline classification models."""

    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    dt = DecisionTreeClassifier(random_state=random_state)
    knn = KNeighborsClassifier()
    rf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    ada = AdaBoostClassifier(n_estimators=100, random_state=random_state)
    try:
        bag_dt = BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        )
    except TypeError:
        # For older scikit-learn versions
        bag_dt = BaggingClassifier(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        )

    hgb = HistGradientBoostingClassifier(max_iter=200, random_state=random_state)

    voting = VotingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('hgb', hgb)],
        voting='soft',
        n_jobs=-1
    )

    return {
        'Logistic': lr,
        'DecisionTree': dt,
        'KNN': knn,
        'RandomForest': rf,
        'AdaBoost': ada,
        'Bagging(DT)': bag_dt,
        'HistGradientBoosting': hgb,
        'Voting(LR+RF+HGB)': voting
    }


# ---------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------
def train_eval(models, X, y, test_size=0.25, random_state=42, cv=5):
    """
    Train and evaluate several models.
    Returns a DataFrame of performance metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Manual cross-validation keeps us in control of what gets called on the
    # estimator (only ``fit`` and ``predict``), so models that lack attributes
    # like ``predict_proba`` or ``classes_`` are still evaluated consistently
    # across scikit-learn versions.
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    results = []

    for name, model in models.items():
        print(f"Training {name} ...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        try:
            proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
        except Exception:
            auc = np.nan

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
                cv_scores.append(accuracy_score(y_test_cv, y_pred_cv))

            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))
        except Exception:
            cv_mean = np.nan
            cv_std = np.nan

        results.append({
            "model": name,
            "accuracy": acc,
            "roc_auc": auc,
            "cv_accuracy_mean": cv_mean,
            "cv_accuracy_std": cv_std,
        })

    return pd.DataFrame(results).set_index("model")
