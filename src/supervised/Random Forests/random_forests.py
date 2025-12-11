# src/rice_ml/supervised_learning/random_forests.py

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


@dataclass
class RandomForestConfig:
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
    return (
        pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
