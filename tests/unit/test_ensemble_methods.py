"""
Unit tests for ensemble_methods.py

We test:
- structure of get_models()
- that each model is an sklearn estimator
- that all models fit a tiny synthetic dataset
- that predict_proba works when appropriate
- that VotingClassifier contains correct sub-estimators
"""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from rice_ml.supervised_learning.ensemble_methods import get_models
from rice_ml.supervised_learning import ensemble_methods


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------
@pytest.fixture
def tiny_dataset():
    # Simple AND-like pattern
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=float)
    y = np.array([0, 0, 0, 1], dtype=int)
    return X, y


@pytest.fixture
def models():
    return get_models(random_state=42)


# ---------------------------------------------------------
# Test 1 — get_models returns a dictionary
# ---------------------------------------------------------
def test_get_models_returns_dict(models):
    assert isinstance(models, dict), "get_models() must return a dictionary."


# ---------------------------------------------------------
# Test 2 — expected model keys exist
# ---------------------------------------------------------
def test_model_keys_present(models):
    expected = {
        'Logistic',
        'DecisionTree',
        'KNN',
        'RandomForest',
        'AdaBoost',
        'Bagging(DT)',
        'HistGradientBoosting',
        'Voting(LR+RF+HGB)',
    }

    assert expected.issubset(models.keys()), (
        f"Missing models. Expected at least: {expected}"
    )


# ---------------------------------------------------------
# Test 3 — all models are sklearn classifier estimators
# ---------------------------------------------------------
def test_models_are_estimators(models):
    for name, model in models.items():
        assert isinstance(model, ClassifierMixin), (
            f"Model '{name}' is not a valid sklearn classifier."
        )


# ---------------------------------------------------------
# Test 4 — every model can fit the tiny dataset without errors
# ---------------------------------------------------------
def test_models_fit_on_tiny_dataset(models, tiny_dataset):
    X, y = tiny_dataset

    for name, model in models.items():

        # Skip KNN because 5-neighbor KNN cannot run on 4 samples
        if name.lower().startswith("knn"):
            continue

        model.fit(X, y)
        assert hasattr(model, "predict")
        preds = model.predict(X)
        assert len(preds) == len(y)



# ---------------------------------------------------------
# Test 5 — check predict_proba where applicable
# ---------------------------------------------------------
def test_predict_proba_works(models, tiny_dataset):
    X, y = tiny_dataset

    for name, model in models.items():

        if name.lower().startswith("knn"):
            continue

        model.fit(X, y)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            assert proba.shape[0] == X.shape[0]


# ---------------------------------------------------------
# Test 6 — VotingClassifier internal structure
# ---------------------------------------------------------
def test_voting_classifier_structure(models):
    voting = models['Voting(LR+RF+HGB)']

    # Check estimator names inside VotingClassifier
    expected_inner = {"lr", "rf", "hgb"}
    actual_inner = {name for name, _ in voting.estimators}

    assert expected_inner == actual_inner, (
        f"VotingClassifier must include lr, rf, hgb. Found: {actual_inner}"
    )

    # Ensure soft voting is being used
    assert voting.voting == "soft", "VotingClassifier must use soft voting."


# ---------------------------------------------------------
# Test 7 — train_eval returns expected metrics for simple models
# ---------------------------------------------------------
def test_train_eval_outputs_dataframe_with_metrics():
    # Use lightweight estimators to keep the test fast
    simple_models = {
        "Logistic": LogisticRegression(max_iter=200, random_state=0),
        "DecisionTree": DecisionTreeClassifier(max_depth=3, random_state=0),
    }

    # Balanced synthetic dataset
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.2, 0.1],
        [0.3, 0.1],
        [0.4, 0.2],
        [1.0, 1.0],
        [1.1, 1.0],
        [1.2, 1.1],
        [1.3, 1.1],
        [1.4, 1.2],
    ])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    results = ensemble_methods.train_eval(
        simple_models, X, y, test_size=0.2, random_state=0, cv=2
    )

    # Expected structure and metrics
    assert list(results.index) == ["Logistic", "DecisionTree"]
    for column in ["accuracy", "roc_auc", "cv_accuracy_mean", "cv_accuracy_std"]:
        assert column in results.columns
        assert np.all(~np.isnan(results[column])), f"{column} should not be NaN"


def test_train_eval_handles_models_without_predict_proba():
    class DummyHardClassifier(BaseEstimator, ClassifierMixin):
        def fit(self, X, y):
            self.constant_ = int(np.bincount(y).argmax())
            return self

        def predict(self, X):
            return np.full(len(X), self.constant_, dtype=int)

    X = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
        [1.5, 1.5],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    results = ensemble_methods.train_eval(
        {"Dummy": DummyHardClassifier()},
        X,
        y,
        test_size=0.33,
        random_state=1,
        cv=2,
    )

    assert list(results.index) == ["Dummy"]
    assert np.isnan(results.loc["Dummy", "roc_auc"])
    assert 0.0 <= results.loc["Dummy", "accuracy"] <= 1.0
    assert results.loc["Dummy", "cv_accuracy_mean"] == pytest.approx(
        results.loc["Dummy", "cv_accuracy_mean"]
    )

