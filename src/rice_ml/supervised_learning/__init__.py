"""
Supervised learning algorithms.

Includes classification and regression models implemented
from scratch and with scikit-learn benchmarks.
"""

from .perceptron import Perceptron
from .multilayer_perceptron import MultilayerPerceptron
from .k_nearest_neighbors import (
    KNNClassifier,
    build_knn_pipeline,
    train_knn_model,
    evaluate_knn_model,
)
from .decision_tree import DecisionTreeClassifier
from .regression_trees import RegressionTree, RegressionTreeConfig
from .random_forests import (
    RandomForestConfig,
    train_random_forest,
    evaluate_random_forest,
    get_feature_importances,
)
from .ensemble_methods import get_models, train_eval

__all__ = [
    "Perceptron",
    "MultilayerPerceptron",
    "DecisionTreeClassifier",
    "RegressionTree",
    "RegressionTreeConfig",
    "RandomForestConfig",
    "train_random_forest",
    "evaluate_random_forest",
    "get_feature_importances",
    "KNNClassifier",
    "build_knn_pipeline",
    "train_knn_model",
    "evaluate_knn_model",
    "get_models",
    "train_eval",
]