"""
rice_ml

Custom machine learning algorithms and utilities developed for
the INDE-577 Data Science & Machine Learning course.
"""

from . import supervised_learning
from . import unsupervised_learning
from . import semi_supervised
from . import processing
from . import utils
from . import visualization

# Expose commonly used functions at the package level
from .supervised_learning.distance_metrics import (
    euclidean_distance,
    manhattan_distance,
)
from .processing.preprocessing import (
    standardize,
    minmax_scale,
    maxabs_scale,
    l2_normalize_rows,
    train_test_split,
    train_val_test_split,
)
from .processing.post_processing import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    log_loss,
    mse,
    rmse,
    mae,
    r2_score,
)

__all__ = [
    "supervised_learning",
    "unsupervised_learning",
    "semi_supervised",
    "processing",
    "utils",
    "visualization",
    # Distance metrics
    "euclidean_distance",
    "manhattan_distance",
    # Preprocessing helpers
    "standardize",
    "minmax_scale",
    "maxabs_scale",
    "l2_normalize_rows",
    "train_test_split",
    "train_val_test_split",
    # Post-processing metrics
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "confusion_matrix",
    "roc_auc_score",
    "log_loss",
    "mse",
    "rmse",
    "mae",
    "r2_score",
]