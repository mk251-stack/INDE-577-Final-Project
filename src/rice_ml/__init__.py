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

__all__ = [
    "supervised_learning",
    "unsupervised_learning",
    "semi_supervised",
    "processing",
    "utils",
    "visualization",
]