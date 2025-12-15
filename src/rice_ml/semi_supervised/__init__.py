"""
Semi-supervised learning algorithms.

Includes graph-based label propagation methods.
"""
from .label_propagation import LabelPropagation
from .utils import make_semi_supervised_labels
from .hp_search import label_propagation_grid_search

__all__ = [
    "LabelPropagation",
    "make_semi_supervised_labels",
    "label_propagation_grid_search",
]