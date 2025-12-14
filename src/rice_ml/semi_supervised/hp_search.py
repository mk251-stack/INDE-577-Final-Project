from __future__ import annotations

from itertools import product
from typing import Iterable, Optional, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from .label_propagation import LabelPropagation
from .utils import make_semi_supervised_labels


def label_propagation_grid_search(
    X_graph: np.ndarray,
    y_graph_true: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_labeled_per_class: int,
    n_neighbors_list: Iterable[int],
    alpha_list: Iterable[float],
    gamma_list: Optional[Iterable[Optional[float]]] = None,
    max_iter: int = 200,
    tol: float = 1e-4,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Grid search for Label Propagation hyperparameters.

    For each combination of (n_neighbors, alpha, gamma), we:
      - create a semi-supervised labeling with n_labeled_per_class points,
      - fit LabelPropagation on X_graph,
      - evaluate test accuracy on (X_test, y_test),
      - record timing, convergence, and number of iterations.

    Parameters
    ----------
    X_graph, y_graph_true : arrays
        Graph training set (features and true labels).

    X_test, y_test : arrays
        Held-out test set used to evaluate inductive performance.

    n_labeled_per_class : int
        Number of labeled samples per class in the semi-supervised setup.

    n_neighbors_list, alpha_list, gamma_list : iterables
        Hyperparameter grids for the label propagation model.
        If gamma_list is None, only a single "auto" (gamma=None) value is used.

    max_iter, tol : see LabelPropagation
        Passed through to the estimator.

    random_state : int, default=42
        Seed for semi-supervised label selection.

    verbose : bool, default=True
        If True, prints simple progress information.

    Returns
    -------
    results : pandas.DataFrame
        One row per hyperparameter combination with columns:
        ["n_neighbors", "alpha", "gamma",
         "test_acc", "converged", "n_iter",
         "graph_build", "propagation", "fit_total",
         "n_labeled", "n_unlabeled", "n_train"].
    """
    if gamma_list is None:
        gamma_list = [None]

    y_graph_true = np.asarray(y_graph_true)
    X_graph = np.asarray(X_graph)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    combos = list(product(n_neighbors_list, alpha_list, gamma_list))
    rows: List[dict] = []

    for i, (n_neighbors, alpha, gamma) in enumerate(combos, start=1):
        if verbose:
            print(
                f"[GridSearch] Combo {i}/{len(combos)}: "
                f"n_neighbors={n_neighbors}, alpha={alpha}, gamma={gamma}"
            )

        y_semi, labeled_mask, unlabeled_mask = make_semi_supervised_labels(
            y_graph_true,
            n_labeled_per_class=n_labeled_per_class,
            random_state=random_state,
        )

        model = LabelPropagation(
            n_neighbors=n_neighbors,
            alpha=alpha,
            gamma=gamma,
            max_iter=max_iter,
            tol=tol,
            verbose=False,
        )
        model.fit(X_graph, y_semi)
        y_test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)

        timing = model.timing_
        rows.append(
            {
                "n_neighbors": n_neighbors,
                "alpha": alpha,
                "gamma": gamma,
                "test_acc": test_acc,
                "converged": model.converged_,
                "n_iter": model.n_iter_,
                "graph_build": timing.get("graph_build", np.nan),
                "propagation": timing.get("propagation", np.nan),
                "fit_total": timing.get("fit_total", np.nan),
                "n_labeled": int(labeled_mask.sum()),
                "n_unlabeled": int(unlabeled_mask.sum()),
                "n_train": int(X_graph.shape[0]),
            }
        )

    return pd.DataFrame(rows)
