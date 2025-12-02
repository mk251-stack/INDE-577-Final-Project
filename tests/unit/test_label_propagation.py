import numpy as np

from rice_ml.semi_supervised.label_propagation import LabelPropagation
from rice_ml.semi_supervised.utils import make_semi_supervised_labels
from rice_ml.semi_supervised.hp_search import label_propagation_grid_search


def create_simple_dataset():
    """
    Two separable clusters for quick testing.
    """
    rng = np.random.RandomState(0)

    X0 = rng.normal(loc=(-2, 0), scale=0.3, size=(25, 2))
    X1 = rng.normal(loc=(2, 0), scale=0.3, size=(25, 2))

    X = np.vstack([X0, X1])
    y = np.array([0] * 25 + [1] * 25)

    return X, y


def test_make_semi_supervised_labels():
    X, y = create_simple_dataset()

    y_semi, labeled_mask, unlabeled_mask = make_semi_supervised_labels(
        y=y,
        n_labeled_per_class=3,
        random_state=42,
    )

    # Exactly 3 labeled points per class
    assert labeled_mask.sum() == 6
    assert unlabeled_mask.sum() == len(y) - 6

    # Labeled values preserved, unlabeled set to -1
    assert np.all(y_semi[labeled_mask] >= 0)
    assert np.all(y_semi[unlabeled_mask] == -1)


def test_label_propagation_fit_and_transductive_predict():
    X, y = create_simple_dataset()
    y_semi, _, _ = make_semi_supervised_labels(y, 3, random_state=42)

    lp = LabelPropagation(
        n_neighbors=5,
        alpha=0.9,
        max_iter=200,
        tol=1e-4,
    )

    lp.fit(X, y_semi)

    preds = lp.predict()

    # Output length sanity
    assert preds.shape == y.shape

    # Predictions only from known classes
    assert set(np.unique(preds)).issubset(set(np.unique(y)))

    # Ensure we converged
    assert lp.n_iter_ > 0


def test_label_propagation_inductive_predict():
    X, y = create_simple_dataset()
    y_semi, _, _ = make_semi_supervised_labels(y, 3, random_state=42)

    lp = LabelPropagation(n_neighbors=5, alpha=0.9)
    lp.fit(X, y_semi)

    X_new = np.array([[2.0, 0.1], [-2.0, -0.1]])
    preds = lp.predict(X_new)

    # Should predict 2 values
    assert preds.shape == (2,)

    # Must be binary labels
    assert set(preds).issubset({0, 1})


def test_grid_search_smoke():
    X, y = create_simple_dataset()
    y_semi, _, _ = make_semi_supervised_labels(y, 3)

    results = label_propagation_grid_search(
        X_graph=X,
        y_graph_true=y,
        X_test=X,
        y_test=y,
        n_labeled_per_class=3,
        n_neighbors_list=[3],
        alpha_list=[0.9],
        gamma_list=[None],
        max_iter=50,
        tol=1e-4,
        verbose=False,
    )

    assert len(results) == 1
    assert "test_acc" in results.columns
