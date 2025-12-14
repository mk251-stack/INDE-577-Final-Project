import numpy as np
import pytest

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

    # Ensure at least one iteration was executed
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


def test_label_propagation_requires_labeled_data():
    """
    LabelPropagation should raise an error if no labeled samples are provided.
    """
    X = np.random.randn(20, 2)
    y = np.full(20, -1)  # all unlabeled

    lp = LabelPropagation(n_neighbors=3)

    with pytest.raises(ValueError):
        lp.fit(X, y)


def test_label_propagation_deterministic():
    """
    Fitting LabelPropagation twice with the same data should
    produce identical transductive predictions.
    """
    X, y = create_simple_dataset()
    y_semi, _, _ = make_semi_supervised_labels(
        y, n_labeled_per_class=3, random_state=42
    )

    lp1 = LabelPropagation(n_neighbors=5, alpha=0.9)
    lp2 = LabelPropagation(n_neighbors=5, alpha=0.9)

    lp1.fit(X, y_semi)
    lp2.fit(X, y_semi)

    preds1 = lp1.predict()
    preds2 = lp2.predict()

    assert np.array_equal(preds1, preds2)


def test_labeled_points_are_respected():
    """
    Labeled points should retain their original labels
    after propagation.
    """
    X, y = create_simple_dataset()
    y_semi, labeled_mask, _ = make_semi_supervised_labels(
        y, n_labeled_per_class=3, random_state=0
    )

    lp = LabelPropagation(n_neighbors=5, alpha=0.9)
    lp.fit(X, y_semi)

    preds = lp.predict()

    assert np.all(preds[labeled_mask] == y[labeled_mask])
