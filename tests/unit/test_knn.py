import numpy as np
import pytest

from rice_ml.supervised_learning.knn import KNNClassifier, KNNRegressor


# ------------------------ Classifier ------------------------

def test_classifier_basic_predict_and_proba_uniform_euclidean():
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0,0,1,1])
    clf = KNNClassifier(n_neighbors=3, metric="euclidean", weights="uniform").fit(X, y)

    preds = clf.predict([[0.1, 0.1], [0.9, 0.9]])
    assert preds.tolist() == [0, 1]

    proba = clf.predict_proba([[0.1, 0.1], [0.9, 0.9]])
    # rows sum to 1
    assert np.allclose(proba.sum(axis=1), 1.0)
    # class order is sorted(unique) = [0,1]
    assert (proba.argmax(axis=1) == preds).all()


def test_classifier_manhattan_distance_weighted():
    X = np.array([[0,0],[2,0],[0,2],[2,2]], dtype=float)
    y = np.array(["A","A","B","B"], dtype=object)
    # Query near (0,0) should favor A
    clf = KNNClassifier(n_neighbors=3, metric="manhattan", weights="distance").fit(X, y)
    pred = clf.predict([[0.1, 0.2]])
    assert pred.tolist() == ["A"]
    # predict_proba should be concentrated on A
    p = clf.predict_proba([[0.1, 0.2]])[0]
    assert p[0] > p[1]  # classes_ sorted -> ["A","B"]


def test_classifier_errors_and_kneighbors():
    X = np.array([[0,0],[1,1],[2,2]], dtype=float)
    y = np.array([0,1,1])
    clf = KNNClassifier(n_neighbors=2).fit(X, y)
    # wrong feature count
    with pytest.raises(ValueError):
        clf.predict([[0.0, 0.0, 0.0]])
    # kneighbors returns shapes (nq, k)
    d, idx = clf.kneighbors([[1.0, 1.0]])
    assert d.shape == (1, 2) and idx.shape == (1, 2)


def test_classifier_score_accuracy():
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0,0,1,1])
    clf = KNNClassifier(n_neighbors=1).fit(X, y)
    assert clf.score(X, y) == 1.0


def test_classifier_zero_distance_with_distance_weights():
    # exact duplicate point in training -> zero distance handling
    X = np.array([[0,0],[1,1],[0,0]], dtype=float)
    y = np.array([0,1,0])
    clf = KNNClassifier(n_neighbors=2, weights="distance").fit(X, y)
    # query exactly matches (0,0); only zero-distance neighbors count -> both class 0
    pred = clf.predict([[0,0]])
    assert pred.tolist() == [0]
    p = clf.predict_proba([[0,0]])[0]
    # full mass on class 0
    assert np.isclose(p[0], 1.0)


# ------------------------ Regressor ------------------------

def test_regressor_basic_predict_and_score():
    X = np.array([[0],[1],[2],[3]], dtype=float)
    y = np.array([0.0, 1.0, 1.5, 3.0])
    reg = KNNRegressor(n_neighbors=2, weights="distance").fit(X, y)
    pred = reg.predict([[1.5]])[0]
    assert 1.2 < pred < 1.3
    # perfect fit at training points with k=1
    reg2 = KNNRegressor(n_neighbors=1).fit(X, y)
    assert reg2.score(X, y) == 1.0


def test_regressor_input_errors():
    X = np.array([[0],[1],[2]], dtype=float)
    y = np.array([0.0, 1.0, 2.0])
    # n_neighbors > n_samples
    with pytest.raises(ValueError):
        KNNRegressor(n_neighbors=5).fit(X, y)
    # non-numeric y
    with pytest.raises(TypeError):
        KNNRegressor(n_neighbors=1).fit(X, np.array(["a","b","c"], dtype=object))


def test_regressor_constant_y_score_error():
    X = np.array([[0],[1],[2]], dtype=float)
    y = np.array([5.0, 5.0, 5.0])
    reg = KNNRegressor(n_neighbors=1).fit(X, y)
    # not perfect predictions off-training -> R^2 undefined; but here using training X, k=1 => perfect
    assert reg.score(X, y) == 1.0
    # perturb X slightly to avoid exact matches -> should raise
    with pytest.raises(ValueError):
        reg.score(X + 0.1, y)
