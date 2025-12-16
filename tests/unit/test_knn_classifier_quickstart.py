import numpy as np

from rice_ml.supervised_learning import KNNClassifier


def test_knn_classifier_quickstart_example_runs():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])

    clf = KNNClassifier(n_neighbors=3).fit(X, y)
    preds = clf.predict([[0.1, 0.1]])

    assert preds.shape == (1,)
    assert preds[0] in y