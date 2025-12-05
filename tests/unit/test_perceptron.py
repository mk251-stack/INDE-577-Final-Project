import numpy as np

from rice_ml.supervised_learning.perceptron import Perceptron


def test_perceptron_learns_linearly_separable_data():
    np.random.seed(0)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([-1, -1, -1, 1])  # AND gate style labels

    clf = Perceptron(eta=0.2, epochs=20).fit(X, y)

    preds = clf.predict(X)
    assert preds.tolist() == y.tolist()
    assert clf.errors_[-1] == 0


def test_net_input_and_predict_with_manual_weights():
    clf = Perceptron()
    clf.w_ = np.array([0.1, 0.8, -0.3])

    x_positive = np.array([2.0, 0.5])
    x_negative = np.array([-1.0, 1.0])

    assert clf.net_input(x_positive) > 0
    assert clf.net_input(x_negative) < 0
    assert clf.predict(x_positive) == 1
    assert clf.predict(x_negative) == -1


def test_errors_and_weight_shape_after_fit():
    np.random.seed(1)
    X = np.array([[1.0, -1.0], [2.0, 1.0], [-1.0, -2.0]])
    y = np.array([1, 1, -1])

    epochs = 5
    clf = Perceptron(eta=0.1, epochs=epochs).fit(X, y)

    assert len(clf.errors_) == epochs
    assert clf.w_.shape == (X.shape[1] + 1,)
    assert all(error >= 0 for error in clf.errors_)