import numpy as np

from rice_ml.supervised_learning.multilayer_perceptron import MultilayerPerceptron


def _separable_dataset(n_per_class: int = 50):
    rng = np.random.default_rng(0)
    class0 = rng.normal(loc=-1.0, scale=0.2, size=(n_per_class, 2))
    class1 = rng.normal(loc=1.0, scale=0.2, size=(n_per_class, 2))

    X = np.vstack([class0, class1])
    y = np.hstack([np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)])
    return X, y


def test_mlp_trains_on_separable_data():
    X, y = _separable_dataset(n_per_class=60)

    model = MultilayerPerceptron(hidden_units=8, learning_rate=0.5, epochs=300, random_state=7)
    model.fit(X, y)

    preds = model.predict(X)
    accuracy = np.mean(preds == y)
    assert accuracy > 0.95


def test_predict_proba_range_and_shape():
    X, y = _separable_dataset(n_per_class=40)
    model = MultilayerPerceptron(hidden_units=6, learning_rate=0.3, epochs=200, random_state=0).fit(X, y)

    probs = model.predict_proba(X)

    assert probs.shape == (X.shape[0],)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_losses_recorded_and_decrease():
    X, y = _separable_dataset(n_per_class=30)
    epochs = 150
    model = MultilayerPerceptron(hidden_units=5, learning_rate=0.3, epochs=epochs, random_state=1).fit(X, y)

    assert len(model.losses_) == epochs
    assert model.losses_[0] > model.losses_[-1]


def test_predict_outputs_binary_labels():
    X, y = _separable_dataset(n_per_class=20)
    model = MultilayerPerceptron(hidden_units=5, learning_rate=0.2, epochs=250, random_state=9).fit(X, y)

    preds = model.predict(X)

    assert set(np.unique(preds)) <= {0, 1}
    assert preds.shape == y.shape