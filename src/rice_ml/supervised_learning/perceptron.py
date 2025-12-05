"""
Perceptron classifier implemented from scratch.

The Perceptron algorithm is a fundamental supervised learning method for binary
classification. This implementation trains a linear decision boundary by iteratively
updating weights based on misclassified examples.

Features:
- Uses the classic perceptron update rule
- Supports adjustable learning rate and number of epochs
- Tracks misclassifications per epoch for learning curve analysis
- Works with numerical input features (X) and binary labels (-1, 1)

Example:
    from perceptron import Perceptron

    model = Perceptron(eta=0.01, epochs=1000)
    model.fit(X, y)
    predictions = model.predict(X)
"""

import numpy as np


class Perceptron:
    """
    Perceptron classifier.

    Parameters
    ----------
    eta : float, default=0.01
        Learning rate.
    epochs : int, default=1000
        Number of passes over the training dataset.

    Attributes
    ----------
    w_ : ndarray of shape (n_features + 1,)
        Weight vector after training (w0 is the bias term).
    errors_ : list
        Number of misclassifications in each epoch.
    """

    def __init__(self, eta=0.01, epochs=1000):
        self.eta = eta
        self.epochs = epochs
        self.w_ = None
        self.errors_ = []

    def fit(self, X, y):
        """
        Fit the model to the training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input feature matrix.
        y : ndarray of shape (n_samples,)
            Target labels (-1 or 1).

        Returns
        -------
        self
        """
        # Random weight initialization
        self.w_ = np.random.randn(X.shape[1] + 1)
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0

            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)

            self.errors_.append(errors)

        return self

    def net_input(self, X):
        """
        Compute the linear combination of inputs and weights.

        Parameters
        ----------
        X : ndarray
            Input vector.

        Returns
        -------
        float
            Net input value.
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        Return class label prediction for input X.

        Parameters
        ----------
        X : ndarray or list
            Input sample.

        Returns
        -------
        int
            Predicted class label (-1 or 1).
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
