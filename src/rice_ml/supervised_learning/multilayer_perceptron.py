"""
Multilayer Perceptron (MLP) classifier implemented from scratch.

This module provides a single-hidden-layer MLP for binary classification using:
- Sigmoid activation functions
- Cross-entropy loss
- Backpropagation
- Gradient descent optimization

This implementation is intended for educational purposes and mirrors the
structure of other supervised learning modules in this package.
"""

import numpy as np


class MultilayerPerceptron:
    """
    Multilayer Perceptron classifier (one hidden layer).

    Parameters
    ----------
    hidden_units : int, default=16
        Number of neurons in the hidden layer.

    learning_rate : float, default=0.01
        Gradient descent step size.

    epochs : int, default=100
        Number of full passes over the training data.

    random_state : int or None, default=None
        Seed for reproducible weight initialization.

    Attributes
    ----------
    W1 : ndarray
        Weights between input layer and hidden layer.

    b1 : ndarray
        Bias vector for hidden layer.

    W2 : ndarray
        Weights between hidden layer and output neuron.

    b2 : ndarray
        Bias for output neuron.

    losses_ : list
        Cross-entropy loss recorded at each epoch.
    """

    def __init__(self, hidden_units=16, learning_rate=0.01, epochs=100, random_state=None):
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        self.losses_ = []

    # -------------------------------------------------------------
    # Activation functions
    # -------------------------------------------------------------
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, a):
        return a * (1 - a)

    # -------------------------------------------------------------
    # Weight initialization
    # -------------------------------------------------------------
    def _initialize_weights(self, n_features):
        rng = np.random.default_rng(self.random_state)

        # Xavier initialization for stable gradients
        self.W1 = rng.normal(
            scale=np.sqrt(1 / n_features),
            size=(n_features, self.hidden_units)
        )
        self.b1 = np.zeros((1, self.hidden_units))

        self.W2 = rng.normal(
            scale=np.sqrt(1 / self.hidden_units),
            size=(self.hidden_units, 1)
        )
        self.b2 = np.zeros((1, 1))

    # -------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------
    def _forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = self._sigmoid(Z1)

        Z2 = A1 @ self.W2 + self.b2
        A2 = self._sigmoid(Z2)

        return A1, A2

    # -------------------------------------------------------------
    # Backpropagation
    # -------------------------------------------------------------
    def _backward(self, X, y, A1, A2):
        m = X.shape[0]
        y = y.reshape(-1, 1)

        # Output layer error
        dZ2 = A2 - y
        dW2 = (A1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Hidden layer error
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self._sigmoid_derivative(A1)
        dW1 = (X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Gradient descent update
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    # -------------------------------------------------------------
    # Loss function (cross-entropy)
    # -------------------------------------------------------------
    def _compute_loss(self, y, y_pred):
        y = y.reshape(-1, 1)
        eps = 1e-10
        m = len(y)
        return -(1 / m) * np.sum(
            y * np.log(y_pred + eps) +
            (1 - y) * np.log(1 - y_pred + eps)
        )

    # -------------------------------------------------------------
    # Fit method
    # -------------------------------------------------------------
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)
        self.losses_ = []

        for _ in range(self.epochs):
            A1, A2 = self._forward(X)
            loss = self._compute_loss(y, A2)
            self.losses_.append(loss)

            self._backward(X, y, A1, A2)

        return self

    # -------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------
    def predict_proba(self, X):
        _, A2 = self._forward(X)
        return A2.flatten()

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)
