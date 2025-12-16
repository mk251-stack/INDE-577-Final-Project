import numpy as np
import warnings
from typing import Optional


class MultilayerPerceptron:
    """
    Single-hidden-layer Multilayer Perceptron (MLP) for binary classification.

    This is a minimal, from-scratch implementation of a feedforward neural
    network trained using gradient descent and backpropagation. It is intended
    for educational and diagnostic purposes rather than production use.

    Architecture
    ------------
    - Input layer
    - One hidden layer with sigmoid activation
    - Output layer with sigmoid activation

    Loss
    ----
    Binary cross-entropy loss with optional L2 regularization.

    Notes
    -----
    - Uses full-batch or mini-batch gradient descent depending on `batch_size`
    - Inputs should be standardized (zero mean, unit variance) and numeric
    - Targets must be binary; {-1, 1} labels are internally mapped to {0, 1}
    - Includes simple stability guards (gradient clipping and early stopping)
    - Designed to illustrate optimization challenges in neural networks
    """

    def __init__(
        self,
        hidden_units: int = 16,
        learning_rate: float = 0.01,
        epochs: int = 100,
        random_state: Optional[int] = None,
        batch_size: Optional[int] = None,
        weight_decay: float = 0.0,
        max_grad_norm: Optional[float] = 5.0,
        tol: float = 1e-4,
        patience: int = 5,
    ):

        """
        Initialize the Multilayer Perceptron.

        Parameters
        ----------
        hidden_units : int, default=16
            Number of neurons in the hidden layer.
        learning_rate : float, default=0.01
            Step size for gradient descent updates.
        epochs : int, default=100
            Number of training epochs.
        random_state : int or None, default=None
            Random seed for weight initialization and shuffling.
        batch_size : int or None, default=None
            Size of mini-batches. If None, full-batch gradient descent is used.
        weight_decay : float, default=0.0
            L2 regularization strength (0 disables regularization).
        max_grad_norm : float or None, default=5.0
            Threshold for gradient clipping. Set to None to disable clipping.
        tol : float, default=1e-4
            Minimum improvement in loss required before triggering early stopping.
        patience : int, default=5
            Number of epochs with loss improvement below `tol` before stopping.
        """
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.tol = tol
        self.patience = patience

        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        self.losses_ = []
        self._eps = 1e-10
        self.stop_reason_ = None
        self.n_epochs_ = 0

    # ------------------------------------------------------------------
    # Activation functions
    # ------------------------------------------------------------------
    def _sigmoid(self, z):
        """
        Sigmoid activation function.

        Parameters
        ----------
        z : ndarray
            Input array.

        Returns
        -------
        a : ndarray
            Element-wise sigmoid activation.
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, a):
        """
        Derivative of the sigmoid function.

        Parameters
        ----------
        a : ndarray
            Sigmoid activation values.

        Returns
        -------
        da : ndarray
            Derivative of sigmoid with respect to its input.
        """
        return a * (1 - a)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _initialize_weights(self, n_features, rng):
        """
        Initialize network weights using Xavier initialization.

        Parameters
        ----------
        n_features : int
            Number of input features.
        rng : numpy.random.Generator
            Random number generator.
        """
        self.W1 = rng.normal(
            scale=np.sqrt(1 / n_features),
            size=(n_features, self.hidden_units),
        )
        self.b1 = np.zeros((1, self.hidden_units))

        self.W2 = rng.normal(
            scale=np.sqrt(1 / self.hidden_units),
            size=(self.hidden_units, 1),
        )
        self.b2 = np.zeros((1, 1))

    # ------------------------------------------------------------------
    # Forward and backward propagation
    # ------------------------------------------------------------------
    def _forward(self, X):
        """
        Perform a forward pass through the network.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        A1 : ndarray of shape (n_samples, hidden_units)
            Hidden-layer activations.
        A2 : ndarray of shape (n_samples, 1)
            Output probabilities.
        """
        Z1 = X @ self.W1 + self.b1
        A1 = self._sigmoid(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = self._sigmoid(Z2)
        return A1, A2

    def _compute_loss(self, y, y_pred):
        """
        Compute binary cross-entropy loss with optional L2 regularization.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True binary labels.
        y_pred : ndarray of shape (n_samples, 1)
            Predicted probabilities.

        Returns
        -------
        loss : float
            Scalar loss value.
        """
        y = y.reshape(-1, 1)
        m = len(y)

        base_loss = -(1 / m) * np.sum(
            y * np.log(y_pred + self._eps)
            + (1 - y) * np.log(1 - y_pred + self._eps)
        )

        if self.weight_decay == 0:
            return base_loss

        l2_term = (self.weight_decay / (2 * m)) * (
            np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2)
        )
        return base_loss + l2_term

    def _backward(self, X, y, A1, A2):
        """
        Perform backpropagation and update weights.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input batch.
        y : ndarray of shape (n_samples,)
            True labels.
        A1 : ndarray
            Hidden-layer activations.
        A2 : ndarray
            Output-layer activations.
        """
        m = X.shape[0]
        y = y.reshape(-1, 1)

        dZ2 = A2 - y
        dW2 = (A1.T @ dZ2) / m
        if self.weight_decay:
            dW2 += (self.weight_decay / m) * self.W2
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self._sigmoid_derivative(A1)
        dW1 = (X.T @ dZ1) / m
        if self.weight_decay:
            dW1 += (self.weight_decay / m) * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        if self.max_grad_norm is not None:
            self._clip_gradients(dW1, dW2)

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def _clip_gradients(self, dW1, dW2):
        """Apply global-norm clipping to stabilize training."""
        for grad in (dW1, dW2):
            norm = np.linalg.norm(grad)
            if norm > self.max_grad_norm > 0:
                grad *= self.max_grad_norm / (norm + self._eps)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X, y):
        """
        Train the MLP using gradient descent.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : ndarray of shape (n_samples,)
            Training labels.

        Returns
        -------
        self : MultilayerPerceptron
            Fitted model.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self._validate_inputs(X, y)

        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        self._initialize_weights(n_features, rng)
        self.losses_ = []
        self.stop_reason_ = None
        self.n_epochs_ = 0

        batch_size = self.batch_size or n_samples

        no_improve_epochs = 0

        for epoch in range(self.epochs):
            indices = rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                A1, A2 = self._forward(X_batch)
                self._backward(X_batch, y_batch, A1, A2)

            _, A2_full = self._forward(X)
            loss = self._compute_loss(y, A2_full)
            self.losses_.append(loss)
            self.n_epochs_ = epoch + 1

            if not np.isfinite(loss) or loss > 1e6:
                self.stop_reason_ = "loss_exploded"
                warnings.warn(
                    "Training stopped because the loss became non-finite or exploded. "
                    "Verify input scaling and reduce the learning rate.",
                    UserWarning,
                )
                break

            if len(self.losses_) > 1:
                improvement = self.losses_[-2] - self.losses_[-1]
                if improvement < self.tol:
                    no_improve_epochs += 1
                else:
                    no_improve_epochs = 0

                if no_improve_epochs >= self.patience:
                    self.stop_reason_ = "early_stopping"
                    break

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        probs : ndarray of shape (n_samples,)
            Predicted probabilities for the positive class.
        """
        _, A2 = self._forward(X)
        return A2.flatten()

    def predict(self, X):
        """
        Predict binary class labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_inputs(self, X, y):
        """Validate feature scaling, numeric types, and binary labels."""
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")

        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("X must contain numeric values. Please encode categorical features.")

        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values. Clean or impute your data before training.")

        feature_means = X.mean(axis=0)
        feature_stds = X.std(axis=0)

        if np.any(feature_stds == 0):
            raise ValueError("At least one feature has zero variance; remove or adjust constant columns.")

        if np.any(np.abs(feature_means) > 5) or np.any(feature_stds > 5):
            warnings.warn(
                "Inputs do not appear standardized (mean far from 0 or std far from 1). "
                "Performance may degrade; please scale your features.",
                UserWarning,
            )

        if y.ndim != 1:
            raise ValueError("y must be a 1D array of binary labels.")

        unique_labels = set(np.unique(y))
        if unique_labels == {-1, 1}:
            warnings.warn(
                "Mapping labels from {-1, 1} to {0, 1} for compatibility.",
                UserWarning,
            )
            y[:] = (y + 1) / 2
        elif unique_labels != {0, 1}:
            warnings.warn(
                "MultilayerPerceptron only supports binary targets encoded as 0/1 or -1/1.",
                UserWarning,
            )
            raise ValueError("Invalid labels: expected binary targets encoded as 0/1 or -1/1.")
