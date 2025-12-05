import numpy as np

class MultilayerPerceptron:

    def __init__(self, hidden_units=16, learning_rate=0.01, epochs=100,
                 random_state=None, batch_size=None, weight_decay=0.0):

        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        self.losses_ = []
        self._eps = 1e-10

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, a):
        return a * (1 - a)

    def _initialize_weights(self, n_features, rng):
        self.W1 = rng.normal(scale=np.sqrt(1 / n_features),
                             size=(n_features, self.hidden_units))
        self.b1 = np.zeros((1, self.hidden_units))

        self.W2 = rng.normal(scale=np.sqrt(1 / self.hidden_units),
                             size=(self.hidden_units, 1))
        self.b2 = np.zeros((1, 1))

    def _forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = self._sigmoid(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = self._sigmoid(Z2)
        return A1, A2

    def _compute_loss(self, y, y_pred):
        y = y.reshape(-1, 1)
        m = len(y)

        base_loss = -(1 / m) * np.sum(
            y * np.log(y_pred + self._eps) +
            (1 - y) * np.log(1 - y_pred + self._eps)
        )

        if self.weight_decay == 0:
            return base_loss

        l2_term = (self.weight_decay / (2 * m)) * \
            (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return base_loss + l2_term

    def _backward(self, X, y, A1, A2):
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

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def fit(self, X, y):
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        self._initialize_weights(n_features, rng)
        self.losses_ = []

        batch_size = self.batch_size or n_samples

        for _ in range(self.epochs):

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

        return self

    def predict_proba(self, X):
        _, A2 = self._forward(X)
        return A2.flatten()

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)
