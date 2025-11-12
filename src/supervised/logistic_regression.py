
import numpy as np

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

class LogisticRegressionGD:
    def __init__(self, lr: float = 0.1, n_iter: int = 2000, fit_intercept: bool = True, random_state: int | None = 42):
        self.lr = lr
        self.n_iter = n_iter
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    def _add_bias(self, X):
        return np.c_[np.ones(len(X)), X] if self.fit_intercept else X

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        Xb = self._add_bias(X)
        rng = np.random.default_rng(self.random_state)
        self.w_ = rng.normal(scale=0.01, size=Xb.shape[1])
        m = len(Xb)
        self.loss_ = []
        for _ in range(self.n_iter):
            z = Xb @ self.w_
            p = _sigmoid(z)
            grad = (1/m) * (Xb.T @ (p - y))
            self.w_ -= self.lr * grad
            eps = 1e-9
            loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            self.loss_.append(loss)
        return self

    def predict_proba(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        Xb = self._add_bias(X)
        return _sigmoid(Xb @ self.w_)

    def predict(self, X: np.ndarray, threshold: float = 0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
