import numpy as np

def soft_threshold(w, lmbda):
    return np.sign(w) * np.maximum(np.abs(w) - lmbda, 0)

class LinearSVM:
    def __init__(self, lr=0.01, epochs=50):
        self.lr = lr
        self.epochs = epochs
        self.W = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)

        # Convert labels from {0,1} to {-1, +1}
        y = np.where(y == 0, -1, 1)

        for _ in range(self.epochs):
            for i in range(n_samples):
                if y[i] * (X[i] @ self.W) < 1:
                    self.W += self.lr * y[i] * X[i].toarray().ravel()

                # Proximal step for L1 regularization (sparsity)
                self.W = soft_threshold(self.W, self.lr * 0.001)

    def predict(self, X):
        return np.sign(X @ self.W)

