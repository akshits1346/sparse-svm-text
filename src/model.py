import numpy as np

class LinearSVM:
    def __init__(self, lr=0.01, epochs=50):
        self.lr = lr
        self.epochs = epochs
        self.W = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)

        y = np.where(y == 0, -1, 1)

        for _ in range(self.epochs):
            for i in range(n_samples):
                if y[i] * (X[i] @ self.W) < 1:
                    self.W += self.lr * y[i] * X[i].toarray().ravel()

    def predict(self, X):
        return np.sign(X @ self.W)

