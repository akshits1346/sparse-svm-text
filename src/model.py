import numpy as np

class SparseSVM:
    def __init__(self, lr=0.01, lmbda=0.001, epochs=50):
        self.lr = lr
        self.lmbda = lmbda
        self.epochs = epochs
        self.W = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)

        y_transformed = np.where(y == 0, -1, 1)

        for epoch in range(self.epochs):
            for i in range(n_samples):
                xi = X[i].toarray().flatten()
                condition = y_transformed[i] * np.dot(xi, self.W)

                if condition < 1:
                    self.W += self.lr * (
                        y_transformed[i] * xi - self.lmbda * np.sign(self.W)
                    )
                else:
                    self.W += -self.lr * self.lmbda * np.sign(self.W)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1} complete")

    def predict(self, X):
        scores = X @ self.W
        return np.where(scores >= 0, 1, 0)

    def sparsity(self):
        return np.mean(self.W == 0)

