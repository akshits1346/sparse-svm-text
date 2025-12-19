import numpy as np

class SparseSVM:
    def __init__(self, lr=0.1, epochs=50, lmbda=0.001):
        self.lr = lr
        self.epochs = epochs
        self.lmbda = lmbda

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        y_transformed = np.where(y == 0, -1, 1)  # Convert labels to {-1,1}
        
        for epoch in range(self.epochs):
            for i in range(n_samples):
                condition = y_transformed[i] * np.dot(X[i].toarray(), self.W)
                if condition < 1:
                    self.W += self.lr * (y_transformed[i]*X[i].toarray().flatten() - self.lmbda*np.sign(self.W))
                else:
                    self.W += -self.lr * self.lmbda*np.sign(self.W)
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1} complete")
                
    def predict(self, X):
        pred = np.dot(X.toarray(), self.W)
        return np.where(pred >= 0, 1, 0)


