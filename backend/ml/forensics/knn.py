import numpy as np


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = np.empty((0, 0), dtype=np.float64)
        self.y_train = np.empty((0,), dtype=np.int64)

    def fit(self, X, y):
        X_array = np.asarray(X, dtype=np.float64)
        y_array = np.asarray(y, dtype=np.int64)

        if X_array.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y_array.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if len(X_array) != len(y_array):
            raise ValueError("X and y must contain the same number of samples.")
        if len(X_array) == 0:
            raise ValueError("Training data cannot be empty.")

        self.X_train = X_array
        self.y_train = y_array
        return self

    def _neighbor_indices(self, x):
        if self.X_train.size == 0:
            raise ValueError("The model must be fitted before prediction.")

        sample = np.asarray(x, dtype=np.float64)
        distances = np.linalg.norm(self.X_train - sample, axis=1)
        neighbor_count = min(self.k, len(distances))

        if neighbor_count == 0:
            return np.empty((0,), dtype=np.int64)

        indices = np.argpartition(distances, neighbor_count - 1)[:neighbor_count]
        return indices[np.argsort(distances[indices])]

    def predict_one(self, x):
        neighbor_indices = self._neighbor_indices(x)
        neighbor_labels = self.y_train[neighbor_indices]
        positive_votes = int(np.sum(neighbor_labels == 1))
        negative_votes = len(neighbor_labels) - positive_votes
        return 1 if positive_votes > negative_votes else 0

    def confidence_one(self, x):
        neighbor_indices = self._neighbor_indices(x)
        if len(neighbor_indices) == 0:
            return None

        neighbor_labels = self.y_train[neighbor_indices]
        prediction = self.predict_one(x)
        return float(np.mean(neighbor_labels == prediction))

    def predict(self, X):
        X_array = np.asarray(X, dtype=np.float64)
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)
        return [self.predict_one(row) for row in X_array]
