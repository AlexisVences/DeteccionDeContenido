import math
import random


class SVM:
    def __init__(self, gamma=0.5, epochs=20):
        self.gamma = gamma
        self.epochs = epochs

        self.X_train = []
        self.y_train = []
        self.alpha = []

    def _rbf(self, x1, x2):
        sq_dist = 0.0

        for i in range(len(x1)):
            sq_dist += (x1[i] - x2[i]) ** 2

        return math.exp(-self.gamma * sq_dist)

    def fit(self, X, y):
        self.X_train = X
        self.y_train = [1 if label == 1 else -1 for label in y]

        n = len(X)
        self.alpha = [0.0] * n

        for epoch in range(self.epochs):
            indices = list(range(n))
            random.shuffle(indices)

            errors = 0

            for i in indices:
                prediction = self._decision_function(X[i])

                if self.y_train[i] * prediction <= 0:
                    self.alpha[i] += 1
                    errors += 1

            print(f"Epoch {epoch+1}/{self.epochs} - errores: {errors}")

            if errors == 0:
                break

    def _decision_function(self, x):
        result = 0.0

        for i in range(len(self.X_train)):
            if self.alpha[i] > 0:
                result += (
                    self.alpha[i]
                    * self.y_train[i]
                    * self._rbf(self.X_train[i], x)
                )

        return result

    def predict(self, X):
        predictions = []

        for x in X:
            val = self._decision_function(x)
            predictions.append(1 if val >= 0 else 0)

        return predictions