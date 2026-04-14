class SVM:

    def __init__(self, lr=0.001, lambda_param=0.01, epochs=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.w = None
        self.b = 0

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])

        # convertir etiquetas a -1 y 1
        y_ = [1 if label == 1 else -1 for label in y]

        # inicializar pesos
        self.w = [0.0] * n_features
        self.b = 0.0

        for _ in range(self.epochs):
            for i in range(n_samples):
                condition = y_[i] * (self._dot(self.w, X[i]) + self.b)

                if condition >= 1:
                    # solo regularización
                    for j in range(n_features):
                        self.w[j] -= self.lr * (2 * self.lambda_param * self.w[j])
                else:
                    # error → actualizar fuerte
                    for j in range(n_features):
                        self.w[j] -= self.lr * (
                            2 * self.lambda_param * self.w[j] - y_[i] * X[i][j]
                        )
                    self.b -= self.lr * y_[i]

    def predict(self, X):
        predictions = []
        for x in X:
            val = self._dot(self.w, x) + self.b
            predictions.append(1 if val >= 0 else 0)
        return predictions

    def _dot(self, a, b):
        return sum(a[i] * b[i] for i in range(len(a)))