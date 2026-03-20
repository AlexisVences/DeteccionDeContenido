class KNN:

    def __init__(self, k=3):
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _euclidean(self, a, b):

        total = 0

        for i in range(len(a)):
            total += (a[i] - b[i]) ** 2

        return total ** 0.5

    def _predict_one(self, x):

        distances = []

        for i in range(len(self.X_train)):

            d = self._euclidean(self.X_train[i], x)

            distances.append((d, self.y_train[i]))

        distances.sort(key=lambda t: t[0])

        neighbors = distances[:self.k]

        votes = {0:0, 1:0}

        for _, label in neighbors:
            votes[label] += 1

        if votes[1] > votes[0]:
            return 1
        else:
            return 0

    def predict(self, X):

        predictions = []

        for x in X:
            predictions.append(self._predict_one(x))

        return predictions