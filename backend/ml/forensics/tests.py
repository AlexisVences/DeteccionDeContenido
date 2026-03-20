from raw_image_reader import PNGReader
from basic_stats import (
    rgb_to_grayscale,
    mean,
    variance,
    entropy,
    laplacian_filter,
    residual_energy
)
import os
from knn import KNN
import random

def extract_features(reader):

    grayscale = rgb_to_grayscale(reader.pixels, reader.width)

    m = mean(grayscale)
    v = variance(grayscale, m)
    e = entropy(grayscale)

    laplacian = laplacian_filter(grayscale)
    energy = residual_energy(laplacian)

    return [m, v, e, energy]

def load_dataset(dataset_path):

    X = []
    y = []

    for label_name in os.listdir(dataset_path):

        folder = os.path.join(dataset_path, label_name)

        if not os.path.isdir(folder):
            continue

        label = 0 if label_name.lower() == "training_real" else 1

        print("Procesando carpeta:", label_name)

        for file in os.listdir(folder):

            if not file.endswith(".png"):
                continue

            path = os.path.join(folder, file)

            reader = PNGReader(path)
            reader.read()

            features = extract_features(reader)

            X.append(features)
            y.append(label)

    return X, y

def normalize(X):

    cols = len(X[0])

    mins = [min(row[i] for row in X) for i in range(cols)]
    maxs = [max(row[i] for row in X) for i in range(cols)]

    Xn = []

    for row in X:

        new_row = []

        for i in range(cols):

            if maxs[i] == mins[i]:
                new_row.append(0)
            else:
                val = (row[i] - mins[i]) / (maxs[i] - mins[i])
                new_row.append(val)

        Xn.append(new_row)

    return Xn

def train_test_split(X, y, ratio=0.8):

    data = list(zip(X, y))

    random.shuffle(data)

    X, y = zip(*data)

    split = int(len(X) * ratio)

    X_train = list(X[:split])
    y_train = list(y[:split])

    X_test = list(X[split:])
    y_test = list(y[split:])

    return X_train, X_test, y_train, y_test

def accuracy(y_true, y_pred):

    correct = 0

    for i in range(len(y_true)):

        if y_true[i] == y_pred[i]:
            correct += 1

    return correct / len(y_true)

if __name__ == "__main__":
    dataset_path = "./test_images/imagenes/real_and_fake_face"

    print("Cargando dataset...")

    X, y = load_dataset(dataset_path)

    print("Total imágenes:", len(X))

    print("Normalizando features...")

    X = normalize(X)

    print("Dividiendo train/test...")

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print("Entrenando KNN...")

    model = KNN(k=5)

    model.fit(X_train, y_train)

    print("Prediciendo...")

    predictions = model.predict(X_test)

    acc = accuracy(y_test, predictions)

    print("Accuracy:", acc)