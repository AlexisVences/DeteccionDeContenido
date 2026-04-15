# python -m forensics.tests

from forensics.raw_image_reader import PNGReader
from features.feature_extractor import FeatureExtractor
from multiprocessing import Pool, cpu_count
import os
from forensics.knn import KNN
from forensics.svm import SVM
import random

def select_features(feature_dict):
    return [
        feature_dict["energy_to_variance_ratio"],
        feature_dict["block_energy_variance"],
        feature_dict["spectral_energy"],
        feature_dict["energia_diagonal_2da_derivada_norm"],

        # 🔥 nuevas
        feature_dict["block_variance_mean"],
        feature_dict["block_variance_var"],
        feature_dict["symmetry_score"],
        feature_dict["local_hf_variance"],
        feature_dict["gradient_direction"],
        feature_dict["entropy_block_var"]
    ]

def process_image(args):
    path, label = args

    try:
        reader = PNGReader(path)
        reader.read()

        features_dict = FeatureExtractor.extract_from_png_reader(reader)
        features = select_features(features_dict)

        return features, label

    except Exception as e:
        print(f"\nError en {path}: {e}")
        return None

def load_dataset(dataset_path):

    tasks = []

    for label_name in os.listdir(dataset_path):

        folder = os.path.join(dataset_path, label_name)

        if not os.path.isdir(folder):
            continue

        label = 0 if label_name.lower() == "real" else 1

        for file in os.listdir(folder):

            if not file.endswith(".png"):
                continue

            path = os.path.join(folder, file)
            tasks.append((path, label))

    print(f"Total tareas: {len(tasks)}")

    # 🔥 usar todos los núcleos
    with Pool(cpu_count() - 1) as p:
        results = []
        total = len(tasks)

        try:
            for i, result in enumerate(p.imap_unordered(process_image, tasks), 1):
                results.append(result)

                # progreso en la misma línea
                print(f"\rProcesadas {i}/{total} imágenes ({(i/total)*100:.1f}%)", end="")

        except KeyboardInterrupt:
            print("\nInterrumpido")
            p.terminate()
            p.join()

    print()

    X = []
    y = []

    for result in results:
        if result is None:
            continue

        features, label = result
        X.append(features)
        y.append(label)

    return X, y
    
def fit_standardizer(X):

    cols = len(X[0])

    means = [0] * cols
    stds = [0] * cols

    # calcular medias
    for i in range(cols):
        means[i] = sum(row[i] for row in X) / len(X)

    # calcular desviaciones estándar
    for i in range(cols):
        variance = sum((row[i] - means[i]) ** 2 for row in X) / len(X)
        stds[i] = variance ** 0.5

    return means, stds


def transform_standardizer(X, means, stds):

    Xn = []

    for row in X:

        new_row = []

        for i in range(len(row)):

            if stds[i] == 0:
                new_row.append(0)
            else:
                val = (row[i] - means[i]) / stds[i]
                new_row.append(val)

        Xn.append(new_row)

    return Xn

def train_test_split(X, y, ratio=0.8):
    data = list(zip(X, y))

    real = [d for d in data if d[1] == 0]
    fake = [d for d in data if d[1] == 1]

    random.shuffle(real)
    random.shuffle(fake)

    split_r = int(len(real) * ratio)
    split_f = int(len(fake) * ratio)

    train = real[:split_r] + fake[:split_f]
    test = real[split_r:] + fake[split_f:]

    random.shuffle(train)
    random.shuffle(test)

    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    return list(X_train), list(X_test), list(y_train), list(y_test)

def accuracy(y_true, y_pred):

    correct = 0

    for i in range(len(y_true)):

        if y_true[i] == y_pred[i]:
            correct += 1

    return correct / len(y_true)

def compute_feature_averages(X, y, feature_names):
    
    # separar por clase
    sums = {
        0: [0] * len(feature_names),
        1: [0] * len(feature_names)
    }
    
    counts = {
        0: 0,
        1: 0
    }

    # acumular
    for i in range(len(X)):
        label = y[i]
        counts[label] += 1
        
        for j in range(len(X[i])):
            sums[label][j] += X[i][j]

    # calcular promedios
    averages = {
        0: [],
        1: []
    }

    for label in [0, 1]:
        for j in range(len(feature_names)):
            if counts[label] == 0:
                avg = 0
            else:
                avg = sums[label][j] / counts[label]
            
            averages[label].append(avg)

    return averages

if __name__ == "__main__":
    dataset_path = "./test_images/dataset"

    print("Cargando dataset...")

    X, y = load_dataset(dataset_path)

    print("Total imágenes:", len(X))

    print("\nCalculando promedios por clase...")

    feature_names = [
        "energy_to_variance_ratio",
        "block_energy_variance",
        "spectral_energy",
        "energia_diagonal_2da_derivada_norm",
        "block_variance_mean",
        "block_variance_var",
        "symmetry_score",
        "local_hf_variance",
        "gradient_direction",
        "entropy_block_var"
    ]

    averages = compute_feature_averages(X, y, feature_names)

    print("\n=== PROMEDIOS ===")

    for label in [0, 1]:
        print("\nClase:", "REAL (0)" if label == 0 else "FAKE (1)")
        
        for i in range(len(feature_names)):
            print(f"{feature_names[i]}: {averages[label][i]:.5f}")

    print("\n=== DIFERENCIA ENTRE CLASES ===")

    for i in range(len(feature_names)):
        diff = abs(averages[0][i] - averages[1][i])
        print(f"{feature_names[i]:40} {diff:.5f}")

    print("Dividiendo train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    
    print("Normalizando...")
    means, stds = fit_standardizer(X_train)
    X_train = transform_standardizer(X_train, means, stds)
    X_test = transform_standardizer(X_test, means, stds)
    print(X_train[0])

    print("Entrenando KNN...")

    model = KNN(k=5)

    model.fit(X_train, y_train)

    print("Prediciendo...")

    predictions = model.predict(X_test)

    acc = accuracy(y_test, predictions)

    print("Accuracy:", acc)