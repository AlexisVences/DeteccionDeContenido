import math, cmath


def rgb_to_grayscale(pixels, width):
    """
    Convierte matriz RGB (lista de filas) a matriz grayscale.
    """
    grayscale = []

    for row in pixels:
        gray_row = []
        for i in range(0, len(row), 3):
            r = row[i]
            g = row[i + 1]
            b = row[i + 2]

            # Fórmula estándar luminancia
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            gray_row.append(gray)

        grayscale.append(gray_row)

    return grayscale


def mean(matrix):
    total = 0
    count = 0

    for row in matrix:
        for value in row:
            total += value
            count += 1

    return total / count


def variance(matrix, mean_value=None):
    if mean_value is None:
        mean_value = mean(matrix)

    total = 0
    count = 0

    for row in matrix:
        for value in row:
            total += (value - mean_value) ** 2
            count += 1

    return total / count


def entropy(matrix):
    histogram = [0] * 256
    total_pixels = 0

    for row in matrix:
        for value in row:
            histogram[value] += 1
            total_pixels += 1

    ent = 0

    for count in histogram:
        if count > 0:
            p = count / total_pixels
            ent -= p * math.log2(p)

    return ent


def laplacian_filter(matrix):
    """
    Filtro Laplaciano 3x3 manual
    """
    height = len(matrix)
    width = len(matrix[0])

    output = [[0 for _ in range(width)] for _ in range(height)]

    kernel = [
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ]

    for i in range(1, height - 1):
        for j in range(1, width - 1):

            value = 0

            for ki in range(3):
                for kj in range(3):
                    value += (
                        matrix[i + ki - 1][j + kj - 1] *
                        kernel[ki][kj]
                    )

            output[i][j] = value

    return output


def residual_energy(matrix):
    total = 0
    count = 0

    for row in matrix:
        for value in row:
            total += value * value
            count += 1

    return total / count


def dft_2d(matrix):
    height = len(matrix)
    width = len(matrix[0])

    result = [[0 for _ in range(width)] for _ in range(height)]

    for u in range(height):
        for v in range(width):

            sum_val = 0

            for x in range(height):
                for y in range(width):

                    angle = -2j * cmath.pi * ((u * x / height) + (v * y / width))
                    sum_val += matrix[x][y] * cmath.exp(angle)

            result[u][v] = abs(sum_val)

    return result


def spectral_energy(spectrum):

    total = 0
    count = 0

    for row in spectrum:
        for value in row:
            total += value
            count += 1

    return total / count


def gradient_magnitude(matrix):

    height = len(matrix)
    width = len(matrix[0])

    gradients = []

    for i in range(1, height - 1):
        for j in range(1, width - 1):

            gx = matrix[i][j+1] - matrix[i][j-1]
            gy = matrix[i+1][j] - matrix[i-1][j]

            mag = (gx*gx + gy*gy) ** 0.5
            gradients.append(mag)

    return gradients


def gradient_stats(matrix):

    grads = gradient_magnitude(matrix)

    mean = sum(grads) / len(grads)

    var = sum((g - mean)**2 for g in grads) / len(grads)

    return mean, var