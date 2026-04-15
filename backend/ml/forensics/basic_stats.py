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


def fft_1d(x):
    N = len(x)

    if N <= 1:
        return x

    if N % 2 != 0:
        raise ValueError("El tamaño debe ser potencia de 2")

    even = fft_1d(x[0::2])
    odd = fft_1d(x[1::2])

    result = [0] * N

    for k in range(N // 2):
        t = cmath.exp(-2j * cmath.pi * k / N) * odd[k]
        result[k] = even[k] + t
        result[k + N // 2] = even[k] - t

    return result

def fft_2d(matrix):
    # FFT por filas
    temp = [fft_1d(row) for row in matrix]

    # transponer
    temp = list(zip(*temp))

    # FFT por columnas
    temp = [fft_1d(list(col)) for col in temp]

    # transponer de regreso
    result = list(zip(*temp))

    return result


def spectral_energy(spectrum):

    total = 0
    count = 0

    for row in spectrum:
        for value in row:
            total += abs(value)   # 🔥 importante (complejos)
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

def ai_derivative_kernel_score(matrix):
    """
    DETECTOR BASADO EN DERIVADAS DE SEGUNDO ORDEN (LAPLACIANO)

    1) MODELO CONTINUO DE UNA IMAGEN
    --------------------------------
    Una imagen en escala de grises puede modelarse como una función:

        I(x, y)
        - x = posición horizontal
        - y = posición vertical
        - I = intensidad

    2) PRIMERA DERIVADA (CAMBIO DE INTENSIDAD)
    La derivada parcial en x se define como:

        ∂I/∂x = lim(h→0) [ I(x+h, y) - I(x, y) ] / h

    usamos una aproximación discreta:

        ∂I/∂x ≈ I(x+1, y) - I(x, y)

    Mejor aún (más simétrica):

        ∂I/∂x ≈ [ I(x+1, y) - I(x-1, y) ] / 2

    Esto mide cambios de intensidad → bordes.

    3) SEGUNDA DERIVADA (CAMBIO DEL CAMBIO)
    derivamos nuevamente:

        ∂²I/∂x²

    Definición continua:

        ∂²I/∂x² = lim(h→0) [ I(x+h) - 2I(x) + I(x-h) ] / h²

    Aproximación discreta (h = 1):

        ∂²I/∂x² ≈ I(x+1) - 2I(x) + I(x-1)

    mide:
        - curvatura
        - irregularidades finas
        - cambios abruptos en la textura

    4) EXTENSIÓN A 2D (LAPLACIANO)
    En 2D:

        ∇²I = ∂²I/∂x² + ∂²I/∂y²

    Sustituyendo las aproximaciones discretas:

        ∇²I ≈
            [I(x+1,y) - 2I(x,y) + I(x-1,y)] + [I(x,y+1) - 2I(x,y) + I(x,y-1)]

    Reordenando:

        ∇²I ≈
            I(x+1,y) + I(x-1,y) + I(x,y+1) + I(x,y-1) - 4I(x,y)

    Esto se traduce directamente al kernel:

        [ 0  1  0 ]
        [ 1 -4  1 ]
        [ 0  1  0 ]


    5) MODIFICACIÓN PARA DETECTAR IA

    Para capturar mejor  irregularidades, extendemos el Laplaciano incluyendo diagonales:

        [ -1  -1  -1 ]
        [ -1   8  -1 ]
        [ -1  -1  -1 ]

    6) CONVOLUCIÓN
    Aplicar el kernel significa:

        R(i,j) = suma de: I vecinos * pesos del kernel

    Esto aproxima el Laplaciano en cada pixel.


    7) ENERGÍA DEL RESULTADO
    Calculamos:

        Energía = promedio de R(i,j)^2

    Esto mide:
        - cantidad de detalle de alta frecuencia
        - irregularidad estructural


    8) INTERPRETACIÓN FINAL
    - Imágenes reales:
        alta variabilidad -> energía moderada-alta

    - Imágenes IA:
        patrones artificiales -> energía atípica

    """

    height = len(matrix)
    width = len(matrix[0])

    # Kernel derivado matemáticamente (Laplaciano extendido)
    kernel = [
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ]

    # Matriz de salida
    response = [[0 for _ in range(width)] for _ in range(height)]

    # Convolución
    for i in range(1, height - 1):
        for j in range(1, width - 1):

            value = 0

            for ki in range(3):
                for kj in range(3):
                    value += (
                        matrix[i + ki - 1][j + kj - 1] *
                        kernel[ki][kj]
                    )

            response[i][j] = value

    # Energía del resultado (R^2 promedio)
    total = 0
    count = 0

    for row in response:
        for value in row:
            total += value * value
            count += 1

    energy = total / count

    return energy