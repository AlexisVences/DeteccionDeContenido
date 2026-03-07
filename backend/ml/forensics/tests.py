from raw_image_reader import PNGReader
from basic_stats import (
    rgb_to_grayscale,
    mean,
    variance,
    entropy,
    laplacian_filter,
    residual_energy
)

if __name__ == "__main__":
    reader = PNGReader("paisaje.png")
    reader.read()

    grayscale = rgb_to_grayscale(reader.pixels, reader.width)

    m = mean(grayscale)
    v = variance(grayscale, m)
    e = entropy(grayscale)

    laplacian = laplacian_filter(grayscale)
    energy = residual_energy(laplacian)

    print("Mean:", m)
    print("Variance:", v)
    print("Entropy:", e)
    print("High-Frequency Energy:", energy)