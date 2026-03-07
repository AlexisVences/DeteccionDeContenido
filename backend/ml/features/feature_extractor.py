from ..forensics.basic_stats import (
    rgb_to_grayscale,
    mean,
    variance,
    entropy,
    laplacian_filter,
    residual_energy,
    dft_2d,
    spectral_energy,
    gradient_stats
)

from ..forensics.block_analysis import (
    split_into_blocks,
    block_energy,
    energy_distribution_variance
)

# siguiente implementacion: A. Análisis del Espectro de Fourier (FFT)
class FeatureExtractor:

    @staticmethod
    def extract_from_png_reader(reader):

        grayscale = rgb_to_grayscale(reader.pixels, reader.width)

        m = mean(grayscale)
        v = variance(grayscale, m)
        e = entropy(grayscale)
        laplacian = laplacian_filter(grayscale)
        energy = residual_energy(laplacian)    
        blocks = split_into_blocks(laplacian, 32)
        block_var = energy_distribution_variance(blocks)
        #metrica del gradiente
        grad_mean, grad_var = gradient_stats(grayscale)
        # espectro de fourier
        spectrum = dft_2d(grayscale[:64][:64])  # limitar tamaño
        spec_energy = spectral_energy(spectrum)

        features = {
            "mean": m,
            "variance": v,
            "entropy": e,
            "hf_energy": energy,
            "energy_to_variance_ratio": energy / (v + 1e-6),
            "block_energy_variance": block_var,
            "gradiente_mean": grad_mean,
            "gradient_variance": grad_var,
            "spectral_energy": spec_energy
        }

        return features