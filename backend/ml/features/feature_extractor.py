from ..forensics.basic_stats import (
    rgb_to_grayscale,
    mean,
    variance,
    entropy,
    laplacian_filter,
    residual_energy,
    dft_2d,
    spectral_energy,
    gradient_stats,
    ai_derivative_kernel_score
)

from ..forensics.block_analysis import (
    split_into_blocks,
    block_energy,
    energy_distribution_variance
)

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
        sub_image = [row[:64] for row in grayscale[:64]]
        spectrum = dft_2d(sub_image)
        spec_energy = spectral_energy(spectrum)
        energia_diag_2da_derivada = ai_derivative_kernel_score(grayscale)
        energia_diag_norm = energia_diag_2da_derivada / (v + 1e-6)    
            
        features = {
            "mean": m,
            "variance": v,
            "entropy": e,
            "hf_energy": energy,
            "energy_to_variance_ratio": energy / (v + 1e-6),
            "block_energy_variance": block_var,
            "gradiente_mean": grad_mean,
            "gradient_variance": grad_var,
            "spectral_energy": spec_energy,
            "energia_diagonal_2da_derivada_norm": energia_diag_norm

        }

        return features