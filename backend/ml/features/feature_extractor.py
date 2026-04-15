from forensics.basic_stats import (
    rgb_to_grayscale,
    mean,
    variance,
    entropy,
    laplacian_filter,
    residual_energy,
    fft_2d,
    spectral_energy,
    gradient_stats,
    ai_derivative_kernel_score
)

from forensics.block_analysis import (
    split_into_blocks,
    block_energy,
    energy_distribution_variance,
    block_variance_stats,
    symmetry_score,
    local_hf_variance,
    gradient_direction_consistency,
    block_entropy_variance
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
        #espectro de fourier
        sub_image = [row[:64] for row in grayscale[:64]]
        spectrum = fft_2d(sub_image)
        spec_energy = spectral_energy(spectrum)
        energia_diag_2da_derivada = ai_derivative_kernel_score(grayscale)
        energia_diag_norm = energia_diag_2da_derivada / (v + 1e-6)  

        blocks = split_into_blocks(grayscale, 32)

        mean_var, var_var = block_variance_stats(blocks)
        sym = symmetry_score(grayscale)
        hf_local = local_hf_variance(blocks)
        grad_dir = gradient_direction_consistency(grayscale)
        ent_mean, ent_var = block_entropy_variance(blocks)  
            
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
            "energia_diagonal_2da_derivada_norm": energia_diag_norm,

            "block_variance_mean": mean_var,
            "block_variance_var": var_var,
            "symmetry_score": sym,
            "local_hf_variance": hf_local,
            "gradient_direction": grad_dir,
            "entropy_block_var": ent_var
        }

        return features