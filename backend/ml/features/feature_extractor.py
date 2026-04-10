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
    energy_distribution_variance
)


class FeatureExtractor:

    @staticmethod
    def _channel_stats(pixels):
        totals = {"r": 0, "g": 0, "b": 0}
        pixel_count = 0

        for row in pixels:
            for i in range(0, len(row), 3):
                totals["r"] += row[i]
                totals["g"] += row[i + 1]
                totals["b"] += row[i + 2]
                pixel_count += 1

        if pixel_count == 0:
            return {
                "avg_red": 0,
                "avg_green": 0,
                "avg_blue": 0,
                "colorfulness": 0,
            }

        avg_red = totals["r"] / pixel_count
        avg_green = totals["g"] / pixel_count
        avg_blue = totals["b"] / pixel_count

        colorfulness = (
            abs(avg_red - avg_green) +
            abs(avg_red - avg_blue) +
            abs(avg_green - avg_blue)
        ) / 3

        return {
            "avg_red": avg_red,
            "avg_green": avg_green,
            "avg_blue": avg_blue,
            "colorfulness": colorfulness,
        }

    @staticmethod
    def _classify_color(r, g, b):
        if max(r, g, b) - min(r, g, b) < 15:
            return "gray"

        if b > g and b > r:
            return "blue"

        if g > r and g > b:
            return "green"

        if r > b and r > g:
            if g > 0.7 * r:
                return "yellow"
            return "red"

        if r >= g >= b:
            return "yellow"

        return "mixed"

    @staticmethod
    def _dominant_color_features(pixels, height):
        counts = {
            "blue": 0,
            "green": 0,
            "yellow": 0,
            "gray": 0,
            "red": 0,
            "mixed": 0,
        }
        total_pixels = 0
        top_blue = 0
        top_pixels = 0
        bottom_green = 0
        bottom_yellow = 0
        bottom_pixels = 0
        split_index = max(1, height // 2)

        for row_index, row in enumerate(pixels):
            for i in range(0, len(row), 3):
                color_name = FeatureExtractor._classify_color(
                    row[i],
                    row[i + 1],
                    row[i + 2],
                )
                counts[color_name] += 1
                total_pixels += 1

                if row_index < split_index:
                    top_pixels += 1
                    if color_name == "blue":
                        top_blue += 1
                else:
                    bottom_pixels += 1
                    if color_name == "green":
                        bottom_green += 1
                    if color_name == "yellow":
                        bottom_yellow += 1

        dominant_color = max(counts, key=counts.get) if total_pixels else "mixed"

        return {
            "dominant_color": dominant_color,
            "dominant_color_ratio": counts[dominant_color] / total_pixels if total_pixels else 0,
            "top_blue_ratio": top_blue / top_pixels if top_pixels else 0,
            "bottom_green_ratio": bottom_green / bottom_pixels if bottom_pixels else 0,
            "bottom_yellow_ratio": bottom_yellow / bottom_pixels if bottom_pixels else 0,
        }

    @staticmethod
    def _spatial_brightness_features(grayscale):
        height = len(grayscale)
        split_index = max(1, height // 2)
        top_half = grayscale[:split_index]
        bottom_half = grayscale[split_index:]

        top_brightness = mean(top_half) if top_half else 0
        bottom_brightness = mean(bottom_half) if bottom_half else 0

        return {
            "top_brightness": top_brightness,
            "bottom_brightness": bottom_brightness,
            "brightness_balance": top_brightness - bottom_brightness,
        }

    @staticmethod
    def _edge_density(grayscale):
        height = len(grayscale)
        width = len(grayscale[0])
        strong_edges = 0
        total = 0

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                gx = grayscale[i][j + 1] - grayscale[i][j - 1]
                gy = grayscale[i + 1][j] - grayscale[i - 1][j]
                magnitude = (gx * gx + gy * gy) ** 0.5
                total += 1

                if magnitude > 20:
                    strong_edges += 1

        return strong_edges / total if total else 0

    @staticmethod
    def extract_from_png_reader(reader):
        grayscale = rgb_to_grayscale(reader.pixels, reader.width)

        m = mean(grayscale)
        v = variance(grayscale, m)
        e = entropy(grayscale)

        laplacian = laplacian_filter(grayscale)
        energy = residual_energy(laplacian)
        blocks = split_into_blocks(laplacian, 32)
        block_var = energy_distribution_variance(blocks) if blocks else 0
        grad_mean, grad_var = gradient_stats(grayscale)

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
            "energia_diagonal_2da_derivada_norm": energia_diag_norm,
            "edge_density": FeatureExtractor._edge_density(grayscale),
            "width": reader.width,
            "height": reader.height,
            "aspect_ratio": reader.width / (reader.height + 1e-6),
        }

        features.update(FeatureExtractor._channel_stats(reader.pixels))
        features.update(
            FeatureExtractor._dominant_color_features(reader.pixels, reader.height)
        )
        features.update(FeatureExtractor._spatial_brightness_features(grayscale))

        return features
