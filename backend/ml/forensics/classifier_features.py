from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image


FEATURE_KEYS = [
    "energy_to_variance_ratio",
    "block_energy_variance",
    "spectral_energy",
    "energia_diagonal_2da_derivada_norm",
    "block_variance_mean",
    "block_variance_var",
    "symmetry_score",
    "local_hf_variance",
    "gradient_direction",
    "entropy_block_var",
]
BLOCK_SIZE = 32


def _load_rgb_array(image_source) -> np.ndarray:
    if isinstance(image_source, (str, Path)):
        with Image.open(image_source) as image:
            return np.asarray(image.convert("RGB"), dtype=np.float64)

    with Image.open(io.BytesIO(image_source)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64)


def _grayscale(rgb: np.ndarray) -> np.ndarray:
    return np.rint(
        0.299 * rgb[:, :, 0] +
        0.587 * rgb[:, :, 1] +
        0.114 * rgb[:, :, 2]
    ).astype(np.uint8)


def _laplacian_cross(gray: np.ndarray) -> np.ndarray:
    gray64 = gray.astype(np.float64)
    response = np.zeros_like(gray64)
    response[1:-1, 1:-1] = (
        gray64[:-2, 1:-1] +
        gray64[2:, 1:-1] +
        gray64[1:-1, :-2] +
        gray64[1:-1, 2:] -
        4.0 * gray64[1:-1, 1:-1]
    )
    return response


def _laplacian_full(gray: np.ndarray) -> np.ndarray:
    gray64 = gray.astype(np.float64)
    response = np.zeros_like(gray64)
    response[1:-1, 1:-1] = (
        -gray64[:-2, :-2] -
        gray64[:-2, 1:-1] -
        gray64[:-2, 2:] -
        gray64[1:-1, :-2] +
        8.0 * gray64[1:-1, 1:-1] -
        gray64[1:-1, 2:] -
        gray64[2:, :-2] -
        gray64[2:, 1:-1] -
        gray64[2:, 2:]
    )
    return response


def _gradient_components(gray: np.ndarray):
    gray64 = gray.astype(np.float64)
    gx = gray64[1:-1, 2:] - gray64[1:-1, :-2]
    gy = gray64[2:, 1:-1] - gray64[:-2, 1:-1]
    return gx, gy


def _split_blocks(matrix: np.ndarray, block_size: int = BLOCK_SIZE) -> np.ndarray:
    height, width = matrix.shape
    trimmed_height = (height // block_size) * block_size
    trimmed_width = (width // block_size) * block_size

    if trimmed_height == 0 or trimmed_width == 0:
        return np.empty((0, block_size, block_size), dtype=np.float64)

    trimmed = matrix[:trimmed_height, :trimmed_width]
    blocks = trimmed.reshape(
        trimmed_height // block_size,
        block_size,
        trimmed_width // block_size,
        block_size,
    )
    blocks = blocks.swapaxes(1, 2)
    return blocks.reshape(-1, block_size, block_size).astype(np.float64)


def _block_entropy_variance(blocks: np.ndarray) -> float:
    if len(blocks) == 0:
        return 0.0

    entropies = []
    for block in blocks:
        flat = np.rint(block).astype(np.uint8).ravel()
        counts = np.bincount(flat, minlength=256)
        probabilities = counts[counts > 0] / flat.size
        entropy = -np.sum(probabilities * np.log2(probabilities))
        entropies.append(entropy)

    return float(np.var(np.asarray(entropies, dtype=np.float64)))


def extract_classifier_feature_dict(image_source) -> dict[str, float]:
    rgb = _load_rgb_array(image_source)
    gray = _grayscale(rgb)
    gray64 = gray.astype(np.float64)

    mean_value = float(np.mean(gray64))
    variance_value = float(np.var(gray64))

    histogram = np.bincount(gray.ravel(), minlength=256)
    probabilities = histogram[histogram > 0] / gray.size
    entropy_value = float(-np.sum(probabilities * np.log2(probabilities)))

    laplacian = _laplacian_cross(gray)
    residual_energy = float(np.mean(np.square(laplacian)))

    laplacian_blocks = _split_blocks(laplacian)
    if len(laplacian_blocks) > 0:
        laplacian_energies = np.mean(np.square(laplacian_blocks), axis=(1, 2))
        block_energy_variance = float(np.var(laplacian_energies))
    else:
        laplacian_energies = np.empty((0,), dtype=np.float64)
        block_energy_variance = 0.0

    gx, gy = _gradient_components(gray)
    gradient_magnitude = np.sqrt(np.square(gx) + np.square(gy))

    sub_image = gray64[:64, :64]
    spectrum = np.fft.fft2(sub_image)
    spectral_energy = float(np.mean(np.abs(spectrum)))

    derivative_response = _laplacian_full(gray)
    derivative_energy = float(np.mean(np.square(derivative_response)))
    derivative_ratio = derivative_energy / (variance_value + 1e-6)

    grayscale_blocks = _split_blocks(gray64)
    if len(grayscale_blocks) > 0:
        block_variances = np.var(grayscale_blocks, axis=(1, 2))
        grayscale_block_energies = np.mean(np.square(grayscale_blocks), axis=(1, 2))
        block_variance_mean = float(np.mean(block_variances))
        block_variance_var = float(np.var(block_variances))
        local_hf_variance = float(
            np.mean(
                np.abs(
                    grayscale_block_energies -
                    np.mean(grayscale_block_energies)
                )
            )
        )
        entropy_block_var = _block_entropy_variance(grayscale_blocks)
    else:
        block_variance_mean = 0.0
        block_variance_var = 0.0
        local_hf_variance = 0.0
        entropy_block_var = 0.0

    half_width = gray.shape[1] // 2
    if half_width > 0:
        left_half = gray64[:, :half_width]
        right_half = gray64[:, -half_width:][:, ::-1]
        symmetry_score = float(np.mean(np.abs(left_half - right_half)))
    else:
        symmetry_score = 0.0

    mask = gx != 0
    if np.any(mask):
        gradient_direction = float(np.mean(np.abs(gy[mask] / gx[mask])))
    else:
        gradient_direction = 0.0

    return {
        "energy_to_variance_ratio": residual_energy / (variance_value + 1e-6),
        "block_energy_variance": block_energy_variance,
        "spectral_energy": spectral_energy,
        "energia_diagonal_2da_derivada_norm": derivative_ratio,
        "block_variance_mean": block_variance_mean,
        "block_variance_var": block_variance_var,
        "symmetry_score": symmetry_score,
        "local_hf_variance": local_hf_variance,
        "gradient_direction": gradient_direction,
        "entropy_block_var": entropy_block_var,
    }


def extract_classifier_features(image_source) -> list[float]:
    feature_dict = extract_classifier_feature_dict(image_source)
    return [feature_dict[key] for key in FEATURE_KEYS]
