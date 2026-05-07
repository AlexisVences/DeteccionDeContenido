from __future__ import annotations

from typing import Any

import numpy as np

from backend.ml.forensics.basic_stats import (
    gradient_stats,
    laplacian_filter,
    residual_energy,
    rgb_to_grayscale,
    variance,
)


DEFAULT_CONFIG = {
    "sharpness_threshold": 0.36,
    "var_laplacian_ref": 1200.0,
    "edge_energy_ref": 450.0,
    "gradient_variance_ref": 900.0,
    "block_size": 32,
    "local_p90_ref": 620.0,
    "local_p95_ref": 860.0,
    "local_focus_ratio_ref": 0.22,
    "local_gate_threshold": 0.45,
    "weights": {
        "var_laplacian": 0.35,
        "edge_energy": 0.25,
        "gradient_variance": 0.15,
        "local_p90": 0.10,
        "local_p95": 0.10,
        "local_focus_ratio": 0.05,
    },
}


def _safe_ratio(value: float, reference: float) -> float:
    if reference <= 0:
        return 0.0
    return min(value / reference, 2.0)


def _laplacian_block_energies(
    laplacian: list[list[float]],
    block_size: int,
) -> np.ndarray:
    arr = np.asarray(laplacian, dtype=np.float64)
    height, width = arr.shape
    trimmed_height = (height // block_size) * block_size
    trimmed_width = (width // block_size) * block_size

    if trimmed_height == 0 or trimmed_width == 0:
        return np.empty((0,), dtype=np.float64)

    trimmed = arr[:trimmed_height, :trimmed_width]
    blocks = trimmed.reshape(
        trimmed_height // block_size,
        block_size,
        trimmed_width // block_size,
        block_size,
    ).swapaxes(1, 2)
    return np.mean(np.square(blocks), axis=(2, 3)).ravel()


def _laplacian_block_energy_grid(
    laplacian: list[list[float]],
    block_size: int,
) -> np.ndarray:
    arr = np.asarray(laplacian, dtype=np.float64)
    height, width = arr.shape
    trimmed_height = (height // block_size) * block_size
    trimmed_width = (width // block_size) * block_size

    if trimmed_height == 0 or trimmed_width == 0:
        return np.empty((0, 0), dtype=np.float64)

    trimmed = arr[:trimmed_height, :trimmed_width]
    blocks = trimmed.reshape(
        trimmed_height // block_size,
        block_size,
        trimmed_width // block_size,
        block_size,
    ).swapaxes(1, 2)
    return np.mean(np.square(blocks), axis=(2, 3))


def suggest_focus_crop_box(
    reader: Any,
    crop_width_blocks: int = 10,
    crop_height_blocks: int = 10,
    block_size: int = 32,
) -> tuple[int, int, int, int] | None:
    grayscale = rgb_to_grayscale(reader.pixels, reader.width)
    laplacian = laplacian_filter(grayscale)
    energy_grid = _laplacian_block_energy_grid(laplacian, block_size)

    if energy_grid.size == 0:
        return None

    rows, cols = energy_grid.shape
    if rows < crop_height_blocks or cols < crop_width_blocks:
        return None

    best_score = None
    best_row = 0
    best_col = 0
    for row in range(0, rows - crop_height_blocks + 1):
        for col in range(0, cols - crop_width_blocks + 1):
            score = float(
                np.sum(
                    energy_grid[
                        row:row + crop_height_blocks,
                        col:col + crop_width_blocks,
                    ]
                )
            )
            if best_score is None or score > best_score:
                best_score = score
                best_row = row
                best_col = col

    left = best_col * block_size
    top = best_row * block_size
    right = left + crop_width_blocks * block_size
    bottom = top + crop_height_blocks * block_size
    return (left, top, right, bottom)


def evaluate_quality_from_reader(
    reader: Any,
    config: dict[str, Any] | None = None,
) -> dict[str, float | bool | str]:
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
        if "weights" in config and isinstance(config["weights"], dict):
            merged_weights = dict(DEFAULT_CONFIG["weights"])
            merged_weights.update(config["weights"])
            cfg["weights"] = merged_weights

    grayscale = rgb_to_grayscale(reader.pixels, reader.width)
    laplacian = laplacian_filter(grayscale)

    var_laplacian = variance(laplacian)
    edge_energy = residual_energy(laplacian)
    _, gradient_variance = gradient_stats(grayscale)
    block_energies = _laplacian_block_energies(laplacian, int(cfg["block_size"]))

    n_var_laplacian = _safe_ratio(var_laplacian, cfg["var_laplacian_ref"])
    n_edge_energy = _safe_ratio(edge_energy, cfg["edge_energy_ref"])
    n_grad_var = _safe_ratio(gradient_variance, cfg["gradient_variance_ref"])

    if len(block_energies) > 0:
        local_p90 = float(np.percentile(block_energies, 90))
        local_p95 = float(np.percentile(block_energies, 95))
        local_focus_ratio = float(
            np.mean(block_energies >= float(cfg["local_p90_ref"]))
        )
    else:
        local_p90 = 0.0
        local_p95 = 0.0
        local_focus_ratio = 0.0

    n_local_p90 = _safe_ratio(local_p90, cfg["local_p90_ref"])
    n_local_p95 = _safe_ratio(local_p95, cfg["local_p95_ref"])
    n_local_focus_ratio = _safe_ratio(
        local_focus_ratio,
        cfg["local_focus_ratio_ref"],
    )

    w = cfg["weights"]
    weighted_sum = (
        (w["var_laplacian"] * n_var_laplacian)
        + (w["edge_energy"] * n_edge_energy)
        + (w["gradient_variance"] * n_grad_var)
        + (w["local_p90"] * n_local_p90)
        + (w["local_p95"] * n_local_p95)
        + (w["local_focus_ratio"] * n_local_focus_ratio)
    )
    sharpness_score = max(0.0, min(weighted_sum / 2.0, 1.0))

    local_weight_total = (
        w["local_p90"] + w["local_p95"] + w["local_focus_ratio"]
    )
    local_focus_score = (
        (
            (w["local_p90"] * n_local_p90)
            + (w["local_p95"] * n_local_p95)
            + (w["local_focus_ratio"] * n_local_focus_ratio)
        ) / max(local_weight_total, 1e-9)
    )
    local_focus_score = max(0.0, min(local_focus_score / 2.0, 1.0))

    threshold = float(cfg["sharpness_threshold"])
    local_gate_threshold = float(cfg["local_gate_threshold"])
    has_usable_focus_region = (
        local_focus_score >= local_gate_threshold and
        local_focus_ratio >= 0.08
    )
    is_acceptable = sharpness_score >= threshold or has_usable_focus_region

    return {
        "var_laplacian": var_laplacian,
        "edge_energy": edge_energy,
        "gradient_variance": gradient_variance,
        "local_p90": local_p90,
        "local_p95": local_p95,
        "local_focus_ratio": local_focus_ratio,
        "local_focus_score": local_focus_score,
        "has_usable_focus_region": has_usable_focus_region,
        "sharpness_score": sharpness_score,
        "threshold": threshold,
        "is_acceptable": is_acceptable,
    }
