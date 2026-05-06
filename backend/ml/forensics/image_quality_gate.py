from __future__ import annotations

import math
from typing import Any

from backend.ml.forensics.basic_stats import (
    gradient_stats,
    laplacian_filter,
    residual_energy,
    rgb_to_grayscale,
    variance,
)


DEFAULT_CONFIG = {
    "sharpness_threshold": 0.42,
    "var_laplacian_ref": 1200.0,
    "edge_energy_ref": 450.0,
    "gradient_variance_ref": 900.0,
    "weights": {
        "var_laplacian": 0.45,
        "edge_energy": 0.35,
        "gradient_variance": 0.20,
    },
}


def _safe_ratio(value: float, reference: float) -> float:
    if reference <= 0:
        return 0.0
    # Cap to reduce outlier dominance and keep the score interpretable.
    return min(value / reference, 2.0)


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

    n_var_laplacian = _safe_ratio(var_laplacian, cfg["var_laplacian_ref"])
    n_edge_energy = _safe_ratio(edge_energy, cfg["edge_energy_ref"])
    n_grad_var = _safe_ratio(gradient_variance, cfg["gradient_variance_ref"])

    w = cfg["weights"]
    weighted_sum = (
        (w["var_laplacian"] * n_var_laplacian)
        + (w["edge_energy"] * n_edge_energy)
        + (w["gradient_variance"] * n_grad_var)
    )
    # weighted_sum in [0, 2], mapped to [0, 1]
    sharpness_score = max(0.0, min(weighted_sum / 2.0, 1.0))

    threshold = float(cfg["sharpness_threshold"])
    is_acceptable = sharpness_score >= threshold

    return {
        "var_laplacian": var_laplacian,
        "edge_energy": edge_energy,
        "gradient_variance": gradient_variance,
        "sharpness_score": sharpness_score,
        "threshold": threshold,
        "is_acceptable": is_acceptable,
    }

