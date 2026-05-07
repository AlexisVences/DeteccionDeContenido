from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ML_ROOT = PROJECT_ROOT / "backend" / "ml"

for candidate in (PROJECT_ROOT, BACKEND_ML_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from backend.ml.forensics.classifier_features import extract_classifier_feature_dict
from backend.ml.forensics.knn import KNN

DEFAULT_DATASET_DIR = (
    PROJECT_ROOT / "backend" / "test_images" / "imagenes" / "real_and_fake_general"
)
DEFAULT_FEATURE_CSV_PATH = (
    PROJECT_ROOT / "backend" / "models" / "real_and_fake_general_features.csv"
)
DEFAULT_FEATURE_META_PATH = (
    PROJECT_ROOT / "backend" / "models" / "real_and_fake_general_features.meta.json"
)
DEFAULT_MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "sherlock_ia_knn.joblib"
EXTRACTION_VERSION = 2
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
CSV_COLUMNS = ["image_path", "label_name", "label", *FEATURE_KEYS]


def select_features(feature_dict: dict[str, float]) -> list[float]:
    return [float(feature_dict[key]) for key in FEATURE_KEYS]


def dataset_label(name: str) -> int:
    normalized = name.lower()
    if "fake" in normalized or "ai" in normalized:
        return 1
    if "real" in normalized:
        return 0
    raise ValueError(f"No se pudo inferir la etiqueta para: {name}")


def iter_labeled_image_paths(dataset_dir: Path) -> Iterable[tuple[Path, int, str]]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"No se encontro el dataset en: {dataset_dir}")

    for image_path in sorted(dataset_dir.rglob("*.png")):
        label_name = None
        for parent in image_path.parents:
            if parent == dataset_dir.parent:
                break
            try:
                dataset_label(parent.name)
                label_name = parent.name
                break
            except ValueError:
                continue

        if label_name is None:
            continue

        yield image_path, dataset_label(label_name), label_name


def extract_feature_row(image_path: Path, dataset_dir: Path) -> dict[str, object]:
    label_name = next(
        parent.name
        for parent in image_path.parents
        if parent != dataset_dir.parent and _is_label_name(parent.name)
    )
    feature_dict = extract_classifier_feature_dict(image_path)

    row = {
        "image_path": image_path.relative_to(dataset_dir).as_posix(),
        "label_name": label_name,
        "label": dataset_label(label_name),
    }
    row.update(feature_dict)
    return row


def _is_label_name(name: str) -> bool:
    try:
        dataset_label(name)
        return True
    except ValueError:
        return False


def build_feature_csv(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    csv_path: Path = DEFAULT_FEATURE_CSV_PATH,
    progress_callback: Callable[[dict[str, float | int | str]], None] | None = None,
    resume: bool = True,
) -> dict[str, float | int | str]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = csv_path.with_suffix(".meta.json")

    start_time = time.perf_counter()
    image_entries = list(iter_labeled_image_paths(dataset_dir))
    total_images = len(image_entries)

    processed_paths = set()
    rows_written = 0
    open_mode = "w"
    metadata = load_feature_metadata(meta_path)

    can_resume = (
        resume and
        csv_path.exists() and
        csv_path.stat().st_size > 0 and
        metadata.get("extraction_version") == EXTRACTION_VERSION and
        metadata.get("feature_keys") == FEATURE_KEYS
    )

    if can_resume:
        with csv_path.open("r", newline="", encoding="utf-8") as existing_file:
            reader = csv.DictReader(existing_file)
            if reader.fieldnames == CSV_COLUMNS:
                for row in reader:
                    processed_paths.add(row["image_path"])
                    rows_written += 1
                open_mode = "a"

    if open_mode == "w":
        rows_written = 0

    with csv_path.open(open_mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if open_mode == "w":
            writer.writeheader()

        for image_path, _, _ in image_entries:
            relative_path = image_path.relative_to(dataset_dir).as_posix()
            if relative_path in processed_paths:
                continue

            row = extract_feature_row(image_path, dataset_dir)
            writer.writerow(row)
            handle.flush()
            rows_written += 1

            if progress_callback is not None:
                progress_callback(
                    _progress_snapshot(
                        processed=rows_written,
                        total=total_images,
                        start_time=start_time,
                        current_path=relative_path,
                        status="processed",
                    )
                )

    if rows_written == 0:
        raise ValueError("No se encontraron imagenes PNG etiquetadas en el dataset.")

    save_feature_metadata(
        meta_path,
        {
            "dataset_dir": str(dataset_dir),
            "csv_path": str(csv_path),
            "rows": rows_written,
            "total_images": total_images,
            "feature_keys": FEATURE_KEYS,
            "extraction_version": EXTRACTION_VERSION,
        },
    )

    return {
        "csv_path": str(csv_path),
        "rows": rows_written,
        "total_images": total_images,
        "elapsed_seconds": round(time.perf_counter() - start_time, 3),
    }


def _progress_snapshot(processed: int, total: int, start_time: float, current_path: str, status: str):
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    rate = processed / elapsed
    remaining = max(total - processed, 0)
    eta_seconds = remaining / rate if rate > 0 else None
    return {
        "processed": processed,
        "total": total,
        "elapsed_seconds": elapsed,
        "rate": rate,
        "eta_seconds": eta_seconds,
        "current_path": current_path,
        "status": status,
    }


def load_feature_dataset(csv_path: Path = DEFAULT_FEATURE_CSV_PATH):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo de features en: {csv_path}. "
            "Ejecuta primero scripts/feature_extraction.py."
        )

    image_paths = []
    labels = []
    features = []

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_paths.append(row["image_path"])
            labels.append(int(row["label"]))
            features.append([float(row[key]) for key in FEATURE_KEYS])

    if not features:
        raise ValueError("El CSV de features esta vacio.")

    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    return image_paths, X, y


def standardize_features(X: np.ndarray):
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    X_scaled = (X - means) / stds
    return X_scaled, means, stds


def train_model_from_csv(
    csv_path: Path = DEFAULT_FEATURE_CSV_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    k: int = 21,
) -> dict[str, object]:
    image_paths, X, y = load_feature_dataset(csv_path)
    X_scaled, means, stds = standardize_features(X)

    model = KNN(k=k)
    model.fit(X_scaled, y)

    resources = {
        "model": model,
        "means": means,
        "stds": stds,
        "samples": int(len(y)),
        "feature_keys": FEATURE_KEYS,
        "csv_path": str(csv_path),
        "image_paths": image_paths,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(resources, model_path)
    return resources


def load_model_resources(
    model_path: Path = DEFAULT_MODEL_PATH,
    csv_path: Path = DEFAULT_FEATURE_CSV_PATH,
    k: int = 21,
) -> dict[str, object]:
    if model_path.exists() and model_path.stat().st_mtime >= csv_path.stat().st_mtime:
        return joblib.load(model_path)
    return train_model_from_csv(csv_path=csv_path, model_path=model_path, k=k)


def load_feature_metadata(meta_path: Path) -> dict[str, object]:
    if not meta_path.exists():
        return {}

    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_feature_metadata(meta_path: Path, metadata: dict[str, object]) -> None:
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
