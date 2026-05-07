from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.forensics.training_pipeline import (
    DEFAULT_DATASET_DIR,
    DEFAULT_FEATURE_CSV_PATH,
    DEFAULT_MODEL_PATH,
    build_feature_csv,
    train_model_from_csv,
)

LAST_PROGRESS_UPDATE = 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extrae features del dataset real_and_fake_general y genera un CSV."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Ruta al dataset con imagenes reales y falsas.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_FEATURE_CSV_PATH,
        help="Ruta de salida para el CSV de features.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Ruta de salida para el modelo KNN serializado.",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Solo genera el CSV y omite el entrenamiento del modelo.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Ignora el CSV existente y vuelve a generarlo desde cero.",
    )
    return parser.parse_args()


def format_eta(seconds):
    if seconds is None:
        return "--:--:--"

    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def render_progress(progress):
    global LAST_PROGRESS_UPDATE

    total = max(int(progress["total"]), 1)
    processed = int(progress["processed"])
    now = time.perf_counter()
    if processed < total and processed > 10 and (now - LAST_PROGRESS_UPDATE) < 0.5:
        return

    LAST_PROGRESS_UPDATE = now
    percent = processed / total
    filled = int(percent * 30)
    bar = "#" * filled + "-" * (30 - filled)
    rate = float(progress["rate"])
    eta = format_eta(progress["eta_seconds"])
    current_path = str(progress["current_path"])

    print(
        f"\r[{bar}] {processed}/{total} "
        f"({percent * 100:6.2f}%) "
        f"| {rate:5.2f} img/s "
        f"| ETA {eta} "
        f"| {current_path[:80]:80}",
        end="",
        flush=True,
    )


def main():
    args = parse_args()
    start_time = time.perf_counter()

    summary = build_feature_csv(
        dataset_dir=args.dataset_dir,
        csv_path=args.csv_path,
        progress_callback=render_progress,
        resume=not args.force_rebuild,
    )
    print()
    print(
        f"CSV generado en {summary['csv_path']} con {summary['rows']} filas "
        f"de {summary['total_images']} imagenes en {summary['elapsed_seconds']:.3f}s."
    )

    if not args.skip_model:
        model_start = time.perf_counter()
        resources = train_model_from_csv(
            csv_path=args.csv_path,
            model_path=args.model_path,
            k=21,
        )
        print(
            f"Modelo guardado en {args.model_path} con {resources['samples']} muestras "
            f"en {time.perf_counter() - model_start:.3f}s."
        )

    print(f"Proceso completo en {time.perf_counter() - start_time:.3f}s.")


if __name__ == "__main__":
    main()
