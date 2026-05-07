from __future__ import annotations

<<<<<<< Updated upstream
import math
import os
import sys
import tempfile
import time
from collections import Counter
=======
import sys
import tempfile
>>>>>>> Stashed changes
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

# Run:
# streamlit run app.py

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_ML_ROOT = PROJECT_ROOT / "backend" / "ml"

for candidate in (PROJECT_ROOT, BACKEND_ML_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from backend.ml.description.image_descriptor import ImageDescriptor
<<<<<<< Updated upstream
from backend.ml.features.feature_extractor import FeatureExtractor
from backend.ml.forensics.image_quality_gate import evaluate_quality_from_reader
from backend.ml.forensics.knn import KNN
=======
from backend.ml.forensics.classifier_features import extract_classifier_feature_dict
>>>>>>> Stashed changes
from backend.ml.forensics.raw_image_reader import PNGReader
from backend.ml.forensics.training_pipeline import (
    DEFAULT_DATASET_DIR,
    DEFAULT_FEATURE_CSV_PATH,
    DEFAULT_MODEL_PATH,
    FEATURE_KEYS,
    load_model_resources,
    select_features,
)

DATASET_DIR = DEFAULT_DATASET_DIR
FEATURE_DATASET_PATH = DEFAULT_FEATURE_CSV_PATH
MODEL_ARTIFACT_PATH = DEFAULT_MODEL_PATH

FEATURE_LABELS = {
    "energy_to_variance_ratio": "Relacion energia/varianza",
    "block_energy_variance": "Variacion de energia por bloques",
    "spectral_energy": "Energia espectral",
    "energia_diagonal_2da_derivada_norm": "Derivada diagonal normalizada",
    "block_variance_mean": "Promedio de varianza por bloque",
    "block_variance_var": "Variacion de varianza por bloque",
    "symmetry_score": "Simetria global",
    "local_hf_variance": "Variacion local de alta frecuencia",
    "gradient_direction": "Consistencia de direccion de gradiente",
    "entropy_block_var": "Variacion de entropia por bloque",
}

DEFAULT_SHARPNESS_THRESHOLD = float(os.getenv("IA_SHARPNESS_THRESHOLD", "0.42"))

# Visual system
THEME = {
    "navy": "#0b1f3a",
    "blue": "#1f6feb",
    "cyan": "#2fc3ff",
    "bg": "#f4f8ff",
    "card": "#ffffff",
    "text": "#11233d",
    "muted": "#4f6786",
    "success": "#1f8f55",
    "danger": "#b42318",
    "warning": "#b06b00",
}


def initialize_state() -> None:
    defaults = {
        "uploaded_name": None,
        "uploaded_bytes": None,
        "classification_result": None,
        "description_result": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# SOLO se modificó la función inject_styles()

def inject_styles() -> None:
<<<<<<< Updated upstream
    """Global UI styles: polished dashboard look with subtle animations."""
    st.markdown(
        f"""
        <style>
            .stApp {
                background:
                    radial-gradient(circle at 5% 0%, rgba(47,195,255,.22), transparent 30%),
                    radial-gradient(circle at 95% 0%, rgba(31,111,235,.18), transparent 28%),
                    linear-gradient(180deg, #ffffff 0%, {THEME['bg']} 100%);
                color: {THEME['text']};
            }}
            .block-container {{
                max-width: 1180px;
                padding-top: 1.4rem;
                padding-bottom: 2.8rem;
            }}
            @keyframes fadeUp {{
                from {{ opacity: 0; transform: translateY(8px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            .hero, .card, .result, .footer {{
                animation: fadeUp .35s ease-out;
                background: {THEME['card']};
                border: 1px solid rgba(11,31,58,.10);
                border-radius: 18px;
                box-shadow: 0 12px 34px rgba(11,31,58,.07);
            }}
            .hero {{
                padding: 1.4rem 1.5rem;
                margin-bottom: 1rem;
            }}
            .hero-title {{
                color: {THEME['navy']};
                font-size: 2.15rem;
                font-weight: 800;
                letter-spacing: -0.02em;
                margin: 0 0 .2rem 0;
            }}
            .hero-sub {{
                margin: 0;
                color: {THEME['muted']};
                line-height: 1.62;
                font-size: 1.02rem;
            }}
            .pill-wrap {{ margin-top: .9rem; display: flex; gap: .45rem; flex-wrap: wrap; }}
            .pill {{
                background: rgba(31,111,235,.10);
                border: 1px solid rgba(31,111,235,.25);
                color: {THEME['navy']};
                font-size: .79rem;
                padding: .27rem .58rem;
                border-radius: 999px;
                font-weight: 700;
            }}
            .warn {{
                margin-top: .9rem;
                background: rgba(176,107,0,.10);
                border: 1px solid rgba(176,107,0,.25);
                color: #7a4d00;
                border-radius: 12px;
                padding: .6rem .72rem;
                font-size: .92rem;
            }}
            .card {{ padding: 1rem 1rem; margin-bottom: .9rem; }}
            .card-title {{
                color: {THEME['navy']};
                font-size: 1rem;
                font-weight: 750;
                margin-bottom: .55rem;
            }}
            .result {{ padding: 1rem 1.1rem; margin-bottom: .85rem; border-left: 7px solid {THEME['blue']}; }}
            .result.ai {{ border-left-color: {THEME['danger']}; }}
            .result.real {{ border-left-color: {THEME['success']}; }}
            .result.rejected {{ border-left-color: {THEME['warning']}; }}
            .kpi {{
                background: #f8fbff;
                border: 1px solid rgba(11,31,58,.08);
                border-radius: 14px;
                padding: .68rem .78rem;
            }}
            .footer {{ margin-top: 1.2rem; padding: .95rem 1rem; font-size: .92rem; color: {THEME['muted']}; }}
            .small-muted {{ color: {THEME['muted']}; font-size: .92rem; line-height: 1.6; }}
=======
    styles = """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(28, 105, 212, 0.12), transparent 30%),
                    linear-gradient(180deg, #ffffff 0%, {FORD_BG} 100%);
                color: {FORD_TEXT};
            }

            /* FIX WARNING TEXT VISIBILITY */
            div[data-testid="stAlert"] {
                color: #10243e !important;
            }
            div[data-testid="stAlert"] p {
                color: #10243e !important;
            }

            /* FIX SECONDARY BUTTON TEXT (Generar descripcion) */
            div.stButton > button {
                color: #10243e !important;
                background-color: #e9eef6 !important;
                border: 1px solid rgba(0, 52, 120, 0.2) !important;
            }

            div.stButton > button:hover {
                color: #ffffff !important;
                background-color: {FORD_BLUE} !important;
            }

            .topbar {
                background: #0d1116;
                border-bottom: 1px solid rgba(255, 255, 255, 0.08);
                color: #ffffff;
                padding: 1rem 1.5rem;
                position: sticky;
                top: 0;
                z-index: 100;
                box-shadow: 0 18px 40px rgba(0, 0, 0, 0.16);
            }
            .brand-title {
                font-size: 0.95rem;
                letter-spacing: 0.8em;
                text-transform: uppercase;
                font-weight: 700;
                margin: 0;
            }
            .stApp .block-container {
                max-width: 980px;
                padding-top: 4rem;
                padding-bottom: 3.5rem;
                padding-left: 1.5rem;
                padding-right: 1.5rem;
            }
            .hero-card, .section-card, .result-card {
                background: rgba(255, 255, 255, 0.94);
                border: 1px solid rgba(0, 52, 120, 0.10);
                border-radius: 24px;
                box-shadow: 0 20px 45px rgba(0, 52, 120, 0.08);
                padding: 1.6rem 1.6rem;
                margin-bottom: 1.5rem;
            }
            .hero-title {
                color: {FORD_BLUE};
                font-size: 2.4rem;
                font-weight: 800;
                margin-bottom: 0.25rem;
                letter-spacing: -0.03em;
            }
            .hero-subtitle {
                color: {FORD_TEXT};
                opacity: 0.88;
                margin: 0;
                font-size: 1rem;
                line-height: 1.7;
            }
            .hero-copy {
                color: #283145;
                opacity: 0.9;
                margin-top: 0.8rem;
                font-size: 0.98rem;
                line-height: 1.75;
            }
            .section-title {
                color: {FORD_BLUE};
                font-size: 1.05rem;
                font-weight: 700;
                margin-bottom: 0.85rem;
            }
            .result-card.ai {
                border-left: 8px solid {FORD_RED};
            }
            .result-card.real {
                border-left: 8px solid {FORD_GREEN};
            }
            .result-label {
                font-size: 0.95rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                opacity: 0.72;
                margin-bottom: 0.3rem;
            }
            .result-value {
                font-size: 1.9rem;
                font-weight: 800;
                margin-bottom: 0.45rem;
            }
            .result-help {
                font-size: 0.98rem;
                line-height: 1.55;
                margin: 0;
            }
            .description-box {
                background: linear-gradient(180deg, #ffffff 0%, {FORD_PALE_BLUE} 100%);
                border: 1px solid rgba(0, 52, 120, 0.12);
                border-radius: 18px;
                padding: 1rem 1.1rem;
                color: {FORD_TEXT};
                line-height: 1.7;
            }
>>>>>>> Stashed changes
        </style>
        """
    st.markdown(
        styles
        .replace("{FORD_BG}", FORD_BG)
        .replace("{FORD_TEXT}", FORD_TEXT)
        .replace("{FORD_BLUE}", FORD_BLUE)
        .replace("{FORD_RED}", FORD_RED)
        .replace("{FORD_GREEN}", FORD_GREEN)
        .replace("{FORD_PALE_BLUE}", FORD_PALE_BLUE),
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
<<<<<<< Updated upstream
        <section class="hero">
            <h1 class="hero-title">AI Image Authenticity Analyzer</h1>
            <p class="hero-sub">
                Plataforma de analisis forense para deteccion de contenido sintetico.
                Evalua patrones estadisticos, texturales y estructurales de imagenes PNG
                para identificar senales compatibles con generacion por IA, incluso en modelos avanzados.
=======
        <div class="topbar">
            <div class="brand-title">S H E R L O C K    &nbsp;   A I</div>
        </div>
        <div class="hero-card">
            <div class="hero-title">Sherlock IA</div>
            <p class="hero-subtitle">
                Sistema de análisis forense visual para clasificar imágenes PNG y generar descripciones contextuales de manera profesional sobre la misma imagen.
            </p>
            <p class="hero-copy">
                El modelo <strong>S H E R L O C K   I A</strong> combina extracción de características de imagen, normalización estadística y clasificación KNN optimizada para ofrecer respuestas rápidas y confiables.
>>>>>>> Stashed changes
            </p>
            <div class="pill-wrap">
                <span class="pill">Vision Forense</span>
                <span class="pill">Analisis Estadistico</span>
                <span class="pill">Deteccion de Contenido Sintetico</span>
                <span class="pill">Calidad de Imagen</span>
            </div>
            <div class="warn">
                Este sistema es de apoyo tecnico y no representa una verdad absoluta.
                El resultado debe interpretarse junto con contexto y evidencia adicional.
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


<<<<<<< Updated upstream
def select_features(feature_dict: dict[str, float]) -> list[float]:
    return [feature_dict[key] for key in FEATURE_KEYS]


def fit_standardizer(rows: list[list[float]]) -> tuple[list[float], list[float]]:
    column_count = len(rows[0])
    means = []
    stds = []
    for index in range(column_count):
        values = [row[index] for row in rows]
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        means.append(mean_value)
        stds.append(math.sqrt(variance))
    return means, stds


def transform_standardizer(rows: list[list[float]], means: list[float], stds: list[float]) -> list[list[float]]:
    transformed = []
    for row in rows:
        normalized = []
        for index, value in enumerate(row):
            if stds[index] == 0:
                normalized.append(0.0)
            else:
                normalized.append((value - means[index]) / stds[index])
        transformed.append(normalized)
    return transformed


def dataset_label(folder_name: str) -> int:
    normalized = folder_name.lower()
    if "fake" in normalized or "ai" in normalized:
        return 1
    if "real" in normalized:
        return 0
    raise ValueError(f"No se pudo inferir la etiqueta para la carpeta: {folder_name}")


def summarize_feature_statistics(rows: list[list[float]], labels: list[int]) -> dict[str, dict[str, float]]:
    real_rows = [row for row, label in zip(rows, labels) if label == 0]
    fake_rows = [row for row, label in zip(rows, labels) if label == 1]
    if not real_rows:
        raise ValueError("No hay muestras reales en el dataset de entrenamiento.")

    summary: dict[str, dict[str, float]] = {}
    for idx, key in enumerate(FEATURE_KEYS):
        real_values = [row[idx] for row in real_rows]
        fake_values = [row[idx] for row in fake_rows] if fake_rows else []

        real_mean = sum(real_values) / len(real_values)
        real_var = sum((v - real_mean) ** 2 for v in real_values) / len(real_values)
        real_std = math.sqrt(real_var)

        stats = {
            "real_mean": real_mean,
            "real_std": real_std,
            "real_min": min(real_values),
            "real_max": max(real_values),
        }
        if fake_values:
            fake_mean = sum(fake_values) / len(fake_values)
            fake_var = sum((v - fake_mean) ** 2 for v in fake_values) / len(fake_values)
            stats["fake_mean"] = fake_mean
            stats["fake_std"] = math.sqrt(fake_var)

        summary[key] = stats

    return summary


def build_real_reference_explanation(
    feature_dict: dict[str, float],
    stats: dict[str, dict[str, float]],
    predicted_label: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in FEATURE_KEYS:
        if key not in feature_dict or key not in stats:
            continue

        value = feature_dict[key]
        real_mean = stats[key]["real_mean"]
        real_std = stats[key]["real_std"]
        real_min = stats[key]["real_min"]
        real_max = stats[key]["real_max"]

        z_real = 0.0 if real_std == 0 else (value - real_mean) / real_std
        out_of_range = value < real_min or value > real_max

        score = abs(z_real) + (1.2 if out_of_range else 0.0)
        if predicted_label == 0:
            score = -score

        rows.append(
            {
                "feature": FEATURE_LABELS.get(key, key),
                "feature_key": key,
                "value": value,
                "real_mean": real_mean,
                "real_min": real_min,
                "real_max": real_max,
                "z_real": z_real,
                "status": "Fuera de rango REAL" if out_of_range else "Dentro de rango REAL",
                "impact_score": score,
            }
        )

    rows.sort(key=lambda item: abs(item["impact_score"]), reverse=True)
    return rows


=======
>>>>>>> Stashed changes
def create_temp_png(image_bytes: bytes) -> Path:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_file.write(image_bytes)
    temp_file.flush()
    temp_file.close()
    return Path(temp_file.name)


def load_image(image_bytes: bytes) -> PNGReader:
    temp_path = create_temp_png(image_bytes)
    try:
        reader = PNGReader(str(temp_path))
        reader.read()
        return reader
    finally:
        temp_path.unlink(missing_ok=True)


def get_model_resources() -> dict[str, Any]:
    resources = st.session_state.get("model_resources")
    if resources is None:
        resources = load_trained_model()
        st.session_state["model_resources"] = resources
    return resources


@st.cache_resource(show_spinner=False)
def load_trained_model() -> dict[str, Any]:
    if not DATASET_DIR.exists():
<<<<<<< Updated upstream
        raise FileNotFoundError(f"No se encontro el dataset de entrenamiento en: {DATASET_DIR}")

    rows: list[list[float]] = []
    labels: list[int] = []

    for folder in sorted(DATASET_DIR.iterdir()):
        if not folder.is_dir():
            continue
        label = dataset_label(folder.name)

        for image_path in sorted(folder.glob("*.png")):
            reader = PNGReader(str(image_path))
            reader.read()
            feature_dict = FeatureExtractor.extract_from_png_reader(reader)
            rows.append(select_features(feature_dict))
            labels.append(label)

    if not rows:
        raise ValueError("El dataset de entrenamiento no contiene imagenes PNG validas.")

    means, stds = fit_standardizer(rows)
    normalized_rows = transform_standardizer(rows, means, stds)

    model = KNN(k=5)
    model.fit(normalized_rows, labels)
    feature_stats = summarize_feature_statistics(rows, labels)

    return {
        "model": model,
        "means": means,
        "stds": stds,
        "samples": len(rows),
        "feature_stats": feature_stats,
    }


def knn_confidence(model: KNN, sample: list[float]) -> float | None:
    if not model.X_train or not model.y_train:
        return None

    distances = []
    for index, trained_row in enumerate(model.X_train):
        distance = model._euclidean(trained_row, sample)
        distances.append((distance, model.y_train[index]))

    distances.sort(key=lambda item: item[0])
    neighbors = distances[: model.k]
    if not neighbors:
        return None

    predicted_label = model.predict([sample])[0]
    votes = Counter(label for _, label in neighbors)
    return votes[predicted_label] / len(neighbors)
=======
        raise FileNotFoundError(
            f"No se encontro el dataset de entrenamiento en: {DATASET_DIR}"
        )
    if not FEATURE_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro el CSV de features en: {FEATURE_DATASET_PATH}. "
            "Ejecuta scripts/feature_extraction.py antes de iniciar la app."
        )
    return load_model_resources(
        model_path=MODEL_ARTIFACT_PATH,
        csv_path=FEATURE_DATASET_PATH,
        k=21,
    )
>>>>>>> Stashed changes


def run_inference(image_bytes: bytes) -> dict[str, Any]:
    reader = load_image(image_bytes)
    resources = get_model_resources()
<<<<<<< Updated upstream

    quality = evaluate_quality_from_reader(
        reader,
        config={"sharpness_threshold": DEFAULT_SHARPNESS_THRESHOLD},
    )

    if not quality["is_acceptable"]:
        return {
            "rejected_by_quality": True,
            "quality_metrics": quality,
            "message": (
                "Imagen demasiado desenfocada para ejecutar el analisis forense. "
                "No contiene suficiente informacion visual para una decision confiable."
            ),
            "dataset_samples": resources["samples"],
        }

    feature_dict = FeatureExtractor.extract_from_png_reader(reader)
    sample = select_features(feature_dict)
    normalized_sample = transform_standardizer([sample], resources["means"], resources["stds"])[0]

    prediction = resources["model"].predict([normalized_sample])[0]
    confidence = knn_confidence(resources["model"], normalized_sample)
    explanation_rows = build_real_reference_explanation(feature_dict, resources["feature_stats"], prediction)
=======

    feature_dict = extract_classifier_feature_dict(image_bytes)
    sample = np.asarray(select_features(feature_dict), dtype=np.float64)
    normalized_sample = (sample - resources["means"]) / resources["stds"]

    prediction = resources["model"].predict([normalized_sample])[0]
    confidence = resources["model"].confidence_one(normalized_sample)
>>>>>>> Stashed changes

    return {
        "rejected_by_quality": False,
        "label": prediction,
        "label_text": "Generada por IA" if prediction == 1 else "Imagen real",
        "confidence": confidence,
        "feature_dict": feature_dict,
        "dataset_samples": resources["samples"],
        "quality_metrics": quality,
        "feature_explanation": explanation_rows,
        "explanation": (
<<<<<<< Updated upstream
            "Patrones compatibles con generacion sintetica detectados"
            if prediction == 1
            else "Las caracteristicas se alinean con distribuciones naturales"
=======
            "La descripción y explicación de la clasificación como ia generated vendra mas adelante"
            if prediction == 1
            else "La descripción y explicación de la clasificación como real vendra mas adelante"
>>>>>>> Stashed changes
        ),
    }


def generate_description(image_bytes: bytes) -> dict[str, Any]:
    reader = load_image(image_bytes)
    descriptor = ImageDescriptor()
    return descriptor.describe_from_reader(reader)


def reset_analysis_state() -> None:
    st.session_state.classification_result = None
    st.session_state.description_result = None


<<<<<<< Updated upstream
def render_upload_analysis_section() -> None:
    left, right = st.columns([1.1, 1], gap="large")
=======
def render_upload_section() -> None:
    st.markdown(
        '<div class="section-card"><div class="section-title">Carge la imagen en la sección a continuación para empezar con el análisis :)</div>',
        unsafe_allow_html=True,
    )
>>>>>>> Stashed changes

    with left:
        st.markdown('<section class="card"><div class="card-title">Carga de Imagen</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Arrastra y suelta una imagen PNG o selecciona un archivo",
            type=["png"],
            help="Solo PNG para mantener compatibilidad con el pipeline forense actual.",
        )
        st.markdown('</section>', unsafe_allow_html=True)

    if uploaded_file is None:
        with right:
            st.markdown(
                '<section class="card"><div class="card-title">Preview</div><p class="small-muted">Carga una imagen para visualizarla y ejecutar el analisis.</p></section>',
                unsafe_allow_html=True,
            )
        return

    file_bytes = uploaded_file.getvalue()
    if st.session_state.uploaded_name != uploaded_file.name or st.session_state.uploaded_bytes != file_bytes:
        st.session_state.uploaded_name = uploaded_file.name
        st.session_state.uploaded_bytes = file_bytes
        reset_analysis_state()

    with right:
        st.markdown('<section class="card"><div class="card-title">Preview</div>', unsafe_allow_html=True)
        st.image(file_bytes, caption=uploaded_file.name, use_container_width=True)
        st.markdown('</section>', unsafe_allow_html=True)

    with left:
        st.markdown('<section class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Pipeline de Analisis</div>', unsafe_allow_html=True)
        st.markdown(
<<<<<<< Updated upstream
            '<p class="small-muted">El sistema evaluara calidad, extraera caracteristicas forenses y emitira una clasificacion con explicacion.</p>',
            unsafe_allow_html=True,
        )
        if st.button("Iniciar Analisis Forense", type="primary", use_container_width=True):
            status = st.empty()
            progress = st.progress(0)
            steps = [
                "Validando calidad de imagen...",
                "Extrayendo caracteristicas forenses...",
                "Calculando metricas y comparando contra dataset...",
                "Generando resultado final...",
            ]
            try:
                for i, msg in enumerate(steps, start=1):
                    status.info(msg)
                    progress.progress(int((i - 1) / len(steps) * 100))
                    time.sleep(0.18)

                st.session_state.classification_result = run_inference(file_bytes)
                st.session_state.description_result = None
                progress.progress(100)
                status.success("Analisis completado.")
            except Exception as error:
                st.session_state.classification_result = None
                progress.empty()
                status.error(f"No fue posible analizar la imagen: {error}")
        st.markdown('</section>', unsafe_allow_html=True)


def render_quality_panel(result: dict[str, Any]) -> None:
    quality = result.get("quality_metrics")
    if not quality:
        return

    with st.expander("Calidad de imagen y nitidez"):
        score = float(quality["sharpness_score"])
        threshold = float(quality["threshold"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Sharpness Score", f"{score:.3f}")
        c2.metric("Threshold", f"{threshold:.3f}")
        c3.metric("Estado", "Apta" if quality["is_acceptable"] else "Insuficiente")
        st.progress(min(100, max(0, int(score * 100))))

        st.write(
            {
                "var_laplacian": round(float(quality["var_laplacian"]), 5),
                "edge_energy": round(float(quality["edge_energy"]), 5),
                "gradient_variance": round(float(quality["gradient_variance"]), 5),
            }
        )
        st.caption("Threshold configurable por entorno con IA_SHARPNESS_THRESHOLD")


def render_explanation_panel(result: dict[str, Any]) -> None:
    rows = result.get("feature_explanation", [])
    if not rows:
        return

    with st.expander("Mas informacion: por que se obtuvo este resultado"):
        st.markdown(
            '<p class="small-muted">Comparacion contra referencia REAL. Se muestran las caracteristicas con mayor desviacion.</p>',
            unsafe_allow_html=True,
        )

        top_rows = rows[:5]
        table = []
        for row in top_rows:
            table.append(
                {
                    "Caracteristica": row["feature"],
                    "Valor imagen": row["value"],
                    "Promedio REAL": row["real_mean"],
                    "Rango REAL": f'{row["real_min"]:.4f} - {row["real_max"]:.4f}',
                    "Estado": row["status"],
                    "Z vs REAL": row["z_real"],
                }
            )

        st.dataframe(table, use_container_width=True)

        bar_data = {item["Caracteristica"]: abs(float(item["Z vs REAL"])) for item in table}
        st.bar_chart(bar_data)

        reasons = []
        for row in top_rows:
            if row["status"] == "Fuera de rango REAL":
                reasons.append(
                    f"- {row['feature']}: fuera del rango esperado REAL ({row['real_min']:.4f} a {row['real_max']:.4f})."
                )
            elif abs(row["z_real"]) >= 1.5:
                reasons.append(
                    f"- {row['feature']}: dentro de rango pero alejada del promedio REAL (z={row['z_real']:+.2f})."
                )

        if reasons:
            st.markdown("**Lectura rapida**")
            st.markdown("\n".join(reasons))
        else:
            st.success("Las principales caracteristicas caen en rangos compatibles con imagenes reales.")


def render_forensic_metrics(result: dict[str, Any]) -> None:
    feature_dict = result.get("feature_dict")
    if not feature_dict:
        return

    st.markdown('<section class="card"><div class="card-title">Metricas Forenses Relevantes</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    featured = [
        ("Energia espectral", "spectral_energy"),
        ("Simetria", "symmetry_score"),
        ("Textura HF", "local_hf_variance"),
        ("Entropia bloque", "entropy_block_var"),
    ]
    for idx, (label, key) in enumerate(featured):
        cols[idx].metric(label, f"{float(feature_dict.get(key, 0.0)):.4f}")

    st.caption("Estas metricas forman parte de las senales usadas por el clasificador para estimar autenticidad.")
    st.markdown('</section>', unsafe_allow_html=True)
=======
            """
            <div class="section-card">
                <div class="section-title">Lista para analizar</div>
                <p style="margin:0; line-height:1.7;">
                    Presiona "Analizar imagen" para ejecutar el analisis forense.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Analizar imagen", type="primary", use_container_width=True):
            with st.spinner("Analizando imagen y ejecutando el clasificador..."):
                try:
                    st.session_state.classification_result = run_inference(file_bytes)
                    st.session_state.description_result = None
                except Exception as error:
                    st.session_state.classification_result = None
                    st.error(f"No fue posible analizar la imagen: {error}")
>>>>>>> Stashed changes


def render_result_section() -> None:
    result = st.session_state.classification_result
    if not result:
        return

    if result.get("rejected_by_quality"):
        st.markdown(
            """
            <section class="result rejected">
                <div style="font-size:.82rem; text-transform:uppercase; opacity:.7;">Estado del analisis</div>
                <div style="font-size:1.75rem; font-weight:800; color:#b06b00; margin:.2rem 0;">Analisis Rechazado por Calidad</div>
                <div style="line-height:1.6;">No se ejecuto la clasificacion IA vs Real porque la imagen no tiene nitidez suficiente.</div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        st.warning(result["message"])
        render_quality_panel(result)
        return

    is_ai = result["label"] == 1
    status_class = "ai" if is_ai else "real"
    status_color = THEME["danger"] if is_ai else THEME["success"]

    st.markdown(
        f"""
<<<<<<< Updated upstream
        <section class="result {status_class}">
            <div style="font-size:.82rem; text-transform:uppercase; opacity:.7;">Resultado Principal</div>
            <div style="font-size:1.9rem; font-weight:800; color:{status_color}; margin:.2rem 0;">{result['label_text']}</div>
            <div style="line-height:1.6;">{result['explanation']}</div>
        </section>
=======
        <div class="result-card {status_class}">
            <div class="section-title">Resultado</div>
            <div class="result-label">Clasificacion</div>
            <div class="result-value" style="color:{status_color};">
                {result["label_text"]}
            </div>
            <p class="result-help">{result["explanation"]}</p>
        </div>
>>>>>>> Stashed changes
        """,
        unsafe_allow_html=True,
    )

<<<<<<< Updated upstream
    k1, k2, k3 = st.columns(3)
    conf = result.get("confidence")
    if conf is None:
        k1.metric("Confianza", "N/A")
    else:
        k1.metric("Confianza", f"{conf * 100:.1f}%")
    k2.metric("Muestras de entrenamiento", str(result["dataset_samples"]))
    q = result.get("quality_metrics", {})
    k3.metric("Sharpness", f"{float(q.get('sharpness_score', 0.0)):.3f}")

    render_forensic_metrics(result)
    render_quality_panel(result)
    render_explanation_panel(result)

    if st.button("Generar descripcion contextual", use_container_width=True):
        with st.spinner("Generando descripcion..."):
=======
    # if is_ai:
    #     st.error("Se detectaron patrones compatibles con generacion sintetica.")
    # else:
    #     st.success("La imagen se alinea mejor con distribuciones naturales.")

    metric_col, secondary_col = st.columns(2, gap="large")
    with metric_col:
        if confidence is not None:
            st.metric("Confianza", f"{confidence * 100:.1f}%")
            st.progress(int(confidence * 100))
        else:
            st.warning("No fue posible estimar una confianza con la implementacion actual.")

    with secondary_col:
        st.metric("Muestras de entrenamiento", str(result["dataset_samples"]))
        st.caption("Análisis impulsado por un modelo de tipo KNN.")

    if st.button("Generar descripcion", use_container_width=True):
        with st.spinner("Generando descripcion contextual..."):
>>>>>>> Stashed changes
            try:
                st.session_state.description_result = generate_description(st.session_state.uploaded_bytes)
            except Exception as error:
                st.session_state.description_result = None
                st.warning(f"No fue posible generar la descripcion: {error}")


def render_description_section() -> None:
    description_result = st.session_state.description_result
    if not description_result:
        return

    description_text = description_result.get("description", "").strip()
<<<<<<< Updated upstream
    st.markdown('<section class="card"><div class="card-title">Descripcion Contextual</div>', unsafe_allow_html=True)
=======
    st.markdown(
        '<div class="section-card"><div class="section-title">Descripcion</div>',
        unsafe_allow_html=True,
    )

>>>>>>> Stashed changes
    if description_text:
        st.write(description_text)
    else:
        st.warning("El modulo de descripcion no devolvio texto para esta imagen.")
    st.markdown('</section>', unsafe_allow_html=True)


def render_info_section() -> None:
    st.markdown('<section class="card"><div class="card-title">Como Funciona el Sistema</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            """
            - **Analisis de bordes y enfoque**: mide nitidez y detalle util para forensia.
            - **Patrones de textura y ruido**: detecta regularidades no naturales.
            - **Senales estadisticas**: compara distribuciones contra referencias reales.
            - **Decision supervisada**: clasificador KNN sobre features forenses.
            """
        )
    with c2:
        st.markdown(
            """
            **Limitaciones**
            - Imagenes demasiado borrosas se rechazan por calidad.
            - El resultado es una estimacion tecnica, no prueba legal definitiva.
            - La precision depende de la cobertura y calidad del dataset de referencia.
            """
        )
    st.markdown('</section>', unsafe_allow_html=True)


def render_footer() -> None:
    st.markdown(
        """
        <section class="footer">
            <strong>AI Image Authenticity Analyzer</strong> · Streamlit · Python · Forensia Digital · Machine Learning (KNN)<br/>
            Enfoque academico y de investigacion aplicada en deteccion de contenido sintetico.
        </section>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
<<<<<<< Updated upstream
        page_title="AI Image Authenticity Analyzer",
        page_icon="AI",
        layout="wide",
=======
        page_title="Sherlock IA",
        page_icon="🔎",
        layout="centered",
>>>>>>> Stashed changes
    )

    initialize_state()
    inject_styles()
    render_hero()

    try:
        with st.spinner("Cargando el modelo Sherlock-IA..."):
            get_model_resources()
<<<<<<< Updated upstream
=======
    except Exception as error:
        st.error(f"Ocurrió un problema al cargar el modelo: {error}")
        return

    try:
        render_upload_section()
>>>>>>> Stashed changes
    except Exception as error:
        st.error(f"Ocurrió un problema al cargar el modelo: {error}")
        return

    try:
        with st.spinner("Cargando el modelo Sherlock-IA..."):
            get_model_resources()
    except Exception as error:
        st.error(f"Ocurrió un problema al cargar el modelo: {error}")
        return

    try:
        render_upload_analysis_section()
    except Exception as error:
        st.error(f"Ocurrio un problema en la interfaz: {error}")
        return

    if st.session_state.uploaded_bytes is None:
<<<<<<< Updated upstream
        st.info("Carga una imagen PNG para habilitar el analisis forense.")
=======
        st.warning("Nota: este modelo es únicamente de consulta y no tiene ninguna validez oficial")
>>>>>>> Stashed changes

    render_result_section()
    render_description_section()
    render_info_section()
    render_footer()


if __name__ == "__main__":
    main()
