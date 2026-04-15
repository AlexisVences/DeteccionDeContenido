from __future__ import annotations

import math
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import streamlit as st

#Correr codigo con: 
#streamlit run app.py

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_ML_ROOT = PROJECT_ROOT / "backend" / "ml"

for candidate in (PROJECT_ROOT, BACKEND_ML_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from backend.ml.description.image_descriptor import ImageDescriptor
from backend.ml.features.feature_extractor import FeatureExtractor
from backend.ml.forensics.knn import KNN
from backend.ml.forensics.raw_image_reader import PNGReader


DATASET_DIR = PROJECT_ROOT / "backend" / "test_images" / "imagenes" / "real_and_fake_face"
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

FORD_BLUE = "#003478"
FORD_LIGHT_BLUE = "#1c69d4"
FORD_PALE_BLUE = "#dce8f8"
FORD_GREEN = "#1f8f55"
FORD_RED = "#b42318"
FORD_BG = "#f5f8fc"
FORD_TEXT = "#10243e"


def initialize_state() -> None:
    defaults = {
        "uploaded_name": None,
        "uploaded_bytes": None,
        "classification_result": None,
        "description_result": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
                background:
                    radial-gradient(circle at top left, rgba(28, 105, 212, 0.12), transparent 30%),
                    linear-gradient(180deg, #ffffff 0%, {FORD_BG} 100%);
                color: {FORD_TEXT};
            }}
            .block-container {{
                max-width: 980px;
                padding-top: 2.5rem;
                padding-bottom: 3rem;
            }}
            .hero-card, .section-card, .result-card {{
                background: rgba(255, 255, 255, 0.94);
                border: 1px solid rgba(0, 52, 120, 0.10);
                border-radius: 22px;
                box-shadow: 0 16px 40px rgba(0, 52, 120, 0.08);
                padding: 1.4rem 1.5rem;
                margin-bottom: 1rem;
            }}
            .hero-title {{
                color: {FORD_BLUE};
                font-size: 2.2rem;
                font-weight: 800;
                margin-bottom: 0.25rem;
                letter-spacing: -0.02em;
            }}
            .hero-subtitle {{
                color: {FORD_TEXT};
                opacity: 0.85;
                margin: 0;
                font-size: 1rem;
                line-height: 1.6;
            }}
            .section-title {{
                color: {FORD_BLUE};
                font-size: 1.05rem;
                font-weight: 700;
                margin-bottom: 0.85rem;
            }}
            .result-card.ai {{
                border-left: 8px solid {FORD_RED};
            }}
            .result-card.real {{
                border-left: 8px solid {FORD_GREEN};
            }}
            .result-label {{
                font-size: 0.95rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                opacity: 0.72;
                margin-bottom: 0.3rem;
            }}
            .result-value {{
                font-size: 1.9rem;
                font-weight: 800;
                margin-bottom: 0.45rem;
            }}
            .result-help {{
                font-size: 0.98rem;
                line-height: 1.55;
                margin: 0;
            }}
            .description-box {{
                background: linear-gradient(180deg, #ffffff 0%, {FORD_PALE_BLUE} 100%);
                border: 1px solid rgba(0, 52, 120, 0.12);
                border-radius: 18px;
                padding: 1rem 1.1rem;
                color: {FORD_TEXT};
                line-height: 1.7;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
        <div class = "hero-title"> ai </div>
            <div class="hero-title">AI Image Detection System</div>
            <p class="hero-subtitle">
                Herramienta de análisis forense para detectar si una imagen PNG parece
                generada por inteligencia artificial y obtener una descripción contextual.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def transform_standardizer(
    rows: list[list[float]],
    means: list[float],
    stds: list[float],
) -> list[list[float]]:
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


@st.cache_resource(show_spinner=False)
def load_trained_model() -> dict[str, Any]:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"No se encontro el dataset de entrenamiento en: {DATASET_DIR}"
        )

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

    return {
        "model": model,
        "means": means,
        "stds": stds,
        "samples": len(rows),
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


def run_inference(image_bytes: bytes) -> dict[str, Any]:
    reader = load_image(image_bytes)
    resources = load_trained_model()

    feature_dict = FeatureExtractor.extract_from_png_reader(reader)
    sample = select_features(feature_dict)
    normalized_sample = transform_standardizer(
        [sample],
        resources["means"],
        resources["stds"],
    )[0]

    prediction = resources["model"].predict([normalized_sample])[0]
    confidence = knn_confidence(resources["model"], normalized_sample)

    return {
        "label": prediction,
        "label_text": "AI Generated" if prediction == 1 else "Real",
        "confidence": confidence,
        "reader": reader,
        "feature_dict": feature_dict,
        "dataset_samples": resources["samples"],
        "explanation": (
            "Patterns consistent with synthetic generation detected"
            if prediction == 1
            else "Statistical features align with natural image distributions"
        ),
    }


def generate_description(image_bytes: bytes) -> dict[str, Any]:
    reader = load_image(image_bytes)
    descriptor = ImageDescriptor()
    return descriptor.describe_from_reader(reader)


def reset_analysis_state() -> None:
    st.session_state.classification_result = None
    st.session_state.description_result = None


def render_upload_section() -> None:
    st.markdown(
        '<div class="section-card"><div class="section-title">Upload</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Selecciona una imagen PNG para analizar",
        type=["png"],
        help="Solo se aceptan archivos PNG para asegurar compatibilidad con el backend.",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is None:
        return

    file_bytes = uploaded_file.getvalue()
    if (
        st.session_state.uploaded_name != uploaded_file.name
        or st.session_state.uploaded_bytes != file_bytes
    ):
        st.session_state.uploaded_name = uploaded_file.name
        st.session_state.uploaded_bytes = file_bytes
        reset_analysis_state()

    preview_col, info_col = st.columns([1.25, 0.95], gap="large")

    with preview_col:
        st.image(file_bytes, caption=uploaded_file.name, use_container_width=True)

    with info_col:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Ready To Analyze</div>
                <p style="margin:0; line-height:1.7;">
                    La imagen cargada se enviara al pipeline forense para extraer
                    caracteristicas y clasificarla como real o generada por IA.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analizando imagen y ejecutando el clasificador..."):
                try:
                    st.session_state.classification_result = run_inference(file_bytes)
                    st.session_state.description_result = None
                except Exception as error:
                    st.session_state.classification_result = None
                    st.error(f"No fue posible analizar la imagen: {error}")


def render_result_section() -> None:
    result = st.session_state.classification_result
    if not result:
        return

    is_ai = result["label"] == 1
    status_color = FORD_RED if is_ai else FORD_GREEN
    status_class = "ai" if is_ai else "real"
    confidence = result["confidence"]

    st.markdown(
        f"""
        <div class="result-card {status_class}">
            <div class="section-title">Result</div>
            <div class="result-label">Classification</div>
            <div class="result-value" style="color:{status_color};">
                {result["label_text"]}
            </div>
            <p class="result-help">{result["explanation"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if is_ai:
        st.error("Se detectaron patrones compatibles con generacion sintetica.")
    else:
        st.success("La imagen se alinea mejor con distribuciones naturales.")

    metric_col, secondary_col = st.columns(2, gap="large")
    with metric_col:
        if confidence is not None:
            st.metric("Confidence", f"{confidence * 100:.1f}%")
            st.progress(int(confidence * 100))
        else:
            st.warning("No fue posible estimar una confianza con la implementacion actual.")

    with secondary_col:
        st.metric("Training Samples", str(result["dataset_samples"]))
        st.caption("Modelo KNN entrenado desde el dataset disponible en el proyecto.")

    if st.button("Generate Description", use_container_width=True):
        with st.spinner("Generando descripcion contextual..."):
            try:
                st.session_state.description_result = generate_description(
                    st.session_state.uploaded_bytes
                )
            except Exception as error:
                st.session_state.description_result = None
                st.warning(f"No fue posible generar la descripcion: {error}")


def render_description_section() -> None:
    description_result = st.session_state.description_result
    if not description_result:
        return

    description_text = description_result.get("description", "").strip()
    st.markdown(
        '<div class="section-card"><div class="section-title">Description</div>',
        unsafe_allow_html=True,
    )

    if description_text:
        st.markdown(
            f'<div class="description-box">{description_text}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("El modulo de descripcion no devolvio texto para esta imagen.")

    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="AI Image Detection System",
        page_icon="🔎",
        layout="centered",
    )

    initialize_state()
    inject_styles()
    render_header()

    try:
        render_upload_section()
    except Exception as error:
        st.error(f"Ocurrio un problema al cargar la interfaz: {error}")
        return

    if st.session_state.uploaded_bytes is None:
        st.warning("Carga una imagen PNG para habilitar el analisis forense.")

    render_result_section()
    render_description_section()


if __name__ == "__main__":
    main()
