from __future__ import annotations

import io
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

# Run:
# streamlit run app.py

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_ML_ROOT = PROJECT_ROOT / "backend" / "ml"

for candidate in (PROJECT_ROOT, BACKEND_ML_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from backend.ml.description.image_descriptor import ImageDescriptor
from backend.ml.forensics.classifier_features import extract_classifier_feature_dict
from backend.ml.forensics.image_quality_gate import (
    evaluate_quality_from_reader,
    suggest_focus_crop_box,
)
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
    "energy_to_variance_ratio": "Microdetalle frente al contraste",
    "block_energy_variance": "Distribucion del detalle",
    "spectral_energy": "Textura general",
    "energia_diagonal_2da_derivada_norm": "Cambios finos en bordes",
    "block_variance_mean": "Contraste local",
    "block_variance_var": "Regularidad entre zonas",
    "symmetry_score": "Balance visual izquierda-derecha",
    "local_hf_variance": "Detalle fino por regiones",
    "gradient_direction": "Direccion de bordes",
    "entropy_block_var": "Variedad de textura por zonas",
}

DEFAULT_SHARPNESS_THRESHOLD = float(os.getenv("IA_SHARPNESS_THRESHOLD", "0.42"))
FULL_IMAGE_MIN_SHARPNESS = float(os.getenv("IA_FULL_IMAGE_MIN_SHARPNESS", "0.16"))
FOCUS_CROP_MIN_GAIN = float(os.getenv("IA_FOCUS_CROP_MIN_GAIN", "0.12"))
FOCUS_CROP_MIN_SCORE = float(os.getenv("IA_FOCUS_CROP_MIN_SCORE", "0.30"))

FEATURE_COPY = {
    "energy_to_variance_ratio": {
        "low": "aparecio poco microdetalle en relacion con el contraste de la imagen",
        "high": "aparecio mucho microdetalle en relacion con el contraste de la imagen",
    },
    "block_energy_variance": {
        "low": "el detalle esta repartido de forma muy pareja",
        "high": "el detalle se concentra de forma desigual entre zonas",
    },
    "spectral_energy": {
        "low": "la textura general se ve mas suave de lo esperado",
        "high": "la textura general se ve mas cargada de detalle fino",
    },
    "energia_diagonal_2da_derivada_norm": {
        "low": "los bordes finos aparecen demasiado suaves",
        "high": "los bordes finos aparecen demasiado marcados",
    },
    "block_variance_mean": {
        "low": "varias zonas tienen poco contraste local",
        "high": "varias zonas tienen contraste local alto",
    },
    "block_variance_var": {
        "low": "las zonas mantienen un contraste bastante uniforme",
        "high": "hay cambios fuertes de contraste entre zonas",
    },
    "symmetry_score": {
        "low": "la composicion se ve mas balanceada de lo normal",
        "high": "la composicion cambia bastante entre izquierda y derecha",
    },
    "local_hf_variance": {
        "low": "hay poca variacion de detalle fino entre regiones",
        "high": "hay mucha variacion de detalle fino entre regiones",
    },
    "gradient_direction": {
        "low": "los bordes siguen direcciones muy consistentes",
        "high": "los bordes cambian de direccion mas de lo esperado",
    },
    "entropy_block_var": {
        "low": "la variedad de textura entre zonas es baja",
        "high": "la variedad de textura entre zonas es alta",
    },
}

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
        "model_resources": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
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
            div[data-testid="stExpander"] {{
                background: rgba(255,255,255,.82);
                border: 1px solid rgba(11,31,58,.08);
                border-radius: 18px;
                overflow: hidden;
                box-shadow: 0 10px 28px rgba(11,31,58,.05);
            }}
            div[data-testid="stExpander"] details summary {{
                padding-top: .2rem;
                padding-bottom: .2rem;
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
            .pill-wrap {{
                margin-top: .9rem;
                display: flex;
                gap: .45rem;
                flex-wrap: wrap;
            }}
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
            .card {{
                padding: 1rem 1rem;
                margin-bottom: .9rem;
            }}
            .card-title {{
                color: {THEME['navy']};
                font-size: 1rem;
                font-weight: 750;
                margin-bottom: .55rem;
            }}
            .result {{
                padding: 1rem 1.1rem;
                margin-bottom: .85rem;
                border-left: 7px solid {THEME['blue']};
            }}
            .result.ai {{ border-left-color: {THEME['danger']}; }}
            .result.real {{ border-left-color: {THEME['success']}; }}
            .result.rejected {{ border-left-color: {THEME['warning']}; }}
            .result-hero {{
                padding: 1.15rem 1.2rem 1rem 1.2rem;
                margin-bottom: 1rem;
                border-radius: 22px;
                border: 1px solid rgba(11,31,58,.08);
                background:
                    radial-gradient(circle at top right, rgba(47,195,255,.12), transparent 28%),
                    linear-gradient(180deg, rgba(255,255,255,.98), rgba(247,250,255,.98));
                box-shadow: 0 18px 38px rgba(11,31,58,.08);
            }}
            .result-hero.ai {{
                border-left: 8px solid {THEME['danger']};
            }}
            .result-hero.real {{
                border-left: 8px solid {THEME['success']};
            }}
            .result-kicker {{
                font-size: .78rem;
                text-transform: uppercase;
                letter-spacing: .08em;
                color: {THEME['muted']};
                margin-bottom: .45rem;
                font-weight: 700;
            }}
            .result-title {{
                font-size: 2rem;
                line-height: 1.05;
                font-weight: 850;
                margin: 0 0 .45rem 0;
            }}
            .result-copy {{
                color: {THEME['text']};
                line-height: 1.58;
                margin: 0;
                max-width: 65ch;
            }}
            .result-badge {{
                display: inline-flex;
                align-items: center;
                gap: .35rem;
                margin-top: .85rem;
                padding: .34rem .62rem;
                border-radius: 999px;
                background: rgba(11,31,58,.05);
                color: {THEME['navy']};
                font-size: .82rem;
                font-weight: 700;
            }}
            .stat-grid {{
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: .85rem;
                margin: .15rem 0 1rem 0;
            }}
            .mini-stat-grid {{
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: .8rem;
                margin-top: .85rem;
            }}
            .stat-card {{
                background: rgba(255,255,255,.92);
                border: 1px solid rgba(11,31,58,.08);
                border-radius: 18px;
                padding: .9rem .95rem;
                box-shadow: 0 10px 24px rgba(11,31,58,.05);
            }}
            .stat-label {{
                color: {THEME['muted']};
                font-size: .78rem;
                text-transform: uppercase;
                letter-spacing: .06em;
                margin-bottom: .35rem;
                font-weight: 700;
            }}
            .stat-value {{
                color: {THEME['navy']};
                font-size: 1.45rem;
                line-height: 1.05;
                font-weight: 800;
                margin-bottom: .18rem;
            }}
            .stat-help {{
                color: {THEME['muted']};
                font-size: .88rem;
                line-height: 1.45;
            }}
            .insight-card {{
                background: linear-gradient(180deg, rgba(255,255,255,.96), rgba(247,250,255,.95));
                border: 1px solid rgba(11,31,58,.08);
                border-radius: 18px;
                padding: 1rem 1.05rem;
                margin-bottom: 1rem;
                box-shadow: 0 12px 28px rgba(11,31,58,.05);
            }}
            .insight-title {{
                color: {THEME['navy']};
                font-size: 1rem;
                font-weight: 800;
                margin-bottom: .45rem;
            }}
            .insight-copy {{
                color: {THEME['text']};
                line-height: 1.62;
                margin: 0;
            }}
            .insight-list {{
                margin: .7rem 0 0 0;
                padding-left: 1.1rem;
                color: {THEME['text']};
            }}
            .insight-list li {{
                margin-bottom: .35rem;
                line-height: 1.55;
            }}
            .footer {{
                margin-top: 1.2rem;
                padding: .95rem 1rem;
                font-size: .92rem;
                color: {THEME['muted']};
            }}
            .small-muted {{
                color: {THEME['muted']};
                font-size: .92rem;
                line-height: 1.6;
            }}
            .pipeline-status {{
                margin: .7rem 0 .55rem 0;
                padding: .72rem .85rem;
                border-radius: 12px;
                border: 1px solid rgba(31,111,235,.22);
                background: rgba(255,255,255,.96);
                color: #111111;
                font-weight: 700;
                line-height: 1.45;
            }}
            .pipeline-status.success {{
                border-color: rgba(31,143,85,.28);
                background: rgba(31,143,85,.08);
                color: #111111;
            }}
            @media (max-width: 900px) {{
                .stat-grid, .mini-stat-grid {{
                    grid-template-columns: 1fr;
                }}
                .result-title {{
                    font-size: 1.6rem;
                }}
            }}
            div.stButton > button {{
                background-color: #363636;
                color: white;
                border: none;
                border-radius: 12px;
                padding: 0.75rem 1.5rem;
                font-size: 16px;
                font-weight: 600;
                transition: 0.3s;
            }}

            div.stButton > button:hover {{
                background-color: #4f4f4f;
                color: white;
                border: none;
            }}

            div.stButton > button:active {{
                background-color: #1e40af;
                color: white;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <section class="hero">
            <h1 class="hero-title">Sherlock IA</h1>
            <p class="hero-sub">
                Plataforma de analisis forense para deteccion de contenido sintetico.
                Evalua patrones estadisticos, texturales y estructurales de imagenes PNG
                para identificar senales compatibles con generacion por IA.
            </p>
            <div class="warn">
                Este sistema es de apoyo tecnico y no representa una verdad absoluta.
                El resultado debe interpretarse junto con contexto y evidencia adicional.
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def create_temp_png(image_bytes: bytes) -> Path:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_file.write(image_bytes)
    temp_file.flush()
    temp_file.close()
    return Path(temp_file.name)


def crop_png_bytes(image_bytes: bytes, bbox: tuple[int, int, int, int]) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as image:
        cropped = image.convert("RGB").crop(bbox)
        buffer = io.BytesIO()
        cropped.save(buffer, format="PNG")
        return buffer.getvalue()


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


def build_feature_explanation(
    feature_dict: dict[str, float],
    means: np.ndarray,
    stds: np.ndarray,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []

    for idx, key in enumerate(FEATURE_KEYS):
        value = float(feature_dict.get(key, 0.0))
        mean_value = float(means[idx])
        std_value = float(stds[idx])
        z_score = 0.0 if std_value == 0 else (value - mean_value) / std_value
        rows.append(
            {
                "feature": FEATURE_LABELS.get(key, key),
                "feature_key": key,
                "value": value,
                "mean": mean_value,
                "std": std_value,
                "z_score": z_score,
                "impact": abs(z_score),
            }
        )

    rows.sort(key=lambda item: float(item["impact"]), reverse=True)
    return rows


def build_feature_summary(
    rows: list[dict[str, float | str]],
    predicted_label: int,
) -> tuple[str, list[str]]:
    if not rows:
        return (
            "No se generaron hallazgos interpretables para resumir esta clasificacion.",
            [],
        )

    top_rows = rows[:3]
    if predicted_label == 1:
        summary = (
            "El modelo encontro una combinacion de textura, bordes y contraste "
            "que se parece mas a imagenes sinteticas dentro de su referencia."
        )
    else:
        summary = (
            "El modelo encontro textura, bordes y contraste mas cercanos a "
            "imagenes reales dentro de su referencia."
        )

    insights = []
    for row in top_rows:
        z_score = float(row["z_score"])
        tendency = "high" if z_score >= 0 else "low"
        feature_name = str(row["feature"])
        feature_key = str(row.get("feature_key", ""))
        magnitude = abs(z_score)
        if magnitude >= 2.0:
            qualifier = "fue una senal fuerte"
        elif magnitude >= 1.2:
            qualifier = "tuvo peso moderado"
        else:
            qualifier = "tuvo peso leve"

        copy = FEATURE_COPY.get(feature_key, {}).get(
            tendency,
            "se alejo de lo que suele verse en el conjunto de referencia",
        )
        insights.append(f"{feature_name}: {qualifier}; {copy}.")

    return summary, insights


def render_stat_grid(cards: list[dict[str, str]], mini: bool = False) -> None:
    grid_class = "mini-stat-grid" if mini else "stat-grid"
    parts = [f'<div class="{grid_class}">']
    for card in cards:
        parts.append(
            f"""
            <div class="stat-card">
                <div class="stat-label">{card['label']}</div>
                <div class="stat-value">{card['value']}</div>
                <div class="stat-help">{card['help']}</div>
            </div>
            """
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def select_analysis_image(image_bytes: bytes) -> dict[str, Any]:
    reader = load_image(image_bytes)
    original_quality = evaluate_quality_from_reader(
        reader,
        config={"sharpness_threshold": DEFAULT_SHARPNESS_THRESHOLD},
    )

    original_score = float(original_quality["sharpness_score"])
    if original_quality["is_acceptable"] or original_score >= FULL_IMAGE_MIN_SHARPNESS:
        selected_quality = dict(original_quality)
        if not selected_quality["is_acceptable"]:
            selected_quality["is_acceptable"] = True
            selected_quality["acceptance_reason"] = "full_image_min_sharpness"
        return {
            "reader": reader,
            "image_bytes": image_bytes,
            "quality": selected_quality,
            "original_quality": original_quality,
            "focused_quality": None,
            "used_focus_crop": False,
            "focus_bbox": None,
            "region_status": "full_image",
        }

    focus_bbox = suggest_focus_crop_box(reader, crop_width_blocks=10, crop_height_blocks=10)

    if focus_bbox is None:
        return {
            "reader": reader,
            "image_bytes": image_bytes,
            "quality": original_quality,
            "original_quality": original_quality,
            "focused_quality": None,
            "used_focus_crop": False,
            "focus_bbox": None,
            "region_status": "full_image",
        }

    focused_bytes = crop_png_bytes(image_bytes, focus_bbox)
    focused_reader = load_image(focused_bytes)
    focused_quality = evaluate_quality_from_reader(
        focused_reader,
        config={"sharpness_threshold": DEFAULT_SHARPNESS_THRESHOLD},
    )

    focused_score = float(focused_quality["sharpness_score"])
    focused_local_score = float(focused_quality.get("local_focus_score", 0.0))
    focused_is_useful = (
        focused_quality["is_acceptable"]
        or focused_score >= FOCUS_CROP_MIN_SCORE
        or focused_local_score >= 0.45
    )
    use_focus_crop = (
        focused_is_useful
        and focused_score >= original_score + FOCUS_CROP_MIN_GAIN
    )

    if use_focus_crop:
        return {
            "reader": focused_reader,
            "image_bytes": focused_bytes,
            "quality": focused_quality,
            "original_quality": original_quality,
            "focused_quality": focused_quality,
            "used_focus_crop": True,
            "focus_bbox": focus_bbox,
            "region_status": "focus_crop",
        }

    return {
        "reader": reader,
        "image_bytes": image_bytes,
        "quality": original_quality,
        "original_quality": original_quality,
        "focused_quality": focused_quality,
        "used_focus_crop": False,
        "focus_bbox": None,
        "region_status": "full_image",
    }


def run_inference(image_bytes: bytes) -> dict[str, Any]:
    analysis_input = select_analysis_image(image_bytes)
    reader = analysis_input["reader"]
    resources = get_model_resources()
    quality = analysis_input["quality"]

    if not quality["is_acceptable"]:
        return {
            "rejected_by_quality": True,
            "quality_metrics": quality,
            "original_quality_metrics": analysis_input.get("original_quality"),
            "focused_quality_metrics": analysis_input.get("focused_quality"),
            "used_focus_crop": analysis_input["used_focus_crop"],
            "focus_bbox": analysis_input["focus_bbox"],
            "region_status": analysis_input["region_status"],
            "message": (
                "Imagen demasiado desenfocada para ejecutar el analisis forense. "
                "No contiene suficiente informacion visual para una decision confiable."
            ),
            "dataset_samples": resources["samples"],
        }

    feature_dict = extract_classifier_feature_dict(analysis_input["image_bytes"])
    sample = np.asarray(select_features(feature_dict), dtype=np.float64)
    normalized_sample = (sample - resources["means"]) / resources["stds"]

    prediction = resources["model"].predict([normalized_sample])[0]
    confidence = resources["model"].confidence_one(normalized_sample)
    explanation_rows = build_feature_explanation(
        feature_dict,
        np.asarray(resources["means"], dtype=np.float64),
        np.asarray(resources["stds"], dtype=np.float64),
    )

    return {
        "rejected_by_quality": False,
        "label": prediction,
        "label_text": "Generada por IA" if prediction == 1 else "Imagen real",
        "confidence": confidence,
        "feature_dict": feature_dict,
        "dataset_samples": resources["samples"],
        "quality_metrics": quality,
        "original_quality_metrics": analysis_input.get("original_quality"),
        "focused_quality_metrics": analysis_input.get("focused_quality"),
        "feature_explanation": explanation_rows,
        "used_focus_crop": analysis_input["used_focus_crop"],
        "focus_bbox": analysis_input["focus_bbox"],
        "region_status": analysis_input["region_status"],
        "explanation": (
            "Patrones compatibles con generacion sintetica detectados."
            if prediction == 1
            else "Las caracteristicas se alinean mejor con distribuciones naturales."
        ),
    }


def generate_description(image_bytes: bytes) -> dict[str, Any]:
    reader = load_image(image_bytes)
    descriptor = ImageDescriptor()
    return descriptor.describe_from_reader(reader)


def reset_analysis_state() -> None:
    st.session_state.classification_result = None
    st.session_state.description_result = None


def render_upload_analysis_section() -> None:
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown(
            '<section class="card"><div class="card-title">Carga de Imagen</div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Arrastra y suelta una imagen PNG o selecciona un archivo",
            type=["png"],
            help="Solo PNG para mantener compatibilidad con el pipeline forense actual.",
        )
        st.markdown("</section>", unsafe_allow_html=True)

    if uploaded_file is None:
        with right:
            st.markdown(
                '<section class="card"><div class="card-title">Preview</div><p class="small-muted">Carga una imagen para visualizarla y ejecutar el analisis.</p></section>',
                unsafe_allow_html=True,
            )
        return

    file_bytes = uploaded_file.getvalue()
    if (
        st.session_state.uploaded_name != uploaded_file.name
        or st.session_state.uploaded_bytes != file_bytes
    ):
        st.session_state.uploaded_name = uploaded_file.name
        st.session_state.uploaded_bytes = file_bytes
        reset_analysis_state()

    with right:
        st.markdown(
            '<section class="card"><div class="card-title">Preview</div>',
            unsafe_allow_html=True,
        )
        st.image(file_bytes, caption=uploaded_file.name, use_container_width=True)
        st.markdown("</section>", unsafe_allow_html=True)

    with left:
        st.markdown('<section class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-title">Pipeline de Analisis</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="small-muted">El sistema evaluara calidad, extraera caracteristicas forenses y emitira una clasificacion con explicacion.</p>',
            unsafe_allow_html=True,
        )
        if st.button("Iniciar Analisis Forense", type="primary", use_container_width=True):
            status = st.empty()
            progress = st.progress(0)
            steps = [
                "Validando calidad de imagen...",
                "Extrayendo caracteristicas forenses...",
                "Normalizando datos y consultando el modelo...",
                "Generando resultado final...",
            ]
            try:
                for i, message in enumerate(steps, start=1):
                    status.markdown(
                        f'<div class="pipeline-status">{message}</div>',
                        unsafe_allow_html=True,
                    )
                    progress.progress(int((i - 1) / len(steps) * 100))
                    time.sleep(0.12)

                st.session_state.classification_result = run_inference(file_bytes)
                st.session_state.description_result = None
                progress.progress(100)
                status.markdown(
                    '<div class="pipeline-status success">Analisis completado.</div>',
                    unsafe_allow_html=True,
                )
            except Exception as error:
                st.session_state.classification_result = None
                progress.empty()
                status.error(f"No fue posible analizar la imagen: {error}")
        st.markdown("</section>", unsafe_allow_html=True)


def render_explanation_panel(result: dict[str, Any]) -> None:
    rows = result.get("feature_explanation", [])
    if not rows:
        return

    summary, insights = build_feature_summary(rows, int(result["label"]))
    bbox = result.get("focus_bbox")
    used_focus_crop = result.get("used_focus_crop", False)
    with st.expander("Explicacion del resultado"):
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">Lectura rapida</div>
                <p class="insight-copy">{summary}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if insights:
            items = "".join(f"<li>{item}</li>" for item in insights)
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-title">Senales que mas influyeron</div>
                    <ul class="insight-list">{items}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if used_focus_crop and bbox:
            left, top, right, bottom = bbox
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-title">Region analizada</div>
                    <p class="insight-copy">
                        La imagen completa tenia zonas con poco detalle util. Se encontro una region mas nitida y el clasificador trabajo sobre esa parte.
                        Area utilizada: {right - left} x {bottom - top} px.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="insight-card">
                    <div class="insight-title">Region analizada</div>
                    <p class="insight-copy">
                        La imagen tenia suficiente informacion visual para analizarse completa. No se aplico recorte por enfoque.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


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
        # render_quality_panel(result)
        return

    is_ai = result["label"] == 1
    status_class = "ai" if is_ai else "real"
    status_color = THEME["danger"] if is_ai else THEME["success"]
    summary, _ = build_feature_summary(
        result.get("feature_explanation", []),
        int(result["label"]),
    )

    st.markdown(
        f"""
        <section class="result-hero {status_class}">
            <div class="result-kicker">Resultado principal</div>
            <div class="result-title" style="color:{status_color};">{result['label_text']}</div>
            <p class="result-copy">{result['explanation']} {summary}</p>
            <div class="result-badge">Modelo KNN · Analisis forense PNG</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    render_explanation_panel(result)

    if st.button("Generar descripcion contextual", use_container_width=True):
        with st.spinner("Generando descripcion..."):
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
        '<section class="card"><div class="card-title">Descripcion Contextual</div>',
        unsafe_allow_html=True,
    )
    if description_text:
        st.write(description_text)
    else:
        st.warning("El modulo de descripcion no devolvio texto para esta imagen.")
    st.markdown("</section>", unsafe_allow_html=True)


def render_info_section() -> None:
    st.markdown(
        '<section class="card"><div class="card-title">Especificaciones del sistema</div>',
        unsafe_allow_html=True,
    )
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
    st.markdown("</section>", unsafe_allow_html=True)


def render_footer() -> None:
    st.markdown(
        """
        <section class="footer">
            <strong>Sherlock IA</strong> · Streamlit · Python · Forensia Digital · Machine Learning (KNN)<br/>
            Enfoque academico y de investigacion aplicada en deteccion de contenido sintetico.
        </section>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Sherlock IA",
        page_icon="AI",
        layout="wide",
    )

    initialize_state()
    inject_styles()
    render_hero()

    try:
        with st.spinner("Cargando el modelo Sherlock IA..."):
            get_model_resources()
    except Exception as error:
        st.error(f"Ocurrio un problema al cargar el modelo: {error}")
        return

    try:
        render_upload_analysis_section()
    except Exception as error:
        st.error(f"Ocurrio un problema en la interfaz: {error}")
        return

    if st.session_state.uploaded_bytes is None:
        st.info("Carga una imagen PNG para habilitar el analisis forense.")

    render_result_section()
    render_description_section()
    render_info_section()
    render_footer()


if __name__ == "__main__":
    main()
