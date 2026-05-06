from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
ML_ROOT = ROOT / "backend" / "ml"
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from forensics.raw_image_reader import PNGReader
from features.feature_extractor import FeatureExtractor

FORENSICS_DIR = ML_ROOT / "forensics"

FAKE_IMAGES = [
    "atleta_imagen_fake.png",
    "paisaje_natural_auto_fake.png",
    "paisaje_urbano_hollywood_fake_gemini.png",
]

CORE_FEATURES = [
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

EXTRA_CANDIDATE_FEATURES = [
    "entropy",
    "variance",
    "edge_density",
    "gradient_variance",
    "entropy_block_mean",
    "horizontal_edge_bias",
    "brightness_balance",
    "center_darkness_bias",
    "colorfulness",
    "dominant_color_ratio",
    "top_blue_ratio",
    "bottom_green_ratio",
    "bottom_yellow_ratio",
]


def extract_features(image_path: Path) -> Dict[str, float]:
    reader = PNGReader(str(image_path))
    reader.read()
    return FeatureExtractor.extract_from_png_reader(reader)


def classify_image(filename: str) -> str:
    if filename in FAKE_IMAGES:
        return "FAKE"
    return "REAL"


def find_real_images() -> List[str]:
    reals = []
    for p in sorted(FORENSICS_DIR.glob("*.png")):
        if p.name in FAKE_IMAGES:
            continue
        if p.name == "negro.png":
            continue
        if p.name.startswith("paisaje") or p.name.startswith("pasisaje"):
            reals.append(p.name)
    return reals


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    image_names = [name for name in FAKE_IMAGES if (FORENSICS_DIR / name).exists()]
    image_names.extend(find_real_images())

    rows: List[Dict[str, float]] = []
    for name in image_names:
        feat = extract_features(FORENSICS_DIR / name)
        row = {"image": name, "label": classify_image(name)}
        row.update(feat)
        rows.append(row)

    return pd.DataFrame(rows)


def select_available_features(df: pd.DataFrame) -> List[str]:
    candidates = CORE_FEATURES + [f for f in EXTRA_CANDIDATE_FEATURES if f not in CORE_FEATURES]
    return [f for f in candidates if f in df.columns]


def separation_table(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    real = df[df["label"] == "REAL"]
    fake = df[df["label"] == "FAKE"]

    rows = []
    for f in features:
        r = pd.to_numeric(real[f], errors="coerce").dropna()
        k = pd.to_numeric(fake[f], errors="coerce").dropna()
        if r.empty or k.empty:
            continue

        real_mean = float(r.mean())
        real_std = float(r.std(ddof=0))
        fake_mean = float(k.mean())
        z_gap = abs(fake_mean - real_mean) / (real_std + 1e-9)

        out_count = 0
        rmin, rmax = float(r.min()), float(r.max())
        for val in k.tolist():
            if val < rmin or val > rmax:
                out_count += 1

        rows.append(
            {
                "feature": f,
                "real_mean": real_mean,
                "fake_mean": fake_mean,
                "abs_diff": abs(fake_mean - real_mean),
                "z_gap_vs_real": z_gap,
                "fake_out_of_real_range": out_count,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["z_gap_vs_real", "fake_out_of_real_range"], ascending=[False, False])


def fake_zscore_table(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    real = df[df["label"] == "REAL"]
    fake = df[df["label"] == "FAKE"]

    stat = {}
    for f in features:
        r = pd.to_numeric(real[f], errors="coerce").dropna()
        if r.empty:
            continue
        stat[f] = {"mean": float(r.mean()), "std": float(r.std(ddof=0)), "min": float(r.min()), "max": float(r.max())}

    rows = []
    for _, row in fake.iterrows():
        image = row["image"]
        for f in stat:
            v = float(row[f])
            mean = stat[f]["mean"]
            std = stat[f]["std"]
            z = 0.0 if std == 0 else (v - mean) / std
            out = v < stat[f]["min"] or v > stat[f]["max"]
            rows.append(
                {
                    "image": image,
                    "feature": f,
                    "value": v,
                    "z_score": z,
                    "status": "OUT_OF_RANGE" if out else "IN_RANGE",
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Forensics Feature Dashboard", layout="wide")
    st.title("Dashboard Forense: REAL vs IA")
    st.caption("Comparativa de features entre imágenes reales de paisaje y fakes (GPT Image 2 / Gemini)")

    df = load_dataset()
    if df.empty:
        st.error(f"No se pudieron cargar imágenes desde {FORENSICS_DIR}")
        return

    fake_count = int((df["label"] == "FAKE").sum())
    real_count = int((df["label"] == "REAL").sum())
    st.write(f"Imágenes cargadas: {len(df)} (REAL={real_count}, FAKE={fake_count})")

    available_features = select_available_features(df)

    default_n = min(8, len(available_features))
    selected_features = st.multiselect(
        "Features a visualizar",
        options=available_features,
        default=available_features[:default_n],
    )

    if not selected_features:
        st.warning("Selecciona al menos una feature")
        return

    sep = separation_table(df, selected_features)
    st.subheader("Features que mejor separan REAL vs FAKE")
    st.dataframe(
        sep.style.format(
            {
                "real_mean": "{:.4f}",
                "fake_mean": "{:.4f}",
                "abs_diff": "{:.4f}",
                "z_gap_vs_real": "{:.3f}",
            }
        ),
        use_container_width=True,
    )

    st.subheader("Distribución por feature (boxplot)")
    long_df = df[["image", "label"] + selected_features].melt(
        id_vars=["image", "label"], var_name="feature", value_name="value"
    )
    st.bar_chart(long_df, x="feature", y="value", color="label")

    st.subheader("Z-score por imagen FAKE contra distribución REAL")
    zdf = fake_zscore_table(df, selected_features)
    if zdf.empty:
        st.info("No fue posible calcular z-scores")
    else:
        st.dataframe(
            zdf.sort_values(["image", "feature"]).style.format({"value": "{:.4f}", "z_score": "{:+.3f}"}),
            use_container_width=True,
        )

        zplot = zdf.copy()
        zplot["abs_z"] = zplot["z_score"].abs()
        st.bar_chart(zplot, x="feature", y="abs_z", color="image")

    st.subheader("Detalle por imagen")
    st.dataframe(df[["image", "label"] + selected_features], use_container_width=True)


if __name__ == "__main__":
    main()
