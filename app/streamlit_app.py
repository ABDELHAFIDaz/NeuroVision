"""NeuroVision AI - Streamlit demo.

Run from the project root:
    uv run streamlit run app/streamlit_app.py

Needs the trained models in models/*.keras. The "Model comparison" tab needs
reports/metrics_<tag>.json (written by evaluation.ipynb), and the "MLflow runs"
tab needs the mlruns/ folder.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurovision.gradcam import make_gradcam_heatmap, overlay_heatmap  # noqa: E402
from neurovision.preprocessing import encoder  # noqa: E402

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MAX_UPLOAD_MB = 10

st.set_page_config(page_title="NeuroVision AI", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)


def read_all_metrics():
    return [json.loads(p.read_text()) for p in sorted(REPORTS_DIR.glob("metrics_*.json"))]


def metrics_for(model_file):
    for metrics in read_all_metrics():
        if metrics.get("model_file") == model_file:
            return metrics
    return None


def prepare_image(uploaded, model, divisor):
    """Open, validate and resize the upload like the training images."""
    image = Image.open(uploaded)
    image.load()
    height, width = model.input_shape[1:3]
    channels = model.input_shape[-1]
    image = image.convert("RGB" if channels == 3 else "L").resize((width, height))
    array = np.asarray(image, dtype="float32")
    if channels == 1:
        array = array[..., None]
    return array / divisor


st.title("NeuroVision AI")
st.caption("Brain MRI classification: glioma, meningioma, no tumor, pituitary.")
st.warning(
    "Educational project. This tool is not a medical device and must not be "
    "used for diagnosis."
)

models = sorted(MODELS_DIR.glob("*.keras"))
if not models:
    st.error(f"No .keras model found in {MODELS_DIR}. Train a model first.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    choice = st.selectbox("Model", models, format_func=lambda p: p.name)
    threshold = st.slider("Warn when confidence is below", 0.50, 0.99, 0.80)
    show_cam = st.checkbox("Show Grad-CAM", value=True)

model = load_model(str(choice))
metrics = metrics_for(choice.name)

if metrics:
    divisor = 255.0 if metrics["input_max"] <= 1.0 else 1.0
else:
    with st.sidebar:
        scale = st.radio(
            "Pixel scale used in training",
            ["0 to 1", "0 to 255"],
            help="Run evaluation.ipynb for this model to detect it automatically.",
        )
    divisor = 255.0 if scale == "0 to 1" else 1.0
    st.info("No evaluation metrics found for this model, so the pixel scale is manual.")

class_names = list(encoder.classes_)
tab_predict, tab_compare, tab_runs = st.tabs(["Predict", "Model comparison", "MLflow runs"])

with tab_predict:
    uploaded = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File too large (limit {MAX_UPLOAD_MB} MB).")
        else:
            try:
                array = prepare_image(uploaded, model, divisor)
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("This file is not a valid image.")
            else:
                probs = model.predict(array[None, ...], verbose=0)[0]
                best = int(np.argmax(probs))
                label = class_names[best] if len(class_names) == len(probs) else str(best)

                left, right = st.columns(2)
                left.image(uploaded, caption="Uploaded image", use_container_width=True)
                right.metric("Prediction", label, f"{probs[best]:.1%} confidence")
                if probs[best] < threshold:
                    right.warning("Low confidence: treat this prediction with caution.")
                right.bar_chart(pd.Series(probs, index=class_names[: len(probs)]))

                if show_cam:
                    try:
                        heatmap, _ = make_gradcam_heatmap(model, array)
                        st.image(
                            overlay_heatmap(array, heatmap),
                            caption="Grad-CAM: regions that influenced the prediction",
                            use_container_width=True,
                        )
                    except Exception as error:  # noqa: BLE001
                        st.warning(f"Grad-CAM unavailable for this model: {error}")

with tab_compare:
    rows = read_all_metrics()
    if not rows:
        st.info("Run evaluation.ipynb for each model to fill this tab.")
    else:
        table = pd.DataFrame(rows)[
            ["tag", "model_file", "test_accuracy", "test_loss", "macro_f1", "weighted_f1", "n_test"]
        ].round(4)
        st.dataframe(table, use_container_width=True, hide_index=True)
        tag = st.selectbox("Confusion matrix of", [row["tag"] for row in rows])
        image_path = REPORTS_DIR / f"confusion_matrix_{tag}.png"
        if image_path.exists():
            st.image(str(image_path), width=520)

with tab_runs:
    runs_dir = ROOT / "mlruns"
    if not runs_dir.exists():
        st.info("No mlruns/ folder found. Train or evaluate a model first.")
    else:
        try:
            import mlflow

            mlflow.set_tracking_uri(runs_dir.as_uri())
            runs = mlflow.search_runs(experiment_names=["NeuroVision"])
            wanted = [
                "tags.mlflow.runName", "tags.stage", "metrics.best_val_accuracy",
                "metrics.test_accuracy", "metrics.macro_f1", "metrics.training_minutes",
                "start_time",
            ]
            st.dataframe(runs[[c for c in wanted if c in runs.columns]], use_container_width=True)
            st.caption("For curves and artifacts: uv run mlflow ui --backend-store-uri ./mlruns")
        except Exception as error:  # noqa: BLE001
            st.warning(f"Could not read MLflow runs: {error}")