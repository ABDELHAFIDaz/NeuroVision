import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neurovision.gradcam import make_gradcam_heatmap, overlay_heatmap
from neurovision.predict import (
    DEFAULT_MODEL_PATH,
    decode_image_bytes,
    load_model,
    predict_array,
    resize_rgb,
)

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MAX_UPLOAD_MB = 10
FRIENDLY_NAMES = {
    "best_model.keras": "CNN from scratch (Part 1)",
    "transfer_best.keras": "MobileNetV2, frozen base (Part 2)",
    "finetuned_best.keras": "MobileNetV2, fine-tuned (Part 2)",
}

st.set_page_config(page_title="NeuroVision AI", page_icon="🧠", layout="wide")


@st.cache_resource
def get_model(path):
    return load_model(path)


def read_all_metrics():
    return [json.loads(p.read_text()) for p in sorted(REPORTS_DIR.glob("metrics_*.json"))]


st.title("NeuroVision AI")
st.caption(
    "Brain tumor classification from MRI images. "
    "This tool supports medical staff and never replaces a doctor's diagnosis."
)

model_files = sorted(MODELS_DIR.glob("*.keras"))
if not model_files:
    st.error(f"No model found in {MODELS_DIR}. Train the model first.")
    st.stop()

default_index = model_files.index(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH in model_files else 0
with st.sidebar:
    st.header("Model")
    chosen = st.selectbox(
        "Model used for the prediction",
        model_files,
        index=default_index,
        format_func=lambda p: FRIENDLY_NAMES.get(p.name, p.name),
    )
    metrics = next((m for m in read_all_metrics() if m.get("model_file") == chosen.name), None)
    if metrics:
        st.metric("Test accuracy", f"{metrics['test_accuracy']:.1%}")
        st.metric("Macro F1-score", f"{metrics['macro_f1']:.3f}")
    else:
        st.caption("No evaluation found for this model (run evaluation.ipynb).")
    threshold = st.slider("Warn when confidence is below", 0.50, 0.99, 0.80)

model = get_model(chosen)

tab_predict, tab_compare, tab_runs = st.tabs(["Predict", "Model comparison", "MLflow runs"])

with tab_predict:
    uploaded = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png", "bmp"])
    show_cam = st.checkbox("Show Grad-CAM heatmap (where the model looked)")

    if uploaded is None:
        st.info("Upload an image to get a prediction.")
    elif uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"File too large (limit {MAX_UPLOAD_MB} MB).")
    else:
        try:
            image = decode_image_bytes(uploaded.getvalue())
        except ValueError as error:
            st.error(str(error))
            st.stop()

        result = predict_array(image, model)

        left, right = st.columns(2)
        left.image(image, caption="Uploaded image")
        right.metric("Predicted class", result["class"])
        right.metric("Confidence", f"{result['confidence']:.1%}")
        if result["confidence"] < threshold:
            right.warning("Low confidence: treat this prediction with caution.")

        st.subheader("Probability per class")
        st.bar_chart(pd.Series(result["probabilities"]))

        if show_cam:
            resized = resize_rgb(image, model)
            try:
                heatmap, _ = make_gradcam_heatmap(model, resized)
            except Exception as error:  # noqa: BLE001
                st.warning(f"Grad-CAM is not available for this model: {error}")
            else:
                st.subheader("Grad-CAM")
                st.image(
                    overlay_heatmap(resized, heatmap),
                    caption="Regions that influenced the prediction",
                )

with tab_compare:
    rows = read_all_metrics()
    if not rows:
        st.info("Run evaluation.ipynb once per model to fill this tab.")
    else:
        table = pd.DataFrame(rows)[
            ["tag", "model_file", "test_accuracy", "test_loss", "macro_f1", "weighted_f1", "n_test"]
        ].round(4)
        st.dataframe(table, width="stretch", hide_index=True)
        tag = st.selectbox("Confusion matrix of", [row["tag"] for row in rows])
        image_path = REPORTS_DIR / f"confusion_matrix_{tag}.png"
        if image_path.exists():
            st.image(str(image_path), width=520)

with tab_runs:
    db_path = ROOT / "mlflow.db"
    if not db_path.exists():
        st.info("No mlflow.db found. Train or evaluate a model first.")
    else:
        try:
            import mlflow

            mlflow.set_tracking_uri(f"sqlite:///{db_path}")
            runs = mlflow.search_runs(experiment_names=["NeuroVision"])
            wanted = [
                "tags.mlflow.runName",
                "tags.stage",
                "metrics.best_val_accuracy",
                "metrics.test_accuracy",
                "metrics.macro_f1",
                "metrics.training_minutes",
                "start_time",
            ]
            st.dataframe(runs[[c for c in wanted if c in runs.columns]], width="stretch")
            st.caption("Curves and artifacts: uv run mlflow ui --backend-store-uri sqlite:///mlflow.db")
        except Exception as error:  # noqa: BLE001
            st.warning(f"Could not read the MLflow runs: {error}")