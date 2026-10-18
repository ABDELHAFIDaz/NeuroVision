import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from neurovision.gradcam import make_gradcam_heatmap, overlay_heatmap
from neurovision.predict import (
    DEFAULT_MODEL_PATH,
    decode_image_bytes,
    load_model,
    predict_array,
    resize_rgb,
)

st.set_page_config(page_title="NeuroVision AI", page_icon="🧠")


@st.cache_resource
def get_model():
    return load_model(DEFAULT_MODEL_PATH)


st.title("NeuroVision AI")
st.caption(
    "Brain tumor classification from MRI images. "
    "This tool supports medical staff and never replaces a doctor's diagnosis."
)

if not DEFAULT_MODEL_PATH.exists():
    st.error(f"Model not found at {DEFAULT_MODEL_PATH}. Train the model first.")
    st.stop()

model = get_model()

uploaded = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png", "bmp"])
show_cam = st.checkbox("Show Grad-CAM heatmap (where the model looked)")

if uploaded is not None:
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

    st.subheader("Probability per class")
    st.bar_chart(pd.Series(result["probabilities"]))

    if show_cam:
        resized = resize_rgb(image, model)
        heatmap, _ = make_gradcam_heatmap(model, resized)
        st.subheader("Grad-CAM")
        st.image(overlay_heatmap(resized, heatmap), caption="Regions that influenced the prediction")
else:
    st.info("Upload an image to get a prediction.")