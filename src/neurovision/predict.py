from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from neurovision.data import CLASSES, VALID_EXTENSIONS

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "best_model.keras"


def load_model(path=DEFAULT_MODEL_PATH):
    return tf.keras.models.load_model(path)


@lru_cache(maxsize=1)
def _default_model():
    return load_model(DEFAULT_MODEL_PATH)


def model_input_size(model):
    height, width = model.input_shape[1:3]
    return width, height


def resize_rgb(img_rgb, model):
    return cv2.resize(img_rgb, model_input_size(model), interpolation=cv2.INTER_AREA)


def decode_image_bytes(data):
    if not data:
        raise ValueError("The file is empty.")
    try:
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise ValueError("The file is not a readable image.") from error
    if img is None:
        raise ValueError("The file is not a readable image.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_image(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() not in VALID_EXTENSIONS:
        allowed = ", ".join(sorted(VALID_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{path.suffix}'. Allowed: {allowed}")
    return decode_image_bytes(path.read_bytes())


def predict_array(img_rgb, model):
    resized = resize_rgb(img_rgb, model)
    probs = model.predict(resized[None, ...], verbose=0)[0]
    index = int(np.argmax(probs))
    return {
        "class": CLASSES[index],
        "confidence": float(probs[index]),
        "probabilities": {name: float(p) for name, p in zip(CLASSES, probs)},
    }


def predict_image(path, model=None):
    img = read_image(path)
    return predict_array(img, model if model is not None else _default_model())