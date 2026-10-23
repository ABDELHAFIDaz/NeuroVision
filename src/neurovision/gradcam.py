
import cv2
import numpy as np
import tensorflow as tf


def _layers(model):
    return [
        layer
        for layer in model.layers
        if not isinstance(layer, tf.keras.layers.InputLayer)
    ]


def _last_feature_map_index(layers, x):
    last = None
    for i, layer in enumerate(layers):
        x = layer(x)
        if len(x.shape) == 4:
            last = i
    return last


def make_gradcam_heatmap(model, img, class_index=None):
   
    x = tf.convert_to_tensor(np.asarray(img, dtype="float32")[None, ...])
    layers = _layers(model)

    try:
        target = _last_feature_map_index(layers, x)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Grad-CAM needs a model whose top-level layers are chained one "
            f"after the other (Sequential-like). Original error: {error}"
        ) from error
    if target is None:
        raise ValueError("No layer of this model outputs a feature map.")

    with tf.GradientTape() as tape:
        h = x
        for i, layer in enumerate(layers):
            h = layer(h)
            if i == target:
                conv_out = h
                
                tape.watch(conv_out)
        preds = h
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        score = preds[:, class_index]

    grads = tape.gradient(score, conv_out)
    if grads is None:
        raise ValueError("Could not compute gradients for the target layer.")

    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy().astype("float32"), int(class_index)


def overlay_heatmap(img, heatmap, alpha=0.4):
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    if img.dtype != np.uint8:
        scale = 255.0 if img.max() <= 1.0 else 1.0
        img = np.clip(img * scale, 0, 255).astype("uint8")
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    resized = cv2.resize(np.asarray(heatmap, dtype="float32"), (img.shape[1], img.shape[0]))
    colored = cv2.applyColorMap(np.uint8(255 * np.clip(resized, 0, 1)), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 1 - alpha, colored, alpha, 0)