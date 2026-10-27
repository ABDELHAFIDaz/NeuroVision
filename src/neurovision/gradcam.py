"""Grad-CAM for the NeuroVision models.

Same public API as before:
    make_gradcam_heatmap(model, img_rgb, class_index=None) -> (heatmap, class_index)
    overlay_heatmap(img_rgb, heatmap, alpha=0.4) -> uint8 RGB image

Target layer:
  * scratch CNN: the last top-level Conv2D layer (as before);
  * transfer / fine-tuned MobileNetV2: the base network is ONE nested layer, so
    there is no top-level Conv2D. The output of that base (its last feature map)
    is used instead.

The model is run layer by layer, so it must be a chain of layers (Sequential-like),
which is the case for both build_model() and build_transfer_model().
"""
import cv2
import numpy as np
import tensorflow as tf


def _layers(model):
    return [
        layer
        for layer in model.layers
        if not isinstance(layer, tf.keras.layers.InputLayer)
    ]


def _target_index(layers, x):
    convs = [i for i, layer in enumerate(layers) if isinstance(layer, tf.keras.layers.Conv2D)]
    if convs:
        return convs[-1]
    last = None
    for i, layer in enumerate(layers):
        x = layer(x, training=False)
        if len(x.shape) == 4:
            last = i
    return last


def make_gradcam_heatmap(model, img_rgb, class_index=None):
    """Return (heatmap in [0, 1] with shape (h, w), predicted class index).

    `img_rgb` is one image as the model receives it (resized, pixels 0-255),
    without the batch axis.
    """
    x = tf.convert_to_tensor(np.asarray(img_rgb, dtype="float32")[None, ...])
    layers = _layers(model)

    try:
        target = _target_index(layers, x)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Grad-CAM needs a model whose layers are chained one after the other. "
            f"Original error: {error}"
        ) from error
    if target is None:
        raise ValueError("No layer of this model outputs a feature map.")

    with tf.GradientTape() as tape:
        h = x
        for i, layer in enumerate(layers):
            h = layer(h, training=False)
            if i == target:
                conv_out = h
                # Needed when the base is frozen (transfer learning): its
                # variables are not trainable, so nothing is watched otherwise.
                tape.watch(conv_out)
        preds = h
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        score = preds[:, class_index]

    grads = tape.gradient(score, conv_out)
    if grads is None:
        raise ValueError("Could not compute gradients for the target layer.")

    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))
    return heatmap.numpy().astype("float32"), int(class_index)


def overlay_heatmap(img_rgb, heatmap, alpha=0.4):
    """Blend the heatmap (jet colours) on the image. Returns uint8 RGB."""
    img = np.asarray(img_rgb)
    if img.dtype != np.uint8:
        scale = 255.0 if img.max() <= 1.0 else 1.0
        img = np.clip(img * scale, 0, 255).astype("uint8")
    heat = cv2.resize(np.asarray(heatmap, dtype="float32"), (img.shape[1], img.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * np.clip(heat, 0, 1)), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 1 - alpha, heat, alpha, 0)