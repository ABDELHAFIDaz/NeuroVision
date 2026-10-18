import cv2
import numpy as np
import tensorflow as tf


def make_gradcam_heatmap(model, img_rgb, class_index=None):
    last_conv = next(
        layer for layer in reversed(model.layers) if isinstance(layer, tf.keras.layers.Conv2D)
    )

    inputs = tf.keras.Input(shape=model.input_shape[1:])
    x = inputs
    conv_output = None
    for layer in model.layers:
        x = layer(x)
        if layer is last_conv:
            conv_output = x
    grad_model = tf.keras.Model(inputs, [conv_output, x])

    x = tf.convert_to_tensor(img_rgb[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        score = preds[:, class_index]

    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), class_index


def overlay_heatmap(img_rgb, heatmap, alpha=0.4):
    heat = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 1 - alpha, heat, alpha, 0)