import tensorflow as tf
from tensorflow.keras import layers, models

from neurovision.data import CLASSES

IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)


def build_augmentation():
    return models.Sequential(
        [
            layers.RandomRotation(0.03),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.05, 0.05),
        ],
        name="augmentation",
    )


def build_mobilenet_preprocessing():
    return layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")


def load_base(trainable=False, weights="imagenet"):
    base = tf.keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE, include_top=False, weights=weights
    )
    base.trainable = trainable
    return base


def build_transfer_model(learning_rate=1e-3, dropout=0.3, weights="imagenet"):
    base = load_base(trainable=False, weights=weights)

    inputs = layers.Input(shape=INPUT_SHAPE)
    x = build_augmentation()(inputs)
    x = build_mobilenet_preprocessing()(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax")(x)

    model = models.Model(inputs, outputs, name="mobilenetv2_transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_base(model):
    return next(
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.Model) and layer.name.startswith("mobilenetv2")
    )


def count_parameters(model):
    trainable = sum(int(tf.size(w)) for w in model.trainable_weights)
    non_trainable = sum(int(tf.size(w)) for w in model.non_trainable_weights)
    return trainable, non_trainable


def unfreeze_last_layers(model, num_layers=30, learning_rate=1e-4):
    base = get_base(model)
    base.trainable = True
    for layer in base.layers[:-num_layers]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model