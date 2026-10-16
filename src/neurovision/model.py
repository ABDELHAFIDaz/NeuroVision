import tensorflow as tf
from tensorflow.keras import layers, models

from neurovision.data import CLASSES


def build_model(input_shape=(224, 224, 3), learning_rate=1e-3, dropout=0.5):
    augmentation = models.Sequential(
        [
            layers.RandomRotation(0.03),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.05, 0.05),
        ],
        name="augmentation",
    )

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Rescaling(1.0 / 255),
            augmentation,
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(len(CLASSES), activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model