import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from neurovision.data import CLASSES, load_dataset, remove_duplicates, scan_dataset, split_readable

encoder = LabelEncoder()
encoder.fit(CLASSES)


def encode_labels(labels):
    return to_categorical(encoder.transform(labels), num_classes=len(CLASSES))


def make_dataset(images, labels, batch_size=32, training=False):
    ds = tf.data.Dataset.from_tensor_slices((images, encode_labels(labels)))
    ds = ds.cache()
    if training:
        ds = ds.shuffle(len(images), seed=42)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def prepare_data(raw_dir, size=(224, 224), seed=42):
    scan = scan_dataset(raw_dir)
    readable, _ = split_readable(scan["valid"])
    files = remove_duplicates(readable)
    X, y, paths = load_dataset(files, size=size, seed=seed)

    X_train, X_temp, y_train, y_temp, p_train, p_temp = train_test_split(
        X, y, paths, test_size=0.30, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test, p_val, p_test = train_test_split(
        X_temp, y_temp, p_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )
    return {
        "train": (X_train, y_train, p_train),
        "val": (X_val, y_val, p_val),
        "test": (X_test, y_test, p_test),
    }