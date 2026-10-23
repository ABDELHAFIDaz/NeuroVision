
import os
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "NeuroVision"
DB_PATH = REPO_ROOT / "mlflow.db"
ARTIFACTS_DIR = REPO_ROOT / "mlartifacts"


def setup_mlflow(experiment=EXPERIMENT):
    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{DB_PATH}")
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=ARTIFACTS_DIR.as_uri())
    mlflow.set_experiment(experiment)
    return uri


def start_run(run_name, tags=None):
    if mlflow.active_run():
        mlflow.end_run()
    run = mlflow.start_run(run_name=run_name)
    if tags:
        mlflow.set_tags(tags)
    return run


class MlflowEpochLogger(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        if mlflow.active_run() and logs:
            mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch)


def log_training_summary(history, duration_s, model_path=None):
    h = history.history
    metrics = {"training_minutes": duration_s / 60}
    if h.get("val_accuracy"):
        metrics["best_val_accuracy"] = float(max(h["val_accuracy"]))
        mlflow.log_param("best_epoch", int(np.argmax(h["val_accuracy"])) + 1)
    if h.get("val_loss"):
        metrics["best_val_loss"] = float(min(h["val_loss"]))
    if h.get("accuracy"):
        metrics["final_train_accuracy"] = float(h["accuracy"][-1])
    mlflow.log_metrics(metrics)
    mlflow.log_param("epochs_run", len(h["loss"]))
    if model_path and Path(model_path).exists():
        mlflow.log_artifact(str(model_path), "model")


def log_evaluation(tag, model_path, test_loss, test_acc, report, files=()):
   
    mlflow.log_param("model_tag", tag)
    mlflow.log_metrics(
        {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
        }
    )
    for name, values in report.items():
        if isinstance(values, dict) and name not in ("macro avg", "weighted avg"):
            for key in ("precision", "recall", "f1-score"):
                mlflow.log_metric(f"{name}_{key.replace('-', '_')}", float(values[key]))
    for path in files:
        if Path(path).exists():
            mlflow.log_artifact(str(path), "figures")
    if Path(model_path).exists():
        mlflow.log_artifact(str(model_path), "model")
