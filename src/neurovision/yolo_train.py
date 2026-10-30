"""Part 3 helpers: YOLOv8 training configuration, timed runs and training-curve diagnostics."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

SEED = 42

# Training hyperparameters (justified in part3_03_training.ipynb)
TRAIN_DEFAULTS = {
    "epochs": 50,          # upper bound: early stopping (patience) usually ends the run sooner
    "imgsz": 640,          # multiple of 32; set from the most common image size found in discovery
    "batch": 16,           # limited by GPU memory; lower it on an out-of-memory error
    "patience": 10,        # stop if the validation fitness does not improve for 10 epochs
    "optimizer": "AdamW",  # stable for fine-tuning a pre-trained model on a small dataset
    "lr0": 0.001,          # initial learning rate (low: we adapt a pre-trained network)
    "lrf": 0.01,           # final learning rate = lr0 * lrf
    "seed": SEED,
    "deterministic": True,
    "workers": 8,          # use 0 on Windows if the DataLoader hangs
    "plots": True,
}

# Augmentation adapted to MRI. Applied by YOLOv8 to the training set only.
AUGMENTATION = {
    "hsv_h": 0.0,          # no hue change: grayscale images
    "hsv_s": 0.0,          # no saturation change: grayscale images
    "hsv_v": 0.3,          # brightness variation (scanner differences)
    "degrees": 10.0,       # light rotation only
    "translate": 0.1,
    "scale": 0.3,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,         # no horizontal flip: brain side carries information
    "mosaic": 0.5,         # applied to half of the batches; see the notebook for the reasoning
    "close_mosaic": 10,    # last 10 epochs without mosaic: finish on natural images
    "mixup": 0.0,
}

HISTORY_COLUMNS = {
    "box": ("train/box_loss", "val/box_loss"),
    "cls": ("train/cls_loss", "val/cls_loss"),
    "dfl": ("train/dfl_loss", "val/dfl_loss"),
}


def check_environment() -> dict:
    """Library versions and GPU availability, without failing when something is missing."""
    info: dict = {}
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        info["ultralytics"] = "NOT INSTALLED (uv add ultralytics)"
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["gpu"] = torch.cuda.get_device_name(0) if info["cuda_available"] else None
    except ImportError:
        info.update(torch="NOT INSTALLED", cuda_available=False, gpu=None)
    return info


def train_model(model_name: str, data_yaml: str | Path, name: str, project: str | Path, **overrides) -> dict:
    """Train one YOLOv8 model and return everything needed to log and compare the run.

    `overrides` replace any value of TRAIN_DEFAULTS / AUGMENTATION (one change at a time
    makes a clean comparison between runs).
    """
    from ultralytics import YOLO

    params = {**TRAIN_DEFAULTS, **AUGMENTATION, **overrides}
    model = YOLO(model_name)  # COCO pre-trained weights: transfer learning

    start = time.perf_counter()
    results = model.train(
        data=str(Path(data_yaml).resolve()),
        project=str(Path(project).resolve()),
        name=name,
        exist_ok=False,  # a new folder per run (name, name2, ...)
        **params,
    )
    duration = time.perf_counter() - start

    save_dir = Path(model.trainer.save_dir)
    return {
        "name": name,
        "model": model_name,
        "params": params,
        "duration_s": duration,
        "save_dir": save_dir,
        "best": save_dir / "weights" / "best.pt",
        "last": save_dir / "weights" / "last.pt",
        "metrics": dict(getattr(results, "results_dict", {}) or {}),
    }


def save_run_summary(run: dict, path: str | Path) -> Path:
    """JSON summary of a run (parameters, duration, paths, final metrics)."""
    summary = {k: (str(v) if isinstance(v, Path) else v) for k, v in run.items()}
    summary["duration_min"] = round(run["duration_s"] / 60, 2)
    path = Path(path)
    path.write_text(json.dumps(summary, indent=2, default=str))
    return path


def load_history(save_dir: str | Path) -> pd.DataFrame:
    """results.csv written by Ultralytics, with clean column names."""
    history = pd.read_csv(Path(save_dir) / "results.csv")
    history.columns = [c.strip() for c in history.columns]
    return history


def overfitting_report(history: pd.DataFrame, tolerance: float = 0.05) -> dict:
    """Simple overfitting diagnosis from the box + class losses.

    Overfitting signal: the validation loss rose by more than `tolerance` above its minimum
    while the training loss kept decreasing.
    """
    train_loss = history["train/box_loss"] + history["train/cls_loss"]
    val_loss = history["val/box_loss"] + history["val/cls_loss"]
    best_val_idx = int(val_loss.idxmin())
    after = slice(best_val_idx, None)

    val_rise = float(val_loss.iloc[-1] / val_loss.min() - 1)
    train_still_dropping = bool(train_loss.iloc[-1] < train_loss.iloc[after].iloc[0])
    overfitting = val_rise > tolerance and train_still_dropping

    report = {
        "epochs_run": int(len(history)),
        "epoch_lowest_val_loss": int(history["epoch"].iloc[best_val_idx]),
        "final_train_loss": round(float(train_loss.iloc[-1]), 4),
        "final_val_loss": round(float(val_loss.iloc[-1]), 4),
        "val_loss_rise_after_minimum": f"{val_rise:.1%}",
        "overfitting_suspected": overfitting,
    }
    recall_col = "metrics/recall(B)"
    if recall_col in history:
        report["best_recall"] = round(float(history[recall_col].max()), 4)
        report["epoch_best_recall"] = int(history["epoch"].iloc[int(history[recall_col].idxmax())])
    map_col = "metrics/mAP50-95(B)"
    if map_col in history:
        report["best_mAP50-95"] = round(float(history[map_col].max()), 4)
        report["epoch_best_mAP50-95"] = int(history["epoch"].iloc[int(history[map_col].idxmax())])
    return report
