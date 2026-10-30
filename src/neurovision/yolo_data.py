"""Part 3 helpers: inventory, cleaning, stratified split and YOLO export for brain tumor detection.

Everything works on a pandas "inventory" table (one row per file) so each cleaning decision
is a visible, countable filter instead of a silent deletion. Raw data is never modified.
"""
from __future__ import annotations

import hashlib
import logging
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

SEED = 42

# Confirmed from the data: folder name -> class id used inside the YOLO .txt files
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
FOLDER_TO_ID = {"glioma": 0, "meningioma": 1, "no tumor": 2, "notumor": 2, "pituitary": 3}

VALID_EXTENSIONS = {".jpeg", ".jpg", ".bmp", ".png"}
SIDE_SUFFIXES = {".txt", ".yaml", ".yml"}  # label / config files, not images
IGNORED_LABEL_FILES = {"classes.txt", "labels.txt", "notes.txt"}
BOX_COLORS = [(230, 57, 70), (42, 157, 143), (150, 150, 150), (69, 123, 157)]  # RGB, one per class

MIXED_POLICIES = ("keep", "drop_inconsistent", "drop_all_mixed")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- paths and labels
def find_label(img_path: str | Path) -> Path | None:
    """Label next to the image, or in a parallel `labels/` folder (images/ -> labels/)."""
    img_path = Path(img_path)
    same_dir = img_path.with_suffix(".txt")
    if same_dir.exists():
        return same_dir
    parts = list(img_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        parallel = Path(*parts).with_suffix(".txt")
        if parallel.exists():
            return parallel
    return None


def folder_class_id(img_path: str | Path, root: str | Path) -> int | None:
    """Class id deduced from the class folder (Glioma, No Tumor...) in the image path."""
    for part in Path(img_path).relative_to(root).parts[:-1]:
        key = part.lower().replace("_", " ").strip()
        if key in FOLDER_TO_ID:
            return FOLDER_TO_ID[key]
    return None


def read_boxes(label_path: str | Path):
    """Parse and validate a YOLO label file.

    Returns (status, reason, boxes) with status in {"ok", "empty", "invalid"} and
    boxes = [(class_id, xc, yc, w, h), ...] (normalized values).
    """
    try:
        lines = [l.strip() for l in Path(label_path).read_text().splitlines() if l.strip()]
    except Exception as exc:  # unreadable / binary file
        return "invalid", f"unreadable label file ({exc})", []
    if not lines:
        return "empty", "empty annotation", []

    boxes = []
    for n, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) != 5:
            return "invalid", f"line {n}: expected 5 values, got {len(parts)}", []
        try:
            cls = float(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            return "invalid", f"line {n}: non-numeric value", []
        if not cls.is_integer() or not 0 <= int(cls) < len(CLASS_NAMES):
            return "invalid", f"line {n}: invalid class id {parts[0]}", []
        if any(not 0.0 <= v <= 1.0 for v in coords):
            return "invalid", f"line {n}: coordinate outside [0, 1]", []
        if coords[2] <= 0 or coords[3] <= 0:
            return "invalid", f"line {n}: box with zero width or height", []
        boxes.append((int(cls), *coords))
    return "ok", "", boxes


# --------------------------------------------------------------------------- images
def is_readable(path: str | Path) -> bool:
    try:
        img = cv2.imread(str(path))
        return img is not None and img.size > 0
    except Exception as exc:  # corrupted / unsupported file
        logger.warning("Cannot read %s: %s", path, exc)
        return False


def file_hash(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def image_size(path: str | Path):
    """(width, height) without decoding the whole image; None if unreadable."""
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def draw_boxes(img_path: str | Path, label_path: str | Path | None = None) -> np.ndarray:
    """RGB image with boxes and class names drawn (normalized coords converted to pixels)."""
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    label_path = label_path or find_label(img_path)
    if label_path is None:
        return img
    _, _, boxes = read_boxes(label_path)
    for cls, xc, yc, bw, bh in boxes:
        x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
        x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        color = BOX_COLORS[cls]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, CLASS_NAMES[cls], (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return img


# --------------------------------------------------------------------------- inventory
def build_inventory(root: str | Path, check_readable: bool = True, compute_hash: bool = True) -> pd.DataFrame:
    """One row per non-label file under `root`, with everything needed to clean the dataset."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}")

    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() in SIDE_SUFFIXES:
            continue
        rel = p.relative_to(root)
        ext = p.suffix.lower()
        row = {
            "path": p, "rel": str(rel),
            "split": rel.parts[0] if len(rel.parts) > 1 else "",
            "folder_class": folder_class_id(p, root),
            "ext": ext, "ext_ok": ext in VALID_EXTENSIONS,
            "readable": None, "label_path": None, "label_status": None,
            "label_reason": "", "boxes": [], "classes": [], "n_boxes": 0, "hash": None,
        }
        if row["ext_ok"]:
            label = find_label(p)
            row["label_path"] = label
            if label is None:
                row["label_status"] = "missing"
            else:
                status, reason, boxes = read_boxes(label)
                row.update(label_status=status, label_reason=reason, boxes=boxes,
                           classes=[b[0] for b in boxes], n_boxes=len(boxes))
            if check_readable:
                row["readable"] = is_readable(p)
            if compute_hash and row["readable"] is not False:
                row["hash"] = file_hash(p)
        rows.append(row)
    return pd.DataFrame(rows)


def find_orphan_labels(root: str | Path, df: pd.DataFrame) -> list[Path]:
    """Label files that no image points to."""
    used = {Path(p) for p in df["label_path"].dropna()}
    return [t for t in sorted(Path(root).rglob("*.txt"))
            if t not in used and t.name.lower() not in IGNORED_LABEL_FILES]


# --------------------------------------------------------------------------- cleaning
def apply_cleaning(df: pd.DataFrame, mixed_policy: str = "drop_inconsistent") -> pd.DataFrame:
    """Add an `excluded` column: empty string = kept, otherwise the first reason found.

    mixed_policy (images whose boxes disagree with their class folder):
      keep               : keep everything
      drop_inconsistent  : drop images whose boxes do NOT include the folder class
      drop_all_mixed     : drop every image having a box of another class
    """
    if mixed_policy not in MIXED_POLICIES:
        raise ValueError(f"mixed_policy must be one of {MIXED_POLICIES}")
    df = df.copy()
    df["excluded"] = ""

    def mark(mask, reason):
        df.loc[mask & (df["excluded"] == ""), "excluded"] = reason

    mark(~df["ext_ok"], "bad_extension")
    mark(df["readable"] == False, "corrupted")  # noqa: E712  (None means "not checked")
    mark(df["label_status"] == "missing", "missing_label")
    mark(df["label_status"] == "invalid", "invalid_annotation")
    mark(df["label_status"] == "empty", "empty_annotation")
    mark(df["folder_class"].isna(), "unknown_class_folder")

    def inconsistent(row) -> bool:
        if mixed_policy == "keep" or pd.isna(row["folder_class"]):
            return False
        classes = set(row["classes"])
        if mixed_policy == "drop_inconsistent":
            return int(row["folder_class"]) not in classes
        return classes != {int(row["folder_class"])}

    mark(df.apply(inconsistent, axis=1).astype(bool), "inconsistent_class")

    # Byte-identical duplicates among the images still kept
    kept = df[df["excluded"] == ""]
    for _, group in kept.groupby("hash"):
        if len(group) < 2:
            continue
        if group["folder_class"].nunique() > 1:
            df.loc[group.index, "excluded"] = "cross_class_duplicate"
        else:
            df.loc[group.index[1:], "excluded"] = "duplicate"
    return df


# --------------------------------------------------------------------------- split and export
def stratified_split(df: pd.DataFrame, seed: int = SEED, ratios=(0.70, 0.15, 0.15)) -> dict[str, pd.DataFrame]:
    """Stratified train/valid/test split of the kept images (stratified on the class folder)."""
    train_r, val_r, test_r = ratios
    if abs(sum(ratios) - 1) > 1e-6:
        raise ValueError("ratios must sum to 1")
    clean = df[df["excluded"] == ""]
    train, temp = train_test_split(clean, test_size=1 - train_r,
                                   stratify=clean["folder_class"], random_state=seed)
    valid, test = train_test_split(temp, test_size=test_r / (val_r + test_r),
                                   stratify=temp["folder_class"], random_state=seed)
    return {"train": train.sort_index(), "valid": valid.sort_index(), "test": test.sort_index()}


def export_name(rel: str) -> str:
    """Unique file name built from the original relative path (avoids collisions between folders)."""
    p = Path(rel)
    return "__".join(p.with_suffix("").parts).replace(" ", "_") + p.suffix.lower()


def write_yolo_dataset(splits: dict[str, pd.DataFrame], out_dir: str | Path, overwrite: bool = False) -> Path:
    """Copy image + label pairs into out_dir/images/<split> and out_dir/labels/<split>."""
    out = Path(out_dir)
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} already exists (use overwrite=True to rebuild it)")
        shutil.rmtree(out)
    for name, part in splits.items():
        img_dir, lbl_dir = out / "images" / name, out / "labels" / name
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for row in part.itertuples():
            new_name = export_name(row.rel)
            shutil.copy2(row.path, img_dir / new_name)
            lines = [l.strip() for l in Path(row.label_path).read_text().splitlines() if l.strip()]
            (lbl_dir / Path(new_name).with_suffix(".txt").name).write_text("\n".join(lines) + "\n")
    return out


def write_data_yaml(out_dir: str | Path) -> Path:
    """data.yaml with train/val/test, nc and names. No `path` key: Ultralytics then uses the
    yaml's own folder, so the dataset stays portable (laptop -> Vast.ai)."""
    cfg = {
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    target = Path(out_dir) / "data.yaml"
    with open(target, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return target


def verify_yolo_dataset(out_dir: str | Path) -> pd.DataFrame:
    """Counts per split: images, labels, unmatched files, invalid labels, boxes per class."""
    out = Path(out_dir)
    rows = []
    for split in ("train", "valid", "test"):
        images = sorted((out / "images" / split).glob("*"))
        labels = sorted((out / "labels" / split).glob("*.txt"))
        img_stems, lbl_stems = {p.stem for p in images}, {p.stem for p in labels}
        invalid, per_class = 0, Counter()
        for label in labels:
            status, _, boxes = read_boxes(label)
            invalid += status != "ok"
            per_class.update(b[0] for b in boxes)
        rows.append({
            "split": split, "images": len(images), "labels": len(labels),
            "images_without_label": len(img_stems - lbl_stems),
            "labels_without_image": len(lbl_stems - img_stems),
            "invalid_labels": invalid,
            **{f"boxes_{n}": per_class.get(i, 0) for i, n in enumerate(CLASS_NAMES)},
        })
    return pd.DataFrame(rows).set_index("split")
