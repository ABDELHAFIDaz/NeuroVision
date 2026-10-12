from __future__ import annotations

import hashlib
import logging
from collections import Counter, defaultdict
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
VALID_EXTENSIONS = {".jpeg", ".jpg", ".bmp", ".png"}


def scan_dataset(raw_dir: str | Path) -> dict:
    
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {raw_dir}")

    valid, bad_ext = [], []
    by_class = {}
    for class_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        label = class_dir.name
        files = [f for f in sorted(class_dir.rglob("*")) if f.is_file()]
        by_class[label] = len(files)
        for f in files:
            if f.suffix.lower() in VALID_EXTENSIONS:
                valid.append((f, label))
            else:
                bad_ext.append(f)
    return {"valid": valid, "bad_ext": bad_ext, "by_class": by_class}


def is_readable(path: str | Path) -> bool:
    try:
        img = cv2.imread(str(path))
        return img is not None and img.size > 0
    except Exception as exc:  # corrupted / unsupported file
        logger.warning("Cannot read %s: %s", path, exc)
        return False


def split_readable(valid: list[tuple[Path, str]]):
    ok, corrupted = [], []
    for path, label in valid:
        (ok if is_readable(path) else corrupted).append((path, label))
    return ok, corrupted


def file_hash(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def find_duplicates(files: list[tuple[Path, str]]) -> list[list[tuple[Path, str]]]:
    groups = defaultdict(list)
    for path, label in files:
        groups[file_hash(path)].append((path, label))
    return [g for g in groups.values() if len(g) > 1]


def class_counts(files: list[tuple[Path, str]]) -> Counter:
    return Counter(label for _, label in files)