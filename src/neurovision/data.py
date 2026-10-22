from __future__ import annotations

import hashlib
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

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


def remove_duplicates(files: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    drop = set()
    for group in find_duplicates(files):
        if len({label for _, label in group}) > 1:
            drop.update(p for p, _ in group)
        else:
            drop.update(p for p, _ in group[1:])
    return [(p, label) for p, label in files if p not in drop]


def load_dataset(files: list[tuple[Path, str]], size: tuple[int, int] = (224, 224), seed: int = 42):
    items = list(files)
    random.Random(seed).shuffle(items)

    images, labels, paths = [], [], []
    for path, label in items:
        try:
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError("unreadable image")
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            logger.warning("Skipping %s: %s", path, exc)
            continue
        images.append(img)
        labels.append(label)
        paths.append(str(path))

    return np.stack(images), np.array(labels), np.array(paths)


def split_fingerprint(paths, raw_dir) -> str:
    relative = sorted(str(Path(p).relative_to(raw_dir)) for p in paths)
    return hashlib.md5("\n".join(relative).encode()).hexdigest()[:12]