from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_image_name(image_name: str) -> str:
    return Path(image_name).name


def list_images(images_dir: Path) -> list[str]:
    if not images_dir.exists():
        return []
    image_names = [
        p.name for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_names)


def image_path(images_dir: Path, image_name: str) -> Path:
    safe_name = safe_image_name(image_name)
    path = images_dir / safe_name
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Image not found: {safe_name}")
    return path


def load_image_rgb(image_path_value: Path) -> np.ndarray:
    with Image.open(image_path_value) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def load_image_size(image_path_value: Path) -> tuple[int, int]:
    with Image.open(image_path_value) as img:
        width, height = img.size
    return height, width


def load_binary_mask(mask_path: Path, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(mask_path) as img:
        arr = np.array(img.convert("L"), dtype=np.uint8)

    if expected_shape is not None and tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(
            f"Mask shape mismatch for {mask_path.name}: got {arr.shape}, expected {expected_shape}"
        )
    return arr > 0


def save_binary_mask(mask_bool: np.ndarray, mask_path: Path) -> None:
    ensure_dir(mask_path.parent)
    arr = (mask_bool.astype(np.uint8)) * 255
    Image.fromarray(arr, mode="L").save(mask_path)


def sanitize_class_name(class_name: str) -> str:
    class_name = class_name.strip()
    if not class_name:
        return "class"
    cleaned = []
    for ch in class_name:
        if ch.isalnum() or ch in {"_", "-", "."}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned)


def class_id_from_entry(class_entry: Any) -> int | None:
    if isinstance(class_entry, dict):
        raw = class_entry.get("id")
    else:
        raw = class_entry
    try:
        class_id = int(raw)
    except (TypeError, ValueError):
        return None
    if class_id <= 0:
        return None
    return class_id


def mask_filename_for_class_id(image_stem: str, class_id: int) -> str:
    return f"{image_stem}__cid_{int(class_id)}.png"


def frozen_mask_filename_for_class_id(image_stem: str, class_id: int) -> str:
    return f"{image_stem}__cid_{int(class_id)}__frozen.png"


def mask_filename(image_stem: str, class_name: str) -> str:
    return f"{image_stem}__{sanitize_class_name(class_name)}.png"


def iter_existing_mask_paths(masks_dir: Path, image_stem: str, classes: Iterable[Any]) -> list[Path]:
    paths = []
    for class_entry in classes:
        class_id = class_id_from_entry(class_entry)
        if class_id is not None:
            mask_path = masks_dir / mask_filename_for_class_id(image_stem, class_id)
            if mask_path.exists():
                paths.append(mask_path)
                continue
            if isinstance(class_entry, dict):
                legacy_name = str(class_entry.get("name", "")).strip()
                if legacy_name:
                    legacy_path = masks_dir / mask_filename(image_stem, legacy_name)
                    if legacy_path.exists():
                        paths.append(legacy_path)
            continue
        class_name = str(class_entry).strip()
        if not class_name:
            continue
        legacy_path = masks_dir / mask_filename(image_stem, class_name)
        if legacy_path.exists():
            paths.append(legacy_path)
    return paths
