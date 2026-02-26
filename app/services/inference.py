"""Inference pipeline for trained segmentation models."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from PIL import Image
from skimage.io import imread

from .features import ensure_rgb, extract_feature_stack, flatten_features
from .schemas import FeatureConfig


@dataclass
class InferenceImageResult:
    """Per-image outputs from an analysis run."""

    input_image: str
    mask_path: str
    overlay_path: str
    summary: dict[str, Any]


@dataclass
class InferenceRunResult:
    """Aggregate outputs from an analysis run."""

    output_dir: str
    items: list[InferenceImageResult]
    summary: dict[str, Any]
    warnings: list[str]


def _log(log_fn: Callable[[str], None] | None, message: str) -> None:
    if log_fn is not None:
        log_fn(message)


def _utc_now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _slugify(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return normalized or "image"


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    rgb = ensure_rgb(image)
    if np.issubdtype(rgb.dtype, np.floating):
        clipped = np.clip(rgb, 0.0, 1.0)
        return (clipped * 255.0).astype(np.uint8)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _build_palette(class_indices: list[int], seed: int) -> dict[int, tuple[int, int, int]]:
    rng = np.random.default_rng(seed)
    palette: dict[int, tuple[int, int, int]] = {0: (0, 0, 0)}
    for class_index in class_indices:
        if class_index == 0:
            continue
        color = tuple(int(v) for v in rng.integers(35, 255, size=3))
        palette[class_index] = color
    return palette


def _colorize_mask(mask: np.ndarray, palette: dict[int, tuple[int, int, int]]) -> np.ndarray:
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_index, color in palette.items():
        color_mask[mask == class_index] = color
    return color_mask


def _save_mask(mask: np.ndarray, path: Path) -> None:
    max_value = int(mask.max()) if mask.size else 0
    if max_value <= 255:
        Image.fromarray(mask.astype(np.uint8), mode="L").save(path)
    else:
        Image.fromarray(mask.astype(np.uint16), mode="I;16").save(path)


def _class_count_summary(
    mask: np.ndarray,
    class_name_by_idx: dict[int, str],
) -> dict[str, dict[str, float | int]]:
    total = int(mask.size) if mask.size else 1
    unique, counts = np.unique(mask, return_counts=True)
    summary: dict[str, dict[str, float | int]] = {}
    for class_idx, count in zip(unique.tolist(), counts.tolist()):
        name = class_name_by_idx.get(int(class_idx), f"class_{class_idx}")
        summary[name] = {
            "pixels": int(count),
            "percent": round((float(count) / total) * 100.0, 3),
        }
    return summary


def run_analysis(
    *,
    run_id: int,
    model_record: dict[str, Any],
    image_paths: list[str],
    runs_dir: Path,
    base_dir: Path,
    random_seed: int = 42,
    log_fn: Callable[[str], None] | None = None,
) -> InferenceRunResult:
    """Run segmentation inference on selected images."""
    model_path = Path(str(model_record["model_path"])).resolve()
    metadata_path = Path(str(model_record["metadata_path"])).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")

    _log(log_fn, f"Loading model from {model_path}")
    classifier = joblib.load(model_path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    feature_config = FeatureConfig.from_dict(metadata.get("feature_config", {}))
    classes = metadata.get("classes", [])
    if not classes:
        raise ValueError("Model metadata has no class definitions.")
    class_name_by_idx = {
        int(entry["class_index"]): str(entry["name"]) for entry in classes
    }
    class_indices = sorted(class_name_by_idx.keys())

    run_dir = runs_dir / f"run_{run_id}_{_utc_now_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _log(log_fn, f"Writing outputs to {run_dir}")

    palette = _build_palette(class_indices, seed=random_seed + int(run_id))
    warnings: list[str] = []
    results: list[InferenceImageResult] = []

    aggregate_counts: dict[str, int] = {}
    total_pixels = 0

    for image_idx, image_path_raw in enumerate(image_paths, start=1):
        image_path = Path(image_path_raw).expanduser().resolve()
        if not image_path.exists():
            warnings.append(f"Image not found and skipped: {image_path}")
            continue

        _log(log_fn, f"Analyzing image {image_idx}/{len(image_paths)}: {image_path.name}")
        image = _to_uint8_rgb(imread(str(image_path)))
        feature_stack, _ = extract_feature_stack(image, feature_config)
        X = flatten_features(feature_stack)
        predicted = classifier.predict(X).reshape(image.shape[0], image.shape[1]).astype(np.int32)

        base_name = f"{image_idx:03d}_{_slugify(image_path.stem)}"
        mask_path = run_dir / f"{base_name}_mask.png"
        overlay_path = run_dir / f"{base_name}_overlay.png"

        _save_mask(predicted, mask_path)
        color_mask = _colorize_mask(predicted, palette)
        overlay = image.copy()
        foreground = predicted > 0
        overlay[foreground] = (
            0.55 * image[foreground] + 0.45 * color_mask[foreground]
        ).astype(np.uint8)
        Image.fromarray(overlay).save(overlay_path)

        summary = _class_count_summary(predicted, class_name_by_idx)
        for class_name, class_values in summary.items():
            aggregate_counts[class_name] = (
                aggregate_counts.get(class_name, 0) + int(class_values["pixels"])
            )
        total_pixels += int(predicted.size)

        try:
            mask_rel = mask_path.relative_to(base_dir).as_posix()
        except ValueError:
            mask_rel = str(mask_path)
        try:
            overlay_rel = overlay_path.relative_to(base_dir).as_posix()
        except ValueError:
            overlay_rel = str(overlay_path)

        results.append(
            InferenceImageResult(
                input_image=str(image_path),
                mask_path=mask_rel,
                overlay_path=overlay_rel,
                summary=summary,
            )
        )

    aggregate_percent = {
        class_name: round((count / total_pixels) * 100.0, 3) if total_pixels else 0.0
        for class_name, count in aggregate_counts.items()
    }

    try:
        run_dir_rel = run_dir.relative_to(base_dir).as_posix()
    except ValueError:
        run_dir_rel = str(run_dir)

    run_summary = {
        "images_requested": len(image_paths),
        "images_processed": len(results),
        "total_pixels": total_pixels,
        "counts_by_class": aggregate_counts,
        "percent_by_class": aggregate_percent,
    }
    return InferenceRunResult(
        output_dir=run_dir_rel,
        items=results,
        summary=run_summary,
        warnings=warnings,
    )

