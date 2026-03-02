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
from .postprocess.graph_smoothing import (
    GraphSmoothConfig,
    compute_area_change,
    compute_area_percentages,
    compute_flip_stats,
    graph_energy_smooth,
)
from .schemas import FeatureConfig

EPS = 1e-8


@dataclass
class InferenceImageResult:
    """Per-image outputs from an analysis run."""

    input_image: str
    mask_path: str
    overlay_path: str
    summary: dict[str, Any]
    raw_mask_path: str | None = None
    conf_path: str | None = None
    raw_overlay_path: str | None = None
    refined_mask_path: str | None = None
    refined_overlay_path: str | None = None
    flip_stats: dict[str, Any] | None = None
    area_raw: dict[str, Any] | None = None
    area_refined: dict[str, Any] | None = None
    area_delta: dict[str, Any] | None = None
    postprocess_applied: bool = False


@dataclass
class InferenceRunResult:
    """Aggregate outputs from an analysis run."""

    output_dir: str
    items: list[InferenceImageResult]
    summary: dict[str, Any]
    warnings: list[str]
    postprocess_enabled: bool = False
    postprocess_config: dict[str, Any] | None = None
    flip_stats: dict[str, Any] | None = None
    area_raw: dict[str, Any] | None = None
    area_refined: dict[str, Any] | None = None
    area_delta: dict[str, Any] | None = None


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


def _save_overlay(
    image_rgb: np.ndarray,
    label_mask: np.ndarray,
    palette: dict[int, tuple[int, int, int]],
    path: Path,
) -> None:
    color_mask = _colorize_mask(label_mask, palette)
    overlay = image_rgb.copy()
    foreground = label_mask > 0
    overlay[foreground] = (0.55 * image_rgb[foreground] + 0.45 * color_mask[foreground]).astype(np.uint8)
    Image.fromarray(overlay).save(path)


def _save_mask(mask: np.ndarray, path: Path) -> None:
    max_value = int(mask.max()) if mask.size else 0
    if max_value <= 255:
        Image.fromarray(mask.astype(np.uint8), mode="L").save(path)
    else:
        Image.fromarray(mask.astype(np.uint16), mode="I;16").save(path)


def _save_confidence(confidence: np.ndarray, path: Path) -> None:
    encoded = np.clip(np.rint(confidence * 255.0), 0.0, 255.0).astype(np.uint8)
    Image.fromarray(encoded, mode="L").save(path)


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


def _safe_relative(path_value: Path, base_dir: Path) -> str:
    try:
        return path_value.relative_to(base_dir).as_posix()
    except ValueError:
        return str(path_value)


def _class_metadata(metadata: dict[str, Any]) -> tuple[np.ndarray, list[str], dict[int, str]]:
    classes = metadata.get("classes", [])
    if not classes:
        raise ValueError("Model metadata has no class definitions.")
    sorted_classes = sorted(classes, key=lambda entry: int(entry.get("class_index", 0)))
    class_values = np.array([int(entry["class_index"]) for entry in sorted_classes], dtype=np.int32)
    class_names = [str(entry.get("name", f"class_{idx}")) for idx, entry in enumerate(sorted_classes)]
    class_name_by_idx = {
        int(entry["class_index"]): str(entry.get("name", f"class_{entry.get('class_index')}"))
        for entry in sorted_classes
    }
    return class_values, class_names, class_name_by_idx


def _predict_proba_aligned(
    classifier: Any,
    X: np.ndarray,
    class_values: np.ndarray,
) -> np.ndarray:
    model_proba = classifier.predict_proba(X)
    if not isinstance(model_proba, np.ndarray):
        raise TypeError("Expected classifier.predict_proba to return a numpy array.")
    model_classes = np.asarray(getattr(classifier, "classes_", []))
    index_by_class = {int(class_value): idx for idx, class_value in enumerate(class_values.tolist())}

    aligned = np.zeros((X.shape[0], class_values.size), dtype=np.float32)
    for source_col, source_class in enumerate(model_classes.tolist()):
        try:
            class_value = int(source_class)
        except (TypeError, ValueError):
            continue
        target_col = index_by_class.get(class_value)
        if target_col is None:
            continue
        if source_col >= model_proba.shape[1]:
            continue
        aligned[:, target_col] = model_proba[:, source_col].astype(np.float32)

    row_sums = aligned.sum(axis=1, keepdims=True)
    missing_rows = row_sums[:, 0] <= EPS
    if np.any(missing_rows):
        predicted = classifier.predict(X)
        for row_idx in np.where(missing_rows)[0]:
            class_value = int(predicted[row_idx]) if row_idx < len(predicted) else int(class_values[0])
            class_pos = index_by_class.get(class_value, 0)
            aligned[row_idx, :] = 0.0
            aligned[row_idx, class_pos] = 1.0
        row_sums = aligned.sum(axis=1, keepdims=True)
    aligned /= np.maximum(row_sums, EPS)
    return aligned


def _aggregate_area_percentages(
    class_names: list[str],
    counts_by_class: dict[str, int],
) -> dict[str, Any]:
    total_pixels = int(sum(max(0, int(counts_by_class.get(name, 0))) for name in class_names))
    by_class: dict[str, dict[str, float | int]] = {}
    for class_name in class_names:
        pixels = int(max(0, int(counts_by_class.get(class_name, 0))))
        by_class[class_name] = {
            "pixels": pixels,
            "percent": (float(pixels) / float(total_pixels) * 100.0) if total_pixels else 0.0,
        }
    return {
        "total_pixels": total_pixels,
        "by_class": by_class,
    }


def _aggregate_flip_stats(
    class_names: list[str],
    *,
    total_pixels: int,
    total_flip_pixels: int,
    raw_pixels_by_class: dict[str, int],
    refined_pixels_by_class: dict[str, int],
    flipped_out_by_class: dict[str, int],
    flipped_in_by_class: dict[str, int],
) -> dict[str, Any]:
    by_class: dict[str, dict[str, float | int]] = {}
    for class_name in class_names:
        raw_pixels = int(raw_pixels_by_class.get(class_name, 0))
        refined_pixels = int(refined_pixels_by_class.get(class_name, 0))
        flipped_out = int(flipped_out_by_class.get(class_name, 0))
        flipped_in = int(flipped_in_by_class.get(class_name, 0))
        by_class[class_name] = {
            "raw_pixels": raw_pixels,
            "refined_pixels": refined_pixels,
            "flipped_out_pixels": flipped_out,
            "flipped_in_pixels": flipped_in,
            "flipped_out_rate": (float(flipped_out) / float(raw_pixels)) if raw_pixels else 0.0,
            "flipped_in_rate": (float(flipped_in) / float(refined_pixels)) if refined_pixels else 0.0,
        }
    return {
        "total_pixels": int(total_pixels),
        "flip_pixels": int(total_flip_pixels),
        "flip_rate": (float(total_flip_pixels) / float(total_pixels)) if total_pixels else 0.0,
        "by_class": by_class,
    }


def run_analysis(
    *,
    run_id: int,
    model_record: dict[str, Any],
    image_paths: list[str],
    runs_dir: Path,
    base_dir: Path,
    random_seed: int = 42,
    log_fn: Callable[[str], None] | None = None,
    postprocess_config: dict[str, Any] | None = None,
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
    class_values, class_names, class_name_by_idx = _class_metadata(metadata)
    class_indices = class_values.tolist()
    smooth_config = GraphSmoothConfig.from_dict(postprocess_config)
    smoothing_enabled = bool(smooth_config.enabled)

    run_dir = runs_dir / f"run_{run_id}_{_utc_now_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _log(log_fn, f"Writing outputs to {run_dir}")

    palette = _build_palette(class_indices, seed=random_seed + int(run_id))
    warnings: list[str] = []
    results: list[InferenceImageResult] = []

    aggregate_raw_counts: dict[str, int] = {class_name: 0 for class_name in class_names}
    aggregate_refined_counts: dict[str, int] = {class_name: 0 for class_name in class_names}
    aggregate_raw_pixels_for_flip: dict[str, int] = {class_name: 0 for class_name in class_names}
    aggregate_refined_pixels_for_flip: dict[str, int] = {class_name: 0 for class_name in class_names}
    aggregate_flipped_out: dict[str, int] = {class_name: 0 for class_name in class_names}
    aggregate_flipped_in: dict[str, int] = {class_name: 0 for class_name in class_names}
    total_raw_pixels = 0
    total_flip_pixels = 0

    for image_idx, image_path_raw in enumerate(image_paths, start=1):
        image_path = Path(image_path_raw).expanduser().resolve()
        if not image_path.exists():
            warnings.append(f"Image not found and skipped: {image_path}")
            continue

        _log(log_fn, f"Analyzing image {image_idx}/{len(image_paths)}: {image_path.name}")
        image = _to_uint8_rgb(imread(str(image_path)))
        feature_stack, _ = extract_feature_stack(image, feature_config)
        X = flatten_features(feature_stack)
        proba_flat = _predict_proba_aligned(classifier, X, class_values)
        proba_raw = proba_flat.reshape(image.shape[0], image.shape[1], class_values.size)
        label_raw_pos = np.argmax(proba_raw, axis=2).astype(np.int32)
        label_raw = class_values[label_raw_pos]
        conf_raw = np.max(proba_raw, axis=2).astype(np.float32, copy=False)

        base_name = f"{image_idx:03d}_{_slugify(image_path.stem)}"
        raw_mask_path = run_dir / f"{base_name}_raw_mask.png"
        pred_mask_path = run_dir / f"{base_name}__pred.png"
        conf_path = run_dir / f"{base_name}__conf.png"
        output_meta_path = run_dir / f"{base_name}__meta.json"
        raw_overlay_path = run_dir / f"{base_name}_raw_overlay.png"

        _save_mask(label_raw, raw_mask_path)
        _save_mask(label_raw, pred_mask_path)
        _save_confidence(conf_raw, conf_path)
        _save_overlay(image, label_raw, palette, raw_overlay_path)

        postprocess_applied = False
        label_refined = label_raw
        label_refined_pos = label_raw_pos
        refined_mask_path: Path | None = None
        refined_overlay_path: Path | None = None
        refined_debug: dict[str, Any] | None = None
        if smoothing_enabled:
            postprocess_applied = True
            proba_refined, label_refined_pos, refined_debug = graph_energy_smooth(
                image_rgb=image,
                proba=proba_raw,
                classes=class_names,
                config=smooth_config,
            )
            label_refined = class_values[label_refined_pos]
            refined_mask_path = run_dir / f"{base_name}_refined_mask.png"
            refined_overlay_path = run_dir / f"{base_name}_refined_overlay.png"
            _save_mask(label_refined, refined_mask_path)
            _save_overlay(image, label_refined, palette, refined_overlay_path)

            # Keep arrays referenced for easier debugging in run summary.
            del proba_refined

        raw_summary = _class_count_summary(label_raw, class_name_by_idx)
        refined_summary = _class_count_summary(label_refined, class_name_by_idx)
        area_raw = compute_area_percentages(label_raw_pos, class_names)
        area_refined = compute_area_percentages(label_refined_pos, class_names)
        area_delta = compute_area_change(area_raw, area_refined)
        flip_stats = compute_flip_stats(label_raw_pos, label_refined_pos, class_names)

        for class_name in class_names:
            aggregate_raw_counts[class_name] += int(raw_summary.get(class_name, {}).get("pixels", 0))
            aggregate_refined_counts[class_name] += int(refined_summary.get(class_name, {}).get("pixels", 0))

            class_flip = flip_stats.get("by_class", {}).get(class_name, {})
            aggregate_raw_pixels_for_flip[class_name] += int(class_flip.get("raw_pixels", 0))
            aggregate_refined_pixels_for_flip[class_name] += int(class_flip.get("refined_pixels", 0))
            aggregate_flipped_out[class_name] += int(class_flip.get("flipped_out_pixels", 0))
            aggregate_flipped_in[class_name] += int(class_flip.get("flipped_in_pixels", 0))

        total_raw_pixels += int(flip_stats.get("total_pixels", 0))
        total_flip_pixels += int(flip_stats.get("flip_pixels", 0))

        item_summary: dict[str, Any] = {
            "raw": raw_summary,
            "refined": refined_summary if postprocess_applied else None,
            "postprocess_applied": postprocess_applied,
            "flip_stats": flip_stats if postprocess_applied else None,
            "area_raw": area_raw,
            "area_refined": area_refined if postprocess_applied else None,
            "area_delta": area_delta if postprocess_applied else None,
            "refined_debug": refined_debug if postprocess_applied else None,
        }

        raw_mask_rel = _safe_relative(raw_mask_path, base_dir)
        pred_mask_rel = _safe_relative(pred_mask_path, base_dir)
        conf_rel = _safe_relative(conf_path, base_dir)
        raw_overlay_rel = _safe_relative(raw_overlay_path, base_dir)
        refined_mask_rel = _safe_relative(refined_mask_path, base_dir) if refined_mask_path is not None else None
        refined_overlay_rel = (
            _safe_relative(refined_overlay_path, base_dir)
            if refined_overlay_path is not None
            else None
        )
        output_meta_rel = _safe_relative(output_meta_path, base_dir)
        output_meta = {
            "input_image": str(image_path),
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
            "num_classes": int(class_values.size),
            "class_index_to_name": {
                str(int(class_value)): str(class_name_by_idx.get(int(class_value), f"class_{int(class_value)}"))
                for class_value in class_values.tolist()
            },
            "prediction_artifacts": {
                "raw_mask_path": raw_mask_rel,
                "pred_path": pred_mask_rel,
                "conf_path": conf_rel,
                "raw_overlay_path": raw_overlay_rel,
                "refined_mask_path": refined_mask_rel,
                "refined_overlay_path": refined_overlay_rel,
            },
            "encoding": {
                "pred_png": "class index map stored as uint8 or uint16",
                "conf_png": "uint8 where value == round(max_probability * 255)",
            },
            "transform": {
                "applied": False,
                "description": "Inference outputs are written at original image resolution.",
            },
        }
        with output_meta_path.open("w", encoding="utf-8") as handle:
            json.dump(output_meta, handle, indent=2)
        item_summary["output_meta_path"] = output_meta_rel
        item_summary["prediction_artifacts"] = output_meta.get("prediction_artifacts")

        primary_mask = refined_mask_rel if refined_mask_rel is not None else raw_mask_rel
        primary_overlay = refined_overlay_rel if refined_overlay_rel is not None else raw_overlay_rel

        results.append(
            InferenceImageResult(
                input_image=str(image_path),
                mask_path=primary_mask,
                overlay_path=primary_overlay,
                summary=item_summary,
                raw_mask_path=raw_mask_rel,
                conf_path=conf_rel,
                raw_overlay_path=raw_overlay_rel,
                refined_mask_path=refined_mask_rel,
                refined_overlay_path=refined_overlay_rel,
                flip_stats=flip_stats if postprocess_applied else None,
                area_raw=area_raw,
                area_refined=area_refined if postprocess_applied else None,
                area_delta=area_delta if postprocess_applied else None,
                postprocess_applied=postprocess_applied,
            )
        )

    run_area_raw = _aggregate_area_percentages(class_names, aggregate_raw_counts)
    run_area_refined = _aggregate_area_percentages(class_names, aggregate_refined_counts)
    run_area_delta = compute_area_change(run_area_raw, run_area_refined)
    run_flip_stats = _aggregate_flip_stats(
        class_names,
        total_pixels=total_raw_pixels,
        total_flip_pixels=total_flip_pixels,
        raw_pixels_by_class=aggregate_raw_pixels_for_flip,
        refined_pixels_by_class=aggregate_refined_pixels_for_flip,
        flipped_out_by_class=aggregate_flipped_out,
        flipped_in_by_class=aggregate_flipped_in,
    )

    run_dir_rel = _safe_relative(run_dir, base_dir)
    run_summary = {
        "images_requested": len(image_paths),
        "images_processed": len(results),
        "total_pixels": total_raw_pixels,
        "counts_by_class": aggregate_raw_counts,
        "percent_by_class": {
            class_name: run_area_raw["by_class"][class_name]["percent"]
            for class_name in class_names
        },
        "postprocess_enabled": smoothing_enabled,
        "postprocess_config": smooth_config.to_dict(),
        "counts_by_class_refined": aggregate_refined_counts if smoothing_enabled else None,
        "percent_by_class_refined": (
            {
                class_name: run_area_refined["by_class"][class_name]["percent"]
                for class_name in class_names
            }
            if smoothing_enabled
            else None
        ),
        "flip_stats": run_flip_stats if smoothing_enabled else None,
        "area_raw": run_area_raw,
        "area_refined": run_area_refined if smoothing_enabled else None,
        "area_delta": run_area_delta if smoothing_enabled else None,
    }

    return InferenceRunResult(
        output_dir=run_dir_rel,
        items=results,
        summary=run_summary,
        warnings=warnings,
        postprocess_enabled=smoothing_enabled,
        postprocess_config=smooth_config.to_dict(),
        flip_stats=run_flip_stats if smoothing_enabled else None,
        area_raw=run_area_raw,
        area_refined=run_area_refined if smoothing_enabled else None,
        area_delta=run_area_delta if smoothing_enabled else None,
    )
