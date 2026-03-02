"""Model training pipeline for Random-Forest image segmentation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from skimage.io import imread

from .coco import (
    annotations_by_image,
    build_mask,
    category_id_to_class_index,
    compute_sha256,
    load_coco,
    parse_categories,
    resolve_image_path,
)
from .features import ensure_rgb, extract_feature_stack, features_from_coordinates
from .schemas import FeatureConfig, TrainConfig


@dataclass
class TrainingResult:
    """Artifact and metric outputs from a training run."""

    classes: list[dict[str, Any]]
    feature_config: dict[str, Any]
    hyperparams: dict[str, Any]
    metrics: dict[str, Any]
    warnings: list[str]
    artifact_dir: str
    model_path: str
    metadata_path: str
    metadata: dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slugify(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return normalized or "model"


def _log(log_fn: Callable[[str], None] | None, message: str) -> None:
    if log_fn is not None:
        log_fn(message)


def _sample_coords(
    mask: np.ndarray,
    class_index: int,
    limit: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask == class_index)
    if coords.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    if coords.shape[0] > limit:
        selected = rng.choice(coords.shape[0], size=limit, replace=False)
        coords = coords[selected]
    return coords[:, 0], coords[:, 1]


def _compute_iou_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
    class_name_by_idx: dict[int, str],
) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pixel_accuracy = float((y_true == y_pred).mean()) if y_true.size else 0.0

    per_class_iou: dict[str, float | None] = {}
    for row_idx, label in enumerate(labels):
        tp = float(cm[row_idx, row_idx])
        fp = float(cm[:, row_idx].sum() - tp)
        fn = float(cm[row_idx, :].sum() - tp)
        denom = tp + fp + fn
        iou = None if denom == 0 else tp / denom
        name = class_name_by_idx.get(label, f"class_{label}")
        per_class_iou[name] = None if iou is None else round(iou, 5)

    return {
        "pixel_accuracy": round(pixel_accuracy, 5),
        "per_class_iou": per_class_iou,
    }


def train_model(
    *,
    model_name: str,
    dataset: dict[str, Any],
    feature_config: FeatureConfig,
    train_config: TrainConfig,
    models_dir: Path,
    code_version: str,
    log_fn: Callable[[str], None] | None = None,
) -> TrainingResult:
    """Train a RandomForest segmentation model from a dataset."""
    dataset_path = Path(str(dataset["dataset_path"])).resolve()
    coco_path = Path(str(dataset["coco_json_path"])).resolve()
    image_root = str(dataset["image_root"])

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    if not coco_path.exists():
        raise FileNotFoundError(f"COCO file not found: {coco_path}")

    _log(log_fn, f"Loading COCO annotations from {coco_path}")
    coco = load_coco(coco_path)
    categories = parse_categories(coco)
    category_to_class = category_id_to_class_index(categories)
    annotations_map = annotations_by_image(coco)
    images = list(coco.get("images", []))
    if not images:
        raise ValueError("No images found in COCO dataset.")

    class_entries: list[dict[str, Any]] = [{"class_index": 0, "category_id": 0, "name": "background"}]
    for class_index, category in enumerate(categories, start=1):
        class_entries.append(
            {
                "class_index": class_index,
                "category_id": int(category["id"]),
                "name": str(category["name"]),
            }
        )

    class_indices = [entry["class_index"] for entry in class_entries]
    class_name_by_idx = {entry["class_index"]: entry["name"] for entry in class_entries}

    image_ids = [int(image["id"]) for image in images]
    rng = np.random.default_rng(train_config.random_state)
    rng.shuffle(image_ids)

    val_ids: set[int] = set()
    if len(image_ids) > 1 and train_config.validation_split > 0:
        val_count = max(1, int(len(image_ids) * train_config.validation_split))
        val_ids = set(image_ids[:val_count])
    train_ids = set(image_ids) - val_ids
    if not train_ids:
        train_ids = set(image_ids)
        val_ids = set()

    _log(
        log_fn,
        (
            "Split images into "
            f"{len(train_ids)} train and {len(val_ids)} validation "
            f"(validation_split={train_config.validation_split})."
        ),
    )

    warnings: list[str] = []
    skipped_rle_total = 0
    used_rle_total = 0
    sampled_counts: dict[int, int] = {class_idx: 0 for class_idx in class_indices}
    val_sampled_counts: dict[int, int] = {class_idx: 0 for class_idx in class_indices}

    train_X_parts: list[np.ndarray] = []
    train_y_parts: list[np.ndarray] = []
    val_X_parts: list[np.ndarray] = []
    val_y_parts: list[np.ndarray] = []

    val_limit_per_class = max(500, train_config.max_samples_per_class // 3)

    for image_info in images:
        image_id = int(image_info["id"])
        subset = "val" if image_id in val_ids else "train"
        file_name = str(image_info["file_name"])
        image_path = resolve_image_path(dataset_path, image_root, file_name)
        if not image_path.exists():
            warnings.append(f"Image not found and skipped: {image_path}")
            continue

        image = ensure_rgb(imread(str(image_path)))
        h, w = image.shape[:2]

        image_annotations = annotations_map.get(image_id, [])
        mask, mask_warnings, skipped_rle, used_rle = build_mask(
            h,
            w,
            image_annotations,
            category_to_class,
            overlap_policy=train_config.overlap_policy,
            use_rle=train_config.use_rle,
        )
        skipped_rle_total += skipped_rle
        used_rle_total += used_rle
        warnings.extend(mask_warnings)

        feature_stack, _ = extract_feature_stack(image, feature_config)

        if subset == "train":
            for class_idx in class_indices:
                remaining = train_config.max_samples_per_class - sampled_counts[class_idx]
                if remaining <= 0:
                    continue
                rows, cols = _sample_coords(mask, class_idx, remaining, rng)
                if rows.size == 0:
                    continue
                train_X_parts.append(features_from_coordinates(feature_stack, rows, cols))
                train_y_parts.append(np.full(rows.shape[0], class_idx, dtype=np.int32))
                sampled_counts[class_idx] += int(rows.shape[0])
        else:
            for class_idx in class_indices:
                remaining = val_limit_per_class - val_sampled_counts[class_idx]
                if remaining <= 0:
                    continue
                rows, cols = _sample_coords(mask, class_idx, remaining, rng)
                if rows.size == 0:
                    continue
                val_X_parts.append(features_from_coordinates(feature_stack, rows, cols))
                val_y_parts.append(np.full(rows.shape[0], class_idx, dtype=np.int32))
                val_sampled_counts[class_idx] += int(rows.shape[0])

    if not train_X_parts:
        raise RuntimeError("No training pixels sampled. Check dataset paths and annotations.")

    train_X = np.concatenate(train_X_parts, axis=0)
    train_y = np.concatenate(train_y_parts, axis=0)
    shuffle_idx = rng.permutation(train_X.shape[0])
    train_X = train_X[shuffle_idx]
    train_y = train_y[shuffle_idx]

    class_weight: str | dict[int, float] | None
    if train_config.class_weight.strip().lower() in {"none", ""}:
        class_weight = None
    else:
        class_weight = train_config.class_weight

    _log(
        log_fn,
        (
            f"Training RandomForestClassifier on {train_X.shape[0]} sampled pixels "
            f"with {train_X.shape[1]} features."
        ),
    )
    classifier = RandomForestClassifier(
        n_estimators=train_config.n_estimators,
        max_depth=train_config.max_depth,
        min_samples_split=train_config.min_samples_split,
        min_samples_leaf=train_config.min_samples_leaf,
        class_weight=class_weight,
        random_state=train_config.random_state,
        n_jobs=1,
    )
    classifier.fit(train_X, train_y)
    _log(log_fn, "Training complete.")

    metrics: dict[str, Any] = {
        "train_image_count": len(train_ids),
        "validation_image_count": len(val_ids),
        "train_samples": int(train_X.shape[0]),
        "feature_count": int(train_X.shape[1]),
        "sampled_pixels_by_class": {
            class_name_by_idx[idx]: count for idx, count in sampled_counts.items()
        },
        "rle_annotations_used": used_rle_total,
        "skipped_rle_annotations": skipped_rle_total,
    }

    if val_X_parts:
        val_X = np.concatenate(val_X_parts, axis=0)
        val_y = np.concatenate(val_y_parts, axis=0)
        _log(log_fn, f"Running validation on {val_X.shape[0]} sampled pixels.")
        val_pred = classifier.predict(val_X)
        metrics["validation_samples"] = int(val_X.shape[0])
        metrics["validation_sampled_pixels_by_class"] = {
            class_name_by_idx[idx]: count for idx, count in val_sampled_counts.items()
        }
        metrics.update(_compute_iou_metrics(val_y, val_pred, class_indices, class_name_by_idx))
    else:
        metrics["validation_samples"] = 0
        metrics["pixel_accuracy"] = None
        metrics["per_class_iou"] = {}

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifact_dir = models_dir / f"{timestamp}_{_slugify(model_name)}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.joblib"
    metadata_path = artifact_dir / "metadata.json"

    _log(log_fn, f"Saving artifacts to {artifact_dir}")
    joblib.dump(classifier, model_path)

    metadata = {
        "model_name": model_name,
        "trained_at": _utc_now_iso(),
        "code_version": code_version,
        "dataset_id": dataset["id"],
        "dataset_path": str(dataset_path),
        "image_root": image_root,
        "coco_json_path": str(coco_path),
        "coco_checksum": compute_sha256(coco_path),
        "classes": class_entries,
        "category_id_to_class_index": category_to_class,
        "feature_config": feature_config.to_dict(),
        "hyperparams": {
            "n_estimators": train_config.n_estimators,
            "max_depth": train_config.max_depth,
            "min_samples_split": train_config.min_samples_split,
            "min_samples_leaf": train_config.min_samples_leaf,
            "class_weight": class_weight,
        },
        "train_config": train_config.to_dict(),
        "metrics": metrics,
        "overlap_resolution": train_config.overlap_policy,
        "warnings": warnings,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return TrainingResult(
        classes=class_entries,
        feature_config=feature_config.to_dict(),
        hyperparams=metadata["hyperparams"],
        metrics=metrics,
        warnings=warnings,
        artifact_dir=str(artifact_dir),
        model_path=str(model_path),
        metadata_path=str(metadata_path),
        metadata=metadata,
    )

