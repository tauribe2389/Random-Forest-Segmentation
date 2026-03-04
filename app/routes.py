"""Flask routes for workspace dashboard, datasets, training, and analysis runs."""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from PIL import Image

from .services.augmentation import augment_coco_dataset
from .services.coco import (
    annotations_by_image,
    build_mask,
    category_id_to_class_index,
    compute_sha256,
    load_coco,
    parse_categories,
)
from .services.job_queue import (
    build_analysis_dedupe_key,
    build_slic_warmup_dedupe_key,
    build_training_dedupe_key,
)
from .services.labeling.class_schema import class_entries as schema_class_entries
from .services.labeling.class_schema import class_names as schema_class_names
from .services.labeling.class_schema import normalize_class_schema, parse_class_names
from .services.labeling.coco_export import export_coco_annotations
from .services.labeling.image_io import (
    ensure_dir,
    list_images,
    load_image_size,
    mask_filename_for_class_id,
    safe_image_name,
    save_binary_mask,
)
from .services.postprocess.bootstrap_masks import (
    BootstrapCleanupConfig,
    cleanup_label_map,
    threshold_label_map,
)
from .services.schemas import DatasetSpec, FeatureConfig, TrainConfig
from .services.storage import Storage

bp = Blueprint("main", __name__)


def _storage() -> Storage:
    return current_app.extensions["storage"]


def _base_dir() -> Path:
    return Path(current_app.config["BASE_DIR"]).resolve()


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _to_relative_under_base(path_value: str) -> str | None:
    base = _base_dir()
    path = Path(path_value)
    if not path.is_absolute():
        return str(path).replace("\\", "/")
    try:
        return path.resolve().relative_to(base).as_posix()
    except ValueError:
        return None


def _workspace_by_id_or_404(workspace_id: int) -> dict[str, Any]:
    workspace = _storage().get_workspace(workspace_id)
    if workspace is None:
        abort(404, description=f"Workspace id={workspace_id} not found.")
    session["workspace_id"] = int(workspace["id"])
    return workspace


def _resolve_workspace(workspace_id: int | None = None) -> dict[str, Any]:
    storage = _storage()
    workspace: dict[str, Any] | None = None
    if workspace_id is not None:
        workspace = storage.get_workspace(workspace_id)
    if workspace is None:
        session_workspace_id = session.get("workspace_id")
        if isinstance(session_workspace_id, int):
            workspace = storage.get_workspace(session_workspace_id)
    if workspace is None:
        workspaces = storage.list_workspaces()
        if workspaces:
            workspace = workspaces[0]
    if workspace is None:
        abort(404, description="No workspaces available.")
    session["workspace_id"] = int(workspace["id"])
    return workspace


def _workspace_redirect_target(workspace_id: int, raw_next: str) -> str:
    candidate = str(raw_next or "").strip()
    if candidate.startswith("/") and not candidate.startswith("//"):
        # Keep users in the same section when switching workspaces, but avoid carrying
        # stale resource IDs (dataset/model/run/image) that belong to another workspace.
        if re.match(r"^/workspace/\d+/datasets/\d+(?:/|$)", candidate):
            return url_for("main.workspace_datasets", workspace_id=workspace_id)
        if re.match(r"^/workspace/\d+/registered-datasets/\d+(?:/|$)", candidate):
            return url_for("main.workspace_datasets", workspace_id=workspace_id)
        if re.match(r"^/workspace/\d+/models/\d+(?:/|$)", candidate):
            return url_for("main.workspace_models", workspace_id=workspace_id)
        if re.match(r"^/workspace/\d+/analysis/\d+(?:/|$)", candidate):
            return url_for("main.workspace_analysis", workspace_id=workspace_id)
        if re.match(r"^/workspace/\d+", candidate):
            return re.sub(r"^/workspace/\d+", f"/workspace/{workspace_id}", candidate, count=1)
    return url_for("main.workspace_dashboard", workspace_id=workspace_id)


def _workspace_image_count(workspace: dict[str, Any]) -> int:
    images_dir = Path(str(workspace.get("images_dir", "")))
    try:
        return len(list_images(images_dir))
    except Exception:
        return 0


def _registered_dataset_image_names(dataset: dict[str, Any]) -> list[str]:
    coco_json_raw = str(dataset.get("coco_json_path", "")).strip()
    if coco_json_raw:
        coco_json_path = Path(coco_json_raw).resolve()
        if coco_json_path.exists() and coco_json_path.is_file():
            try:
                coco_payload = load_coco(coco_json_path)
            except Exception:
                coco_payload = {}
            images = coco_payload.get("images", [])
            if isinstance(images, list):
                ordered_names: list[str] = []
                seen: set[str] = set()
                for item in images:
                    if not isinstance(item, dict):
                        continue
                    file_name = str(item.get("file_name", "")).strip()
                    if not file_name:
                        continue
                    normalized_key = file_name.lower()
                    if normalized_key in seen:
                        continue
                    seen.add(normalized_key)
                    ordered_names.append(file_name)
                if ordered_names:
                    return ordered_names

    dataset_root_raw = str(dataset.get("dataset_path", "")).strip()
    image_root_raw = str(dataset.get("image_root", "")).strip()
    if not dataset_root_raw or not image_root_raw:
        return []
    dataset_root = Path(dataset_root_raw).resolve()
    image_root_path = Path(image_root_raw)
    if not image_root_path.is_absolute():
        image_root_path = (dataset_root / image_root_path).resolve()
    try:
        return list_images(image_root_path)
    except Exception:
        return []


def _registered_dataset_image_count(dataset: dict[str, Any]) -> int:
    return len(_registered_dataset_image_names(dataset))


def _workspace_images_dir(workspace: dict[str, Any]) -> Path:
    return Path(str(workspace.get("images_dir", ""))).resolve()


def _workspace_datasets_root(workspace: dict[str, Any]) -> Path:
    project_dir = Path(str(workspace.get("project_dir", ""))).resolve()
    return (project_dir / "datasets").resolve()


DATASET_CLASS_EDIT_STATE_PREFIX = "dataset_class_edit_state:"


def _dataset_class_edit_session_key(dataset_id: int) -> str:
    return f"{DATASET_CLASS_EDIT_STATE_PREFIX}{int(dataset_id)}"


def _collect_masked_image_stems(masks_dir: Path) -> set[str]:
    if not masks_dir.exists() or not masks_dir.is_dir():
        return set()
    stems: set[str] = set()
    for mask_path in masks_dir.glob("*__*.png"):
        if not mask_path.is_file():
            continue
        image_stem = str(mask_path.stem).split("__", 1)[0].strip()
        if image_stem:
            stems.add(image_stem)
    return stems


def _remove_dataset_image_assets(
    *,
    storage: Storage,
    dataset_project: dict[str, Any],
    dataset_id: int,
    image_name: str,
) -> dict[str, Any]:
    safe_name = safe_image_name(image_name)
    if not safe_name:
        return {"ok": False, "reason": "invalid_name", "image_name": image_name}

    draft_images_dir = Path(str(dataset_project["images_dir"])).resolve()
    target_image_path = (draft_images_dir / safe_name).resolve()
    if not target_image_path.exists() or not target_image_path.is_file():
        return {"ok": False, "reason": "missing_image", "image_name": safe_name}

    target_image_path.unlink()

    image_stem = Path(safe_name).stem
    masks_dir = Path(str(dataset_project["masks_dir"])).resolve()
    removed_mask_files = 0
    for mask_path in masks_dir.glob(f"{image_stem}__*.png"):
        if mask_path.exists() and mask_path.is_file():
            mask_path.unlink()
            removed_mask_files += 1

    storage.delete_image_slic_override(dataset_id, safe_name)

    cache_dir = Path(str(dataset_project["cache_dir"])).resolve()
    removed_cache_files = 0
    for cache_path in cache_dir.glob(f"{image_stem}*"):
        if cache_path.exists() and cache_path.is_file():
            cache_path.unlink()
            removed_cache_files += 1

    return {
        "ok": True,
        "image_name": safe_name,
        "removed_mask_files": removed_mask_files,
        "removed_cache_files": removed_cache_files,
        "had_masks": removed_mask_files > 0,
    }


MODEL_HEADLINE_METRIC_SPECS: list[tuple[str, str]] = [
    ("feature_count", "Feature Count"),
    ("pixel_accuracy", "Pixel Accuracy"),
    ("train_image_count", "Train Images"),
    ("validation_image_count", "Validation Images"),
    ("train_samples", "Train Samples"),
    ("validation_samples", "Validation Samples"),
    ("rle_annotations_used", "RLE Used"),
    ("skipped_rle_annotations", "RLE Skipped"),
]


def _format_metric_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:.5f}".rstrip("0").rstrip(".")
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _flatten_mapping_rows(payload: Any, prefix: str = "") -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if isinstance(payload, dict):
        for key in sorted(payload.keys(), key=lambda item: str(item)):
            item_key = str(key)
            joined_key = f"{prefix}.{item_key}" if prefix else item_key
            value = payload[key]
            if isinstance(value, dict):
                rows.extend(_flatten_mapping_rows(value, joined_key))
            elif isinstance(value, list):
                rows.append({"key": joined_key, "value": json.dumps(value, sort_keys=True)})
            else:
                rows.append({"key": joined_key, "value": _format_metric_value(value)})
    elif payload is not None:
        rows.append({"key": prefix or "value", "value": _format_metric_value(payload)})
    return rows


def _normalize_named_counts(raw_counts: Any) -> dict[str, int]:
    if not isinstance(raw_counts, dict):
        return {}
    normalized: dict[str, int] = {}
    for raw_name, raw_value in raw_counts.items():
        class_name = str(raw_name).strip()
        if not class_name:
            continue
        try:
            numeric = int(float(raw_value))
        except (TypeError, ValueError):
            numeric = 0
        normalized[class_name] = max(0, numeric)
    return normalized


def _ordered_class_names(
    model_classes: list[dict[str, Any]],
    *extra_name_groups: list[str] | tuple[str, ...],
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    sorted_classes = sorted(
        model_classes,
        key=lambda item: int(item.get("class_index", 0)) if str(item.get("class_index", "")).strip() else 0,
    )
    for item in sorted_classes:
        class_name = str(item.get("name", "")).strip()
        if not class_name or class_name in seen:
            continue
        ordered.append(class_name)
        seen.add(class_name)

    for group in extra_name_groups:
        for raw_name in group:
            class_name = str(raw_name).strip()
            if not class_name or class_name in seen:
                continue
            ordered.append(class_name)
            seen.add(class_name)

    return ordered


def _build_distribution_rows(counts: dict[str, int], class_order: list[str]) -> list[dict[str, Any]]:
    if not counts:
        return []
    ordered_names = _ordered_class_names([], class_order, tuple(counts.keys()))
    max_count = max(counts.values()) if counts else 0
    rows: list[dict[str, Any]] = []
    for class_name in ordered_names:
        count = int(counts.get(class_name, 0))
        percent = (count / max_count * 100.0) if max_count > 0 else 0.0
        rows.append(
            {
                "class_name": class_name,
                "count": count,
                "count_display": f"{count:,}",
                "percent": round(percent, 4),
            }
        )
    return rows


def _build_per_class_iou_rows(per_class_iou: Any, class_order: list[str]) -> list[dict[str, Any]]:
    iou_map = per_class_iou if isinstance(per_class_iou, dict) else {}
    class_names = _ordered_class_names([], class_order, tuple(iou_map.keys()))
    rows: list[dict[str, Any]] = []
    for class_name in class_names:
        raw_value = iou_map.get(class_name)
        numeric_value: float | None = None
        if raw_value is not None:
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                numeric_value = None
        if numeric_value is None:
            status = "incomplete"
            display = "-"
        else:
            status = "zero" if numeric_value == 0 else "ok"
            display = f"{numeric_value:.5f}".rstrip("0").rstrip(".")
        rows.append(
            {
                "class_name": class_name,
                "value": numeric_value,
                "value_display": display,
                "status": status,
            }
        )
    return rows


def _build_sampling_rows(
    train_counts: dict[str, int],
    validation_counts: dict[str, int],
    class_order: list[str],
) -> list[dict[str, Any]]:
    class_names = _ordered_class_names([], class_order, tuple(train_counts.keys()), tuple(validation_counts.keys()))
    rows: list[dict[str, Any]] = []
    for class_name in class_names:
        train_value = int(train_counts.get(class_name, 0))
        validation_value = int(validation_counts.get(class_name, 0))
        rows.append(
            {
                "class_name": class_name,
                "train_sampled_pixels": train_value,
                "validation_sampled_pixels": validation_value,
                "train_display": f"{train_value:,}",
                "validation_display": f"{validation_value:,}",
                "has_zero": train_value == 0 or validation_value == 0,
            }
        )
    return rows


def _coerce_nonnegative_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric < 0:
        return 0.0
    return numeric


def _job_is_terminal(status_value: Any) -> bool:
    return str(status_value or "").strip().lower() in {"completed", "failed", "canceled"}


def _job_last_log(job: dict[str, Any]) -> str:
    logs = job.get("logs_json")
    if not isinstance(logs, list) or not logs:
        return ""
    last_entry = logs[-1]
    if isinstance(last_entry, dict):
        return str(last_entry.get("message", "")).strip()
    return str(last_entry).strip()


_ANALYSIS_IMAGE_PROGRESS_PATTERN = re.compile(r"analyzing image\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_TRAINING_IMAGE_PROGRESS_PATTERN = re.compile(r"sampling image\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)


def _job_log_entry_message(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("message", "")).strip()
    return str(entry).strip()


def _parse_image_progress(message: str, *, pattern: re.Pattern[str]) -> tuple[int, int] | None:
    match = pattern.search(message)
    if match is None:
        return None
    try:
        current = int(match.group(1))
        total = int(match.group(2))
    except (TypeError, ValueError):
        return None
    if total <= 0:
        return None
    current = max(0, min(current, total))
    return current, total


def _job_progress_counter(job: dict[str, Any]) -> dict[str, Any] | None:
    logs = job.get("logs_json")
    if not isinstance(logs, list) or not logs:
        return None
    job_type = str(job.get("job_type", "")).strip().lower()
    if job_type == "analysis":
        candidates = [_ANALYSIS_IMAGE_PROGRESS_PATTERN]
    elif job_type == "training":
        candidates = [_TRAINING_IMAGE_PROGRESS_PATTERN]
    else:
        candidates = [_ANALYSIS_IMAGE_PROGRESS_PATTERN, _TRAINING_IMAGE_PROGRESS_PATTERN]
    for entry in reversed(logs):
        message = _job_log_entry_message(entry)
        if not message:
            continue
        for pattern in candidates:
            parsed = _parse_image_progress(message, pattern=pattern)
            if parsed is None:
                continue
            current, total = parsed
            return {
                "current": current,
                "total": total,
                "label": f"{current}/{total} images",
            }
    return None


def _job_progress_percent(job: dict[str, Any]) -> float:
    try:
        numeric = float(job.get("progress", 0.0))
    except (TypeError, ValueError):
        numeric = 0.0
    if numeric < 0:
        numeric = 0.0
    if numeric > 1:
        numeric = 1.0
    return round(numeric * 100.0, 2)


def _serialize_job(job: dict[str, Any], *, queue_positions: dict[int, int] | None = None) -> dict[str, Any]:
    job_id = int(job["id"])
    queue_position = queue_positions.get(job_id) if isinstance(queue_positions, dict) else None
    progress_counter = _job_progress_counter(job)
    payload = {
        "id": job_id,
        "workspace_id": int(job.get("workspace_id", 0) or 0),
        "job_type": str(job.get("job_type", "")),
        "status": str(job.get("status", "")),
        "stage": str(job.get("stage", "")),
        "progress": float(job.get("progress", 0.0) or 0.0),
        "progress_percent": _job_progress_percent(job),
        "created_at": str(job.get("created_at", "")),
        "started_at": str(job.get("started_at", "")),
        "finished_at": str(job.get("finished_at", "")),
        "heartbeat_at": str(job.get("heartbeat_at", "")),
        "worker_id": str(job.get("worker_id", "")),
        "error_message": str(job.get("error_message", "") or ""),
        "entity_type": str(job.get("entity_type", "") or ""),
        "entity_id": int(job.get("entity_id", 0) or 0),
        "result_ref_type": str(job.get("result_ref_type", "") or ""),
        "result_ref_id": int(job.get("result_ref_id", 0) or 0),
        "cancel_requested": bool(job.get("cancel_requested")),
        "priority": int(job.get("priority", 0) or 0),
        "rerun_of_job_id": int(job.get("rerun_of_job_id", 0) or 0),
        "last_log": _job_last_log(job),
    }
    if progress_counter is not None:
        payload["progress_counter_current"] = int(progress_counter["current"])
        payload["progress_counter_total"] = int(progress_counter["total"])
        payload["progress_counter_label"] = str(progress_counter["label"])
    if queue_position is not None:
        payload["queue_position"] = int(queue_position)
    return payload


def _extract_confusion_matrix(metrics: dict[str, Any], class_order: list[str]) -> dict[str, Any] | None:
    raw_matrix = metrics.get("confusion_matrix")
    if raw_matrix is None:
        raw_matrix = metrics.get("pixel_confusion")
    if raw_matrix is None:
        return None

    labels: list[str] = []
    matrix_rows: list[list[Any]] = []

    if isinstance(raw_matrix, dict):
        if isinstance(raw_matrix.get("matrix"), list):
            matrix_rows = [row for row in raw_matrix.get("matrix", []) if isinstance(row, list)]
            candidate_labels = raw_matrix.get("labels")
            if isinstance(candidate_labels, list):
                labels = [str(item).strip() for item in candidate_labels if str(item).strip()]
        elif raw_matrix and all(isinstance(value, dict) for value in raw_matrix.values()):
            labels = _ordered_class_names([], class_order)
            normalized_matrix: dict[str, dict[str, Any]] = {}
            for true_label, pred_map in raw_matrix.items():
                true_name = str(true_label).strip()
                if not true_name:
                    continue
                if true_name not in labels:
                    labels.append(true_name)

                normalized_preds: dict[str, Any] = {}
                if isinstance(pred_map, dict):
                    for pred_label, pred_value in pred_map.items():
                        pred_name = str(pred_label).strip()
                        if not pred_name:
                            continue
                        normalized_preds[pred_name] = pred_value
                        if pred_name not in labels:
                            labels.append(pred_name)
                normalized_matrix[true_name] = normalized_preds

            for true_name in labels:
                pred_map = normalized_matrix.get(true_name, {})
                row_values: list[Any] = []
                for pred_name in labels:
                    row_values.append(pred_map.get(pred_name, 0))
                matrix_rows.append(row_values)
        else:
            return None
    elif isinstance(raw_matrix, list):
        matrix_rows = [row for row in raw_matrix if isinstance(row, list)]
    else:
        return None

    if not matrix_rows:
        return None

    max_cols = max(len(row) for row in matrix_rows)
    size = max(len(matrix_rows), max_cols)
    if size <= 0:
        return None

    if not labels:
        labels = _ordered_class_names([], class_order)
    if len(labels) < size:
        for class_name in class_order:
            normalized_name = str(class_name).strip()
            if normalized_name and normalized_name not in labels:
                labels.append(normalized_name)
            if len(labels) >= size:
                break
    while len(labels) < size:
        labels.append(f"class_{len(labels)}")
    if len(labels) > size:
        labels = labels[:size]

    padded_matrix: list[list[float]] = []
    for row_index in range(size):
        source_row = matrix_rows[row_index] if row_index < len(matrix_rows) else []
        normalized_row: list[float] = []
        for column_index in range(size):
            value = source_row[column_index] if column_index < len(source_row) else 0
            normalized_row.append(_coerce_nonnegative_float(value))
        padded_matrix.append(normalized_row)

    max_count = max((max(row) for row in padded_matrix), default=0.0)
    rows: list[dict[str, Any]] = []
    for row_index, true_label in enumerate(labels):
        row_counts = padded_matrix[row_index]
        row_total = float(sum(row_counts))
        cells: list[dict[str, Any]] = []
        for count in row_counts:
            count_display = _format_metric_value(count)
            raw_intensity = (count / max_count) if max_count > 0 else 0.0
            cells.append(
                {
                    "count_value": count,
                    "count_display": count_display,
                    "row_total": row_total,
                    "raw_intensity": round(raw_intensity, 6),
                }
            )
        rows.append(
            {
                "true_label": true_label,
                "row_total": row_total,
                "row_total_display": _format_metric_value(row_total),
                "cells": cells,
            }
        )

    return {
        "labels": labels,
        "rows": rows,
        "max_count": max_count,
    }


def _slugify(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower()).strip("_")
    return normalized or "dataset"


def _default_labeler_categories() -> list[str]:
    configured = [
        str(item).strip()
        for item in current_app.config.get("LABELER_CATEGORIES", [])
        if str(item).strip()
    ]
    return configured or ["class_1", "class_2", "class_3"]


def _parse_categories_text(raw_value: str) -> list[str]:
    return parse_class_names(raw_value)


def _dataset_project_categories(dataset_project: dict[str, Any]) -> list[str]:
    return schema_class_names(
        dataset_project.get("categories_json"),
        fallback_names=_default_labeler_categories(),
    )


def _dataset_project_classes(dataset_project: dict[str, Any]) -> list[dict[str, Any]]:
    return schema_class_entries(
        dataset_project.get("categories_json"),
        fallback_names=_default_labeler_categories(),
    )


def _draft_version(project: dict[str, Any]) -> int:
    try:
        return max(1, int(project.get("draft_version") or 1))
    except (TypeError, ValueError):
        return 1


def _draft_origin_summary(
    *,
    storage: Storage,
    workspace_id: int,
    draft_project: dict[str, Any],
) -> dict[str, Any]:
    origin_type = str(draft_project.get("origin_type", "manual") or "manual").strip().lower()
    if origin_type == "branched_from_draft":
        origin_draft_id = draft_project.get("origin_draft_project_id")
        origin_version = draft_project.get("origin_draft_version")
        origin_version_text = str(origin_version if origin_version is not None else "?")
        try:
            origin_id = int(origin_draft_id)
        except (TypeError, ValueError):
            origin_id = None
        if origin_id is not None:
            origin_project = storage.get_workspace_dataset(workspace_id, origin_id)
            if origin_project is not None:
                return {
                    "label": f"Branched from {origin_project['name']} v{origin_version_text}",
                    "url": url_for(
                        "main.workspace_dataset_detail",
                        workspace_id=workspace_id,
                        dataset_id=origin_id,
                    ),
                }
            return {
                "label": f"Branched from draft #{origin_id} v{origin_version_text}",
                "url": None,
            }
        return {"label": f"Branched from draft v{origin_version_text}", "url": None}

    if origin_type == "augmented_from_registered":
        source_dataset_id = draft_project.get("origin_registered_dataset_id")
        try:
            registered_id = int(source_dataset_id)
        except (TypeError, ValueError):
            registered_id = None
        if registered_id is not None:
            source_dataset = storage.get_dataset(registered_id, workspace_id=workspace_id)
            if source_dataset is not None:
                return {
                    "label": f"Augmented from {source_dataset['name']}",
                    "url": url_for(
                        "main.workspace_registered_dataset_detail",
                        workspace_id=workspace_id,
                        dataset_id=registered_id,
                    ),
                }
            return {"label": f"Augmented from registered #{registered_id}", "url": None}
        return {"label": "Augmented from registered dataset", "url": None}

    return {"label": "Manual draft", "url": None}


def _seed_masks_from_coco_annotations(
    *,
    coco_payload: dict[str, Any],
    images_dir: Path,
    masks_dir: Path,
    parsed_categories: list[dict[str, Any]],
) -> tuple[int, list[str]]:
    image_rows = list(coco_payload.get("images", []))
    if not image_rows or not parsed_categories:
        return 0, []

    class_names: list[str] = [str(item.get("name", "")).strip() for item in parsed_categories]
    if not any(class_names):
        return 0, []
    class_names = [name if name else f"class_{idx + 1}" for idx, name in enumerate(class_names)]

    category_to_class = category_id_to_class_index(parsed_categories)
    annotations_map = annotations_by_image(coco_payload)
    warnings: list[str] = []
    saved_masks = 0

    for image_row in image_rows:
        try:
            image_id = int(image_row.get("id"))
        except (TypeError, ValueError):
            continue

        image_name = safe_image_name(str(image_row.get("file_name", "")))
        if not image_name:
            continue
        image_annotations = annotations_map.get(image_id, [])
        if not image_annotations:
            continue

        image_path_value = (images_dir / image_name).resolve()
        if not image_path_value.exists() or not image_path_value.is_file():
            warnings.append(f"Skipped mask seed for missing image: {image_name}")
            continue

        try:
            height = int(image_row.get("height") or 0)
            width = int(image_row.get("width") or 0)
        except (TypeError, ValueError):
            height = 0
            width = 0
        if height <= 0 or width <= 0:
            height, width = load_image_size(image_path_value)

        class_mask, mask_warnings, _, _ = build_mask(
            height,
            width,
            image_annotations,
            category_to_class,
            overlap_policy="higher_area_wins",
            use_rle=True,
        )
        warnings.extend(mask_warnings)
        stem = Path(image_name).stem
        for class_index, _class_name in enumerate(class_names, start=1):
            mask_bool = class_mask == class_index
            if not mask_bool.any():
                continue
            output_path = masks_dir / mask_filename_for_class_id(stem, class_index)
            save_binary_mask(mask_bool, output_path)
            saved_masks += 1

    return saved_masks, warnings


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "on", "yes", "y"}


def _workspace_augmentation_seed(workspace: dict[str, Any]) -> int:
    seed_value = workspace.get("augmentation_seed")
    try:
        return int(seed_value)
    except (TypeError, ValueError):
        return int(current_app.config["RANDOM_SEED"])


AUGMENT_RECIPE_TRANSFORMS = {
    "flip_h",
    "flip_v",
    "rot90",
    "rot180",
    "color_jitter",
    "grayscale",
    "channel_shuffle",
    "gaussian_noise",
    "speckle_noise",
    "gaussian_blur",
}


def _parse_selected_image_names(raw_value: Any) -> list[str]:
    if isinstance(raw_value, str):
        raw_items = raw_value.replace(",", "\n").splitlines()
    elif isinstance(raw_value, list):
        raw_items = [str(item) for item in raw_value]
    else:
        raw_items = []
    selected: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        item = str(raw).strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
    return selected


def _resolve_stored_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    candidate = Path(str(path_value))
    if candidate.is_absolute():
        return candidate.resolve()
    return (_base_dir() / candidate).resolve()


def _save_indexed_label_map(label_map: np.ndarray, output_path: Path) -> None:
    max_value = int(np.max(label_map)) if label_map.size else 0
    if max_value <= 255:
        Image.fromarray(label_map.astype(np.uint8), mode="L").save(output_path)
    else:
        Image.fromarray(label_map.astype(np.uint16), mode="I;16").save(output_path)


def _parse_int_in_range(
    raw_value: Any,
    *,
    field_label: str,
    minimum: int,
    maximum: int | None = None,
    default: int,
) -> tuple[int, str | None]:
    value_text = str(raw_value if raw_value is not None else "").strip()
    if not value_text:
        parsed = int(default)
    else:
        try:
            parsed = int(value_text)
        except ValueError:
            return int(default), f"{field_label} must be an integer."
    if parsed < minimum:
        bound_text = f"at least {minimum}"
        return int(default), f"{field_label} must be {bound_text}."
    if maximum is not None and parsed > maximum:
        return int(default), f"{field_label} must be at most {maximum}."
    return int(parsed), None


def _model_editable_classes(model: dict[str, Any]) -> list[dict[str, Any]]:
    raw_classes = model.get("classes_json")
    if not isinstance(raw_classes, list):
        raw_classes = []
    parsed: list[dict[str, Any]] = []
    for entry in raw_classes:
        if not isinstance(entry, dict):
            continue
        try:
            class_index = int(entry.get("class_index"))
        except (TypeError, ValueError):
            continue
        class_name = str(entry.get("name", f"class_{class_index}")).strip() or f"class_{class_index}"
        parsed.append({"class_index": class_index, "name": class_name})
    parsed.sort(key=lambda item: int(item["class_index"]))
    return [item for item in parsed if int(item["class_index"]) > 0]


def _build_promote_item_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        try:
            item_id = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        input_image = str(item.get("input_image", "")).strip()
        input_name = safe_image_name(Path(input_image).name) if input_image else ""
        if not input_name:
            continue
        source_mask_path = (
            item.get("refined_mask_path")
            or item.get("mask_path")
            or item.get("raw_mask_path")
        )
        if not source_mask_path:
            continue
        rows.append(
            {
                "item_id": item_id,
                "input_image": input_image,
                "input_image_name": input_name,
                "source_mask_path": str(source_mask_path),
                "source_label": "Refined" if item.get("refined_mask_path") else "Raw",
                "conf_path": str(item.get("conf_path") or "").strip(),
                "has_conf": bool(item.get("conf_path")),
                "raw_overlay_path": str(item.get("raw_overlay_path") or ""),
                "refined_overlay_path": str(item.get("refined_overlay_path") or ""),
            }
        )
    return rows


def _parse_augmentation_recipe(raw_recipe_json: str) -> list[dict[str, Any]]:
    raw_text = str(raw_recipe_json or "").strip()
    if not raw_text:
        return []
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Recipe JSON is invalid: {exc.msg}") from exc
    if not isinstance(payload, list):
        raise ValueError("Recipe must be a list of steps.")

    parsed_steps: list[dict[str, Any]] = []
    for index, step in enumerate(payload, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"Recipe step {index} is invalid.")
        transform = str(step.get("transform", "")).strip().lower()
        if transform not in AUGMENT_RECIPE_TRANSFORMS:
            raise ValueError(f"Recipe step {index} uses unsupported transform '{transform}'.")

        selection_mode = str(step.get("selection_mode", "fifo")).strip().lower()
        if selection_mode not in {"fifo", "random", "selected"}:
            selection_mode = "fifo"

        try:
            selection_percent = float(step.get("selection_percent", 100.0))
        except (TypeError, ValueError):
            selection_percent = 100.0
        selection_percent = max(0.0, min(100.0, selection_percent))

        settings = step.get("settings", {})
        if not isinstance(settings, dict):
            settings = {}
        selected_images = _parse_selected_image_names(step.get("selected_images", []))

        parsed_steps.append(
            {
                "transform": transform,
                "selection_mode": selection_mode,
                "selection_percent": selection_percent,
                "settings": settings,
                "selected_images": selected_images,
            }
        )

    return parsed_steps


@bp.app_template_filter("prettyjson")
def pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


@bp.app_template_filter("human_datetime")
def human_datetime(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "-"

    parsed: datetime | None = None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                parsed = datetime.strptime(raw, fmt)
                break
            except ValueError:
                continue

    if parsed is None:
        return raw

    formatted = parsed.strftime("%b %d, %Y %I:%M %p")
    tz_name = parsed.tzname() if parsed.tzinfo is not None else ""
    if tz_name:
        return f"{formatted} {tz_name}"
    return formatted


@bp.app_context_processor
def inject_template_helpers() -> dict[str, Any]:
    def file_url(path_value: str | None) -> str | None:
        if not path_value:
            return None
        relative = _to_relative_under_base(path_value)
        if relative is None:
            return None
        return url_for("main.serve_file", relative_path=relative)

    return {
        "file_url": file_url,
        "app_name": str(current_app.config.get("APP_NAME", "Model Foundry")),
    }


@bp.route("/workspaces/switch", methods=["POST"])
def switch_workspace() -> str:
    workspace_id_raw = str(request.form.get("workspace_id", "")).strip()
    next_url = str(request.form.get("next_url", "")).strip()
    try:
        workspace_id = int(workspace_id_raw)
    except ValueError:
        flash("Select a valid workspace.", "error")
        workspaces = _storage().list_workspaces()
        if workspaces:
            fallback_workspace_id = int(workspaces[0]["id"])
            session["workspace_id"] = fallback_workspace_id
            return redirect(url_for("main.workspace_dashboard", workspace_id=fallback_workspace_id))
        return redirect(url_for("main.dashboard_root"))

    workspace = _storage().get_workspace(workspace_id)
    if workspace is None:
        flash("Selected workspace was not found.", "error")
        workspaces = _storage().list_workspaces()
        if workspaces:
            fallback_workspace_id = int(workspaces[0]["id"])
            session["workspace_id"] = fallback_workspace_id
            return redirect(url_for("main.workspace_dashboard", workspace_id=fallback_workspace_id))
        return redirect(url_for("main.dashboard_root"))

    session["workspace_id"] = workspace_id
    return redirect(_workspace_redirect_target(workspace_id, next_url))


@bp.route("/", methods=["GET"])
def dashboard_root() -> str:
    workspace_rows = _storage().list_workspaces()
    selected_workspace_id: int | None = None
    session_workspace_id = session.get("workspace_id")
    if isinstance(session_workspace_id, int):
        selected_workspace_id = int(session_workspace_id)
    elif workspace_rows:
        selected_workspace_id = int(workspace_rows[0]["id"])
    if selected_workspace_id is not None and not any(
        int(row["id"]) == selected_workspace_id for row in workspace_rows
    ):
        selected_workspace_id = int(workspace_rows[0]["id"]) if workspace_rows else None
    if selected_workspace_id is not None:
        session["workspace_id"] = selected_workspace_id

    return render_template(
        "workspace_landing.html",
        workspace_rows=workspace_rows,
        selected_workspace_id=selected_workspace_id,
    )


@bp.route("/workspace/<int:workspace_id>", methods=["GET"])
def workspace_dashboard(workspace_id: int) -> str:
    workspace = _workspace_by_id_or_404(workspace_id)
    storage = _storage()
    dataset_projects = storage.list_workspace_datasets(workspace_id)
    registered_datasets = storage.list_datasets(workspace_id=workspace_id)
    models = storage.list_models(workspace_id=workspace_id)
    runs = storage.list_runs(workspace_id=workspace_id, limit=8)
    image_count = _workspace_image_count(workspace)

    summary = {
        "images": image_count,
        "datasets": len(dataset_projects),
        "models": len(models),
        "analysis": len(runs),
    }

    if image_count == 0:
        primary_action = {
            "title": "Upload images",
            "description": "Start by adding raw images to this workspace.",
            "label": "Upload Images",
            "url": url_for("labeler.workspace_images", workspace_id=workspace_id) + "#upload",
        }
    elif not dataset_projects:
        primary_action = {
            "title": "Create dataset",
            "description": "Select workspace images and create a dataset labeling project.",
            "label": "Create Dataset",
            "url": url_for("main.register_dataset", workspace_id=workspace_id),
        }
    elif not registered_datasets:
        primary_action = {
            "title": "Label and export dataset",
            "description": "Open a dataset project, label with superpixels, then register for training.",
            "label": "Open Dataset",
            "url": url_for(
                "main.workspace_dataset_detail",
                workspace_id=workspace_id,
                dataset_id=int(dataset_projects[0]["id"]),
            ),
        }
    elif not models:
        primary_action = {
            "title": "Train model",
            "description": "Train your first model from a registered dataset.",
            "label": "Train Model",
            "url": url_for("main.workspace_models", workspace_id=workspace_id) + "#train-model",
        }
    else:
        primary_action = {
            "title": "Run analysis",
            "description": "Use a trained model to run analysis on an image set.",
            "label": "Run Analysis",
            "url": url_for("main.new_analysis", workspace_id=workspace_id, model_id=int(models[0]["id"])),
        }

    return render_template(
        "dashboard.html",
        workspace=workspace,
        summary=summary,
        primary_action=primary_action,
        recent_models=models[:5],
        recent_runs=runs[:5],
    )


@bp.route("/workspace/<int:workspace_id>/datasets", methods=["GET"])
def workspace_datasets(workspace_id: int) -> str:
    workspace = _workspace_by_id_or_404(workspace_id)
    storage = _storage()
    dataset_projects = storage.list_workspace_datasets(workspace_id)
    registered_datasets_raw = storage.list_datasets(workspace_id=workspace_id)
    registered_datasets: list[dict[str, Any]] = []
    for item in registered_datasets_raw:
        row = dict(item)
        row["image_count"] = _registered_dataset_image_count(row)
        registered_datasets.append(row)

    registered_by_id = {int(item["id"]): item for item in registered_datasets}
    registered_by_draft: dict[int, list[dict[str, Any]]] = {}
    for item in registered_datasets:
        source_draft_id = item.get("source_draft_project_id")
        try:
            normalized_draft_id = int(source_draft_id)
        except (TypeError, ValueError):
            continue
        registered_by_draft.setdefault(normalized_draft_id, []).append(item)

    project_rows: list[dict[str, Any]] = []
    for project in dataset_projects:
        images_dir = Path(str(project.get("images_dir", ""))).resolve()
        try:
            image_count = len(list_images(images_dir))
        except Exception:
            image_count = 0
        linked_dataset = None
        linked_id = project.get("dataset_id")
        if linked_id is not None:
            try:
                linked_dataset = registered_by_id.get(int(linked_id))
            except (TypeError, ValueError):
                linked_dataset = None
        row = dict(project)
        row["image_count"] = image_count
        row["categories"] = _dataset_project_categories(project)
        row["linked_dataset"] = linked_dataset
        row["draft_version"] = _draft_version(project)
        row["origin"] = _draft_origin_summary(storage=storage, workspace_id=workspace_id, draft_project=project)
        row["registration_history"] = registered_by_draft.get(int(project["id"]), [])
        project_rows.append(row)

    return render_template(
        "datasets_index.html",
        workspace=workspace,
        dataset_projects=project_rows,
        registered_datasets=registered_datasets,
    )


@bp.route("/workspace/<int:workspace_id>/datasets/new", methods=["GET", "POST"])
@bp.route("/datasets/new", methods=["GET", "POST"])
def register_dataset(workspace_id: int | None = None) -> str:
    workspace = _resolve_workspace(workspace_id)
    workspace_id = int(workspace["id"])
    workspace_images = list_images(_workspace_images_dir(workspace))
    form_name = ""
    form_categories_text = ""
    selected_images: list[str] = []

    if request.method == "GET":
        return render_template(
            "dataset_new.html",
            workspace=workspace,
            workspace_images=workspace_images,
            form_name=form_name,
            form_categories_text="\n".join(_default_labeler_categories()),
            selected_images=selected_images,
        )

    form_name = str(request.form.get("name", "")).strip()
    form_categories_text = str(request.form.get("categories_text", "")).strip()
    categories = _parse_categories_text(form_categories_text) or _default_labeler_categories()
    available_images = set(workspace_images)
    seen_images: set[str] = set()
    for item in request.form.getlist("selected_images"):
        image_name = safe_image_name(item)
        if image_name not in available_images:
            continue
        if image_name in seen_images:
            continue
        seen_images.add(image_name)
        selected_images.append(image_name)

    if not form_name:
        flash("Dataset name is required.", "error")
        return (
            render_template(
                "dataset_new.html",
                workspace=workspace,
                workspace_images=workspace_images,
                form_name=form_name,
                form_categories_text=form_categories_text,
                selected_images=selected_images,
            ),
            400,
        )

    if not selected_images:
        flash("Select at least one workspace image.", "error")
        return (
            render_template(
                "dataset_new.html",
                workspace=workspace,
                workspace_images=workspace_images,
                form_name=form_name,
                form_categories_text=form_categories_text,
                selected_images=selected_images,
            ),
            400,
        )

    dataset_root = (_workspace_datasets_root(workspace) / f"{_slugify(form_name)}_{uuid.uuid4().hex[:8]}").resolve()
    images_dir = (dataset_root / "images").resolve()
    masks_dir = (dataset_root / "masks").resolve()
    coco_dir = (dataset_root / "coco").resolve()
    cache_dir = (masks_dir / "_cache").resolve()
    ensure_dir(images_dir)
    ensure_dir(masks_dir)
    ensure_dir(coco_dir)
    ensure_dir(cache_dir)

    workspace_images_dir = _workspace_images_dir(workspace)
    copied_count = 0
    try:
        for image_name in selected_images:
            source_path = (workspace_images_dir / image_name).resolve()
            if not source_path.exists() or not source_path.is_file():
                continue
            shutil.copy2(source_path, images_dir / image_name)
            copied_count += 1
    except Exception as exc:
        flash(f"Failed while copying selected images: {exc}", "error")
        return (
            render_template(
                "dataset_new.html",
                workspace=workspace,
                workspace_images=workspace_images,
                form_name=form_name,
                form_categories_text=form_categories_text,
                selected_images=selected_images,
            ),
            500,
        )

    if copied_count == 0:
        flash("None of the selected images could be copied into the dataset.", "error")
        return (
            render_template(
                "dataset_new.html",
                workspace=workspace,
                workspace_images=workspace_images,
                form_name=form_name,
                form_categories_text=form_categories_text,
                selected_images=selected_images,
            ),
            400,
        )

    try:
        dataset_project_id = _storage().create_workspace_dataset(
            workspace_id=workspace_id,
            name=form_name,
            project_dir=str(dataset_root),
            images_dir=str(images_dir),
            masks_dir=str(masks_dir),
            coco_dir=str(coco_dir),
            cache_dir=str(cache_dir),
            categories=categories,
            slic_algorithm="slic",
            slic_preset_name="medium",
            slic_detail_level="medium",
            slic_n_segments=int(current_app.config["LABELER_SLIC_N_SEGMENTS"]),
            slic_compactness=float(current_app.config["LABELER_SLIC_COMPACTNESS"]),
            slic_sigma=float(current_app.config["LABELER_SLIC_SIGMA"]),
            slic_colorspace="lab",
        )
    except sqlite3.IntegrityError:
        if dataset_root.exists():
            shutil.rmtree(dataset_root, ignore_errors=True)
        flash(f"A dataset named '{form_name}' already exists.", "error")
        return (
            render_template(
                "dataset_new.html",
                workspace=workspace,
                workspace_images=workspace_images,
                form_name=form_name,
                form_categories_text=form_categories_text,
                selected_images=selected_images,
            ),
            400,
        )
    except Exception as exc:
        if dataset_root.exists():
            shutil.rmtree(dataset_root, ignore_errors=True)
        flash(f"Failed to create dataset project: {exc}", "error")
        return (
            render_template(
                "dataset_new.html",
                workspace=workspace,
                workspace_images=workspace_images,
                form_name=form_name,
                form_categories_text=form_categories_text,
                selected_images=selected_images,
            ),
            500,
        )

    flash(
        (
            f"Dataset '{form_name}' created with {copied_count} image(s). "
            "Open images below to label with superpixels."
        ),
        "success",
    )
    return redirect(
        url_for(
            "main.workspace_dataset_detail",
            workspace_id=workspace_id,
            dataset_id=dataset_project_id,
        )
    )


@bp.route("/workspace/<int:workspace_id>/datasets/<int:dataset_id>", methods=["GET"])
def workspace_dataset_detail(workspace_id: int, dataset_id: int) -> str:
    workspace = _workspace_by_id_or_404(workspace_id)
    storage = _storage()
    dataset_project = storage.get_workspace_dataset(workspace_id, dataset_id)
    if dataset_project is None:
        abort(404, description=f"Dataset project id={dataset_id} not found in workspace.")

    class_entries = _dataset_project_classes(dataset_project)
    categories = [str(entry.get("name", "")) for entry in class_entries if str(entry.get("name", "")).strip()]
    if not categories:
        categories = _default_labeler_categories()
    images_dir = Path(str(dataset_project["images_dir"])).resolve()
    masks_dir = Path(str(dataset_project["masks_dir"])).resolve()
    images = list_images(images_dir)
    masked_stems = _collect_masked_image_stems(masks_dir)
    draft_image_rows: list[dict[str, Any]] = []
    labeled_image_count = 0
    next_unlabeled_image_name: str | None = None
    for image_name in images:
        image_stem = Path(image_name).stem
        is_labeled = image_stem in masked_stems
        if is_labeled:
            labeled_image_count += 1
        elif next_unlabeled_image_name is None:
            next_unlabeled_image_name = image_name
        draft_image_rows.append(
            {
                "name": image_name,
                "is_labeled": is_labeled,
            }
        )

    first_image_name = images[0] if images else None
    start_label_image_name = next_unlabeled_image_name or first_image_name
    start_label_url = (
        url_for(
            "labeler.dataset_page",
            workspace_id=workspace_id,
            dataset_id=dataset_id,
            image_name=start_label_image_name,
        )
        if start_label_image_name
        else None
    )
    label_next_unlabeled_url = (
        url_for(
            "labeler.dataset_page",
            workspace_id=workspace_id,
            dataset_id=dataset_id,
            image_name=next_unlabeled_image_name,
        )
        if next_unlabeled_image_name
        else start_label_url
    )

    registration_history = storage.list_datasets_for_draft(dataset_id, workspace_id=workspace_id)

    registered_dataset: dict[str, Any] | None = None
    linked_dataset_id = dataset_project.get("dataset_id")
    if linked_dataset_id is not None:
        try:
            registered_dataset = storage.get_dataset(int(linked_dataset_id), workspace_id=workspace_id)
        except (TypeError, ValueError):
            registered_dataset = None
    if registered_dataset is None and registration_history:
        registered_dataset = registration_history[0]

    registration_ids = {int(item["id"]) for item in registration_history}
    if registered_dataset is not None:
        registration_ids.add(int(registered_dataset["id"]))
    related_models = [
        model
        for model in storage.list_models(workspace_id=workspace_id)
        if int(model["dataset_id"]) in registration_ids
    ]
    registered_name_by_id = {int(item["id"]): str(item["name"]) for item in registration_history}
    if registered_dataset is not None:
        registered_name_by_id[int(registered_dataset["id"])] = str(registered_dataset["name"])
    for model in related_models:
        model["dataset_name"] = registered_name_by_id.get(int(model["dataset_id"]), f"Dataset #{model['dataset_id']}")

    workspace_images = list_images(_workspace_images_dir(workspace))
    image_set = set(images)
    available_workspace_images = [name for name in workspace_images if name not in image_set]
    available_workspace_image_rows = [{"name": image_name} for image_name in available_workspace_images]

    class_edit_state_raw = session.pop(_dataset_class_edit_session_key(dataset_id), None)
    class_edit_state = class_edit_state_raw if isinstance(class_edit_state_raw, dict) else {}
    class_edit_categories_text = str(class_edit_state.get("categories_text", ""))
    if not class_edit_categories_text.strip():
        class_edit_categories_text = "\n".join(categories)

    class_edit_diff_raw = class_edit_state.get("diff")
    class_edit_diff = class_edit_diff_raw if isinstance(class_edit_diff_raw, dict) else None

    stats = {
        "draft_image_count": len(images),
        "labeled_image_count": labeled_image_count,
        "unlabeled_image_count": max(0, len(images) - labeled_image_count),
        "class_count": len(categories),
        "registration_count": len(registration_history),
        "has_masks": bool(masked_stems),
    }
    active_slic_warmup_job = storage.find_active_job_by_dedupe(
        dedupe_key=build_slic_warmup_dedupe_key(workspace_id=workspace_id, dataset_id=dataset_id),
        workspace_id=workspace_id,
        job_type="slic_warmup",
    )
    class_rows = [
        {
            "order": idx + 1,
            "id": int(entry.get("id", idx + 1)),
            "name": str(entry.get("name", "")),
        }
        for idx, entry in enumerate(class_entries)
        if str(entry.get("name", "")).strip()
    ]

    return render_template(
        "dataset_detail.html",
        workspace=workspace,
        dataset_project=dataset_project,
        draft_version=_draft_version(dataset_project),
        origin=_draft_origin_summary(storage=storage, workspace_id=workspace_id, draft_project=dataset_project),
        categories=categories,
        images=images,
        draft_image_rows=draft_image_rows,
        available_workspace_images=available_workspace_images,
        available_workspace_image_rows=available_workspace_image_rows,
        registered_dataset=registered_dataset,
        registration_history=registration_history,
        related_models=related_models,
        stats=stats,
        class_rows=class_rows,
        class_edit_categories_text=class_edit_categories_text,
        class_edit_diff=class_edit_diff,
        start_label_url=start_label_url,
        label_next_unlabeled_url=label_next_unlabeled_url,
        next_unlabeled_image_name=next_unlabeled_image_name,
        active_slic_warmup_job=active_slic_warmup_job,
    )


@bp.route("/workspace/<int:workspace_id>/datasets/<int:dataset_id>/slic-warmup", methods=["POST"])
def queue_dataset_slic_warmup(workspace_id: int, dataset_id: int) -> str:
    _workspace_by_id_or_404(workspace_id)
    storage = _storage()
    dataset_project = storage.get_workspace_dataset(workspace_id, dataset_id)
    if dataset_project is None:
        abort(404, description=f"Dataset project id={dataset_id} not found in workspace.")

    images_dir = Path(str(dataset_project.get("images_dir", ""))).resolve()
    image_count = len(list_images(images_dir)) if images_dir.exists() and images_dir.is_dir() else 0
    if image_count <= 0:
        flash("This draft has no images to warm up.", "warning")
        return redirect(
            url_for("main.workspace_dataset_detail", workspace_id=workspace_id, dataset_id=dataset_id)
            + "#draft-images-section"
        )

    dedupe_key = build_slic_warmup_dedupe_key(workspace_id=workspace_id, dataset_id=dataset_id)
    existing = storage.find_active_job_by_dedupe(
        dedupe_key=dedupe_key,
        workspace_id=workspace_id,
        job_type="slic_warmup",
    )
    if existing is not None:
        existing_id = int(existing["id"])
        existing_status = str(existing.get("status", "")).strip().lower() or "active"
        flash(f"SLIC warmup job #{existing_id} is already {existing_status}.", "warning")
        return redirect(
            url_for("main.workspace_dataset_detail", workspace_id=workspace_id, dataset_id=dataset_id)
            + "#draft-images-section"
        )

    payload = {
        "workspace_id": workspace_id,
        "dataset_id": dataset_id,
    }
    job_id, created = storage.enqueue_job(
        workspace_id=workspace_id,
        job_type="slic_warmup",
        payload=payload,
        dedupe_key=dedupe_key,
        entity_type="dataset",
        entity_id=dataset_id,
    )
    if created:
        flash(f"Queued SLIC warmup job #{job_id} for {image_count} image(s).", "success")
    else:
        flash(f"SLIC warmup job #{job_id} is already queued or running.", "warning")
    return redirect(
        url_for("main.workspace_dataset_detail", workspace_id=workspace_id, dataset_id=dataset_id)
        + "#draft-images-section"
    )


@bp.route("/workspace/<int:workspace_id>/datasets/<int:dataset_id>/images/add", methods=["POST"])
def add_workspace_dataset_images(workspace_id: int, dataset_id: int) -> str:
    workspace = _workspace_by_id_or_404(workspace_id)
    storage = _storage()
    dataset_project = storage.get_workspace_dataset(workspace_id, dataset_id)
    if dataset_project is None:
        abort(404, description=f"Dataset project id={dataset_id} not found in workspace.")

    selected: list[str] = []
    seen: set[str] = set()
    for item in request.form.getlist("selected_images"):
        image_name = safe_image_name(item)
        if not image_name or image_name in seen:
            continue
        seen.add(image_name)
        selected.append(image_name)
    if not selected:
        flash("Select at least one workspace image to add.", "error")
        return redirect(url_for("main.workspace_dataset_detail", workspace_id=workspace_id, dataset_id=dataset_id))

    workspace_images_dir = _workspace_images_dir(workspace)
    draft_images_dir = Path(str(dataset_project["images_dir"])).resolve()
    existing = set(list_images(draft_images_dir))
    copied_count = 0
    skipped_count = 0
    for image_name in selected:
        source_path = (workspace_images_dir / image_name).resolve()
        destination = (draft_images_dir / image_name).resolve()
        if image_name in existing:
            skipped_count += 1
            continue
        if not source_path.exists() or not source_path.is_file():
            skipped_count += 1
            continue
        shutil.copy2(source_path, destination)
        copied_count += 1

    if copied_count > 0:
        storage.bump_labeler_project_version(dataset_id)
        flash(f"Added {copied_count} image(s) to draft dataset.", "success")
    else:
        flash("No new images were added to the draft dataset.", "warning")
    if skipped_count > 0:
        flash(f"Skipped {skipped_count} image(s) already present or unavailable.", "warning")
    return redirect(url_for("main.workspace_dataset_detail", workspace_id=workspace_id, dataset_id=dataset_id))


@bp.route("/workspace/<int:workspace_id>/datasets/<int:dataset_id>/images/<path:image_name>/remove", methods=["POST"])
def remove_workspace_dataset_image(workspace_id: int, dataset_id: int, image_name: str) -> str:
    _workspace_by_id_or_404(workspace_id)
    storage = _storage()
    dataset_project = storage.get_workspace_dataset(workspace_id, dataset_id)
    if dataset_project is None:
        abort(404, description=f"Dataset project id={dataset_id} not found in workspace.")

    outcome = _remove_dataset_image_assets(
        storage=storage,
        dataset_project=dataset_project,
        dataset_id=dataset_id,
        image_name=image_name,
    )
    if not outcome.get("ok"):
        reason = str(outcome.get("reason", "unknown"))
        if reason == "invalid_name":
            flash("Select a valid image to remove.", "error")
        elif reason == "missing_image":
            flash(f"Image '{outcome.get('image_name', image_name)}' was not found in this draft dataset.", "error")
        else:
            flash(f"Failed to remove image '{image_name}'.", "error")
        return redirect(url_for("main.workspace_dataset_detail", workspace_id=workspace_id, dataset_id=dataset_id))

    storage.bump_labeler_project_version(dataset_id)
    safe_name = str(outcome.get("image_name", image_name))
    removed_mask_files = int(outcome.get("removed_mask_files", 0))
    removed_cache_files = int(outcome.get("removed_cache_files", 0))
    flash(
        (
            f"Removed image '{safe_name}' from the draft "
            f"(mask files removed={removed_mask_files}, cache files removed={removed_cache_files})."
        ),
        "success",
    )
    return redirect(url_for("main.workspace_dataset_detail", workspace_id=workspace_id, dataset_id=dataset_id))


@bp.route("/workspace/<int:workspace_id>/datasets/<int:dataset_id>/images/remove", methods=["POST"])
def remove_workspace_dataset_images(workspace_id: int, dataset_id: int) -> str:
    _workspace_by_id_or_404(workspace_id)
    storage = _storage()
    dataset_project = storage.get_workspace_dataset(workspace_id, dataset_id)
    if dataset_project is None:
        abort(404, description=f"Dataset project id={dataset_id} not found in workspace.")

    selected: list[str] = []
    seen: set[str] = set()
    for item in request.form.getlist("selected_images"):
        image_name = safe_image_name(item)
        if not image_name:
            continue
        key = image_name.lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append(image_name)

    if not selected:
        flash("Select at least one draft image to remove.", "error")
        return redirect(url_for("main.workspace_dataset_detail", workspace_id=workspace_id, dataset_id=dataset_id))

    if not _is_truthy(request.form.get("confirm_remove")):
        flash("Confirm bulk removal before deleting draft images.", "error")
        return redirect(url_for("main.workspace_dataset_detail", workspace_id=workspace_id, dataset_id=dataset_id))

    removed_count = 0
    skipped_count = 0
    removed_mask_files = 0
    removed_cache_files = 0
    removed_with_masks = 0
    for image_name in selected:
        outcome = _remove_dataset_image_assets(
            storage=storage,
            dataset_project=dataset_project,
            dataset_id=dataset_id,
            image_name=image_name,
        )
        if not outcome.get("ok"):
            skipped_count += 1
            continue
        removed_count += 1
        removed_mask_files += int(outcome.get("removed_mask_files", 0))
        removed_cache_files += int(outcome.get("removed_cache_files", 0))
        if outcome.get("had_masks"):
            removed_with_masks += 1

    if removed_count > 0:
        storage.bump_labeler_project_version(dataset_id)
        flash(
            (
                f"Removed {removed_count} draft image(s). "
                f"Mask files removed={removed_mask_files}, cache files removed={removed_cache_files}. "
                "Workspace originals were not deleted."
            ),
            "success",
        )
        if removed_with_masks > 0:
            flash(
                f"{removed_with_masks} removed image(s) had existing masks; those masks were also removed.",
                "warning",
            )
    if skipped_count > 0:
        flash(f"Skipped {skipped_count} image(s) that were missing or invalid.", "warning")
    if removed_count <= 0 and skipped_count <= 0:
        flash("No draft images were removed.", "warning")

    return redirect(url_for("main.workspace_dataset_detail", workspace_id=workspace_id, dataset_id=dataset_id))


@bp.route("/workspace/<int:workspace_id>/datasets/<int:dataset_id>/branch", methods=["POST"])
def branch_workspace_dataset(workspace_id: int, dataset_id: int) -> str:
    workspace = _workspace_by_id_or_404(workspace_id)
    storage = _storage()
    source_draft = storage.get_workspace_dataset(workspace_id, dataset_id)
    if source_draft is None:
        abort(404, description=f"Dataset project id={dataset_id} not found in workspace.")

    branch_name = str(request.form.get("branch_name", "")).strip()
    if not branch_name:
        flash("Branch draft name is required.", "error")
        return redirect(url_for("main.workspace_dataset_detail", workspace_id=workspace_id, dataset_id=dataset_id))

    source_images_dir = Path(str(source_draft["images_dir"])).resolve()
    source_masks_dir = Path(str(source_draft["masks_dir"])).resolve()
    source_coco_dir = Path(str(source_draft["coco_dir"])).resolve()
    branch_root = (_workspace_datasets_root(workspace) / f"{_slugify(branch_name)}_{uuid.uuid4().hex[:8]}").resolve()
    branch_images_dir = (branch_root / "images").resolve()
    branch_masks_dir = (branch_root / "masks").resolve()
    branch_coco_dir = (branch_root / "coco").resolve()
    branch_cache_dir = (branch_masks_dir / "_cache").resolve()

    ensure_dir(branch_images_dir)
    ensure_dir(branch_masks_dir)
    ensure_dir(branch_coco_dir)
    ensure_dir(branch_cache_dir)

    try:
        for image_name in list_images(source_images_dir):
            shutil.copy2(source_images_dir / image_name, branch_images_dir / image_name)
        for mask_path in source_masks_dir.glob("*.png"):
            if mask_path.is_file():
                shutil.copy2(mask_path, branch_masks_dir / mask_path.name)
        for coco_path in source_coco_dir.glob("*"):
            if coco_path.is_file():
                shutil.copy2(coco_path, branch_coco_dir / coco_path.name)
    except Exception as exc:
        if branch_root.exists():
            shutil.rmtree(branch_root, ignore_errors=True)
        flash(f"Failed to create branch draft content: {exc}", "error")
        return redirect(url_for("main.workspace_dataset_detail", workspace_id=workspace_id, dataset_id=dataset_id))

    try:
        branched_project_id = storage.create_workspace_dataset(
            workspace_id=workspace_id,
            name=branch_name,
            project_dir=str(branch_root),
            images_dir=str(branch_images_dir),
            masks_dir=str(branch_masks_dir),
            coco_dir=str(branch_coco_dir),
            cache_dir=str(branch_cache_dir),
            categories=_dataset_project_classes(source_draft),
            slic_algorithm=str(source_draft.get("slic_algorithm", "slic")),
            slic_preset_name=str(source_draft.get("slic_preset_name", "medium")),
            slic_detail_level=str(source_draft.get("slic_detail_level", "medium")),
            slic_n_segments=int(source_draft.get("slic_n_segments") or current_app.config["LABELER_SLIC_N_SEGMENTS"]),
            slic_compactness=float(
                source_draft.get("slic_compactness") or current_app.config["LABELER_SLIC_COMPACTNESS"]
            ),
            slic_sigma=float(source_draft.get("slic_sigma") or current_app.config["LABELER_SLIC_SIGMA"]),
            slic_colorspace=str(source_draft.get("slic_colorspace", "lab")),
            quickshift_ratio=float(source_draft.get("quickshift_ratio") or 1.0),
            quickshift_kernel_size=int(source_draft.get("quickshift_kernel_size") or 5),
            quickshift_max_dist=float(source_draft.get("quickshift_max_dist") or 10.0),
            quickshift_sigma=float(source_draft.get("quickshift_sigma") or 0.0),
            felzenszwalb_scale=float(source_draft.get("felzenszwalb_scale") or 100.0),
            felzenszwalb_sigma=float(source_draft.get("felzenszwalb_sigma") or 0.8),
            felzenszwalb_min_size=int(source_draft.get("felzenszwalb_min_size") or 50),
            texture_enabled=bool(source_draft.get("texture_enabled")),
            texture_mode=str(source_draft.get("texture_mode", "append_to_color")),
            texture_lbp_enabled=bool(source_draft.get("texture_lbp_enabled")),
            texture_lbp_points=int(source_draft.get("texture_lbp_points") or 8),
            texture_lbp_radii=list(source_draft.get("texture_lbp_radii_json") or [1]),
            texture_lbp_method=str(source_draft.get("texture_lbp_method", "uniform")),
            texture_lbp_normalize=bool(source_draft.get("texture_lbp_normalize", True)),
            texture_gabor_enabled=bool(source_draft.get("texture_gabor_enabled")),
            texture_gabor_frequencies=list(source_draft.get("texture_gabor_frequencies_json") or [0.1, 0.2]),
            texture_gabor_thetas=list(source_draft.get("texture_gabor_thetas_json") or [0.0, 45.0, 90.0, 135.0]),
            texture_gabor_bandwidth=float(source_draft.get("texture_gabor_bandwidth") or 1.0),
            texture_gabor_include_real=bool(source_draft.get("texture_gabor_include_real")),
            texture_gabor_include_imag=bool(source_draft.get("texture_gabor_include_imag")),
            texture_gabor_include_magnitude=bool(source_draft.get("texture_gabor_include_magnitude", True)),
            texture_gabor_normalize=bool(source_draft.get("texture_gabor_normalize", True)),
            texture_weight_color=float(source_draft.get("texture_weight_color") or 1.0),
            texture_weight_lbp=float(source_draft.get("texture_weight_lbp") or 0.25),
            texture_weight_gabor=float(source_draft.get("texture_weight_gabor") or 0.25),
            draft_version=1,
            origin_type="branched_from_draft",
            origin_draft_project_id=int(source_draft["id"]),
            origin_draft_version=_draft_version(source_draft),
            origin_registered_dataset_id=None,
        )
    except sqlite3.IntegrityError:
        if branch_root.exists():
            shutil.rmtree(branch_root, ignore_errors=True)
        flash(f"A draft dataset named '{branch_name}' already exists.", "error")
        return redirect(url_for("main.workspace_dataset_detail", workspace_id=workspace_id, dataset_id=dataset_id))
    except Exception as exc:
        if branch_root.exists():
            shutil.rmtree(branch_root, ignore_errors=True)
        flash(f"Failed to create branch draft: {exc}", "error")
        return redirect(url_for("main.workspace_dataset_detail", workspace_id=workspace_id, dataset_id=dataset_id))

    flash(f"Created branch draft '{branch_name}' from '{source_draft['name']}'.", "success")
    return redirect(
        url_for(
            "main.workspace_dataset_detail",
            workspace_id=workspace_id,
            dataset_id=branched_project_id,
        )
    )


@bp.route("/workspace/<int:workspace_id>/datasets/<int:dataset_id>/register", methods=["POST"])
def register_workspace_dataset(workspace_id: int, dataset_id: int) -> str:
    workspace = _workspace_by_id_or_404(workspace_id)
    storage = _storage()
    dataset_project = storage.get_workspace_dataset(workspace_id, dataset_id)
    if dataset_project is None:
        abort(404, description=f"Dataset project id={dataset_id} not found in workspace.")

    if not _is_truthy(request.form.get("confirm_immutable")):
        flash("Confirm snapshot immutability before registering this draft.", "error")
        return redirect(
            url_for("main.workspace_dataset_detail", workspace_id=workspace_id, dataset_id=dataset_id)
            + "#snapshots-lineage"
        )

    if not _dataset_project_categories(dataset_project):
        flash("Add at least one class before registering this dataset.", "error")
        return redirect(url_for("main.workspace_dataset_detail", workspace_id=workspace_id, dataset_id=dataset_id))
    category_entries = _dataset_project_classes(dataset_project)

    requested_name = str(request.form.get("registered_name", "")).strip()
    dataset_name = requested_name or str(dataset_project["name"])
    draft_version = _draft_version(dataset_project)

    try:
        coco_output_path, image_count, annotation_count = export_coco_annotations(
            images_dir=Path(str(dataset_project["images_dir"])).resolve(),
            masks_dir=Path(str(dataset_project["masks_dir"])).resolve(),
            coco_dir=Path(str(dataset_project["coco_dir"])).resolve(),
            categories=category_entries,
            min_area=int(current_app.config["LABELER_MIN_COCO_AREA"]),
        )
        parsed_categories = parse_categories(load_coco(coco_output_path))
        checksum = compute_sha256(coco_output_path)
        project_dir = Path(str(dataset_project["project_dir"])).resolve()
        images_dir = Path(str(dataset_project["images_dir"])).resolve()
        try:
            image_root = images_dir.relative_to(project_dir).as_posix()
        except ValueError:
            image_root = str(images_dir)
        spec = DatasetSpec(
            name=dataset_name,
            dataset_path=str(project_dir),
            image_root=image_root,
            coco_json_path=str(coco_output_path),
        )
        registered_dataset_id = storage.create_dataset(
            spec,
            coco_checksum=checksum,
            categories=parsed_categories,
            workspace_id=workspace_id,
            source_draft_project_id=int(dataset_project["id"]),
            source_draft_version=draft_version,
        )
        storage.link_labeler_project_dataset(dataset_id, registered_dataset_id)
    except ModuleNotFoundError as exc:
        flash(f"Missing dependency: {exc.name}. Install requirements first.", "error")
        return redirect(url_for("main.workspace_dataset_detail", workspace_id=workspace_id, dataset_id=dataset_id))
    except sqlite3.IntegrityError:
        flash(
            (
                f"A registered dataset named '{dataset_name}' already exists. "
                "Choose a different name to register another snapshot."
            ),
            "error",
        )
        return redirect(url_for("main.workspace_dataset_detail", workspace_id=workspace_id, dataset_id=dataset_id))
    except Exception as exc:
        flash(f"Failed to register dataset: {exc}", "error")
        return redirect(url_for("main.workspace_dataset_detail", workspace_id=workspace_id, dataset_id=dataset_id))

    flash(
        (
            f"Dataset '{dataset_name}' registered for training "
            f"from draft v{draft_version} (images={image_count}, annotations={annotation_count})."
        ),
        "success",
    )
    return redirect(
        url_for(
            "main.workspace_registered_dataset_detail",
            workspace_id=workspace_id,
            dataset_id=registered_dataset_id,
        )
    )


@bp.route("/workspace/<int:workspace_id>/registered-datasets/<int:dataset_id>", methods=["GET"])
def workspace_registered_dataset_detail(workspace_id: int, dataset_id: int) -> str:
    workspace = _workspace_by_id_or_404(workspace_id)
    storage = _storage()
    dataset = storage.get_dataset(dataset_id, workspace_id=workspace_id)
    if dataset is None:
        abort(404, description=f"Registered dataset id={dataset_id} not found in workspace.")
    dataset = dict(dataset)
    dataset["image_count"] = _registered_dataset_image_count(dataset)

    source_draft_project = None
    source_draft_id = dataset.get("source_draft_project_id")
    try:
        normalized_source_draft_id = int(source_draft_id)
    except (TypeError, ValueError):
        normalized_source_draft_id = None
    if normalized_source_draft_id is not None:
        source_draft_project = storage.get_workspace_dataset(workspace_id, normalized_source_draft_id)

    related_models = [
        model
        for model in storage.list_models(workspace_id=workspace_id)
        if int(model["dataset_id"]) == dataset_id
    ]
    return render_template(
        "registered_dataset_detail.html",
        workspace=workspace,
        dataset=dataset,
        source_draft_project=source_draft_project,
        related_models=related_models,
    )


@bp.route("/workspace/<int:workspace_id>/registered-datasets/images", methods=["GET"])
def workspace_registered_dataset_images(workspace_id: int):
    _workspace_by_id_or_404(workspace_id)
    storage = _storage()
    dataset_id = request.args.get("dataset_id", type=int)
    if dataset_id is None or dataset_id <= 0:
        return jsonify({"error": "dataset_id query param is required."}), 400
    dataset = storage.get_dataset(dataset_id, workspace_id=workspace_id)
    if dataset is None:
        abort(404, description=f"Registered dataset id={dataset_id} not found in workspace.")
    image_names = _registered_dataset_image_names(dataset)
    return jsonify(
        {
            "workspace_id": workspace_id,
            "dataset_id": dataset_id,
            "count": len(image_names),
            "images": image_names,
        }
    )


@bp.route("/workspace/<int:workspace_id>/datasets/augment", methods=["GET", "POST"])
@bp.route("/augment/new", methods=["GET", "POST"])
def augment_new(workspace_id: int | None = None) -> str:
    workspace = _resolve_workspace(workspace_id)
    workspace_id = int(workspace["id"])
    storage = _storage()
    datasets = storage.list_datasets(workspace_id=workspace_id)
    selected_source_dataset_id = request.args.get("source_dataset_id", type=int)

    if request.method == "GET":
        return render_template(
            "augment_new.html",
            workspace=workspace,
            datasets=datasets,
            selected_source_dataset_id=selected_source_dataset_id,
        )

    if not datasets:
        flash("Register a dataset before running augmentation.", "error")
        return (
            render_template(
                "augment_new.html",
                workspace=workspace,
                datasets=datasets,
                selected_source_dataset_id=selected_source_dataset_id,
            ),
            400,
        )

    source_dataset_id_raw = str(request.form.get("source_dataset_id", "")).strip()
    output_name = str(request.form.get("output_name", "")).strip() or "augmented_dataset"

    try:
        source_dataset_id = int(source_dataset_id_raw)
    except ValueError:
        flash("Select a valid source dataset.", "error")
        return (
            render_template(
                "augment_new.html",
                workspace=workspace,
                datasets=datasets,
                selected_source_dataset_id=selected_source_dataset_id,
            ),
            400,
        )

    source_dataset = storage.get_dataset(source_dataset_id, workspace_id=workspace_id)
    if source_dataset is None:
        flash("Selected source dataset was not found.", "error")
        return (
            render_template(
                "augment_new.html",
                workspace=workspace,
                datasets=datasets,
                selected_source_dataset_id=selected_source_dataset_id,
            ),
            400,
        )

    include_original = _is_truthy(request.form.get("include_original"))
    recipe_json = str(request.form.get("recipe_json", "")).strip()
    try:
        min_area = int(str(request.form.get("min_area", "1")).strip() or "1")
    except ValueError:
        flash("Minimum area must be an integer.", "error")
        return (
            render_template(
                "augment_new.html",
                workspace=workspace,
                datasets=datasets,
                selected_source_dataset_id=source_dataset_id,
            ),
            400,
        )

    try:
        recipe_steps = _parse_augmentation_recipe(recipe_json)
    except ValueError as exc:
        flash(str(exc), "error")
        return (
            render_template(
                "augment_new.html",
                workspace=workspace,
                datasets=datasets,
                selected_source_dataset_id=source_dataset_id,
            ),
            400,
        )

    if not include_original and not recipe_steps:
        flash("Add at least one recipe step or include originals.", "error")
        return (
            render_template(
                "augment_new.html",
                workspace=workspace,
                datasets=datasets,
                selected_source_dataset_id=source_dataset_id,
            ),
            400,
        )

    logs: list[str] = []

    def log(message: str) -> None:
        logs.append(message)

    output_root = (
        Path(current_app.config["BASE_DIR"]).resolve()
        / "workspace"
        / f"workspace_{workspace_id}"
        / "augmentations"
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        result = augment_coco_dataset(
            dataset=source_dataset,
            output_root=output_root,
            output_name=output_name,
            recipe_steps=recipe_steps,
            include_original=include_original,
            min_area=max(1, min_area),
            random_seed=_workspace_augmentation_seed(workspace),
            log_fn=log,
        )
    except ModuleNotFoundError as exc:
        flash(f"Missing dependency: {exc.name}. Install requirements first.", "error")
        return (
            render_template(
                "augment_new.html",
                workspace=workspace,
                datasets=datasets,
                selected_source_dataset_id=source_dataset_id,
            ),
            500,
        )
    except Exception as exc:
        flash(f"Augmentation failed: {exc}", "error")
        return (
            render_template(
                "augment_new.html",
                workspace=workspace,
                datasets=datasets,
                selected_source_dataset_id=source_dataset_id,
            ),
            500,
        )

    draft_dataset_id: int | None = None
    draft_project: dict[str, Any] | None = None
    seeded_mask_files = 0
    seed_warnings: list[str] = []
    try:
        augmented_coco = load_coco(Path(result.coco_json_path))
        parsed_categories = parse_categories(augmented_coco)
        category_names = [str(item.get("name", "")).strip() for item in parsed_categories if str(item.get("name", "")).strip()]
        if not category_names:
            category_names = _default_labeler_categories()

        draft_root = Path(result.output_dir).resolve()
        draft_images_dir = Path(result.images_dir).resolve()
        draft_masks_dir = (draft_root / "masks").resolve()
        draft_coco_dir = (draft_root / "coco").resolve()
        draft_cache_dir = (draft_masks_dir / "_cache").resolve()
        ensure_dir(draft_masks_dir)
        ensure_dir(draft_coco_dir)
        ensure_dir(draft_cache_dir)

        source_annotations_copy = draft_coco_dir / "source_augmented_annotations.json"
        shutil.copy2(Path(result.coco_json_path).resolve(), source_annotations_copy)

        seeded_mask_files, seed_warnings = _seed_masks_from_coco_annotations(
            coco_payload=augmented_coco,
            images_dir=draft_images_dir,
            masks_dir=draft_masks_dir,
            parsed_categories=parsed_categories,
        )

        draft_dataset_id = storage.create_workspace_dataset(
            workspace_id=workspace_id,
            name=output_name,
            project_dir=str(draft_root),
            images_dir=str(draft_images_dir),
            masks_dir=str(draft_masks_dir),
            coco_dir=str(draft_coco_dir),
            cache_dir=str(draft_cache_dir),
            categories=category_names,
            slic_algorithm="slic",
            slic_preset_name="medium",
            slic_detail_level="medium",
            slic_n_segments=int(current_app.config["LABELER_SLIC_N_SEGMENTS"]),
            slic_compactness=float(current_app.config["LABELER_SLIC_COMPACTNESS"]),
            slic_sigma=float(current_app.config["LABELER_SLIC_SIGMA"]),
            slic_colorspace="lab",
            draft_version=1,
            origin_type="augmented_from_registered",
            origin_draft_project_id=None,
            origin_draft_version=None,
            origin_registered_dataset_id=source_dataset_id,
        )
        draft_project = storage.get_workspace_dataset(workspace_id, draft_dataset_id)
        flash(
            (
                f"Augmentation complete. Created editable draft '{output_name}' "
                f"with {result.image_count} image(s)."
            ),
            "success",
        )
    except sqlite3.IntegrityError:
        flash(
            (
                f"Augmentation completed, but a draft dataset named '{output_name}' already exists. "
                "Use a different output name."
            ),
            "error",
        )
    except Exception as exc:
        flash(f"Augmentation completed, but draft creation failed: {exc}", "warning")

    return render_template(
        "augment_result.html",
        workspace=workspace,
        source_dataset=source_dataset,
        result=result,
        logs=logs,
        draft_dataset=draft_project,
        seeded_mask_files=seeded_mask_files,
        seed_warnings=seed_warnings,
    )


@bp.route("/workspace/<int:workspace_id>/models", methods=["GET"])
def workspace_models(workspace_id: int) -> str:
    workspace = _workspace_by_id_or_404(workspace_id)
    storage = _storage()
    models = storage.list_models(workspace_id=workspace_id)
    datasets = storage.list_datasets(workspace_id=workspace_id)
    return render_template(
        "models_index.html",
        workspace=workspace,
        models=models,
        datasets=datasets,
        default_seed=current_app.config["RANDOM_SEED"],
    )


@bp.route("/workspace/<int:workspace_id>/models/train", methods=["POST"])
@bp.route("/models/train", methods=["POST"])
def train_new_model(workspace_id: int | None = None) -> str:
    workspace = _resolve_workspace(workspace_id)
    workspace_id = int(workspace["id"])
    storage = _storage()
    model_name = str(request.form.get("model_name", "")).strip()
    dataset_id_raw = str(request.form.get("dataset_id", "")).strip()
    if not model_name or not dataset_id_raw:
        flash("Model name and dataset are required.", "error")
        return redirect(url_for("main.workspace_models", workspace_id=workspace_id))

    try:
        dataset_id = int(dataset_id_raw)
    except ValueError:
        flash("Invalid dataset selection.", "error")
        return redirect(url_for("main.workspace_models", workspace_id=workspace_id))

    dataset = storage.get_dataset(dataset_id, workspace_id=workspace_id)
    if dataset is None:
        flash("Selected dataset does not exist in this workspace.", "error")
        return redirect(url_for("main.workspace_models", workspace_id=workspace_id))

    try:
        feature_config = FeatureConfig.from_form(request.form)
        train_config = TrainConfig.from_form(
            request.form,
            default_seed=int(current_app.config["RANDOM_SEED"]),
        )
    except Exception as exc:
        flash(f"Invalid training configuration: {exc}", "error")
        return redirect(url_for("main.workspace_models", workspace_id=workspace_id))

    feature_payload = feature_config.to_dict()
    train_payload = train_config.to_dict()
    dedupe_key = build_training_dedupe_key(
        workspace_id=workspace_id,
        model_name=model_name,
        dataset_id=dataset_id,
        feature_config=feature_payload,
        train_config=train_payload,
    )
    existing_job = storage.find_active_job_by_dedupe(
        dedupe_key=dedupe_key,
        workspace_id=workspace_id,
        job_type="training",
    )
    if existing_job is not None:
        existing_id = int(existing_job["id"])
        existing_status = str(existing_job.get("status", "queued"))
        flash(
            f"Training job #{existing_id} is already {existing_status} for this configuration.",
            "warning",
        )
        return redirect(url_for("main.workspace_jobs", workspace_id=workspace_id))

    hyperparams_payload = {
        "n_estimators": int(train_config.n_estimators),
        "max_depth": train_config.max_depth,
        "min_samples_split": int(train_config.min_samples_split),
        "min_samples_leaf": int(train_config.min_samples_leaf),
        "class_weight": str(train_config.class_weight),
    }
    model_id = storage.create_model(
        name=model_name,
        dataset_id=dataset_id,
        classes=[],
        feature_config=feature_payload,
        hyperparams=hyperparams_payload,
        metrics={},
        artifact_dir="",
        model_path="",
        metadata_path="",
        status="queued",
        error_message=None,
        workspace_id=workspace_id,
    )
    payload = {
        "workspace_id": workspace_id,
        "model_id": model_id,
        "model_name": model_name,
        "dataset_id": dataset_id,
        "feature_config": feature_payload,
        "train_config": train_payload,
    }
    job_id, created = storage.enqueue_job(
        workspace_id=workspace_id,
        job_type="training",
        payload=payload,
        dedupe_key=dedupe_key,
        entity_type="model",
        entity_id=model_id,
        rerun_of_job_id=None,
    )
    if not created:
        storage.update_model(
            model_id,
            status="canceled",
            error_message="Duplicate training request suppressed.",
            workspace_id=workspace_id,
        )
        flash(f"Training job #{job_id} is already queued or running.", "warning")
    else:
        flash(f"Training job queued as #{job_id}.", "success")
    return redirect(url_for("main.workspace_models", workspace_id=workspace_id))


@bp.route("/workspace/<int:workspace_id>/models/<int:model_id>", methods=["GET"])
@bp.route("/models/<int:model_id>", methods=["GET"])
def model_details(model_id: int, workspace_id: int | None = None) -> str:
    workspace = _resolve_workspace(workspace_id)
    workspace_id = int(workspace["id"])
    storage = _storage()
    model = storage.get_model(model_id, workspace_id=workspace_id)
    if model is None:
        abort(404, description=f"Model id={model_id} not found.")

    metrics = model.get("metrics_json")
    if not isinstance(metrics, dict):
        metrics = {}

    feature_config_rows = _flatten_mapping_rows(model.get("feature_config_json"))
    hyperparam_rows = _flatten_mapping_rows(model.get("hyperparams_json"))

    headline_metrics = [
        {
            "key": key,
            "label": label,
            "value": metrics.get(key),
            "display": _format_metric_value(metrics.get(key)),
        }
        for key, label in MODEL_HEADLINE_METRIC_SPECS
    ]

    sampled_pixels_by_class = _normalize_named_counts(metrics.get("sampled_pixels_by_class"))
    validation_sampled_pixels_by_class = _normalize_named_counts(metrics.get("validation_sampled_pixels_by_class"))

    model_classes = model.get("classes_json")
    if not isinstance(model_classes, list):
        model_classes = []

    class_order = _ordered_class_names(
        model_classes,
        tuple(sampled_pixels_by_class.keys()),
        tuple(validation_sampled_pixels_by_class.keys()),
    )

    per_class_iou_rows = _build_per_class_iou_rows(metrics.get("per_class_iou"), class_order)
    sampling_rows = _build_sampling_rows(
        sampled_pixels_by_class,
        validation_sampled_pixels_by_class,
        class_order,
    )
    train_distribution_rows = _build_distribution_rows(sampled_pixels_by_class, class_order)
    validation_distribution_rows = _build_distribution_rows(validation_sampled_pixels_by_class, class_order)
    confusion_matrix = _extract_confusion_matrix(metrics, class_order)

    has_incomplete_metrics = any(row["status"] == "incomplete" for row in per_class_iou_rows)

    dataset_id = int(model["dataset_id"])
    dataset_record = storage.get_dataset(dataset_id, workspace_id=workspace_id)
    dataset_detail_url = (
        url_for(
            "main.workspace_registered_dataset_detail",
            workspace_id=workspace_id,
            dataset_id=dataset_id,
        )
        if dataset_record is not None
        else None
    )

    path_rows = [
        {"label": "Dataset Path", "value": str(model.get("dataset_path", ""))},
        {"label": "Image Root", "value": str(model.get("dataset_image_root", ""))},
        {"label": "COCO JSON", "value": str(model.get("dataset_coco_json_path", ""))},
        {"label": "Artifact Dir", "value": str(model.get("artifact_dir", ""))},
        {"label": "Model File", "value": str(model.get("model_path", ""))},
        {"label": "Metadata File", "value": str(model.get("metadata_path", ""))},
    ]

    return render_template(
        "model_detail.html",
        workspace=workspace,
        model=model,
        metrics=metrics,
        dataset_detail_url=dataset_detail_url,
        headline_metrics=headline_metrics,
        per_class_iou_rows=per_class_iou_rows,
        sampling_rows=sampling_rows,
        train_distribution_rows=train_distribution_rows,
        validation_distribution_rows=validation_distribution_rows,
        confusion_matrix=confusion_matrix,
        has_incomplete_metrics=has_incomplete_metrics,
        feature_config_rows=feature_config_rows,
        hyperparam_rows=hyperparam_rows,
        path_rows=path_rows,
    )


@bp.route("/workspace/<int:workspace_id>/analysis", methods=["GET"])
def workspace_analysis(workspace_id: int) -> str:
    workspace = _workspace_by_id_or_404(workspace_id)
    runs = _storage().list_runs(workspace_id=workspace_id, limit=100)
    return render_template("analysis_index.html", workspace=workspace, runs=runs)


@bp.route("/workspace/<int:workspace_id>/analysis/new", methods=["GET", "POST"])
@bp.route("/analysis/new", methods=["GET", "POST"])
def new_analysis(workspace_id: int | None = None) -> str:
    workspace = _resolve_workspace(workspace_id)
    workspace_id = int(workspace["id"])
    storage = _storage()
    models = storage.list_models(workspace_id=workspace_id)
    workspace_images_dir = _workspace_images_dir(workspace)
    workspace_image_names = list_images(workspace_images_dir)
    workspace_images = [
        {
            "filename": image_name,
            "thumbnail_url": url_for(
                "labeler.api_thumbnail",
                workspace_id=workspace_id,
                image_name=image_name,
            ),
        }
        for image_name in workspace_image_names
    ]
    available_images_by_lower = {name.lower(): name for name in workspace_image_names}
    selected_model_id = request.args.get("model_id", type=int)
    selected_images: list[str] = []
    selection_mode = "manual"
    dataset_id_raw = ""
    postprocess_form: dict[str, Any] = {
        "enabled": False,
        "slic_target_superpixel_area_px": "900",
        "slic_n_segments": "",
        "slic_compactness": "10",
        "lambda_smooth": "0.7",
        "edge_awareness": "0.5",
        "iterations": "8",
        "temperature": "1.0",
        "min_region_area_px": "0",
    }

    def _coerce_int(raw_value: Any, default_value: int, minimum: int, maximum: int | None = None) -> int:
        try:
            parsed = int(str(raw_value).strip())
        except (TypeError, ValueError):
            parsed = int(default_value)
        parsed = max(minimum, parsed)
        if maximum is not None:
            parsed = min(maximum, parsed)
        return parsed

    def _coerce_float(
        raw_value: Any,
        default_value: float,
        minimum: float,
        maximum: float | None = None,
    ) -> float:
        try:
            parsed = float(str(raw_value).strip())
        except (TypeError, ValueError):
            parsed = float(default_value)
        parsed = max(minimum, parsed)
        if maximum is not None:
            parsed = min(maximum, parsed)
        return parsed

    def _build_postprocess_config() -> tuple[dict[str, Any], list[str]]:
        enabled = str(request.form.get("postprocess_enabled", "")).strip() == "1"
        postprocess_form["enabled"] = enabled
        for key in (
            "slic_target_superpixel_area_px",
            "slic_n_segments",
            "slic_compactness",
            "lambda_smooth",
            "edge_awareness",
            "iterations",
            "temperature",
            "min_region_area_px",
        ):
            postprocess_form[key] = str(request.form.get(key, postprocess_form[key])).strip()

        config = {
            "enabled": enabled,
            "slic_target_superpixel_area_px": _coerce_int(
                postprocess_form["slic_target_superpixel_area_px"],
                default_value=900,
                minimum=1,
                maximum=250000,
            ),
            "slic_n_segments": None,
            "slic_compactness": _coerce_float(
                postprocess_form["slic_compactness"],
                default_value=10.0,
                minimum=0.01,
                maximum=10000.0,
            ),
            "slic_sigma": 0.0,
            "lambda_smooth": _coerce_float(
                postprocess_form["lambda_smooth"],
                default_value=0.7,
                minimum=0.0,
                maximum=100.0,
            ),
            "edge_awareness": _coerce_float(
                postprocess_form["edge_awareness"],
                default_value=0.5,
                minimum=0.0,
                maximum=1.0,
            ),
            "iterations": _coerce_int(
                postprocess_form["iterations"],
                default_value=8,
                minimum=1,
                maximum=200,
            ),
            "temperature": _coerce_float(
                postprocess_form["temperature"],
                default_value=1.0,
                minimum=0.01,
                maximum=50.0,
            ),
            "min_region_area_px": _coerce_int(
                postprocess_form["min_region_area_px"],
                default_value=0,
                minimum=0,
                maximum=200000,
            ),
        }

        raw_n_segments = str(postprocess_form.get("slic_n_segments", "")).strip()
        if raw_n_segments:
            config["slic_n_segments"] = _coerce_int(
                raw_n_segments,
                default_value=0,
                minimum=1,
                maximum=200000,
            )
        else:
            config["slic_n_segments"] = None

        if not enabled:
            return config, []

        errors: list[str] = []

        raw_target_area = str(postprocess_form.get("slic_target_superpixel_area_px", "")).strip()
        try:
            target_area = int(raw_target_area)
            if target_area <= 0:
                raise ValueError
        except ValueError:
            errors.append("Superpixel target area must be a positive integer.")

        if raw_n_segments:
            try:
                n_segments_value = int(raw_n_segments)
                if n_segments_value <= 0:
                    raise ValueError
            except ValueError:
                errors.append("Advanced n_segments override must be a positive integer when provided.")

        raw_compactness = str(postprocess_form.get("slic_compactness", "")).strip()
        try:
            compactness_value = float(raw_compactness)
            if compactness_value <= 0:
                raise ValueError
        except ValueError:
            errors.append("SLIC compactness must be greater than 0.")

        raw_lambda = str(postprocess_form.get("lambda_smooth", "")).strip()
        try:
            lambda_value = float(raw_lambda)
            if lambda_value < 0:
                raise ValueError
        except ValueError:
            errors.append("Lambda must be 0 or greater.")

        raw_edge_awareness = str(postprocess_form.get("edge_awareness", "")).strip()
        try:
            edge_awareness_value = float(raw_edge_awareness)
            if edge_awareness_value < 0 or edge_awareness_value > 1:
                raise ValueError
        except ValueError:
            errors.append("Edge awareness must be between 0 and 1.")

        raw_iterations = str(postprocess_form.get("iterations", "")).strip()
        try:
            iterations_value = int(raw_iterations)
            if iterations_value <= 0:
                raise ValueError
        except ValueError:
            errors.append("Iterations must be a positive integer.")

        raw_temperature = str(postprocess_form.get("temperature", "")).strip()
        try:
            temperature_value = float(raw_temperature)
            if temperature_value <= 0:
                raise ValueError
        except ValueError:
            errors.append("Temperature must be greater than 0.")

        raw_min_region_area = str(postprocess_form.get("min_region_area_px", "")).strip()
        try:
            min_region_area_value = int(raw_min_region_area)
            if min_region_area_value < 0:
                raise ValueError
        except ValueError:
            errors.append("Minimum cleanup area must be 0 or greater.")

        return config, errors

    def _render_analysis_form(status_code: int = 200):
        rendered = render_template(
            "analysis_new.html",
            workspace=workspace,
            models=models,
            selected_model_id=selected_model_id,
            workspace_images=workspace_images,
            selected_images=selected_images,
            form_selection_mode=selection_mode,
            form_dataset_id=dataset_id_raw,
            postprocess_form=postprocess_form,
        )
        if status_code == 200:
            return rendered
        return rendered, status_code

    if request.method == "GET":
        return _render_analysis_form()

    selection_mode = str(request.form.get("selection_mode", "manual")).strip().lower()
    if selection_mode not in {"manual", "dataset"}:
        selection_mode = "manual"
    dataset_id_raw = str(request.form.get("dataset_id", "")).strip()

    seen_images: set[str] = set()
    invalid_submissions: list[str] = []
    for item in request.form.getlist("selected_images"):
        image_name = safe_image_name(item)
        if not image_name:
            continue
        lowered = image_name.lower()
        if lowered in seen_images:
            continue
        seen_images.add(lowered)
        canonical_name = available_images_by_lower.get(lowered)
        if canonical_name is None:
            invalid_submissions.append(image_name)
            continue
        selected_images.append(canonical_name)

    model_id_raw = str(request.form.get("model_id", "")).strip()
    try:
        model_id = int(model_id_raw)
    except ValueError:
        flash("Please select a valid trained model.", "error")
        selected_model_id = None
        return _render_analysis_form(400)
    selected_model_id = model_id

    model = storage.get_model(model_id, workspace_id=workspace_id)
    if model is None:
        flash("Selected model does not exist.", "error")
        selected_model_id = None
        return _render_analysis_form(400)

    if invalid_submissions:
        invalid_preview = ", ".join(invalid_submissions[:5])
        suffix = " ..." if len(invalid_submissions) > 5 else ""
        flash(
            (
                "Some submitted images were invalid or outside this workspace: "
                f"{invalid_preview}{suffix}"
            ),
            "error",
        )
        return _render_analysis_form(400)

    if selection_mode == "dataset":
        flash("Dataset selection mode is not yet supported. Use Manual selection.", "error")
        return _render_analysis_form(400)

    if not selected_images:
        flash("Select at least one workspace image.", "error")
        return _render_analysis_form(400)

    postprocess_config, postprocess_errors = _build_postprocess_config()
    if postprocess_errors:
        for error in postprocess_errors:
            flash(error, "error")
        return _render_analysis_form(400)

    input_images: list[str] = []
    unavailable_images: list[str] = []
    for image_name in selected_images:
        candidate = (workspace_images_dir / image_name).resolve()
        try:
            candidate.relative_to(workspace_images_dir)
        except ValueError:
            unavailable_images.append(image_name)
            continue
        if not candidate.exists() or not candidate.is_file():
            unavailable_images.append(image_name)
            continue
        input_images.append(str(candidate))

    if unavailable_images:
        missing_preview = ", ".join(unavailable_images[:5])
        suffix = " ..." if len(unavailable_images) > 5 else ""
        flash(
            (
                "Some selected images are no longer available in this workspace: "
                f"{missing_preview}{suffix}"
            ),
            "error",
        )
        return _render_analysis_form(400)

    dedupe_key = build_analysis_dedupe_key(
        workspace_id=workspace_id,
        model_id=model_id,
        input_images=input_images,
        postprocess_config=postprocess_config,
    )
    existing_job = storage.find_active_job_by_dedupe(
        dedupe_key=dedupe_key,
        workspace_id=workspace_id,
        job_type="analysis",
    )
    if existing_job is not None:
        existing_id = int(existing_job["id"])
        existing_status = str(existing_job.get("status", "queued"))
        flash(
            f"Analysis job #{existing_id} is already {existing_status} for this selection.",
            "warning",
        )
        return redirect(url_for("main.workspace_jobs", workspace_id=workspace_id))

    run_id = storage.create_analysis_run(
        model_id=model_id,
        input_images=input_images,
        workspace_id=workspace_id,
        postprocess_enabled=bool(postprocess_config.get("enabled")),
        postprocess_config=postprocess_config,
        status="queued",
    )
    payload = {
        "workspace_id": workspace_id,
        "run_id": run_id,
        "model_id": model_id,
        "input_images": input_images,
        "postprocess_config": postprocess_config,
    }
    job_id, created = storage.enqueue_job(
        workspace_id=workspace_id,
        job_type="analysis",
        payload=payload,
        dedupe_key=dedupe_key,
        entity_type="analysis_run",
        entity_id=run_id,
        rerun_of_job_id=None,
    )
    if not created:
        storage.update_analysis_run_state(
            run_id,
            status="canceled",
            error_message="Duplicate analysis request suppressed.",
            workspace_id=workspace_id,
        )
        flash(f"Analysis job #{job_id} is already queued or running.", "warning")
    else:
        flash(f"Analysis job queued as #{job_id}.", "success")
    return redirect(url_for("main.workspace_analysis", workspace_id=workspace_id))


@bp.route("/workspace/<int:workspace_id>/analysis/<int:run_id>", methods=["GET"])
@bp.route("/analysis/<int:run_id>", methods=["GET"])
def analysis_details(run_id: int, workspace_id: int | None = None) -> str:
    workspace = _resolve_workspace(workspace_id)
    workspace_id = int(workspace["id"])
    storage = _storage()
    run = storage.get_analysis_run(run_id, workspace_id=workspace_id)
    if run is None:
        abort(404, description=f"Analysis run id={run_id} not found.")

    items = storage.get_analysis_items(run_id)
    model = storage.get_model(int(run["model_id"]), workspace_id=workspace_id)
    run_postprocess_enabled = bool(run.get("postprocess_enabled"))

    def _sorted_area_delta_rows(payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        by_class = payload.get("by_class")
        if not isinstance(by_class, dict):
            return []
        rows: list[dict[str, Any]] = []
        for class_name, values in by_class.items():
            if not isinstance(values, dict):
                continue
            rows.append(
                {
                    "class_name": str(class_name),
                    "raw_percent": float(values.get("raw_percent", 0.0)),
                    "refined_percent": float(values.get("refined_percent", 0.0)),
                    "delta_percent": float(values.get("delta_percent", 0.0)),
                    "raw_pixels": int(values.get("raw_pixels", 0)),
                    "refined_pixels": int(values.get("refined_pixels", 0)),
                    "delta_pixels": int(values.get("delta_pixels", 0)),
                }
            )
        rows.sort(key=lambda row: abs(float(row["delta_percent"])), reverse=True)
        return rows

    run_flip_stats = run.get("flip_stats_json") if isinstance(run.get("flip_stats_json"), dict) else None
    run_area_delta = run.get("area_delta_json") if isinstance(run.get("area_delta_json"), dict) else None
    run_area_rows = _sorted_area_delta_rows(run_area_delta)
    run_top_area_changes = run_area_rows[:3]

    enriched_items: list[dict[str, Any]] = []
    for item in items:
        enriched = dict(item)
        input_image = str(enriched.get("input_image", ""))
        input_image_name = Path(input_image).name if input_image else "image"
        flip_payload = enriched.get("flip_stats_json")
        if not isinstance(flip_payload, dict):
            flip_payload = None
        area_delta_payload = enriched.get("area_delta_json")
        if not isinstance(area_delta_payload, dict):
            area_delta_payload = None
        area_rows = _sorted_area_delta_rows(area_delta_payload)
        enriched["input_image_name"] = input_image_name
        enriched["flip_rate_percent"] = (
            float(flip_payload.get("flip_rate", 0.0)) * 100.0 if flip_payload is not None else None
        )
        enriched["area_delta_rows"] = area_rows
        enriched["top_area_changes"] = area_rows[:3]
        enriched_items.append(enriched)

    return render_template(
        "analysis_detail.html",
        workspace=workspace,
        run=run,
        items=enriched_items,
        model=model,
        run_postprocess_enabled=run_postprocess_enabled,
        run_flip_stats=run_flip_stats,
        run_area_rows=run_area_rows,
        run_top_area_changes=run_top_area_changes,
    )


@bp.route("/workspace/<int:workspace_id>/analysis/<int:run_id>/promote", methods=["GET", "POST"])
def promote_analysis_run(workspace_id: int, run_id: int) -> str:
    workspace = _workspace_by_id_or_404(workspace_id)
    storage = _storage()
    run = storage.get_analysis_run(run_id, workspace_id=workspace_id)
    if run is None:
        abort(404, description=f"Analysis run id={run_id} not found.")

    model = storage.get_model(int(run["model_id"]), workspace_id=workspace_id)
    if model is None:
        flash("This run references a missing model.", "error")
        return redirect(url_for("main.analysis_details", workspace_id=workspace_id, run_id=run_id))

    item_rows = _build_promote_item_rows(storage.get_analysis_items(run_id))
    if not item_rows:
        flash("This run has no usable prediction masks to promote.", "error")
        return redirect(url_for("main.analysis_details", workspace_id=workspace_id, run_id=run_id))

    class_entries = _model_editable_classes(model)
    if not class_entries:
        class_entries = [
            {"class_index": idx + 1, "name": class_name}
            for idx, class_name in enumerate(_default_labeler_categories())
        ]
    class_index_to_name = {
        "0": "background",
        **{str(int(entry["class_index"])): str(entry["name"]) for entry in class_entries},
    }
    category_names = [str(entry["name"]) for entry in class_entries]
    category_schema = normalize_class_schema(
        {
            "schema_version": 1,
            "next_class_id": (max((int(entry["class_index"]) for entry in class_entries), default=0) + 1),
            "classes": [
                {
                    "id": int(entry["class_index"]),
                    "name": str(entry["name"]),
                    "order": idx,
                }
                for idx, entry in enumerate(class_entries)
                if int(entry["class_index"]) > 0
            ],
        },
        fallback_names=category_names or _default_labeler_categories(),
    )
    item_by_id = {int(row["item_id"]): row for row in item_rows}
    default_selected_ids = [int(row["item_id"]) for row in item_rows]

    form_state: dict[str, Any] = {
        "draft_name": f"Bootstrap from Run {run_id}",
        "threshold_pct": 70,
        "cleanup_remove_small_blobs": True,
        "min_blob_px": 200,
        "cleanup_fill_small_holes": True,
        "max_hole_px": 200,
        "cleanup_boundary_ignore": False,
        "boundary_px": 2,
    }
    selected_item_ids = list(default_selected_ids)

    def _render(status_code: int = 200) -> tuple[str, int] | str:
        selected_rows = [item_by_id[item_id] for item_id in selected_item_ids if item_id in item_by_id]
        missing_conf_total = sum(1 for row in item_rows if not row["has_conf"])
        missing_conf_selected = sum(1 for row in selected_rows if not row["has_conf"])
        rendered = render_template(
            "analysis_promote.html",
            workspace=workspace,
            run=run,
            model=model,
            item_rows=item_rows,
            class_entries=class_entries,
            class_index_to_name=class_index_to_name,
            selected_item_ids=selected_item_ids,
            form_state=form_state,
            missing_conf_total=missing_conf_total,
            missing_conf_selected=missing_conf_selected,
        )
        if status_code == 200:
            return rendered
        return rendered, status_code

    if request.method == "GET":
        return _render()

    form_state["draft_name"] = str(request.form.get("draft_name", "")).strip() or form_state["draft_name"]
    threshold_pct, threshold_error = _parse_int_in_range(
        request.form.get("threshold_pct"),
        field_label="Confidence threshold",
        minimum=0,
        maximum=100,
        default=int(form_state["threshold_pct"]),
    )
    form_state["threshold_pct"] = threshold_pct
    form_state["cleanup_remove_small_blobs"] = _is_truthy(request.form.get("cleanup_remove_small_blobs"))
    form_state["cleanup_fill_small_holes"] = _is_truthy(request.form.get("cleanup_fill_small_holes"))
    form_state["cleanup_boundary_ignore"] = _is_truthy(request.form.get("cleanup_boundary_ignore"))

    min_blob_px, min_blob_error = _parse_int_in_range(
        request.form.get("min_blob_px"),
        field_label="Min blob size",
        minimum=1,
        default=int(form_state["min_blob_px"]),
    )
    max_hole_px, max_hole_error = _parse_int_in_range(
        request.form.get("max_hole_px"),
        field_label="Max hole size",
        minimum=1,
        default=int(form_state["max_hole_px"]),
    )
    boundary_px, boundary_error = _parse_int_in_range(
        request.form.get("boundary_px"),
        field_label="Boundary ignore band",
        minimum=0,
        default=int(form_state["boundary_px"]),
    )
    form_state["min_blob_px"] = min_blob_px
    form_state["max_hole_px"] = max_hole_px
    form_state["boundary_px"] = boundary_px

    selected_item_ids = []
    seen_selected: set[int] = set()
    for raw in request.form.getlist("selected_item_ids"):
        try:
            item_id = int(str(raw).strip())
        except ValueError:
            continue
        if item_id in seen_selected or item_id not in item_by_id:
            continue
        seen_selected.add(item_id)
        selected_item_ids.append(item_id)

    validation_errors: list[str] = []
    if threshold_error:
        validation_errors.append(threshold_error)
    if min_blob_error:
        validation_errors.append(min_blob_error)
    if max_hole_error:
        validation_errors.append(max_hole_error)
    if boundary_error:
        validation_errors.append(boundary_error)
    if not selected_item_ids:
        validation_errors.append("Select at least one analysis output to promote.")
    if not str(form_state["draft_name"]).strip():
        validation_errors.append("Draft dataset name is required.")

    if validation_errors:
        for message in validation_errors:
            flash(message, "error")
        return _render(400)

    selected_rows = [item_by_id[item_id] for item_id in selected_item_ids]
    cleanup_config = BootstrapCleanupConfig(
        remove_small_blobs=bool(form_state["cleanup_remove_small_blobs"]),
        min_blob_px=int(form_state["min_blob_px"]),
        fill_small_holes=bool(form_state["cleanup_fill_small_holes"]),
        max_hole_px=int(form_state["max_hole_px"]),
        boundary_ignore=bool(form_state["cleanup_boundary_ignore"]),
        boundary_px=int(form_state["boundary_px"]),
    )
    threshold_pct = int(form_state["threshold_pct"])
    threshold_u8 = int(round((threshold_pct / 100.0) * 255.0))

    draft_root = (
        _workspace_datasets_root(workspace)
        / f"{_slugify(str(form_state['draft_name']))}_{uuid.uuid4().hex[:8]}"
    ).resolve()
    images_dir = (draft_root / "images").resolve()
    masks_dir = (draft_root / "masks").resolve()
    coco_dir = (draft_root / "coco").resolve()
    cache_dir = (masks_dir / "_cache").resolve()
    labels_dir = (draft_root / "labels").resolve()
    provenance_dir = (draft_root / "bootstrap").resolve()
    for path in (images_dir, masks_dir, coco_dir, cache_dir, labels_dir, provenance_dir):
        ensure_dir(path)

    processed_count = 0
    skipped_count = 0
    skipped_messages: list[str] = []
    missing_conf_count = 0
    threshold_applied_count = 0
    item_provenance_paths: list[str] = []
    workspace_images_dir = _workspace_images_dir(workspace)
    base_dir = _base_dir()

    for row in selected_rows:
        item_id = int(row["item_id"])
        image_source_path = _resolve_stored_path(str(row["input_image"]))
        mask_source_path = _resolve_stored_path(str(row["source_mask_path"]))
        if (
            image_source_path is None
            or not image_source_path.exists()
            or not image_source_path.is_file()
            or mask_source_path is None
            or not mask_source_path.exists()
            or not mask_source_path.is_file()
        ):
            skipped_count += 1
            skipped_messages.append(f"Skipped item #{item_id}: missing image or mask artifact.")
            continue
        try:
            image_source_path.relative_to(workspace_images_dir)
            mask_source_path.relative_to(base_dir)
        except ValueError:
            skipped_count += 1
            skipped_messages.append(
                f"Skipped item #{item_id}: source assets must be inside workspace/base directories."
            )
            continue

        try:
            with Image.open(mask_source_path) as mask_img:
                label_map_raw = np.asarray(mask_img)
        except Exception as exc:
            skipped_count += 1
            skipped_messages.append(f"Skipped item #{item_id}: failed to read mask ({exc}).")
            continue

        if label_map_raw.ndim == 3:
            label_map_raw = label_map_raw[..., 0]
        if label_map_raw.ndim != 2:
            skipped_count += 1
            skipped_messages.append(f"Skipped item #{item_id}: mask must be 2D.")
            continue
        label_map = label_map_raw.astype(np.int32, copy=False)

        conf_arr: np.ndarray | None = None
        conf_path_value = str(row["conf_path"]).strip()
        conf_source_path = _resolve_stored_path(conf_path_value) if conf_path_value else None
        if conf_source_path is None or not conf_source_path.exists() or not conf_source_path.is_file():
            missing_conf_count += 1
        else:
            try:
                conf_source_path.relative_to(base_dir)
            except ValueError:
                conf_source_path = None
                missing_conf_count += 1
                skipped_messages.append(
                    f"Item #{item_id}: confidence map path is outside base directory; threshold skipped."
                )
                conf_arr = None
            if conf_source_path is None:
                pass
            else:
                try:
                    with Image.open(conf_source_path) as conf_img:
                        conf_arr = np.asarray(conf_img)
                    if conf_arr.ndim == 3:
                        conf_arr = conf_arr[..., 0]
                    conf_arr = conf_arr.astype(np.uint8, copy=False)
                    if conf_arr.shape != label_map.shape:
                        conf_arr = None
                        missing_conf_count += 1
                        skipped_messages.append(
                            f"Item #{item_id}: confidence shape mismatch; threshold skipped."
                        )
                except Exception:
                    conf_arr = None
                    missing_conf_count += 1
                    skipped_messages.append(f"Item #{item_id}: failed to read confidence; threshold skipped.")

        thresholded_map, threshold_stats = threshold_label_map(
            label_map,
            conf_arr,
            threshold_pct=threshold_pct,
        )
        if bool(threshold_stats.get("applied")):
            threshold_applied_count += 1
        cleaned_map, cleanup_stats = cleanup_label_map(thresholded_map, cleanup_config)

        output_name = safe_image_name(str(row["input_image_name"]))
        if not output_name:
            skipped_count += 1
            skipped_messages.append(f"Skipped item #{item_id}: invalid image filename.")
            continue
        output_image_path = (images_dir / output_name).resolve()
        try:
            shutil.copy2(image_source_path, output_image_path)
        except Exception as exc:
            skipped_count += 1
            skipped_messages.append(f"Skipped item #{item_id}: failed to copy image ({exc}).")
            continue

        image_stem = Path(output_name).stem
        label_output_path = (labels_dir / f"{image_stem}__label.png").resolve()
        _save_indexed_label_map(cleaned_map, label_output_path)

        saved_mask_files: list[str] = []
        for class_entry in class_entries:
            class_index = int(class_entry["class_index"])
            class_mask = cleaned_map == class_index
            class_mask_path = (masks_dir / mask_filename_for_class_id(image_stem, class_index)).resolve()
            if np.any(class_mask):
                save_binary_mask(class_mask, class_mask_path)
                saved_mask_files.append(class_mask_path.name)
            elif class_mask_path.exists() and class_mask_path.is_file():
                class_mask_path.unlink()

        provenance_payload = {
            "source_run_id": int(run_id),
            "source_run_item_id": int(item_id),
            "source_model_id": int(run["model_id"]),
            "source_model_name": str(model.get("name", "")),
            "source_variant": "refined_if_available",
            "selected_source_mask": str(row["source_mask_path"]),
            "raw_conf_path": conf_path_value or None,
            "threshold": threshold_stats,
            "cleanup_config": cleanup_config.to_dict(),
            "cleanup_stats": cleanup_stats,
            "class_index_to_name": class_index_to_name,
            "background_class_id": 0,
            "outputs": {
                "image_name": output_name,
                "label_path": _to_relative_under_base(str(label_output_path)) or str(label_output_path),
                "mask_files": saved_mask_files,
            },
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        provenance_path = (provenance_dir / f"{image_stem}__provenance.json").resolve()
        with provenance_path.open("w", encoding="utf-8") as handle:
            json.dump(provenance_payload, handle, indent=2)
        item_provenance_paths.append(_to_relative_under_base(str(provenance_path)) or str(provenance_path))
        processed_count += 1

    if processed_count <= 0:
        if draft_root.exists():
            shutil.rmtree(draft_root, ignore_errors=True)
        flash("Promotion failed: no selected outputs could be converted.", "error")
        for warning in skipped_messages[:5]:
            flash(warning, "warning")
        return _render(400)

    manifest_payload = {
        "source_run_id": int(run_id),
        "source_model_id": int(run["model_id"]),
        "source_model_name": str(model.get("name", "")),
        "selected_item_count": len(selected_rows),
        "processed_item_count": processed_count,
        "skipped_item_count": skipped_count,
        "threshold_pct": threshold_pct,
        "threshold_u8": threshold_u8,
        "threshold_applied_count": threshold_applied_count,
        "missing_conf_count": missing_conf_count,
        "cleanup_config": cleanup_config.to_dict(),
        "class_index_to_name": class_index_to_name,
        "item_provenance_paths": item_provenance_paths,
        "warnings": skipped_messages,
    }
    manifest_path = (provenance_dir / "bootstrap_manifest.json").resolve()
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)

    try:
        dataset_project_id = storage.create_workspace_dataset(
            workspace_id=workspace_id,
            name=str(form_state["draft_name"]),
            project_dir=str(draft_root),
            images_dir=str(images_dir),
            masks_dir=str(masks_dir),
            coco_dir=str(coco_dir),
            cache_dir=str(cache_dir),
            categories=category_schema,
            slic_algorithm="slic",
            slic_preset_name="medium",
            slic_detail_level="medium",
            slic_n_segments=int(current_app.config["LABELER_SLIC_N_SEGMENTS"]),
            slic_compactness=float(current_app.config["LABELER_SLIC_COMPACTNESS"]),
            slic_sigma=float(current_app.config["LABELER_SLIC_SIGMA"]),
            slic_colorspace="lab",
        )
    except sqlite3.IntegrityError:
        if draft_root.exists():
            shutil.rmtree(draft_root, ignore_errors=True)
        flash(f"A draft dataset named '{form_state['draft_name']}' already exists.", "error")
        return _render(400)
    except Exception as exc:
        if draft_root.exists():
            shutil.rmtree(draft_root, ignore_errors=True)
        flash(f"Failed to create promoted draft dataset: {exc}", "error")
        return _render(500)

    if missing_conf_count > 0:
        flash(
            (
                f"Promoted {processed_count} image(s). "
                f"Threshold was skipped for {missing_conf_count} image(s) without confidence maps."
            ),
            "warning",
        )
    else:
        flash(f"Promoted {processed_count} image(s) into draft dataset '{form_state['draft_name']}'.", "success")
    if skipped_count > 0:
        flash(f"Skipped {skipped_count} image(s) due to missing or invalid artifacts.", "warning")
    return redirect(
        url_for(
            "main.workspace_dataset_detail",
            workspace_id=workspace_id,
            dataset_id=dataset_project_id,
        )
    )


def _job_status_class(status_value: str) -> str:
    status_key = str(status_value or "").strip().lower()
    if status_key in {"completed", "trained", "success"}:
        return "is-good"
    if status_key in {"queued", "running", "training", "cancel_requested"}:
        return "is-progress"
    if status_key in {"failed", "error", "canceled"}:
        return "is-bad"
    return "is-neutral"


def _job_type_label(job_type_value: Any) -> str:
    job_type = str(job_type_value or "").strip().lower()
    if job_type == "training":
        return "Training"
    if job_type == "analysis":
        return "Analysis"
    if job_type == "slic_warmup":
        return "SLIC Warmup"
    if not job_type:
        return "-"
    return job_type.replace("_", " ").title()


def _job_target_details(
    storage: Storage,
    *,
    workspace_id: int,
    job: dict[str, Any],
) -> tuple[str, str | None]:
    entity_type = str(job.get("entity_type", "")).strip().lower()
    entity_id_raw = job.get("entity_id")
    try:
        entity_id = int(entity_id_raw)
    except (TypeError, ValueError):
        entity_id = 0
    if entity_type == "model" and entity_id > 0:
        model = storage.get_model(entity_id, workspace_id=workspace_id)
        if model is not None:
            return (
                str(model.get("name", f"Model #{entity_id}")),
                url_for("main.model_details", workspace_id=workspace_id, model_id=entity_id),
            )
        return (f"Model #{entity_id}", None)
    if entity_type == "analysis_run" and entity_id > 0:
        run = storage.get_analysis_run(entity_id, workspace_id=workspace_id)
        if run is None:
            return (f"Run #{entity_id}", None)
        model_name = str(run.get("model_name", "")).strip()
        if model_name:
            return (
                f"Run #{entity_id} ({model_name})",
                url_for("main.analysis_details", workspace_id=workspace_id, run_id=entity_id),
            )
        return (f"Run #{entity_id}", url_for("main.analysis_details", workspace_id=workspace_id, run_id=entity_id))
    if entity_type == "dataset" and entity_id > 0:
        dataset_project = storage.get_workspace_dataset(workspace_id, entity_id)
        if dataset_project is None:
            return (f"Dataset #{entity_id}", None)
        dataset_name = str(dataset_project.get("name", "")).strip() or f"Dataset #{entity_id}"
        return (
            dataset_name,
            url_for("main.workspace_dataset_detail", workspace_id=workspace_id, dataset_id=entity_id),
        )
    return ("-", None)


def _apply_immediate_cancel_to_entity(
    storage: Storage,
    *,
    workspace_id: int,
    job: dict[str, Any],
) -> None:
    entity_type = str(job.get("entity_type", "")).strip().lower()
    entity_id_raw = job.get("entity_id")
    try:
        entity_id = int(entity_id_raw)
    except (TypeError, ValueError):
        entity_id = 0
    if entity_id <= 0:
        return
    if entity_type == "model":
        storage.update_model(
            entity_id,
            status="canceled",
            error_message="Canceled by user.",
            workspace_id=workspace_id,
        )
    elif entity_type == "analysis_run":
        storage.update_analysis_run_state(
            entity_id,
            status="canceled",
            error_message="Canceled by user.",
            workspace_id=workspace_id,
        )


@bp.route("/workspace/<int:workspace_id>/jobs", methods=["GET"])
def workspace_jobs(workspace_id: int) -> str:
    workspace = _workspace_by_id_or_404(workspace_id)
    storage = _storage()
    jobs = storage.list_jobs(workspace_id=workspace_id, limit=250)
    queued_global = storage.list_queued_jobs(limit=1000)
    queue_positions = {int(item["id"]): idx + 1 for idx, item in enumerate(queued_global)}

    rows: list[dict[str, Any]] = []
    for job in jobs:
        row = _serialize_job(job, queue_positions=queue_positions)
        status_key = str(row.get("status", "")).strip().lower()
        job_type = str(row.get("job_type", "")).strip().lower()
        target_label, target_url = _job_target_details(storage, workspace_id=workspace_id, job=job)
        row["target_label"] = target_label
        row["target_url"] = target_url
        row["status_class"] = _job_status_class(status_key)
        row["job_type_label"] = _job_type_label(job_type)
        row["is_terminal"] = _job_is_terminal(status_key)
        row["can_cancel"] = status_key in {"queued", "running"}
        row["can_rerun"] = status_key in {"completed", "failed", "canceled"}
        row["stage_label"] = str(row.get("stage", "")).replace("_", " ").strip().title() or "-"
        rows.append(row)
    queued_rows = [row for row in rows if str(row.get("status", "")).strip().lower() == "queued"]
    non_queued_rows = [row for row in rows if str(row.get("status", "")).strip().lower() != "queued"]
    queued_rows.sort(key=lambda item: int(item.get("queue_position", 999999)))
    non_queued_rows.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    rows = queued_rows + non_queued_rows

    return render_template(
        "jobs_index.html",
        workspace=workspace,
        jobs=rows,
        jobs_poll_url=url_for("main.workspace_jobs_poll", workspace_id=workspace_id),
        jobs_reorder_url=url_for("main.workspace_jobs_reorder", workspace_id=workspace_id),
    )


@bp.route("/workspace/<int:workspace_id>/jobs/poll", methods=["GET"])
def workspace_jobs_poll(workspace_id: int):
    _workspace_by_id_or_404(workspace_id)
    storage = _storage()
    job_type = str(request.args.get("job_type", "")).strip().lower()
    if job_type not in {"training", "analysis", "slic_warmup"}:
        job_type = ""
    limit = request.args.get("limit", type=int) or 120
    limit = max(1, min(500, int(limit)))
    jobs = storage.list_jobs(
        workspace_id=workspace_id,
        limit=limit,
        job_type=job_type or None,
    )
    queued_global = storage.list_queued_jobs(limit=1000)
    queue_positions = {int(item["id"]): idx + 1 for idx, item in enumerate(queued_global)}
    serialized = [_serialize_job(item, queue_positions=queue_positions) for item in jobs]
    return jsonify(
        {
            "generated_at": _utc_now_iso(),
            "workspace_id": workspace_id,
            "job_type": job_type or None,
            "jobs": serialized,
        }
    )


@bp.route("/workspace/<int:workspace_id>/jobs/<int:job_id>/cancel", methods=["POST"])
def workspace_job_cancel(workspace_id: int, job_id: int) -> str:
    _workspace_by_id_or_404(workspace_id)
    storage = _storage()
    job = storage.get_job(job_id, workspace_id=workspace_id)
    if job is None:
        abort(404, description=f"Job id={job_id} not found in workspace.")
    status_key = str(job.get("status", "")).strip().lower()
    if status_key in {"completed", "failed", "canceled"}:
        response_payload = {
            "ok": False,
            "job_id": job_id,
            "status": status_key,
            "message": f"Job #{job_id} is already {status_key}.",
        }
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify(response_payload), 409
        flash(response_payload["message"], "warning")
        return redirect(url_for("main.workspace_jobs", workspace_id=workspace_id))

    result = storage.request_job_cancel(job_id, workspace_id=workspace_id)
    if result is None:
        abort(404, description=f"Job id={job_id} not found in workspace.")
    immediate = bool(result.get("immediate"))
    if immediate:
        _apply_immediate_cancel_to_entity(storage, workspace_id=workspace_id, job=job)
        message = f"Job #{job_id} canceled."
    else:
        message = f"Cancellation requested for running job #{job_id}."

    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify(
            {
                "ok": True,
                "job_id": job_id,
                "status": str(result.get("status", "")),
                "immediate": immediate,
                "message": message,
            }
        )
    flash(message, "success")
    return redirect(url_for("main.workspace_jobs", workspace_id=workspace_id))


@bp.route("/workspace/<int:workspace_id>/jobs/<int:job_id>/rerun", methods=["POST"])
def workspace_job_rerun(workspace_id: int, job_id: int) -> str:
    _workspace_by_id_or_404(workspace_id)
    storage = _storage()
    original_job = storage.get_job(job_id, workspace_id=workspace_id)
    if original_job is None:
        abort(404, description=f"Job id={job_id} not found in workspace.")
    original_status = str(original_job.get("status", "")).strip().lower()
    if original_status not in {"completed", "failed", "canceled"}:
        flash(f"Only completed/failed/canceled jobs can be re-run. Job #{job_id} is {original_status}.", "warning")
        return redirect(url_for("main.workspace_jobs", workspace_id=workspace_id))

    payload = original_job.get("payload_json")
    if not isinstance(payload, dict):
        flash("Unable to re-run job: payload metadata is missing.", "error")
        return redirect(url_for("main.workspace_jobs", workspace_id=workspace_id))

    job_type = str(original_job.get("job_type", "")).strip().lower()
    if job_type == "training":
        model_name = str(payload.get("model_name", "")).strip() or f"rerun_model_{job_id}"
        dataset_id = int(payload.get("dataset_id", 0) or 0)
        feature_config = payload.get("feature_config")
        train_config = payload.get("train_config")
        if dataset_id <= 0 or not isinstance(feature_config, dict) or not isinstance(train_config, dict):
            flash("Unable to re-run training job: payload is incomplete.", "error")
            return redirect(url_for("main.workspace_jobs", workspace_id=workspace_id))
        dataset = storage.get_dataset(dataset_id, workspace_id=workspace_id)
        if dataset is None:
            flash(f"Unable to re-run training job: dataset #{dataset_id} no longer exists.", "error")
            return redirect(url_for("main.workspace_jobs", workspace_id=workspace_id))
        dedupe_key = build_training_dedupe_key(
            workspace_id=workspace_id,
            model_name=model_name,
            dataset_id=dataset_id,
            feature_config=feature_config,
            train_config=train_config,
        )
        existing = storage.find_active_job_by_dedupe(
            dedupe_key=dedupe_key,
            workspace_id=workspace_id,
            job_type="training",
        )
        if existing is not None:
            flash(
                f"Training job #{int(existing['id'])} is already queued/running with this configuration.",
                "warning",
            )
            return redirect(url_for("main.workspace_jobs", workspace_id=workspace_id))

        train_cfg = TrainConfig.from_dict(train_config)
        model_id = storage.create_model(
            name=model_name,
            dataset_id=dataset_id,
            classes=[],
            feature_config=feature_config,
            hyperparams={
                "n_estimators": int(train_cfg.n_estimators),
                "max_depth": train_cfg.max_depth,
                "min_samples_split": int(train_cfg.min_samples_split),
                "min_samples_leaf": int(train_cfg.min_samples_leaf),
                "class_weight": str(train_cfg.class_weight),
            },
            metrics={},
            artifact_dir="",
            model_path="",
            metadata_path="",
            status="queued",
            error_message=None,
            workspace_id=workspace_id,
        )
        new_payload = dict(payload)
        new_payload["workspace_id"] = workspace_id
        new_payload["model_id"] = model_id
        new_job_id, _ = storage.enqueue_job(
            workspace_id=workspace_id,
            job_type="training",
            payload=new_payload,
            dedupe_key=dedupe_key,
            entity_type="model",
            entity_id=model_id,
            rerun_of_job_id=job_id,
        )
        flash(f"Queued training re-run as job #{new_job_id}.", "success")
        return redirect(url_for("main.workspace_jobs", workspace_id=workspace_id))

    if job_type == "analysis":
        model_id = int(payload.get("model_id", 0) or 0)
        input_images = payload.get("input_images")
        postprocess_config = payload.get("postprocess_config")
        if model_id <= 0 or not isinstance(input_images, list):
            flash("Unable to re-run analysis job: payload is incomplete.", "error")
            return redirect(url_for("main.workspace_jobs", workspace_id=workspace_id))
        model_record = storage.get_model(model_id, workspace_id=workspace_id)
        if model_record is None:
            flash(f"Unable to re-run analysis job: model #{model_id} no longer exists.", "error")
            return redirect(url_for("main.workspace_jobs", workspace_id=workspace_id))
        if not isinstance(postprocess_config, dict):
            postprocess_config = {}
        normalized_images = [str(item) for item in input_images if str(item).strip()]
        if not normalized_images:
            flash("Unable to re-run analysis job: no valid input images remain.", "error")
            return redirect(url_for("main.workspace_jobs", workspace_id=workspace_id))
        dedupe_key = build_analysis_dedupe_key(
            workspace_id=workspace_id,
            model_id=model_id,
            input_images=normalized_images,
            postprocess_config=postprocess_config,
        )
        existing = storage.find_active_job_by_dedupe(
            dedupe_key=dedupe_key,
            workspace_id=workspace_id,
            job_type="analysis",
        )
        if existing is not None:
            flash(
                f"Analysis job #{int(existing['id'])} is already queued/running with this selection.",
                "warning",
            )
            return redirect(url_for("main.workspace_jobs", workspace_id=workspace_id))
        run_id = storage.create_analysis_run(
            model_id=model_id,
            input_images=normalized_images,
            workspace_id=workspace_id,
            postprocess_enabled=bool(postprocess_config.get("enabled")),
            postprocess_config=postprocess_config,
            status="queued",
        )
        new_payload = {
            "workspace_id": workspace_id,
            "run_id": run_id,
            "model_id": model_id,
            "input_images": normalized_images,
            "postprocess_config": postprocess_config,
        }
        new_job_id, _ = storage.enqueue_job(
            workspace_id=workspace_id,
            job_type="analysis",
            payload=new_payload,
            dedupe_key=dedupe_key,
            entity_type="analysis_run",
            entity_id=run_id,
            rerun_of_job_id=job_id,
        )
        flash(f"Queued analysis re-run as job #{new_job_id}.", "success")
        return redirect(url_for("main.workspace_jobs", workspace_id=workspace_id))

    if job_type == "slic_warmup":
        dataset_id = int(payload.get("dataset_id", 0) or original_job.get("entity_id", 0) or 0)
        if dataset_id <= 0:
            flash("Unable to re-run SLIC warmup job: payload is incomplete.", "error")
            return redirect(url_for("main.workspace_jobs", workspace_id=workspace_id))
        dataset_project = storage.get_workspace_dataset(workspace_id, dataset_id)
        if dataset_project is None:
            flash(f"Unable to re-run SLIC warmup job: dataset #{dataset_id} no longer exists.", "error")
            return redirect(url_for("main.workspace_jobs", workspace_id=workspace_id))
        dedupe_key = build_slic_warmup_dedupe_key(workspace_id=workspace_id, dataset_id=dataset_id)
        existing = storage.find_active_job_by_dedupe(
            dedupe_key=dedupe_key,
            workspace_id=workspace_id,
            job_type="slic_warmup",
        )
        if existing is not None:
            flash(
                f"SLIC warmup job #{int(existing['id'])} is already queued/running for this dataset.",
                "warning",
            )
            return redirect(url_for("main.workspace_jobs", workspace_id=workspace_id))
        new_payload = {
            "workspace_id": workspace_id,
            "dataset_id": dataset_id,
        }
        new_job_id, _ = storage.enqueue_job(
            workspace_id=workspace_id,
            job_type="slic_warmup",
            payload=new_payload,
            dedupe_key=dedupe_key,
            entity_type="dataset",
            entity_id=dataset_id,
            rerun_of_job_id=job_id,
        )
        flash(f"Queued SLIC warmup re-run as job #{new_job_id}.", "success")
        return redirect(url_for("main.workspace_jobs", workspace_id=workspace_id))

    flash(f"Unsupported job type for re-run: {job_type}", "error")
    return redirect(url_for("main.workspace_jobs", workspace_id=workspace_id))


@bp.route("/workspace/<int:workspace_id>/jobs/reorder", methods=["POST"])
def workspace_jobs_reorder(workspace_id: int):
    _workspace_by_id_or_404(workspace_id)
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "message": "Expected JSON payload."}), 400
    ordered_ids_raw = payload.get("ordered_job_ids")
    if not isinstance(ordered_ids_raw, list):
        return jsonify({"ok": False, "message": "ordered_job_ids must be an array."}), 400
    ordered_ids: list[int] = []
    for item in ordered_ids_raw:
        try:
            ordered_ids.append(int(item))
        except (TypeError, ValueError):
            continue
    changed = _storage().reorder_workspace_queued_jobs(workspace_id, ordered_ids)
    return jsonify({"ok": True, "changed": changed})


@bp.route("/workspace/<int:workspace_id>/settings", methods=["GET", "POST"])
def workspace_settings(workspace_id: int) -> str:
    workspace = _workspace_by_id_or_404(workspace_id)
    storage = _storage()

    if request.method == "POST":
        seed_raw = str(request.form.get("augmentation_seed", "")).strip()
        try:
            seed_value = int(seed_raw)
        except ValueError:
            flash("Augmentation seed must be an integer.", "error")
            return redirect(url_for("main.workspace_settings", workspace_id=workspace_id))
        storage.update_workspace_augmentation_seed(workspace_id, seed_value)
        flash("Workspace augmentation seed updated.", "success")
        return redirect(url_for("main.workspace_settings", workspace_id=workspace_id))

    return render_template(
        "settings.html",
        workspace=workspace,
        effective_augmentation_seed=_workspace_augmentation_seed(workspace),
    )


@bp.route("/files/<path:relative_path>", methods=["GET"])
def serve_file(relative_path: str):
    base = _base_dir()
    target = (base / relative_path).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        abort(404)

    if not target.exists() or not target.is_file():
        abort(404)
    return send_file(target)
