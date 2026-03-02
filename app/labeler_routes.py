"""Blueprint for workspace-scoped superpixel labeling workflows."""

from __future__ import annotations

import base64
import io
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
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

from .services.labeling.coco_export import export_coco_annotations
from .services.labeling.class_schema import (
    build_schema_from_names,
    class_entries,
    class_name_by_id,
    class_names,
    normalize_class_schema,
    parse_class_names,
    resolve_class_id,
    update_schema_with_names,
)
from .services.labeling.image_io import (
    class_id_from_entry,
    ensure_dir,
    image_path,
    iter_existing_mask_paths,
    list_images,
    mask_filename,
    mask_filename_for_class_id,
    safe_image_name,
    sanitize_class_name,
    save_binary_mask,
)
from .services.labeling.mask_state import ImageMaskState, selected_from_saved_masks
from .services.labeling.slic_cache import clear_slic_cache, load_or_create_slic_cache
from .services.storage import Storage

bp = Blueprint("labeler", __name__)

SESSION_STATES: dict[str, dict[str, ImageMaskState]] = {}
STATE_LOCK = threading.Lock()

CLASS_COLORS = [
    (255, 59, 48, 120),
    (52, 199, 89, 120),
    (0, 122, 255, 120),
    (255, 149, 0, 120),
]
SELECTION_COLOR = (255, 223, 0, 130)

SLIC_DETAIL_PRESETS: dict[str, dict[str, float]] = {
    "low": {"n_segments": 700, "compactness": 12.0, "sigma": 1.2},
    "medium": {"n_segments": 1200, "compactness": 10.0, "sigma": 1.0},
    "high": {"n_segments": 2200, "compactness": 8.0, "sigma": 0.8},
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DATASET_CLASS_EDIT_STATE_PREFIX = "dataset_class_edit_state:"


def _storage() -> Storage:
    return current_app.extensions["storage"]


def _labeler_root_dir() -> Path:
    return Path(current_app.config["LABELER_DIR"]).resolve()


def _default_labeler_categories() -> list[str]:
    categories = [str(item).strip() for item in current_app.config["LABELER_CATEGORIES"] if str(item).strip()]
    return categories or ["class_1", "class_2", "class_3"]


def _project_class_schema(project: dict[str, Any]) -> dict[str, Any]:
    return normalize_class_schema(project.get("categories_json"), fallback_names=_default_labeler_categories())


def _project_classes(project: dict[str, Any]) -> list[dict[str, Any]]:
    return class_entries(_project_class_schema(project))


def _project_categories(project: dict[str, Any]) -> list[str]:
    return class_names(_project_class_schema(project))


def _project_images_dir(project: dict[str, Any]) -> Path:
    return Path(str(project["images_dir"])).resolve()


def _project_masks_dir(project: dict[str, Any]) -> Path:
    return Path(str(project["masks_dir"])).resolve()


def _project_coco_dir(project: dict[str, Any]) -> Path:
    return Path(str(project["coco_dir"])).resolve()


def _project_cache_dir(project: dict[str, Any]) -> Path:
    return Path(str(project["cache_dir"])).resolve()


def _labeler_session_id() -> str:
    sid = session.get("labeler_sid")
    if not sid:
        sid = uuid.uuid4().hex
        session["labeler_sid"] = sid
    return sid


def _state_key(project_id: int, image_name: str) -> str:
    return f"{project_id}:{image_name}"


def _slugify(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower()).strip("_")
    return normalized or "workspace"


def _is_allowed_image_filename(filename: str) -> bool:
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


def _unique_image_path(images_dir: Path, filename: str) -> Path:
    safe_name = Path(filename).name
    stem = Path(safe_name).stem or "image"
    suffix = Path(safe_name).suffix.lower()
    if not suffix:
        suffix = ".png"
    candidate = images_dir / f"{stem}{suffix}"
    index = 1
    while candidate.exists():
        candidate = images_dir / f"{stem}_{index}{suffix}"
        index += 1
    return candidate


def _save_uploaded_images(images_dir: Path, files: list[Any]) -> tuple[int, int]:
    saved_count = 0
    skipped_count = 0
    for file_storage in files:
        if file_storage is None:
            continue
        raw_name = str(file_storage.filename or "").strip()
        if not raw_name:
            continue
        if not _is_allowed_image_filename(raw_name):
            skipped_count += 1
            continue
        destination = _unique_image_path(images_dir, raw_name)
        file_storage.save(str(destination))
        saved_count += 1
    return saved_count, skipped_count


def _parse_categories_text(raw_value: str) -> list[str]:
    return parse_class_names(raw_value)


def _dataset_class_edit_session_key(project_id: int) -> str:
    return f"{DATASET_CLASS_EDIT_STATE_PREFIX}{int(project_id)}"


def _class_change_summary(existing: Any, proposed: list[str]) -> dict[str, Any]:
    _, diff = update_schema_with_names(existing, proposed)
    return diff


def _set_dataset_class_edit_state(project_id: int, *, categories_text: str, diff: dict[str, Any]) -> None:
    session[_dataset_class_edit_session_key(project_id)] = {
        "categories_text": str(categories_text or ""),
        "diff": diff,
    }


def _clear_dataset_class_edit_state(project_id: int) -> None:
    session.pop(_dataset_class_edit_session_key(project_id), None)


def _dataset_categories(dataset: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for category in dataset.get("categories_json", []):
        name = str(category.get("name", "")).strip()
        if name:
            names.append(name)
    return names


def _to_relative_under_base(path_value: str) -> str | None:
    base = Path(current_app.config["BASE_DIR"]).resolve()
    path = Path(path_value)
    if not path.is_absolute():
        return str(path).replace("\\", "/")
    try:
        return path.resolve().relative_to(base).as_posix()
    except ValueError:
        return None


def _json_error(message: str, status: int = 400):
    return jsonify({"error": message}), status


def _mask_overlay_base64(mask_bool: np.ndarray, color: tuple[int, int, int, int]) -> str:
    h, w = mask_bool.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask_bool, :3] = color[:3]
    rgba[mask_bool, 3] = color[3]
    with io.BytesIO() as buffer:
        Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def _class_color(classes: list[dict[str, Any]], class_id: int | None) -> tuple[int, int, int, int]:
    idx = 0
    if class_id is not None:
        for pos, class_entry in enumerate(classes):
            if int(class_entry["id"]) == int(class_id):
                idx = pos
                break
    return CLASS_COLORS[idx % len(CLASS_COLORS)]


def _default_class_id(classes: list[dict[str, Any]]) -> int | None:
    if not classes:
        return None
    try:
        return int(classes[0]["id"])
    except (TypeError, ValueError, KeyError):
        return None


def _resolve_payload_class_id(
    raw_value: Any,
    classes: list[dict[str, Any]],
    *,
    fallback_to_default: bool = True,
) -> int | None:
    payload = {"classes": classes}
    resolved = resolve_class_id(raw_value, payload)
    if resolved is not None:
        return int(resolved)
    if fallback_to_default:
        return _default_class_id(classes)
    return None


def _normalize_detail_level(value: Any, *, fallback: str = "medium") -> str:
    candidate = str(value or "").strip().lower()
    if candidate in SLIC_DETAIL_PRESETS:
        return candidate
    return fallback


def _normalize_colorspace(value: Any) -> str:
    candidate = str(value or "").strip().lower()
    if candidate in {"lab", "rgb"}:
        return candidate
    return "lab"


def _normalize_slic_values(
    *,
    n_segments: Any,
    compactness: Any,
    sigma: Any,
    colorspace: Any,
) -> tuple[int, float, float, str]:
    try:
        normalized_segments = int(float(n_segments))
    except (TypeError, ValueError):
        normalized_segments = 1200
    normalized_segments = max(50, min(5000, normalized_segments))

    try:
        normalized_compactness = float(compactness)
    except (TypeError, ValueError):
        normalized_compactness = 10.0
    normalized_compactness = max(0.01, min(200.0, normalized_compactness))

    try:
        normalized_sigma = float(sigma)
    except (TypeError, ValueError):
        normalized_sigma = 1.0
    normalized_sigma = max(0.0, min(10.0, normalized_sigma))

    normalized_colorspace = _normalize_colorspace(colorspace)
    return normalized_segments, normalized_compactness, normalized_sigma, normalized_colorspace


def _project_default_slic(project: dict[str, Any]) -> dict[str, Any]:
    detail_level = _normalize_detail_level(project.get("slic_detail_level"), fallback="medium")
    n_segments, compactness, sigma, colorspace = _normalize_slic_values(
        n_segments=project.get("slic_n_segments"),
        compactness=project.get("slic_compactness"),
        sigma=project.get("slic_sigma"),
        colorspace=project.get("slic_colorspace"),
    )
    preset_name = str(project.get("slic_preset_name", "medium")).strip().lower() or "medium"
    return {
        "algorithm": "slic",
        "preset_name": preset_name,
        "detail_level": detail_level,
        "n_segments": n_segments,
        "compactness": compactness,
        "sigma": sigma,
        "colorspace": colorspace,
    }


def _effective_slic_for_image(project: dict[str, Any], image_name: str) -> dict[str, Any]:
    project_default = _project_default_slic(project)
    override = _storage().get_image_slic_override(int(project["id"]), image_name)
    if override is None:
        return {
            **project_default,
            "preset_mode": "dataset_default",
        }

    detail_level = _normalize_detail_level(override.get("detail_level"), fallback=project_default["detail_level"])
    n_segments, compactness, sigma, colorspace = _normalize_slic_values(
        n_segments=override.get("n_segments"),
        compactness=override.get("compactness"),
        sigma=override.get("sigma"),
        colorspace=override.get("colorspace"),
    )
    preset_name = str(override.get("preset_name", "custom")).strip().lower() or "custom"
    if preset_name in SLIC_DETAIL_PRESETS:
        preset_mode = "detail"
    elif preset_name == "dataset_default":
        preset_mode = "dataset_default"
    else:
        preset_mode = "custom"
    return {
        "algorithm": "slic",
        "preset_mode": preset_mode,
        "preset_name": preset_name,
        "detail_level": detail_level,
        "n_segments": n_segments,
        "compactness": compactness,
        "sigma": sigma,
        "colorspace": colorspace,
    }


def _resolve_requested_slic_settings(
    project: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    project_default = _project_default_slic(project)
    preset_mode = str(payload.get("preset_mode", "dataset_default")).strip().lower()
    detail_level = _normalize_detail_level(payload.get("detail_level"), fallback=project_default["detail_level"])

    if preset_mode == "dataset_default":
        return {
            **project_default,
            "preset_mode": "dataset_default",
            "preset_name": "dataset_default",
        }

    if preset_mode == "detail":
        preset = SLIC_DETAIL_PRESETS[detail_level]
        n_segments, compactness, sigma, colorspace = _normalize_slic_values(
            n_segments=preset["n_segments"],
            compactness=preset["compactness"],
            sigma=preset["sigma"],
            colorspace=payload.get("colorspace", project_default["colorspace"]),
        )
        return {
            "algorithm": "slic",
            "preset_mode": "detail",
            "preset_name": detail_level,
            "detail_level": detail_level,
            "n_segments": n_segments,
            "compactness": compactness,
            "sigma": sigma,
            "colorspace": colorspace,
        }

    n_segments, compactness, sigma, colorspace = _normalize_slic_values(
        n_segments=payload.get("n_segments"),
        compactness=payload.get("compactness"),
        sigma=payload.get("sigma"),
        colorspace=payload.get("colorspace", project_default["colorspace"]),
    )
    return {
        "algorithm": "slic",
        "preset_mode": "custom",
        "preset_name": "custom",
        "detail_level": detail_level,
        "n_segments": n_segments,
        "compactness": compactness,
        "sigma": sigma,
        "colorspace": colorspace,
    }


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _normalize_selected_ids(raw_ids: Any, *, max_count: int = 20000) -> list[int]:
    if not isinstance(raw_ids, list):
        return []
    selected: list[int] = []
    seen: set[int] = set()
    for item in raw_ids[:max_count]:
        sp_id = _to_int(item)
        if sp_id is None:
            continue
        if sp_id < 0 or sp_id in seen:
            continue
        seen.add(sp_id)
        selected.append(sp_id)
    return selected


def _clip_point(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    return (
        max(0, min(width - 1, x)),
        max(0, min(height - 1, y)),
    )


def _sample_line_points(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))
    if steps <= 0:
        return [(x0, y0)]
    points: list[tuple[int, int]] = []
    for step in range(steps + 1):
        ratio = step / steps
        x = int(round(x0 + dx * ratio))
        y = int(round(y0 + dy * ratio))
        points.append((x, y))
    return points


def _selection_ids_from_brush(
    segments: np.ndarray,
    raw_points: Any,
) -> list[int]:
    if not isinstance(raw_points, list) or not raw_points:
        return []

    height, width = segments.shape
    points: list[tuple[int, int]] = []
    for item in raw_points[:3000]:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        x = _to_int(item[0])
        y = _to_int(item[1])
        if x is None or y is None:
            continue
        points.append(_clip_point(x, y, width, height))

    if not points:
        return []

    touched: set[int] = set()
    prev_x, prev_y = points[0]
    touched.add(int(segments[prev_y, prev_x]))
    for x, y in points[1:]:
        for sx, sy in _sample_line_points(prev_x, prev_y, x, y):
            cx, cy = _clip_point(sx, sy, width, height)
            touched.add(int(segments[cy, cx]))
        prev_x, prev_y = x, y
    return sorted(touched)


def _selection_ids_from_marquee(
    state: ImageMaskState,
    rect: Any,
    rule: str,
) -> list[int]:
    if not isinstance(rect, dict):
        return []
    x1 = _to_int(rect.get("x1"))
    y1 = _to_int(rect.get("y1"))
    x2 = _to_int(rect.get("x2"))
    y2 = _to_int(rect.get("y2"))
    if None in {x1, y1, x2, y2}:
        return []

    segments = state.segments
    height, width = segments.shape
    x_min = max(0, min(int(x1), int(x2)))
    x_max = min(width - 1, max(int(x1), int(x2)))
    y_min = max(0, min(int(y1), int(y2)))
    y_max = min(height - 1, max(int(y1), int(y2)))
    if x_min > x_max or y_min > y_max:
        return []

    if rule == "intersects":
        region = segments[y_min : y_max + 1, x_min : x_max + 1]
        ids = np.unique(region)
        return sorted(int(v) for v in ids.tolist())

    centroids = state.segment_centroids()
    selected: list[int] = []
    for sp_id, (cx, cy) in centroids.items():
        if x_min <= cx <= x_max and y_min <= cy <= y_max:
            selected.append(int(sp_id))
    return sorted(selected)


def _clear_project_state(project_id: int) -> None:
    sid = _labeler_session_id()
    prefix = f"{project_id}:"
    with STATE_LOCK:
        bucket = SESSION_STATES.get(sid, {})
        for key in list(bucket.keys()):
            if key.startswith(prefix):
                del bucket[key]


def _clear_project_image_state(project_id: int, image_name: str) -> None:
    sid = _labeler_session_id()
    safe_name = safe_image_name(image_name)
    target_key = _state_key(project_id, safe_name)
    with STATE_LOCK:
        bucket = SESSION_STATES.get(sid, {})
        if target_key in bucket:
            del bucket[target_key]


def _mask_files_for_image(project: dict[str, Any], image_name: str) -> list[Path]:
    classes = _project_classes(project)
    masks_dir = _project_masks_dir(project)
    stem = Path(safe_image_name(image_name)).stem
    return iter_existing_mask_paths(masks_dir, stem, classes)


def _image_file_cards(project: dict[str, Any]) -> list[dict[str, Any]]:
    images_dir = _project_images_dir(project)
    cards: list[dict[str, Any]] = []
    for image_name in list_images(images_dir):
        file_path = images_dir / image_name
        try:
            stat_info = file_path.stat()
        except OSError:
            continue
        modified_at = datetime.fromtimestamp(stat_info.st_mtime, tz=timezone.utc).isoformat()
        cards.append(
            {
                "image_name": image_name,
                "image_extension": Path(image_name).suffix.lower().lstrip("."),
                "image_size_bytes": int(stat_info.st_size),
                "modified_at": modified_at,
                "modified_epoch": float(stat_info.st_mtime),
            }
        )
    return cards


def _resolve_workspace(workspace_id: int | None = None) -> dict[str, Any]:
    storage = _storage()
    workspace: dict[str, Any] | None = None
    if workspace_id is not None:
        workspace = storage.get_workspace(workspace_id)
    if workspace is None:
        session_id = session.get("workspace_id")
        if isinstance(session_id, int):
            workspace = storage.get_workspace(session_id)
    if workspace is None:
        workspaces = storage.list_workspaces()
        if workspaces:
            workspace = workspaces[0]
    if workspace is None:
        abort(404, description="No workspaces available.")
    session["workspace_id"] = int(workspace["id"])
    session["labeler_project_id"] = int(workspace["id"])
    return workspace


def _get_workspace_or_404(workspace_id: int) -> dict[str, Any]:
    workspace = _storage().get_workspace(workspace_id)
    if workspace is None:
        abort(404, description=f"Workspace id={workspace_id} not found.")
    session["workspace_id"] = int(workspace["id"])
    session["labeler_project_id"] = int(workspace["id"])
    return workspace


def _get_dataset_project_or_404(workspace_id: int, dataset_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
    workspace = _get_workspace_or_404(workspace_id)
    dataset_project = _storage().get_workspace_dataset(workspace_id, dataset_id)
    if dataset_project is None:
        abort(404, description=f"Dataset project id={dataset_id} not found in workspace.")
    session["workspace_id"] = workspace_id
    session["labeler_project_id"] = int(dataset_project["id"])
    return workspace, dataset_project


def _get_or_create_state(
    project: dict[str, Any],
    image_name: str,
) -> tuple[str, ImageMaskState, list[dict[str, Any]]]:
    safe_name = safe_image_name(image_name)
    images_dir = _project_images_dir(project)
    masks_dir = _project_masks_dir(project)
    cache_dir = _project_cache_dir(project)
    classes = _project_classes(project)
    img_path = image_path(images_dir, safe_name)

    sid = _labeler_session_id()
    state_key = _state_key(int(project["id"]), safe_name)
    with STATE_LOCK:
        existing = SESSION_STATES.get(sid, {}).get(state_key)
    if existing is not None:
        return safe_name, existing, classes

    slic_settings = _effective_slic_for_image(project, safe_name)
    segments, _ = load_or_create_slic_cache(
        img_path,
        cache_dir,
        n_segments=int(slic_settings["n_segments"]),
        compactness=float(slic_settings["compactness"]),
        sigma=float(slic_settings["sigma"]),
        colorspace=str(slic_settings["colorspace"]),
    )
    selected = selected_from_saved_masks(safe_name, classes, segments, masks_dir)
    created = ImageMaskState(segments=segments, selected=selected)

    with STATE_LOCK:
        bucket = SESSION_STATES.setdefault(sid, {})
        state = bucket.get(state_key)
        if state is None:
            bucket[state_key] = created
            state = created
    return safe_name, state, classes


def _image_navigation_urls(
    project: dict[str, Any],
    current_image_name: str,
    build_label_url,
) -> tuple[str | None, str | None]:
    safe_current = safe_image_name(current_image_name)
    images = list_images(_project_images_dir(project))
    if not images:
        return None, None
    try:
        idx = images.index(safe_current)
    except ValueError:
        return None, None

    prev_url = build_label_url(images[idx - 1]) if idx > 0 else None
    next_url = build_label_url(images[idx + 1]) if idx + 1 < len(images) else None
    return prev_url, next_url


def _filmstrip_items(
    project: dict[str, Any],
    current_image_name: str,
    build_label_url,
    build_thumbnail_url,
) -> list[dict[str, Any]]:
    safe_current = safe_image_name(current_image_name)
    images = list_images(_project_images_dir(project))
    classes = _project_classes(project)
    masks_dir = _project_masks_dir(project)
    items: list[dict[str, Any]] = []
    for candidate in images:
        stem = Path(candidate).stem
        is_labeled = bool(iter_existing_mask_paths(masks_dir, stem, classes))
        items.append(
            {
                "image_name": candidate,
                "label_url": build_label_url(candidate),
                "thumbnail_url": build_thumbnail_url(candidate),
                "is_current": candidate == safe_current,
                "is_labeled": is_labeled,
            }
        )
    return items


def _render_labeler_page(
    *,
    workspace: dict[str, Any],
    project: dict[str, Any],
    image_name: str,
    back_url: str,
    api_base: str,
    image_url: str,
    boundary_url: str,
    context_label: str,
    prev_image_url: str | None,
    next_image_url: str | None,
    filmstrip_items: list[dict[str, Any]],
) -> str:
    try:
        safe_name, state, classes = _get_or_create_state(project, image_name)
    except FileNotFoundError:
        abort(404)
    if not classes:
        abort(400, description="No classes configured for this project.")
    project_default_slic = _project_default_slic(project)
    current_slic = _effective_slic_for_image(project, safe_name)
    default_class_id = int(classes[0]["id"])
    class_name_map = class_name_by_id({"classes": classes})
    with STATE_LOCK:
        initial_overlay = _mask_overlay_base64(
            state.class_mask(default_class_id),
            _class_color(classes, default_class_id),
        )
    return render_template(
        "labeler_page.html",
        workspace=workspace,
        project_id=int(project["id"]),
        image_name=safe_name,
        classes=classes,
        categories=[str(item["name"]) for item in classes],
        class_name_map=class_name_map,
        default_class_id=default_class_id,
        initial_mask_base64=initial_overlay,
        back_url=back_url,
        api_base=api_base,
        image_url=image_url,
        boundary_url=boundary_url,
        context_label=context_label,
        prev_image_url=prev_image_url,
        next_image_url=next_image_url,
        filmstrip_items=filmstrip_items,
        slic_default=project_default_slic,
        slic_current=current_slic,
        slic_detail_presets=SLIC_DETAIL_PRESETS,
        project=project,
    )


def _api_project_image(project: dict[str, Any], image_name: str):
    try:
        img_path = image_path(_project_images_dir(project), image_name)
    except FileNotFoundError:
        abort(404)
    return send_file(img_path)


def _api_project_thumbnail(project: dict[str, Any], image_name: str):
    try:
        safe_name = safe_image_name(image_name)
        img_path = image_path(_project_images_dir(project), safe_name)
    except FileNotFoundError:
        abort(404)

    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    with Image.open(img_path) as source:
        rgb = source.convert("RGB")
        rgb.thumbnail((96, 96), resample=resample)
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=82, optimize=True)
        buffer.seek(0)
        return send_file(buffer, mimetype="image/jpeg")


def _api_project_boundary(project: dict[str, Any], image_name: str):
    try:
        safe_name = safe_image_name(image_name)
        img_path = image_path(_project_images_dir(project), safe_name)
    except FileNotFoundError:
        abort(404)
    slic_settings = _effective_slic_for_image(project, safe_name)
    _, boundary_path = load_or_create_slic_cache(
        img_path,
        _project_cache_dir(project),
        n_segments=int(slic_settings["n_segments"]),
        compactness=float(slic_settings["compactness"]),
        sigma=float(slic_settings["sigma"]),
        colorspace=str(slic_settings["colorspace"]),
    )
    return send_file(boundary_path, mimetype="image/png")


def _api_project_mask(project: dict[str, Any], image_name: str):
    classes = _project_classes(project)
    class_id = _resolve_payload_class_id(
        request.args.get("class_id"),
        classes,
        fallback_to_default=False,
    )
    if class_id is None:
        class_id = _resolve_payload_class_id(request.args.get("class_name"), classes)
    if class_id is None:
        return _json_error("Unknown class selection.", 400)
    class_name = class_name_by_id({"classes": classes}).get(int(class_id), "")
    try:
        _, state, _ = _get_or_create_state(project, image_name)
    except FileNotFoundError:
        return _json_error("Image not found.", 404)
    with STATE_LOCK:
        overlay = _mask_overlay_base64(state.class_mask(int(class_id)), _class_color(classes, int(class_id)))
    return jsonify({"mask_png_base64": overlay, "class_id": int(class_id), "class_name": class_name})


def _api_project_context(project: dict[str, Any], image_name: str, build_label_url):
    safe_name = safe_image_name(image_name)
    images = list_images(_project_images_dir(project))
    if safe_name not in images:
        return _json_error("Image not found.", 404)

    prev_image_url, next_image_url = _image_navigation_urls(
        project,
        safe_name,
        build_label_url,
    )
    return jsonify(
        {
            "image_name": safe_name,
            "prev_image_url": prev_image_url,
            "next_image_url": next_image_url,
            "slic_current": _effective_slic_for_image(project, safe_name),
        }
    )


def _api_project_click(project: dict[str, Any]):
    classes = _project_classes(project)
    if not classes:
        return _json_error("No classes configured for this project.", 400)
    payload = request.get_json(silent=True) or {}
    image_name = payload.get("image_name", "")
    class_id = _resolve_payload_class_id(payload.get("class_id"), classes, fallback_to_default=False)
    if class_id is None:
        class_id = _resolve_payload_class_id(payload.get("class_name"), classes)
    mode = payload.get("mode", "add")
    if class_id is None:
        return _json_error("Unknown class selection.", 400)
    if mode not in {"add", "remove"}:
        return _json_error("mode must be 'add' or 'remove'", 400)
    try:
        x = int(float(payload.get("x")))
        y = int(float(payload.get("y")))
    except (TypeError, ValueError):
        return _json_error("x and y must be numeric", 400)
    try:
        _, state, _ = _get_or_create_state(project, image_name)
    except FileNotFoundError:
        return _json_error("Image not found.", 404)
    h, w = state.segments.shape
    if not (0 <= x < w and 0 <= y < h):
        return _json_error(f"Coordinates out of bounds: ({x}, {y}) for image size ({w}, {h})", 400)
    with STATE_LOCK:
        sp_id = int(state.segments[y, x])
        changed = state.apply_click(int(class_id), sp_id, mode)
        overlay = _mask_overlay_base64(state.class_mask(int(class_id)), _class_color(classes, int(class_id)))
    return jsonify(
        {
            "mask_png_base64": overlay,
            "class_id": int(class_id),
            "class_name": class_name_by_id({"classes": classes}).get(int(class_id), ""),
            "mode": mode,
            "superpixel_id": sp_id,
            "changed": changed,
        }
    )


def _api_project_select_ids(project: dict[str, Any]):
    payload = request.get_json(silent=True) or {}
    image_name = payload.get("image_name", "")
    tool = str(payload.get("tool", "brush")).strip().lower()
    marquee_rule = str(payload.get("marquee_rule", "centroid")).strip().lower()
    if tool not in {"brush", "marquee"}:
        return _json_error("tool must be 'brush' or 'marquee'", 400)
    if marquee_rule not in {"centroid", "intersects"}:
        marquee_rule = "centroid"

    try:
        _, state, _ = _get_or_create_state(project, image_name)
    except FileNotFoundError:
        return _json_error("Image not found.", 404)

    with STATE_LOCK:
        if tool == "brush":
            selected_ids = _selection_ids_from_brush(state.segments, payload.get("points", []))
        else:
            selected_ids = _selection_ids_from_marquee(state, payload.get("rect"), marquee_rule)
    return jsonify({"selected_ids": selected_ids, "count": len(selected_ids), "tool": tool})


def _api_project_selection_preview(project: dict[str, Any]):
    payload = request.get_json(silent=True) or {}
    image_name = payload.get("image_name", "")
    selected_ids = _normalize_selected_ids(payload.get("selected_ids", []))

    try:
        _, state, _ = _get_or_create_state(project, image_name)
    except FileNotFoundError:
        return _json_error("Image not found.", 404)

    with STATE_LOCK:
        if not selected_ids:
            mask_bool = np.zeros_like(state.segments, dtype=bool)
        else:
            selected_arr = np.array(selected_ids, dtype=state.segments.dtype)
            mask_bool = np.isin(state.segments, selected_arr)
        overlay = _mask_overlay_base64(mask_bool, SELECTION_COLOR)
    return jsonify({"mask_png_base64": overlay, "count": len(selected_ids)})


def _api_project_bulk_apply(project: dict[str, Any]):
    classes = _project_classes(project)
    if not classes:
        return _json_error("No classes configured for this project.", 400)
    payload = request.get_json(silent=True) or {}
    image_name = payload.get("image_name", "")
    class_id = _resolve_payload_class_id(payload.get("class_id"), classes, fallback_to_default=False)
    if class_id is None:
        class_id = _resolve_payload_class_id(payload.get("class_name"), classes)
    mode = str(payload.get("mode", "add")).strip().lower()
    if class_id is None:
        return _json_error("Unknown class selection.", 400)
    if mode not in {"add", "remove"}:
        return _json_error("mode must be 'add' or 'remove'", 400)

    selected_ids = _normalize_selected_ids(payload.get("selected_ids", []))
    if not selected_ids:
        return _json_error("selected_ids cannot be empty.", 400)

    try:
        _, state, _ = _get_or_create_state(project, image_name)
    except FileNotFoundError:
        return _json_error("Image not found.", 404)

    h, w = state.segments.shape
    max_sp_id = int(state.segments.max()) if h > 0 and w > 0 else -1
    valid_ids = [sp_id for sp_id in selected_ids if 0 <= sp_id <= max_sp_id]

    changed_count = 0
    with STATE_LOCK:
        for sp_id in valid_ids:
            if state.apply_click(int(class_id), sp_id, mode):
                changed_count += 1
        overlay = _mask_overlay_base64(state.class_mask(int(class_id)), _class_color(classes, int(class_id)))
    return jsonify(
        {
            "mask_png_base64": overlay,
            "class_id": int(class_id),
            "class_name": class_name_by_id({"classes": classes}).get(int(class_id), ""),
            "mode": mode,
            "selected_count": len(valid_ids),
            "changed_count": changed_count,
        }
    )


def _api_project_recompute_superpixels(project: dict[str, Any]):
    payload = request.get_json(silent=True) or {}
    image_name = safe_image_name(payload.get("image_name", ""))
    if not image_name:
        return _json_error("image_name is required.", 400)

    try:
        image_path_value = image_path(_project_images_dir(project), image_name)
    except FileNotFoundError:
        return _json_error("Image not found.", 404)

    apply_remaining = _is_truthy(payload.get("apply_remaining"))
    force_overwrite = _is_truthy(payload.get("force_overwrite"))
    requested_slic = _resolve_requested_slic_settings(project, payload)
    project_id = int(project["id"])

    images = list_images(_project_images_dir(project))
    if apply_remaining:
        target_images = images
    else:
        target_images = [image_name]

    masks_to_remove: list[Path] = []
    for name in target_images:
        masks_to_remove.extend(_mask_files_for_image(project, name))

    if masks_to_remove and not force_overwrite:
        return jsonify(
            {
                "requires_confirmation": True,
                "message": (
                    f"Recomputing will invalidate {len(masks_to_remove)} saved mask file(s) "
                    f"across {len(target_images)} image(s)."
                ),
                "affected_images": len(target_images),
                "saved_mask_files": len(masks_to_remove),
            }
        )

    removed_mask_files = 0
    if masks_to_remove and force_overwrite:
        for mask_path in masks_to_remove:
            if mask_path.exists():
                mask_path.unlink()
                removed_mask_files += 1

    storage = _storage()
    preset_mode = requested_slic["preset_mode"]
    if preset_mode == "dataset_default":
        storage.delete_image_slic_overrides_for_images(project_id, target_images)
    else:
        for name in target_images:
            storage.upsert_image_slic_override(
                project_id,
                name,
                slic_algorithm=requested_slic["algorithm"],
                preset_name=requested_slic["preset_name"],
                detail_level=requested_slic["detail_level"],
                n_segments=requested_slic["n_segments"],
                compactness=requested_slic["compactness"],
                sigma=requested_slic["sigma"],
                colorspace=requested_slic["colorspace"],
            )

    recomputed = 0
    cache_dir = _project_cache_dir(project)
    for name in target_images:
        safe_name = safe_image_name(name)
        clear_slic_cache(cache_dir, Path(safe_name).stem)
        _clear_project_image_state(project_id, safe_name)
        try:
            img_path = image_path(_project_images_dir(project), safe_name)
        except FileNotFoundError:
            continue
        effective = _effective_slic_for_image(project, safe_name)
        load_or_create_slic_cache(
            img_path,
            cache_dir,
            n_segments=int(effective["n_segments"]),
            compactness=float(effective["compactness"]),
            sigma=float(effective["sigma"]),
            colorspace=str(effective["colorspace"]),
        )
        recomputed += 1

    current_effective = _effective_slic_for_image(project, image_name)
    if removed_mask_files > 0:
        storage.bump_labeler_project_version(project_id)
    return jsonify(
        {
            "message": f"Recomputed superpixels for {recomputed} image(s).",
            "recomputed_images": recomputed,
            "removed_mask_files": removed_mask_files,
            "saved_mask_files": len(masks_to_remove),
            "current_slic": current_effective,
            "requires_confirmation": False,
        }
    )


def _api_project_undo(project: dict[str, Any]):
    classes = _project_classes(project)
    if not classes:
        return _json_error("No classes configured for this project.", 400)
    payload = request.get_json(silent=True) or {}
    image_name = payload.get("image_name", "")
    class_id = _resolve_payload_class_id(payload.get("class_id"), classes, fallback_to_default=False)
    if class_id is None:
        class_id = _resolve_payload_class_id(payload.get("class_name"), classes)
    if class_id is None:
        return _json_error("Unknown class selection.", 400)
    try:
        _, state, _ = _get_or_create_state(project, image_name)
    except FileNotFoundError:
        return _json_error("Image not found.", 404)
    with STATE_LOCK:
        action = state.undo()
        overlay = _mask_overlay_base64(state.class_mask(int(class_id)), _class_color(classes, int(class_id)))
    class_name_map = class_name_by_id({"classes": classes})
    action_payload = None
    if action is not None:
        action_payload = {
            "class_id": int(action.class_id),
            "class_name": class_name_map.get(int(action.class_id), ""),
            "superpixel_id": action.sp_id,
            "previous_present": action.previous_present,
        }
    return jsonify(
        {
            "mask_png_base64": overlay,
            "class_id": int(class_id),
            "class_name": class_name_map.get(int(class_id), ""),
            "undone": action is not None,
            "action": action_payload,
        }
    )


def _api_project_redo(project: dict[str, Any]):
    classes = _project_classes(project)
    if not classes:
        return _json_error("No classes configured for this project.", 400)
    payload = request.get_json(silent=True) or {}
    image_name = payload.get("image_name", "")
    class_id = _resolve_payload_class_id(payload.get("class_id"), classes, fallback_to_default=False)
    if class_id is None:
        class_id = _resolve_payload_class_id(payload.get("class_name"), classes)
    if class_id is None:
        return _json_error("Unknown class selection.", 400)
    try:
        _, state, _ = _get_or_create_state(project, image_name)
    except FileNotFoundError:
        return _json_error("Image not found.", 404)
    with STATE_LOCK:
        action = state.redo()
        overlay = _mask_overlay_base64(state.class_mask(int(class_id)), _class_color(classes, int(class_id)))
    class_name_map = class_name_by_id({"classes": classes})
    action_payload = None
    if action is not None:
        action_payload = {
            "class_id": int(action.class_id),
            "class_name": class_name_map.get(int(action.class_id), ""),
            "superpixel_id": action.sp_id,
            "previous_present": action.previous_present,
        }
    return jsonify(
        {
            "mask_png_base64": overlay,
            "class_id": int(class_id),
            "class_name": class_name_map.get(int(class_id), ""),
            "redone": action is not None,
            "action": action_payload,
        }
    )


def _api_project_save(project: dict[str, Any]):
    payload = request.get_json(silent=True) or {}
    image_name = payload.get("image_name", "")
    try:
        safe_name, state, classes = _get_or_create_state(project, image_name)
    except FileNotFoundError:
        return _json_error("Image not found.", 404)
    masks_dir = _project_masks_dir(project)
    stem = Path(safe_name).stem
    saved_files: list[str] = []
    removed_files: list[str] = []
    with STATE_LOCK:
        for class_entry in classes:
            class_id = class_id_from_entry(class_entry)
            if class_id is None:
                continue
            class_name = str(class_entry.get("name", "")).strip()
            mask_path = masks_dir / mask_filename_for_class_id(stem, int(class_id))
            legacy_mask_path = masks_dir / mask_filename(stem, class_name) if class_name else None
            mask_bool = state.class_mask(int(class_id))
            if mask_bool.any():
                save_binary_mask(mask_bool, mask_path)
                saved_files.append(mask_path.name)
                if legacy_mask_path is not None and legacy_mask_path.exists() and legacy_mask_path.is_file():
                    legacy_mask_path.unlink()
                    removed_files.append(legacy_mask_path.name)
            else:
                if mask_path.exists() and mask_path.is_file():
                    mask_path.unlink()
                    removed_files.append(mask_path.name)
                if legacy_mask_path is not None and legacy_mask_path.exists() and legacy_mask_path.is_file():
                    legacy_mask_path.unlink()
                    removed_files.append(legacy_mask_path.name)
    _storage().bump_labeler_project_version(int(project["id"]))
    return jsonify(
        {
            "message": f"Saved {len(saved_files)} mask file(s).",
            "saved_files": saved_files,
            "removed_files": removed_files,
        }
    )


def _api_project_export_coco(project: dict[str, Any]):
    classes = _project_classes(project)
    try:
        output_path, image_count, annotation_count = export_coco_annotations(
            images_dir=_project_images_dir(project),
            masks_dir=_project_masks_dir(project),
            coco_dir=_project_coco_dir(project),
            categories=classes,
            min_area=int(current_app.config["LABELER_MIN_COCO_AREA"]),
        )
    except ModuleNotFoundError as exc:
        return _json_error(f"Missing dependency: {exc.name}. Run pip install -r requirements.txt", 500)
    relative_output = _to_relative_under_base(str(output_path)) or str(output_path)
    return jsonify(
        {
            "message": (
                f"COCO export complete: {annotation_count} annotation(s) across "
                f"{image_count} image(s)."
            ),
            "output_path": str(relative_output),
        }
    )


@bp.app_context_processor
def inject_context() -> dict[str, Any]:
    return {"app_name": str(current_app.config.get("APP_NAME", "Model Foundry"))}


@bp.before_app_request
def ensure_dirs() -> None:
    ensure_dir(_labeler_root_dir())
    ensure_dir(_labeler_root_dir() / "workspaces")
    _labeler_session_id()


@bp.route("/labeler", methods=["GET"])
def index() -> str:
    if not _storage().list_workspaces():
        flash("Create a workspace before opening the image labeler.", "warning")
        return redirect(url_for("main.dashboard_root"))
    workspace = _resolve_workspace()
    return redirect(url_for("labeler.workspace_images", workspace_id=int(workspace["id"])))


@bp.route("/workspace/<int:workspace_id>/images", methods=["GET"])
def workspace_images(workspace_id: int) -> str:
    workspace = _get_workspace_or_404(workspace_id)
    image_cards = _image_file_cards(workspace)
    return render_template(
        "labeler_index.html",
        workspace=workspace,
        image_cards=image_cards,
        image_count=len(image_cards),
    )


@bp.route("/workspaces", methods=["POST"])
@bp.route("/collections", methods=["POST"])
@bp.route("/labeler/projects", methods=["POST"])
def create_workspace() -> str:
    storage = _storage()
    return_to = str(request.form.get("return_to", "dashboard")).strip().lower()
    name = str(request.form.get("name", "")).strip()
    if not name:
        flash("Workspace name is required.", "error")
        workspaces = storage.list_workspaces()
        if workspaces:
            workspace_id = int(workspaces[0]["id"])
            session["workspace_id"] = workspace_id
            session["labeler_project_id"] = workspace_id
            return redirect(url_for("main.workspace_dashboard", workspace_id=workspace_id))
        return redirect(url_for("main.dashboard_root"))

    categories = _default_labeler_categories()
    class_schema = build_schema_from_names(categories)
    project_root = (_labeler_root_dir() / "workspaces" / f"{_slugify(name)}_{uuid.uuid4().hex[:6]}").resolve()
    images_dir = (project_root / "raw_images").resolve()
    ensure_dir(images_dir)

    masks_dir = (project_root / "masks").resolve()
    coco_dir = (project_root / "coco").resolve()
    cache_dir = (masks_dir / "_cache").resolve()
    ensure_dir(masks_dir)
    ensure_dir(coco_dir)
    ensure_dir(cache_dir)

    try:
        workspace_id = storage.create_labeler_project(
            name=name,
            dataset_id=None,
            project_dir=str(project_root),
            images_dir=str(images_dir),
            masks_dir=str(masks_dir),
            coco_dir=str(coco_dir),
            cache_dir=str(cache_dir),
            categories=class_schema,
            augmentation_seed=int(current_app.config["RANDOM_SEED"]),
            slic_algorithm="slic",
            slic_preset_name="medium",
            slic_detail_level="medium",
            slic_n_segments=int(current_app.config["LABELER_SLIC_N_SEGMENTS"]),
            slic_compactness=float(current_app.config["LABELER_SLIC_COMPACTNESS"]),
            slic_sigma=float(current_app.config["LABELER_SLIC_SIGMA"]),
            slic_colorspace="lab",
        )
    except sqlite3.IntegrityError:
        flash(f"A workspace named '{name}' already exists.", "error")
        workspaces = storage.list_workspaces()
        if workspaces:
            workspace_id = int(workspaces[0]["id"])
            session["workspace_id"] = workspace_id
            session["labeler_project_id"] = workspace_id
            return redirect(url_for("main.workspace_dashboard", workspace_id=workspace_id))
        return redirect(url_for("main.dashboard_root"))

    files = []
    files.extend(request.files.getlist("image_files"))
    files.extend(request.files.getlist("folder_files"))
    saved_count, skipped_count = _save_uploaded_images(images_dir, files)
    if saved_count:
        storage.touch_labeler_project(workspace_id)

    storage.backfill_workspace_links(workspace_id)
    session["workspace_id"] = workspace_id
    session["labeler_project_id"] = workspace_id
    if saved_count > 0 and skipped_count > 0:
        flash(
            (
                f"Workspace '{name}' created. Uploaded {saved_count} image(s); "
                f"skipped {skipped_count} non-image file(s)."
            ),
            "warning",
        )
    elif saved_count > 0:
        flash(f"Workspace '{name}' created with {saved_count} uploaded image(s).", "success")
    elif skipped_count > 0:
        flash(
            f"Workspace '{name}' created. No valid images uploaded ({skipped_count} skipped).",
            "warning",
        )
    else:
        flash(f"Workspace '{name}' created.", "success")

    if return_to == "images":
        return redirect(url_for("labeler.workspace_images", workspace_id=workspace_id))
    if return_to == "landing":
        return redirect(url_for("main.dashboard_root"))
    return redirect(url_for("main.workspace_dashboard", workspace_id=workspace_id))


@bp.route("/workspace/<int:workspace_id>/images/upload", methods=["POST"])
@bp.route("/collections/<int:workspace_id>/upload", methods=["POST"])
def upload_images(workspace_id: int) -> str:
    workspace = _get_workspace_or_404(workspace_id)
    return_to = str(request.form.get("return_to", "images")).strip().lower()
    images_dir = _project_images_dir(workspace)
    ensure_dir(images_dir)

    files = []
    files.extend(request.files.getlist("image_files"))
    files.extend(request.files.getlist("folder_files"))
    saved_count, skipped_count = _save_uploaded_images(images_dir, files)

    if saved_count == 0 and skipped_count == 0:
        flash("No image files were selected.", "warning")
    elif saved_count > 0:
        _storage().touch_labeler_project(workspace_id)
        if skipped_count:
            flash(
                f"Uploaded {saved_count} image(s). Skipped {skipped_count} non-image file(s).",
                "warning",
            )
        else:
            flash(f"Uploaded {saved_count} image(s) to workspace '{workspace['name']}'.", "success")
    else:
        flash("No valid image files were uploaded.", "error")

    if return_to == "dashboard":
        return redirect(url_for("main.workspace_dashboard", workspace_id=workspace_id))
    return redirect(url_for("labeler.workspace_images", workspace_id=workspace_id))


@bp.route("/labeler/projects/<int:project_id>/classes", methods=["POST"])
def update_project_classes(project_id: int) -> str:
    storage = _storage()
    project = storage.get_labeler_project(project_id)
    if project is None:
        abort(404, description=f"Labeling project id={project_id} not found.")
    categories_text = str(request.form.get("categories_text", ""))
    categories = _parse_categories_text(categories_text)
    existing_schema = _project_class_schema(project)
    next_schema, class_diff = update_schema_with_names(existing_schema, categories)
    is_dataset_draft = str(project.get("kind", "")).lower() == "dataset"

    if not categories:
        flash("Provide at least one class name.", "error")
        if is_dataset_draft:
            _set_dataset_class_edit_state(project_id, categories_text=categories_text, diff=class_diff)
            parent_workspace_id = int(project.get("parent_workspace_id") or 0)
            return redirect(
                url_for(
                    "main.workspace_dataset_detail",
                    workspace_id=parent_workspace_id,
                    dataset_id=int(project["id"]),
                )
            )
        return redirect(url_for("labeler.workspace_images", workspace_id=int(project["id"])))

    if is_dataset_draft and class_diff["destructive"]:
        confirmed_destructive = _is_truthy(request.form.get("confirm_destructive"))
        if not confirmed_destructive:
            flash(
                (
                    "Destructive class edits detected (class removals). "
                    "Check destructive-change confirmation to proceed."
                ),
                "error",
            )
            _set_dataset_class_edit_state(project_id, categories_text=categories_text, diff=class_diff)
            parent_workspace_id = int(project.get("parent_workspace_id") or 0)
            return redirect(
                url_for(
                    "main.workspace_dataset_detail",
                    workspace_id=parent_workspace_id,
                    dataset_id=int(project["id"]),
                )
            )

    if is_dataset_draft:
        _clear_dataset_class_edit_state(project_id)

    removed_mask_files = 0
    removed_legacy_mask_files = 0
    if class_diff.get("removed_ids"):
        masks_dir = _project_masks_dir(project)
        for raw_id in class_diff.get("removed_ids", []):
            try:
                class_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            for mask_path in masks_dir.glob(f"*__cid_{class_id}.png"):
                if mask_path.exists() and mask_path.is_file():
                    mask_path.unlink()
                    removed_mask_files += 1
        for raw_name in class_diff.get("removed", []):
            sanitized = sanitize_class_name(str(raw_name))
            if not sanitized:
                continue
            for mask_path in masks_dir.glob(f"*__{sanitized}.png"):
                if re.search(r"__cid_\d+\.png$", mask_path.name):
                    continue
                if mask_path.exists() and mask_path.is_file():
                    mask_path.unlink()
                    removed_legacy_mask_files += 1

    storage.update_labeler_project_categories(project_id, next_schema, bump_version=is_dataset_draft)
    _clear_project_state(project_id)
    if class_diff["destructive"]:
        flash(
            (
                f"Updated classes for '{project['name']}'. "
                f"Removed {removed_mask_files + removed_legacy_mask_files} mask file(s) for deleted classes."
            ),
            "warning",
        )
    elif class_diff["added"] and class_diff["append_only"]:
        flash(f"Updated classes for '{project['name']}' (append-only update).", "success")
    else:
        flash(f"Updated classes for '{project['name']}'.", "success")

    return_to = str(request.form.get("return_to", "")).strip()
    if return_to.startswith("/") and not return_to.startswith("//"):
        return redirect(return_to)

    if is_dataset_draft:
        parent_workspace_id = int(project.get("parent_workspace_id") or 0)
        return redirect(
            url_for(
                "main.workspace_dataset_detail",
                workspace_id=parent_workspace_id,
                dataset_id=int(project["id"]),
            )
        )
    workspace_id = int(project.get("parent_workspace_id") or project["id"])
    return redirect(url_for("labeler.workspace_images", workspace_id=workspace_id))


@bp.route("/labeler/projects/<int:workspace_id>/sync-dataset-classes", methods=["POST"])
def sync_dataset_classes(workspace_id: int) -> str:
    storage = _storage()
    workspace = _get_workspace_or_404(workspace_id)
    dataset_id = workspace.get("dataset_id")
    if dataset_id is None:
        flash("This workspace is not linked to a dataset.", "error")
        return redirect(url_for("labeler.workspace_images", workspace_id=workspace_id))
    dataset = storage.get_dataset(int(dataset_id), workspace_id=workspace_id)
    if dataset is None:
        flash("Linked dataset was not found.", "error")
        return redirect(url_for("labeler.workspace_images", workspace_id=workspace_id))
    categories = _dataset_categories(dataset)
    if not categories:
        flash("Linked dataset has no categories.", "error")
        return redirect(url_for("labeler.workspace_images", workspace_id=workspace_id))
    workspace_project = storage.get_workspace(workspace_id)
    if workspace_project is None:
        abort(404, description=f"Workspace id={workspace_id} not found.")
    next_schema, _ = update_schema_with_names(workspace_project.get("categories_json"), categories)
    storage.update_labeler_project_categories(workspace_id, next_schema)
    _clear_project_state(workspace_id)
    flash(f"Classes synced from dataset '{dataset['name']}'.", "success")
    return redirect(url_for("labeler.workspace_images", workspace_id=workspace_id))


@bp.route("/workspace/<int:workspace_id>/images/<path:image_name>/label", methods=["GET"])
@bp.route("/labeler/project/<int:workspace_id>/<path:image_name>", methods=["GET"])
def page(workspace_id: int, image_name: str) -> str:
    _get_workspace_or_404(workspace_id)
    flash("Open the labeler from a dataset page. Workspace images are inspect-only.", "warning")
    return redirect(url_for("main.workspace_datasets", workspace_id=workspace_id))


@bp.route("/workspace/<int:workspace_id>/images/api/image/<path:image_name>", methods=["GET"])
@bp.route("/labeler/project/<int:workspace_id>/api/image/<path:image_name>", methods=["GET"])
def api_image(workspace_id: int, image_name: str):
    workspace = _get_workspace_or_404(workspace_id)
    return _api_project_image(workspace, image_name)


@bp.route("/workspace/<int:workspace_id>/images/api/thumb/<path:image_name>", methods=["GET"])
@bp.route("/labeler/project/<int:workspace_id>/api/thumb/<path:image_name>", methods=["GET"])
def api_thumbnail(workspace_id: int, image_name: str):
    workspace = _get_workspace_or_404(workspace_id)
    return _api_project_thumbnail(workspace, image_name)


@bp.route("/workspace/<int:workspace_id>/images/api/boundary/<path:image_name>", methods=["GET"])
@bp.route("/labeler/project/<int:workspace_id>/api/boundary/<path:image_name>", methods=["GET"])
def api_boundary(workspace_id: int, image_name: str):
    workspace = _get_workspace_or_404(workspace_id)
    return _api_project_boundary(workspace, image_name)


@bp.route("/workspace/<int:workspace_id>/images/api/mask/<path:image_name>", methods=["GET"])
@bp.route("/labeler/project/<int:workspace_id>/api/mask/<path:image_name>", methods=["GET"])
def api_mask(workspace_id: int, image_name: str):
    workspace = _get_workspace_or_404(workspace_id)
    return _api_project_mask(workspace, image_name)


@bp.route("/workspace/<int:workspace_id>/images/api/context/<path:image_name>", methods=["GET"])
@bp.route("/labeler/project/<int:workspace_id>/api/context/<path:image_name>", methods=["GET"])
def api_context(workspace_id: int, image_name: str):
    workspace = _get_workspace_or_404(workspace_id)
    return _api_project_context(
        workspace,
        image_name,
        lambda candidate: url_for("labeler.page", workspace_id=workspace_id, image_name=candidate),
    )


@bp.route("/workspace/<int:workspace_id>/images/api/click", methods=["POST"])
@bp.route("/labeler/project/<int:workspace_id>/api/click", methods=["POST"])
def api_click(workspace_id: int):
    workspace = _get_workspace_or_404(workspace_id)
    return _api_project_click(workspace)


@bp.route("/workspace/<int:workspace_id>/images/api/select_ids", methods=["POST"])
@bp.route("/labeler/project/<int:workspace_id>/api/select_ids", methods=["POST"])
def api_select_ids(workspace_id: int):
    workspace = _get_workspace_or_404(workspace_id)
    return _api_project_select_ids(workspace)


@bp.route("/workspace/<int:workspace_id>/images/api/selection_preview", methods=["POST"])
@bp.route("/labeler/project/<int:workspace_id>/api/selection_preview", methods=["POST"])
def api_selection_preview(workspace_id: int):
    workspace = _get_workspace_or_404(workspace_id)
    return _api_project_selection_preview(workspace)


@bp.route("/workspace/<int:workspace_id>/images/api/bulk_apply", methods=["POST"])
@bp.route("/labeler/project/<int:workspace_id>/api/bulk_apply", methods=["POST"])
def api_bulk_apply(workspace_id: int):
    workspace = _get_workspace_or_404(workspace_id)
    return _api_project_bulk_apply(workspace)


@bp.route("/workspace/<int:workspace_id>/images/api/recompute_superpixels", methods=["POST"])
@bp.route("/labeler/project/<int:workspace_id>/api/recompute_superpixels", methods=["POST"])
def api_recompute_superpixels(workspace_id: int):
    workspace = _get_workspace_or_404(workspace_id)
    return _api_project_recompute_superpixels(workspace)


@bp.route("/workspace/<int:workspace_id>/images/api/undo", methods=["POST"])
@bp.route("/labeler/project/<int:workspace_id>/api/undo", methods=["POST"])
def api_undo(workspace_id: int):
    workspace = _get_workspace_or_404(workspace_id)
    return _api_project_undo(workspace)


@bp.route("/workspace/<int:workspace_id>/images/api/redo", methods=["POST"])
@bp.route("/labeler/project/<int:workspace_id>/api/redo", methods=["POST"])
def api_redo(workspace_id: int):
    workspace = _get_workspace_or_404(workspace_id)
    return _api_project_redo(workspace)


@bp.route("/workspace/<int:workspace_id>/images/api/save", methods=["POST"])
@bp.route("/labeler/project/<int:workspace_id>/api/save", methods=["POST"])
def api_save(workspace_id: int):
    workspace = _get_workspace_or_404(workspace_id)
    return _api_project_save(workspace)


@bp.route("/workspace/<int:workspace_id>/images/api/export_coco", methods=["POST"])
@bp.route("/labeler/project/<int:workspace_id>/api/export_coco", methods=["POST"])
def api_export_coco(workspace_id: int):
    workspace = _get_workspace_or_404(workspace_id)
    return _api_project_export_coco(workspace)


@bp.route("/workspace/<int:workspace_id>/datasets/<int:dataset_id>/images/<path:image_name>/label", methods=["GET"])
def dataset_page(workspace_id: int, dataset_id: int, image_name: str) -> str:
    workspace, dataset_project = _get_dataset_project_or_404(workspace_id, dataset_id)
    api_base = url_for("labeler.dataset_api_click", workspace_id=workspace_id, dataset_id=dataset_id).rsplit(
        "/click",
        1,
    )[0]
    build_dataset_label_url = lambda candidate: url_for(
        "labeler.dataset_page",
        workspace_id=workspace_id,
        dataset_id=dataset_id,
        image_name=candidate,
    )
    prev_image_url, next_image_url = _image_navigation_urls(
        dataset_project,
        image_name,
        build_dataset_label_url,
    )
    filmstrip_items = _filmstrip_items(
        dataset_project,
        image_name,
        build_dataset_label_url,
        lambda candidate: url_for(
            "labeler.dataset_api_thumbnail",
            workspace_id=workspace_id,
            dataset_id=dataset_id,
            image_name=candidate,
        ),
    )
    return _render_labeler_page(
        workspace=workspace,
        project=dataset_project,
        image_name=image_name,
        back_url=url_for("main.workspace_dataset_detail", workspace_id=workspace_id, dataset_id=dataset_id),
        api_base=api_base,
        image_url=url_for(
            "labeler.dataset_api_image",
            workspace_id=workspace_id,
            dataset_id=dataset_id,
            image_name=image_name,
        ),
        boundary_url=url_for(
            "labeler.dataset_api_boundary",
            workspace_id=workspace_id,
            dataset_id=dataset_id,
            image_name=image_name,
        ),
        context_label=f"Dataset: {dataset_project['name']}",
        prev_image_url=prev_image_url,
        next_image_url=next_image_url,
        filmstrip_items=filmstrip_items,
    )


@bp.route("/workspace/<int:workspace_id>/datasets/<int:dataset_id>/images/api/image/<path:image_name>", methods=["GET"])
def dataset_api_image(workspace_id: int, dataset_id: int, image_name: str):
    _, dataset_project = _get_dataset_project_or_404(workspace_id, dataset_id)
    return _api_project_image(dataset_project, image_name)


@bp.route("/workspace/<int:workspace_id>/datasets/<int:dataset_id>/images/api/thumb/<path:image_name>", methods=["GET"])
def dataset_api_thumbnail(workspace_id: int, dataset_id: int, image_name: str):
    _, dataset_project = _get_dataset_project_or_404(workspace_id, dataset_id)
    return _api_project_thumbnail(dataset_project, image_name)


@bp.route("/workspace/<int:workspace_id>/datasets/<int:dataset_id>/images/api/boundary/<path:image_name>", methods=["GET"])
def dataset_api_boundary(workspace_id: int, dataset_id: int, image_name: str):
    _, dataset_project = _get_dataset_project_or_404(workspace_id, dataset_id)
    return _api_project_boundary(dataset_project, image_name)


@bp.route("/workspace/<int:workspace_id>/datasets/<int:dataset_id>/images/api/mask/<path:image_name>", methods=["GET"])
def dataset_api_mask(workspace_id: int, dataset_id: int, image_name: str):
    _, dataset_project = _get_dataset_project_or_404(workspace_id, dataset_id)
    return _api_project_mask(dataset_project, image_name)


@bp.route("/workspace/<int:workspace_id>/datasets/<int:dataset_id>/images/api/context/<path:image_name>", methods=["GET"])
def dataset_api_context(workspace_id: int, dataset_id: int, image_name: str):
    _, dataset_project = _get_dataset_project_or_404(workspace_id, dataset_id)
    return _api_project_context(
        dataset_project,
        image_name,
        lambda candidate: url_for(
            "labeler.dataset_page",
            workspace_id=workspace_id,
            dataset_id=dataset_id,
            image_name=candidate,
        ),
    )


@bp.route("/workspace/<int:workspace_id>/datasets/<int:dataset_id>/images/api/click", methods=["POST"])
def dataset_api_click(workspace_id: int, dataset_id: int):
    _, dataset_project = _get_dataset_project_or_404(workspace_id, dataset_id)
    return _api_project_click(dataset_project)


@bp.route("/workspace/<int:workspace_id>/datasets/<int:dataset_id>/images/api/select_ids", methods=["POST"])
def dataset_api_select_ids(workspace_id: int, dataset_id: int):
    _, dataset_project = _get_dataset_project_or_404(workspace_id, dataset_id)
    return _api_project_select_ids(dataset_project)


@bp.route("/workspace/<int:workspace_id>/datasets/<int:dataset_id>/images/api/selection_preview", methods=["POST"])
def dataset_api_selection_preview(workspace_id: int, dataset_id: int):
    _, dataset_project = _get_dataset_project_or_404(workspace_id, dataset_id)
    return _api_project_selection_preview(dataset_project)


@bp.route("/workspace/<int:workspace_id>/datasets/<int:dataset_id>/images/api/bulk_apply", methods=["POST"])
def dataset_api_bulk_apply(workspace_id: int, dataset_id: int):
    _, dataset_project = _get_dataset_project_or_404(workspace_id, dataset_id)
    return _api_project_bulk_apply(dataset_project)


@bp.route("/workspace/<int:workspace_id>/datasets/<int:dataset_id>/images/api/recompute_superpixels", methods=["POST"])
def dataset_api_recompute_superpixels(workspace_id: int, dataset_id: int):
    _, dataset_project = _get_dataset_project_or_404(workspace_id, dataset_id)
    return _api_project_recompute_superpixels(dataset_project)


@bp.route("/workspace/<int:workspace_id>/datasets/<int:dataset_id>/images/api/undo", methods=["POST"])
def dataset_api_undo(workspace_id: int, dataset_id: int):
    _, dataset_project = _get_dataset_project_or_404(workspace_id, dataset_id)
    return _api_project_undo(dataset_project)


@bp.route("/workspace/<int:workspace_id>/datasets/<int:dataset_id>/images/api/redo", methods=["POST"])
def dataset_api_redo(workspace_id: int, dataset_id: int):
    _, dataset_project = _get_dataset_project_or_404(workspace_id, dataset_id)
    return _api_project_redo(dataset_project)


@bp.route("/workspace/<int:workspace_id>/datasets/<int:dataset_id>/images/api/save", methods=["POST"])
def dataset_api_save(workspace_id: int, dataset_id: int):
    _, dataset_project = _get_dataset_project_or_404(workspace_id, dataset_id)
    return _api_project_save(dataset_project)


@bp.route("/workspace/<int:workspace_id>/datasets/<int:dataset_id>/images/api/export_coco", methods=["POST"])
def dataset_api_export_coco(workspace_id: int, dataset_id: int):
    _, dataset_project = _get_dataset_project_or_404(workspace_id, dataset_id)
    return _api_project_export_coco(dataset_project)
