"""Dataset augmentation workflow for COCO segmentation datasets."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .coco import load_coco, resolve_image_path

try:
    from pycocotools import mask as mask_utils
except ModuleNotFoundError:  # pragma: no cover - checked at runtime
    mask_utils = None


GEOMETRIC_TRANSFORMS = {"orig", "flip_h", "flip_v", "rot90", "rot180"}
SUPPORTED_TRANSFORMS = GEOMETRIC_TRANSFORMS | {
    "color_jitter",
    "grayscale",
    "channel_shuffle",
    "gaussian_noise",
    "speckle_noise",
    "gaussian_blur",
}


@dataclass
class AugmentResult:
    output_dir: str
    images_dir: str
    coco_json_path: str
    image_count: int
    annotation_count: int
    transform_counts: dict[str, int]
    warnings: list[str]
    step_summaries: list[dict[str, Any]]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _slugify(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return normalized or "dataset"


def _log(log_fn: Callable[[str], None] | None, message: str) -> None:
    if log_fn is not None:
        log_fn(message)


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.ndim == 3 and image.shape[2] >= 3:
        image = image[:, :, :3]
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _decode_segmentation(segmentation: Any, height: int, width: int) -> np.ndarray:
    if mask_utils is None:
        raise ModuleNotFoundError("pycocotools")

    if segmentation is None:
        return np.zeros((height, width), dtype=bool)

    if isinstance(segmentation, dict):
        normalized = dict(segmentation)
        counts = normalized.get("counts")
        if isinstance(counts, str):
            normalized["counts"] = counts.encode("utf-8")
            counts = normalized["counts"]
        if isinstance(counts, list):
            rle = mask_utils.frPyObjects(normalized, height, width)
        else:
            rle = normalized
    elif isinstance(segmentation, list):
        if not segmentation:
            return np.zeros((height, width), dtype=bool)
        polygons = segmentation
        if segmentation and isinstance(segmentation[0], (int, float)):
            polygons = [segmentation]
        rles = mask_utils.frPyObjects(polygons, height, width)
        rle = mask_utils.merge(rles) if isinstance(rles, list) else rles
    else:
        raise ValueError(f"Unsupported segmentation type: {type(segmentation)}")

    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(bool)


def _encode_mask(mask_bool: np.ndarray) -> dict[str, Any]:
    if mask_utils is None:
        raise ModuleNotFoundError("pycocotools")
    encoded = mask_utils.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return encoded


def _transform_array(array: np.ndarray, transform: str) -> np.ndarray:
    if transform == "orig":
        return array
    if transform == "flip_h":
        return np.fliplr(array)
    if transform == "flip_v":
        return np.flipud(array)
    if transform == "rot90":
        return np.rot90(array, k=1)
    if transform == "rot180":
        return np.rot90(array, k=2)
    raise ValueError(f"Unsupported geometric transform: {transform}")


def _transform_suffix(transform: str) -> str:
    return {
        "orig": "orig",
        "flip_h": "fh",
        "flip_v": "fv",
        "rot90": "r90",
        "rot180": "r180",
        "color_jitter": "cj",
        "grayscale": "gs",
        "channel_shuffle": "cs",
        "gaussian_noise": "gn",
        "speckle_noise": "sn",
        "gaussian_blur": "gb",
    }[transform]


def _clamp_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(low, min(high, numeric))


def _normalize_selected_names(value: Any) -> list[str]:
    raw_items: list[str] = []
    if isinstance(value, str):
        raw_items = value.replace(",", "\n").splitlines()
    elif isinstance(value, list):
        raw_items = [str(item) for item in value]
    selected: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        item = str(raw).strip()
        if not item:
            continue
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        selected.append(item)
    return selected


def _parse_fixed_channel_order(raw_value: Any) -> tuple[int, int, int] | None:
    if isinstance(raw_value, (list, tuple)):
        tokens = list(raw_value)
    else:
        tokens = str(raw_value or "").replace(" ", "").split(",")
    if len(tokens) != 3:
        return None
    try:
        parsed = tuple(int(item) for item in tokens)
    except (TypeError, ValueError):
        return None
    if sorted(parsed) != [0, 1, 2]:
        return None
    return parsed


def _apply_color_jitter(image_array: np.ndarray, settings: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    brightness = _clamp_float(settings.get("brightness"), 0.0, 1.0, 0.2)
    contrast = _clamp_float(settings.get("contrast"), 0.0, 1.0, 0.2)
    saturation = _clamp_float(settings.get("saturation"), 0.0, 1.0, 0.2)
    hue = _clamp_float(settings.get("hue"), -0.5, 0.5, 0.05)

    image = Image.fromarray(image_array, mode="RGB")
    if brightness > 0:
        factor = float(1.0 + rng.uniform(-brightness, brightness))
        image = ImageEnhance.Brightness(image).enhance(factor)
    if contrast > 0:
        factor = float(1.0 + rng.uniform(-contrast, contrast))
        image = ImageEnhance.Contrast(image).enhance(factor)
    if saturation > 0:
        factor = float(1.0 + rng.uniform(-saturation, saturation))
        image = ImageEnhance.Color(image).enhance(factor)

    if abs(hue) > 1e-8:
        hue_shift = float(rng.uniform(-abs(hue), abs(hue)))
        hsv = np.array(image.convert("HSV"), dtype=np.uint8)
        shift_value = int(round(hue_shift * 255.0))
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + shift_value) % 256
        image = Image.fromarray(hsv, mode="HSV").convert("RGB")

    return _to_uint8_rgb(np.asarray(image))


def _apply_grayscale(image_array: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    alpha = _clamp_float(settings.get("alpha"), 0.0, 1.0, 1.0)
    if alpha <= 0:
        return image_array
    gray = np.dot(image_array[..., :3].astype(np.float32), np.array([0.299, 0.587, 0.114], dtype=np.float32))
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    blended = (1.0 - alpha) * image_array.astype(np.float32) + alpha * gray_rgb
    return np.clip(blended, 0, 255).astype(np.uint8)


def _apply_channel_shuffle(image_array: np.ndarray, settings: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    mode = str(settings.get("mode", "random")).strip().lower()
    if mode == "fixed":
        fixed_order = _parse_fixed_channel_order(settings.get("order"))
        order = fixed_order if fixed_order is not None else (2, 1, 0)
    else:
        order = tuple(int(v) for v in rng.permutation(3))
    return image_array[..., list(order)]


def _apply_gaussian_noise(image_array: np.ndarray, settings: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    mean = _clamp_float(settings.get("mean"), -255.0, 255.0, 0.0)
    std = _clamp_float(settings.get("std"), 0.0, 255.0, 8.0)
    if std <= 0:
        return image_array
    noise = rng.normal(loc=mean, scale=std, size=image_array.shape)
    return np.clip(image_array.astype(np.float32) + noise.astype(np.float32), 0, 255).astype(np.uint8)


def _apply_speckle_noise(image_array: np.ndarray, settings: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    std = _clamp_float(settings.get("std"), 0.0, 3.0, 0.15)
    if std <= 0:
        return image_array
    noise = rng.normal(loc=0.0, scale=std, size=image_array.shape).astype(np.float32)
    source = image_array.astype(np.float32)
    return np.clip(source + source * noise, 0, 255).astype(np.uint8)


def _apply_gaussian_blur(image_array: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    sigma = _clamp_float(settings.get("sigma"), 0.0, 10.0, 1.0)
    if sigma <= 0:
        return image_array
    image = Image.fromarray(image_array, mode="RGB")
    blurred = image.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    return _to_uint8_rgb(np.asarray(blurred))


def _apply_image_transform(
    image_array: np.ndarray,
    transform: str,
    settings: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    if transform in GEOMETRIC_TRANSFORMS:
        return _to_uint8_rgb(_transform_array(image_array, transform))
    if transform == "color_jitter":
        return _apply_color_jitter(image_array, settings, rng)
    if transform == "grayscale":
        return _apply_grayscale(image_array, settings)
    if transform == "channel_shuffle":
        return _apply_channel_shuffle(image_array, settings, rng)
    if transform == "gaussian_noise":
        return _apply_gaussian_noise(image_array, settings, rng)
    if transform == "speckle_noise":
        return _apply_speckle_noise(image_array, settings, rng)
    if transform == "gaussian_blur":
        return _apply_gaussian_blur(image_array, settings)
    raise ValueError(f"Unsupported transform: {transform}")


def _normalize_recipe_steps(recipe_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, step in enumerate(recipe_steps, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"Recipe step {index} must be an object.")
        transform = str(step.get("transform", "")).strip().lower()
        if transform not in SUPPORTED_TRANSFORMS or transform == "orig":
            raise ValueError(f"Unsupported recipe transform at step {index}: {transform}")
        selection_mode = str(step.get("selection_mode", "fifo")).strip().lower()
        if selection_mode not in {"fifo", "random", "selected"}:
            selection_mode = "fifo"
        selection_percent = _clamp_float(step.get("selection_percent"), 0.0, 100.0, 100.0)
        settings = step.get("settings", {})
        if not isinstance(settings, dict):
            settings = {}
        selected_images = _normalize_selected_names(step.get("selected_images", []))
        normalized.append(
            {
                "transform": transform,
                "selection_mode": selection_mode,
                "selection_percent": selection_percent,
                "settings": settings,
                "selected_images": selected_images,
            }
        )
    return normalized


def _select_source_images_for_step(
    source_images: list[dict[str, Any]],
    *,
    selection_mode: str,
    selection_percent: float,
    selected_images: list[str],
    rng: np.random.Generator,
    warnings: list[str],
    step_label: str,
) -> list[dict[str, Any]]:
    ordered_images = sorted(source_images, key=lambda item: str(item.get("file_name", "")))
    total = len(ordered_images)
    if total == 0:
        return []

    if selection_mode == "selected":
        if not selected_images:
            warnings.append(f"{step_label}: selected mode has no image names.")
            return []
        by_full_name: dict[str, dict[str, Any]] = {}
        by_base_name: dict[str, list[dict[str, Any]]] = {}
        for image_row in ordered_images:
            file_name = str(image_row.get("file_name", ""))
            base_name = Path(file_name).name
            by_full_name[file_name] = image_row
            by_base_name.setdefault(base_name, []).append(image_row)

        picked: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        for requested in selected_images:
            key = str(requested).strip()
            if not key:
                continue
            candidates: list[dict[str, Any]]
            if key in by_full_name:
                candidates = [by_full_name[key]]
            else:
                candidates = by_base_name.get(Path(key).name, [])
            if not candidates:
                warnings.append(f"{step_label}: selected image '{key}' not found in source dataset.")
                continue
            for candidate in candidates:
                try:
                    image_id = int(candidate.get("id"))
                except (TypeError, ValueError):
                    continue
                if image_id in seen_ids:
                    continue
                seen_ids.add(image_id)
                picked.append(candidate)
        return picked

    selected_count = int(round((selection_percent / 100.0) * total))
    if selection_percent > 0 and selected_count == 0:
        selected_count = 1
    selected_count = max(0, min(total, selected_count))

    if selection_mode == "random":
        if selected_count >= total:
            return ordered_images
        chosen_indices = np.sort(rng.choice(total, size=selected_count, replace=False))
        return [ordered_images[int(idx)] for idx in chosen_indices.tolist()]

    return ordered_images[:selected_count]


def _append_annotations_for_image(
    *,
    source_annotations: list[dict[str, Any]],
    source_height: int,
    source_width: int,
    image_id: int,
    transform: str,
    min_area: int,
    next_annotation_id: int,
    warnings: list[str],
    output_annotations: list[dict[str, Any]],
) -> int:
    for annotation in source_annotations:
        try:
            mask = _decode_segmentation(annotation.get("segmentation"), source_height, source_width)
        except Exception as exc:
            warnings.append(f"Annotation id={annotation.get('id')} decode failed ({exc}); skipped.")
            continue

        if not mask.any():
            continue

        if transform in GEOMETRIC_TRANSFORMS:
            aug_mask = _transform_array(mask, transform)
        else:
            aug_mask = mask

        area = int(aug_mask.sum())
        if area < int(min_area):
            continue

        encoded = _encode_mask(aug_mask)
        bbox = mask_utils.toBbox(encoded).tolist()
        output_annotations.append(
            {
                "id": next_annotation_id,
                "image_id": image_id,
                "category_id": int(annotation.get("category_id", 0)),
                "segmentation": encoded,
                "area": float(area),
                "bbox": [float(value) for value in bbox],
                "iscrowd": 1,
            }
        )
        next_annotation_id += 1
    return next_annotation_id


def augment_coco_dataset(
    *,
    dataset: dict[str, Any],
    output_root: Path,
    output_name: str,
    recipe_steps: list[dict[str, Any]],
    include_original: bool,
    min_area: int = 1,
    random_seed: int = 42,
    log_fn: Callable[[str], None] | None = None,
) -> AugmentResult:
    if mask_utils is None:
        raise ModuleNotFoundError("pycocotools")

    dataset_path = Path(str(dataset["dataset_path"])).resolve()
    image_root = str(dataset["image_root"])
    coco_path = Path(str(dataset["coco_json_path"])).resolve()

    coco = load_coco(coco_path)
    source_images = list(coco.get("images", []))
    source_annotations = list(coco.get("annotations", []))
    categories = list(coco.get("categories", []))
    if not source_images:
        raise ValueError("Source dataset has no images.")

    normalized_recipe_steps = _normalize_recipe_steps(recipe_steps)
    if not include_original and not normalized_recipe_steps:
        raise ValueError("Recipe is empty. Add at least one augmentation step.")

    stamp = _utc_stamp()
    run_dir = output_root / f"{stamp}_{_slugify(output_name)}"
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    _log(log_fn, f"Creating augmented dataset in {run_dir}")

    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in source_annotations:
        image_id = int(annotation.get("image_id", -1))
        annotations_by_image.setdefault(image_id, []).append(annotation)

    new_images: list[dict[str, Any]] = []
    new_annotations: list[dict[str, Any]] = []
    warnings: list[str] = []
    transform_counts: dict[str, int] = {}
    step_summaries: list[dict[str, Any]] = []

    next_image_id = 1
    next_annotation_id = 1

    ordered_source_images = sorted(source_images, key=lambda item: str(item.get("file_name", "")))
    if include_original:
        added_count = 0
        for image_row in ordered_source_images:
            source_image_id = int(image_row.get("id"))
            file_name = str(image_row.get("file_name", ""))
            image_path = resolve_image_path(dataset_path, image_root, file_name)
            if not image_path.exists():
                warnings.append(f"Missing source image skipped: {image_path}")
                continue

            with Image.open(image_path) as pil_image:
                image_array = _to_uint8_rgb(np.asarray(pil_image))
            source_h, source_w = image_array.shape[:2]

            out_name = f"{_slugify(Path(file_name).stem)}__i{source_image_id}__orig.png"
            out_path = images_dir / out_name
            Image.fromarray(image_array, mode="RGB").save(out_path)

            new_images.append(
                {
                    "id": next_image_id,
                    "file_name": f"images/{out_name}",
                    "width": int(source_w),
                    "height": int(source_h),
                }
            )
            next_annotation_id = _append_annotations_for_image(
                source_annotations=annotations_by_image.get(source_image_id, []),
                source_height=source_h,
                source_width=source_w,
                image_id=next_image_id,
                transform="orig",
                min_area=min_area,
                next_annotation_id=next_annotation_id,
                warnings=warnings,
                output_annotations=new_annotations,
            )
            next_image_id += 1
            added_count += 1
        transform_counts["orig"] = added_count

    for step_index, step in enumerate(normalized_recipe_steps, start=1):
        transform = str(step["transform"])
        selection_mode = str(step["selection_mode"])
        selection_percent = float(step["selection_percent"])
        settings = dict(step.get("settings", {}))
        selected_names = list(step.get("selected_images", []))
        step_label = f"step_{step_index:02d}_{transform}"
        step_rng = np.random.default_rng(int(random_seed) + (step_index * 10007))
        selected_source_images = _select_source_images_for_step(
            ordered_source_images,
            selection_mode=selection_mode,
            selection_percent=selection_percent,
            selected_images=selected_names,
            rng=step_rng,
            warnings=warnings,
            step_label=step_label,
        )

        _log(
            log_fn,
            (
                f"{step_label}: transform={transform}, mode={selection_mode}, "
                f"percent={selection_percent:.2f}, selected={len(selected_source_images)}"
            ),
        )
        generated_count = 0
        for image_row in selected_source_images:
            source_image_id = int(image_row.get("id"))
            file_name = str(image_row.get("file_name", ""))
            image_path = resolve_image_path(dataset_path, image_root, file_name)
            if not image_path.exists():
                warnings.append(f"{step_label}: missing source image skipped: {image_path}")
                continue

            with Image.open(image_path) as pil_image:
                image_array = _to_uint8_rgb(np.asarray(pil_image))
            source_h, source_w = image_array.shape[:2]

            transformed_image = _apply_image_transform(image_array, transform, settings, step_rng)
            aug_h, aug_w = transformed_image.shape[:2]
            suffix = _transform_suffix(transform)
            out_name = f"{_slugify(Path(file_name).stem)}__i{source_image_id}__s{step_index:02d}_{suffix}.png"
            out_path = images_dir / out_name
            Image.fromarray(transformed_image, mode="RGB").save(out_path)

            new_images.append(
                {
                    "id": next_image_id,
                    "file_name": f"images/{out_name}",
                    "width": int(aug_w),
                    "height": int(aug_h),
                }
            )
            next_annotation_id = _append_annotations_for_image(
                source_annotations=annotations_by_image.get(source_image_id, []),
                source_height=source_h,
                source_width=source_w,
                image_id=next_image_id,
                transform=transform,
                min_area=min_area,
                next_annotation_id=next_annotation_id,
                warnings=warnings,
                output_annotations=new_annotations,
            )
            next_image_id += 1
            generated_count += 1

        transform_counts[step_label] = generated_count
        step_summaries.append(
            {
                "step": step_index,
                "transform": transform,
                "selection_mode": selection_mode,
                "selection_percent": selection_percent,
                "selected_count": len(selected_source_images),
                "generated_count": generated_count,
                "selected_images": selected_names if selection_mode == "selected" else [],
                "settings": settings,
            }
        )

    payload = {
        "info": {
            "description": "Model Foundry Augmented Dataset",
            "version": "2.0",
            "date_created": datetime.now(timezone.utc).isoformat(),
            "source_dataset_id": int(dataset["id"]),
            "source_dataset_name": str(dataset["name"]),
            "random_seed": int(random_seed),
            "include_original": bool(include_original),
            "recipe": step_summaries,
        },
        "licenses": [],
        "images": new_images,
        "annotations": new_annotations,
        "categories": categories,
    }

    coco_out = run_dir / "annotations.json"
    with coco_out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    _log(
        log_fn,
        f"Augmentation complete: {len(new_images)} images and {len(new_annotations)} annotations.",
    )

    return AugmentResult(
        output_dir=str(run_dir),
        images_dir=str(images_dir),
        coco_json_path=str(coco_out),
        image_count=len(new_images),
        annotation_count=len(new_annotations),
        transform_counts=transform_counts,
        warnings=warnings,
        step_summaries=step_summaries,
    )
