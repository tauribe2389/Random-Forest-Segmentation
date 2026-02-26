"""Helpers for parsing COCO annotations and rasterizing polygons."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from skimage.draw import polygon as sk_polygon


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum for a file."""
    sha = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_coco(coco_json_path: Path) -> dict[str, Any]:
    """Read and parse a COCO annotations JSON file."""
    with coco_json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "images" not in data or "annotations" not in data or "categories" not in data:
        raise ValueError("COCO JSON must include images, annotations, and categories.")
    return data


def parse_categories(coco: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract ordered categories from COCO payload."""
    categories = coco.get("categories", [])
    if not isinstance(categories, list) or not categories:
        raise ValueError("COCO JSON has no categories.")

    parsed: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for category in categories:
        category_id = int(category["id"])
        if category_id in seen_ids:
            raise ValueError(f"Duplicate category id in COCO categories: {category_id}")
        seen_ids.add(category_id)
        parsed.append(
            {
                "id": category_id,
                "name": str(category.get("name", f"category_{category_id}")),
                "supercategory": category.get("supercategory"),
            }
        )
    return parsed


def category_id_to_class_index(
    categories: list[dict[str, Any]],
) -> dict[int, int]:
    """Map COCO category IDs to contiguous class indices starting from 1."""
    mapping: dict[int, int] = {}
    for offset, category in enumerate(categories, start=1):
        mapping[int(category["id"])] = offset
    return mapping


def annotations_by_image(coco: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """Group annotations by image_id."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco.get("annotations", []):
        image_id = int(annotation["image_id"])
        grouped.setdefault(image_id, []).append(annotation)
    return grouped


def resolve_image_path(dataset_path: Path, image_root: str, file_name: str) -> Path:
    """Resolve an image path from COCO file_name and dataset settings."""
    file_path = Path(file_name)
    if file_path.is_absolute():
        return file_path

    root_candidate = (dataset_path / image_root / file_path).resolve()
    if root_candidate.exists():
        return root_candidate

    fallback = (dataset_path / file_path).resolve()
    return fallback


def _segmentation_to_polygons(segmentation: Any) -> list[list[float]]:
    if not isinstance(segmentation, list):
        return []
    if segmentation and isinstance(segmentation[0], (int, float)):
        return [segmentation]
    polygons: list[list[float]] = []
    for candidate in segmentation:
        if isinstance(candidate, list):
            polygons.append(candidate)
    return polygons


def _polygon_area(flat_points: list[float]) -> float:
    coords = np.asarray(flat_points, dtype=np.float64).reshape(-1, 2)
    if coords.shape[0] < 3:
        return 0.0
    x = coords[:, 0]
    y = coords[:, 1]
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5


def build_mask(
    height: int,
    width: int,
    annotations: list[dict[str, Any]],
    category_to_class: dict[int, int],
    overlap_policy: str = "higher_area_wins",
) -> tuple[np.ndarray, list[str], int]:
    """Rasterize annotation polygons into a class-index mask.

    Returns:
        mask: (H, W) uint16 array where 0 is background.
        warnings: warning text generated during parsing.
        skipped_rle: number of RLE annotations skipped.
    """
    mask = np.zeros((height, width), dtype=np.uint16)
    warnings: list[str] = []
    skipped_rle = 0

    valid_entries: list[tuple[int, float, dict[str, Any], list[list[float]]]] = []
    for order_idx, annotation in enumerate(annotations):
        category_id = int(annotation.get("category_id", -1))
        if category_id not in category_to_class:
            continue

        segmentation = annotation.get("segmentation")
        if isinstance(segmentation, dict):
            skipped_rle += 1
            warnings.append(
                f"Skipped RLE annotation id={annotation.get('id')} (polygon-only MVP)."
            )
            continue

        polygons = _segmentation_to_polygons(segmentation)
        if not polygons:
            continue

        area = float(annotation.get("area", 0.0))
        if area <= 0:
            area = sum(_polygon_area(poly) for poly in polygons)

        valid_entries.append((order_idx, area, annotation, polygons))

    if overlap_policy == "higher_area_wins":
        valid_entries.sort(key=lambda row: (row[1], row[0]))
    elif overlap_policy == "last_annotation_wins":
        valid_entries.sort(key=lambda row: row[0])
    else:
        raise ValueError(f"Unsupported overlap policy: {overlap_policy}")

    for _, _, annotation, polygons in valid_entries:
        class_index = category_to_class[int(annotation["category_id"])]
        for poly_points in polygons:
            try:
                coords = np.asarray(poly_points, dtype=np.float64).reshape(-1, 2)
            except ValueError:
                warnings.append(
                    f"Invalid polygon for annotation id={annotation.get('id')}; skipped."
                )
                continue
            if coords.shape[0] < 3:
                continue

            rr, cc = sk_polygon(coords[:, 1], coords[:, 0], shape=(height, width))
            mask[rr, cc] = class_index

    return mask, warnings, skipped_rle

