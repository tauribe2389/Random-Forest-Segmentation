from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_MASK_UTILS = None


def get_mask_utils():
    global _MASK_UTILS
    if _MASK_UTILS is not None:
        return _MASK_UTILS
    try:
        from pycocotools import mask as mask_utils
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing dependency: {exc.name}. Run: pip install -r requirements.txt")
    _MASK_UTILS = mask_utils
    return _MASK_UTILS


@dataclass
class CandidateAnnotation:
    category_id: int
    segmentation: Any
    iscrowd: int
    dataset_index: int
    order_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple COCO annotation datasets into one. "
            "Supports overlap handling for conflicting pixel coverage."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input COCO annotation.json files, in merge priority order.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output merged COCO annotation.json path.",
    )
    parser.add_argument(
        "--overlap-action",
        choices=["trim", "skip", "allow"],
        default="trim",
        help=(
            "How to resolve overlapping pixels in the same image: "
            "trim=new annotation loses overlapping pixels; "
            "skip=new annotation is dropped if any overlap; "
            "allow=keep overlaps."
        ),
    )
    parser.add_argument(
        "--priority",
        choices=["first", "last"],
        default="first",
        help=(
            "Which dataset order wins overlap conflicts. "
            "'first' means earlier --inputs win; 'last' means later --inputs win."
        ),
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=1,
        help="Drop annotations with final mask area below this pixel threshold.",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Optional path to write merge statistics as JSON.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def decode_to_bool_mask(segmentation: Any, height: int, width: int) -> np.ndarray:
    mask_utils = get_mask_utils()

    if segmentation is None:
        return np.zeros((height, width), dtype=bool)

    if isinstance(segmentation, list):
        if not segmentation:
            return np.zeros((height, width), dtype=bool)
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles) if isinstance(rles, list) else rles
    elif isinstance(segmentation, dict):
        counts = segmentation.get("counts")
        if isinstance(counts, list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            rle = segmentation
    else:
        raise ValueError(f"Unsupported segmentation type: {type(segmentation)}")

    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(bool)


def encode_mask(mask_bool: np.ndarray) -> dict[str, Any]:
    mask_utils = get_mask_utils()
    encoded = mask_utils.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return encoded


def build_merged_structures(
    input_paths: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, list[CandidateAnnotation]]]:
    category_name_to_id: dict[str, int] = {}
    merged_categories: list[dict[str, Any]] = []

    image_name_to_record: dict[str, dict[str, Any]] = {}
    merged_images: list[dict[str, Any]] = []

    candidates_by_image_id: dict[int, list[CandidateAnnotation]] = defaultdict(list)

    for dataset_index, input_path in enumerate(input_paths):
        data = load_json(input_path)
        source_categories = data.get("categories", [])
        source_images = data.get("images", [])
        source_annotations = data.get("annotations", [])

        local_category_id_to_merged: dict[int, int] = {}
        for category in source_categories:
            source_category_id = int(category["id"])
            category_name = str(category["name"])
            merged_category_id = category_name_to_id.get(category_name)
            if merged_category_id is None:
                merged_category_id = len(merged_categories) + 1
                category_name_to_id[category_name] = merged_category_id
                merged_categories.append(
                    {
                        "id": merged_category_id,
                        "name": category_name,
                        "supercategory": category.get("supercategory", "object"),
                    }
                )
            local_category_id_to_merged[source_category_id] = merged_category_id

        local_image_id_to_merged: dict[int, int] = {}
        for image in source_images:
            source_image_id = int(image["id"])
            file_name = str(image["file_name"])
            width = int(image["width"])
            height = int(image["height"])

            existing = image_name_to_record.get(file_name)
            if existing is None:
                merged_image_id = len(merged_images) + 1
                record = {
                    "id": merged_image_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                }
                image_name_to_record[file_name] = record
                merged_images.append(record)
            else:
                if int(existing["width"]) != width or int(existing["height"]) != height:
                    raise ValueError(
                        "Image dimension clash for file_name "
                        f"'{file_name}': existing=({existing['width']}x{existing['height']}), "
                        f"new=({width}x{height}) from {input_path}"
                    )
                merged_image_id = int(existing["id"])

            local_image_id_to_merged[source_image_id] = merged_image_id

        for order_index, annotation in enumerate(source_annotations):
            source_image_id = int(annotation["image_id"])
            source_category_id = int(annotation["category_id"])

            merged_image_id = local_image_id_to_merged.get(source_image_id)
            merged_category_id = local_category_id_to_merged.get(source_category_id)
            if merged_image_id is None or merged_category_id is None:
                continue

            candidates_by_image_id[merged_image_id].append(
                CandidateAnnotation(
                    category_id=merged_category_id,
                    segmentation=annotation.get("segmentation"),
                    iscrowd=int(annotation.get("iscrowd", 1)),
                    dataset_index=dataset_index,
                    order_index=order_index,
                )
            )

    return merged_images, merged_categories, candidates_by_image_id


def merge_annotations_for_image(
    image_record: dict[str, Any],
    candidates: list[CandidateAnnotation],
    overlap_action: str,
    priority: str,
    min_area: int,
    stats: dict[str, int],
) -> list[dict[str, Any]]:
    mask_utils = get_mask_utils()

    width = int(image_record["width"])
    height = int(image_record["height"])

    if priority == "last":
        ordered_candidates = sorted(candidates, key=lambda c: (-c.dataset_index, c.order_index))
    else:
        ordered_candidates = sorted(candidates, key=lambda c: (c.dataset_index, c.order_index))

    occupancy = np.zeros((height, width), dtype=bool)
    kept_annotations: list[dict[str, Any]] = []

    for candidate in ordered_candidates:
        try:
            mask_bool = decode_to_bool_mask(candidate.segmentation, height, width)
        except Exception:
            stats["decode_errors"] += 1
            continue

        if not mask_bool.any():
            stats["empty_masks"] += 1
            continue

        overlap_mask = mask_bool & occupancy
        if overlap_mask.any():
            stats["overlap_annotations"] += 1
            stats["overlap_pixels"] += int(overlap_mask.sum())
            if overlap_action == "skip":
                stats["dropped_overlap"] += 1
                continue
            if overlap_action == "trim":
                mask_bool = mask_bool & ~occupancy
                if not mask_bool.any():
                    stats["dropped_overlap"] += 1
                    continue

        area_px = int(mask_bool.sum())
        if area_px < min_area:
            stats["dropped_small"] += 1
            continue

        encoded = encode_mask(mask_bool)
        bbox = mask_utils.toBbox(encoded).tolist()
        area = float(mask_utils.area(encoded))
        if area < float(min_area):
            stats["dropped_small"] += 1
            continue

        occupancy |= mask_bool
        kept_annotations.append(
            {
                "image_id": int(image_record["id"]),
                "category_id": int(candidate.category_id),
                "segmentation": encoded,
                "bbox": [float(v) for v in bbox],
                "area": area,
                "iscrowd": int(candidate.iscrowd),
            }
        )
        stats["kept_annotations"] += 1

    return kept_annotations


def main() -> None:
    args = parse_args()

    input_paths = [Path(p).resolve() for p in args.inputs]
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve() if args.report else None

    for path in input_paths:
        if not path.exists():
            raise SystemExit(f"Input not found: {path}")

    if args.min_area < 1:
        raise SystemExit("--min-area must be >= 1")

    merged_images, merged_categories, candidates_by_image_id = build_merged_structures(input_paths)

    stats: dict[str, int] = defaultdict(int)
    merged_annotations: list[dict[str, Any]] = []

    image_id_to_record = {int(img["id"]): img for img in merged_images}
    for image_id in sorted(candidates_by_image_id.keys()):
        image_record = image_id_to_record[image_id]
        image_candidates = candidates_by_image_id[image_id]
        merged_annotations.extend(
            merge_annotations_for_image(
                image_record=image_record,
                candidates=image_candidates,
                overlap_action=args.overlap_action,
                priority=args.priority,
                min_area=args.min_area,
                stats=stats,
            )
        )

    for ann_id, annotation in enumerate(merged_annotations, start=1):
        annotation["id"] = ann_id

    merged_payload = {
        "info": {
            "description": "Merged COCO dataset",
            "version": "1.0",
            "date_created": datetime.now().isoformat(),
            "merge_inputs": [str(p) for p in input_paths],
            "overlap_action": args.overlap_action,
            "priority": args.priority,
            "min_area": int(args.min_area),
        },
        "licenses": [],
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": merged_categories,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged_payload, f, indent=2)

    summary = {
        "output_path": str(output_path),
        "input_count": len(input_paths),
        "images": len(merged_images),
        "categories": len(merged_categories),
        "annotations": len(merged_annotations),
        "stats": dict(stats),
    }

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print("Merge complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
