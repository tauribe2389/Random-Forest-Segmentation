from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .class_schema import class_entries
from .image_io import list_images, load_binary_mask, mask_filename, mask_filename_for_class_id


def export_coco_annotations(
    images_dir: Path,
    masks_dir: Path,
    coco_dir: Path,
    categories: Any,
    min_area: int = 50,
) -> tuple[Path, int, int]:
    from pycocotools import mask as mask_utils

    coco_dir.mkdir(parents=True, exist_ok=True)
    image_names = list_images(images_dir)

    images_payload = []
    annotations_payload = []
    class_rows = class_entries(categories)
    categories_payload = [
        {"id": int(class_row["id"]), "name": str(class_row["name"]), "supercategory": "object"}
        for class_row in class_rows
    ]

    annotation_id = 1
    for image_id, image_name in enumerate(image_names, start=1):
        image_path = images_dir / image_name
        with Image.open(image_path) as img:
            width, height = img.size

        images_payload.append(
            {
                "id": image_id,
                "file_name": image_name,
                "width": width,
                "height": height,
            }
        )

        stem = Path(image_name).stem
        for class_row in class_rows:
            category_id = int(class_row["id"])
            class_name = str(class_row["name"])
            mask_path = masks_dir / mask_filename_for_class_id(stem, category_id)
            if not mask_path.exists() and class_name:
                mask_path = masks_dir / mask_filename(stem, class_name)
            if not mask_path.exists() or not mask_path.is_file():
                continue

            try:
                mask_bool = load_binary_mask(mask_path, expected_shape=(height, width))
            except ValueError:
                continue

            if int(mask_bool.sum()) < min_area:
                continue

            encoded = mask_utils.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
            encoded["counts"] = encoded["counts"].decode("utf-8")
            area = float(mask_utils.area(encoded))
            if area < float(min_area):
                continue
            bbox = mask_utils.toBbox(encoded).tolist()

            annotations_payload.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": encoded,
                    "area": area,
                    "bbox": [float(v) for v in bbox],
                    "iscrowd": 1,
                }
            )
            annotation_id += 1

    payload = {
        "info": {
            "description": "Superpixel Labeler Export",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": images_payload,
        "annotations": annotations_payload,
        "categories": categories_payload,
    }

    output_path = coco_dir / "annotations.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return output_path, len(images_payload), len(annotations_payload)
