"""Generate a small synthetic COCO dataset for local testing.

The generated dataset layout:
    <output_dir>/
      images/
        img_001.png
        ...
      annotations.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _flatten_points(points: list[tuple[float, float]]) -> list[float]:
    flat: list[float] = []
    for x, y in points:
        flat.extend([float(x), float(y)])
    return flat


def _circle_polygon(cx: float, cy: float, radius: float, n_points: int = 28) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for idx in range(n_points):
        theta = (2.0 * math.pi * idx) / n_points
        points.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta)))
    return points


def _square_polygon(x0: float, y0: float, size: float) -> list[tuple[float, float]]:
    return [
        (x0, y0),
        (x0 + size, y0),
        (x0 + size, y0 + size),
        (x0, y0 + size),
    ]


def _triangle_polygon(cx: float, cy: float, size: float) -> list[tuple[float, float]]:
    half = size / 2.0
    return [
        (cx, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    ]


def _draw_shape(
    draw: ImageDraw.ImageDraw,
    category_id: int,
    canvas_size: int,
    rng: np.random.Generator,
) -> tuple[list[float], list[float], float]:
    if category_id == 1:
        size = float(rng.integers(canvas_size // 8, canvas_size // 3))
        x0 = float(rng.integers(5, canvas_size - int(size) - 5))
        y0 = float(rng.integers(5, canvas_size - int(size) - 5))
        points = _square_polygon(x0, y0, size)
        draw.polygon(points, fill=(225, 70, 70))
        area = size * size
        bbox = [x0, y0, size, size]
    elif category_id == 2:
        radius = float(rng.integers(canvas_size // 10, canvas_size // 5))
        cx = float(rng.integers(int(radius) + 5, canvas_size - int(radius) - 5))
        cy = float(rng.integers(int(radius) + 5, canvas_size - int(radius) - 5))
        points = _circle_polygon(cx, cy, radius)
        draw.polygon(points, fill=(70, 180, 90))
        area = math.pi * radius * radius
        bbox = [cx - radius, cy - radius, radius * 2.0, radius * 2.0]
    else:
        size = float(rng.integers(canvas_size // 8, canvas_size // 3))
        cx = float(rng.integers(int(size // 2) + 5, canvas_size - int(size // 2) - 5))
        cy = float(rng.integers(int(size // 2) + 5, canvas_size - int(size // 2) - 5))
        points = _triangle_polygon(cx, cy, size)
        draw.polygon(points, fill=(60, 110, 225))
        area = (math.sqrt(3.0) / 4.0) * (size**2)
        bbox = [cx - size / 2.0, cy - size / 2.0, size, size]

    return _flatten_points(points), bbox, float(area)


def generate_demo_dataset(
    output_dir: Path,
    *,
    num_images: int = 8,
    image_size: int = 256,
    seed: int = 42,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    categories = [
        {"id": 1, "name": "red_square"},
        {"id": 2, "name": "green_circle"},
        {"id": 3, "name": "blue_triangle"},
    ]
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    # Force each category to appear early in the dataset.
    forced_categories = [1, 2, 3]

    for image_id in range(1, num_images + 1):
        bg_value = int(rng.integers(210, 245))
        image = Image.new("RGB", (image_size, image_size), (bg_value, bg_value, bg_value))
        draw = ImageDraw.Draw(image)

        shapes_in_image = int(rng.integers(2, 5))
        categories_for_image: list[int] = []
        if forced_categories:
            categories_for_image.append(forced_categories.pop(0))
        while len(categories_for_image) < shapes_in_image:
            categories_for_image.append(int(rng.choice([1, 2, 3])))

        for category_id in categories_for_image:
            segmentation, bbox, area = _draw_shape(draw, category_id, image_size, rng)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [segmentation],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

        file_name = f"img_{image_id:03d}.png"
        image.save(images_dir / file_name)
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": image_size,
                "height": image_size,
            }
        )

    coco_payload = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    with (output_dir / "annotations.json").open("w", encoding="utf-8") as handle:
        json.dump(coco_payload, handle, indent=2)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a synthetic COCO dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo_dataset"),
        help="Directory to write dataset files.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=8,
        help="Number of images to generate.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Width/height of generated images in pixels.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    generate_demo_dataset(
        args.output_dir,
        num_images=args.num_images,
        image_size=args.image_size,
        seed=args.seed,
    )
    print(f"Demo dataset created at: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

