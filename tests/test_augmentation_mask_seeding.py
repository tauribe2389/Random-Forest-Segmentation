from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.routes import _seed_masks_from_coco_annotations
from app.services.labeling.image_io import (
    frozen_mask_filename_for_class_id,
    load_binary_mask,
    mask_filename_for_class_id,
)


class AugmentationMaskSeedingTests(unittest.TestCase):
    @staticmethod
    def _sample_coco_payload() -> dict:
        return {
            "images": [
                {
                    "id": 1,
                    "file_name": "sample.png",
                    "height": 8,
                    "width": 8,
                }
            ],
            "annotations": [
                {
                    "id": 101,
                    "image_id": 1,
                    "category_id": 7,
                    "segmentation": [[1, 1, 6, 1, 6, 6, 1, 6]],
                    "area": 25,
                }
            ],
            "categories": [
                {"id": 7, "name": "target"},
            ],
        }

    @staticmethod
    def _parsed_categories() -> list[dict]:
        return [{"id": 7, "name": "target"}]

    def _make_image(self, images_dir: Path) -> None:
        images_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(0, 0, 0)).save(images_dir / "sample.png")

    def test_seed_masks_without_frozen_outputs_only_base_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            masks_dir = root / "masks"
            masks_dir.mkdir(parents=True, exist_ok=True)
            self._make_image(images_dir)

            seeded_masks, seeded_frozen, warnings = _seed_masks_from_coco_annotations(
                coco_payload=self._sample_coco_payload(),
                images_dir=images_dir,
                masks_dir=masks_dir,
                parsed_categories=self._parsed_categories(),
                seed_frozen_masks=False,
            )

            base_path = masks_dir / mask_filename_for_class_id("sample", 1)
            frozen_path = masks_dir / frozen_mask_filename_for_class_id("sample", 1)
            self.assertEqual(seeded_masks, 1)
            self.assertEqual(seeded_frozen, 0)
            self.assertEqual(warnings, [])
            self.assertTrue(base_path.exists())
            self.assertFalse(frozen_path.exists())

    def test_seed_masks_with_frozen_outputs_both_base_and_frozen_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            masks_dir = root / "masks"
            masks_dir.mkdir(parents=True, exist_ok=True)
            self._make_image(images_dir)

            seeded_masks, seeded_frozen, warnings = _seed_masks_from_coco_annotations(
                coco_payload=self._sample_coco_payload(),
                images_dir=images_dir,
                masks_dir=masks_dir,
                parsed_categories=self._parsed_categories(),
                seed_frozen_masks=True,
            )

            base_path = masks_dir / mask_filename_for_class_id("sample", 1)
            frozen_path = masks_dir / frozen_mask_filename_for_class_id("sample", 1)
            self.assertEqual(seeded_masks, 1)
            self.assertEqual(seeded_frozen, 1)
            self.assertEqual(warnings, [])
            self.assertTrue(base_path.exists())
            self.assertTrue(frozen_path.exists())

            base_mask = load_binary_mask(base_path, expected_shape=(8, 8))
            frozen_mask = load_binary_mask(frozen_path, expected_shape=(8, 8))
            self.assertTrue(bool(base_mask.any()))
            self.assertTrue(bool(frozen_mask.any()))
            self.assertTrue((base_mask == frozen_mask).all())


if __name__ == "__main__":
    unittest.main()
