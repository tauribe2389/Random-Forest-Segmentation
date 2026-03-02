import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.services.labeling.class_schema import build_schema_from_names, update_schema_with_names
from app.services.storage import Storage


def _write_binary_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((4, 4), dtype=np.uint8)
    arr[1:3, 1:3] = 255
    Image.fromarray(arr, mode="L").save(path)


class ClassSchemaTests(unittest.TestCase):
    def test_rename_keeps_existing_id_and_is_not_destructive(self) -> None:
        existing = build_schema_from_names(["cat", "dog"])
        next_schema, diff = update_schema_with_names(existing, ["cat", "hound"])

        self.assertFalse(bool(diff["destructive"]))
        self.assertEqual(diff["removed_ids"], [])
        self.assertEqual([row["id"] for row in next_schema["classes"]], [1, 2])
        self.assertEqual([row["name"] for row in next_schema["classes"]], ["cat", "hound"])

    def test_reorder_keeps_ids_and_is_not_destructive(self) -> None:
        existing = build_schema_from_names(["cat", "dog", "bird"])
        next_schema, diff = update_schema_with_names(existing, ["bird", "cat", "dog"])

        self.assertFalse(bool(diff["destructive"]))
        self.assertTrue(bool(diff["reordered"]))
        self.assertEqual([row["id"] for row in next_schema["classes"]], [3, 1, 2])

    def test_remove_is_destructive(self) -> None:
        existing = build_schema_from_names(["cat", "dog", "bird"])
        next_schema, diff = update_schema_with_names(existing, ["cat", "bird"])

        self.assertTrue(bool(diff["destructive"]))
        self.assertEqual(diff["removed_ids"], [2])
        self.assertEqual([row["id"] for row in next_schema["classes"]], [1, 3])


class StorageMigrationTests(unittest.TestCase):
    def test_migration_normalizes_schema_and_renames_legacy_masks(self) -> None:
        tmpdir = tempfile.mkdtemp(prefix="mf_schema_migration_")
        try:
            root = Path(tmpdir)
            db_path = root / "test.sqlite3"
            project_root = root / "project"
            images_dir = project_root / "images"
            masks_dir = project_root / "masks"
            coco_dir = project_root / "coco"
            cache_dir = masks_dir / "_cache"
            for path in (images_dir, masks_dir, coco_dir, cache_dir):
                path.mkdir(parents=True, exist_ok=True)

            _write_binary_mask(masks_dir / "img1__cat.png")
            _write_binary_mask(masks_dir / "img1__dog.png")
            _write_binary_mask(masks_dir / "img2__cat.png")
            _write_binary_mask(masks_dir / "img2__cid_1.png")

            storage = Storage(db_path=db_path)
            storage.init_db()
            project_id = storage.create_labeler_project(
                name="test_project",
                dataset_id=None,
                project_dir=str(project_root),
                images_dir=str(images_dir),
                masks_dir=str(masks_dir),
                coco_dir=str(coco_dir),
                cache_dir=str(cache_dir),
                categories=["cat", "dog"],
            )

            with storage._connect() as conn:  # noqa: SLF001 - test-only DB setup for legacy simulation
                conn.execute(
                    "UPDATE labeler_projects SET categories_json = ? WHERE id = ?",
                    (json.dumps(["cat", "dog"]), project_id),
                )
                conn.commit()

            stats = storage.migrate_labeler_class_schema_and_masks(fallback_names=["cat", "dog"])
            self.assertGreaterEqual(int(stats["projects_scanned"]), 1)
            self.assertGreaterEqual(int(stats["masks_renamed"]), 2)

            project = storage.get_labeler_project(project_id)
            self.assertIsInstance(project, dict)
            schema = project["categories_json"]
            self.assertIsInstance(schema, dict)
            self.assertEqual([row["id"] for row in schema["classes"]], [1, 2])
            self.assertEqual([row["name"] for row in schema["classes"]], ["cat", "dog"])
            self.assertGreaterEqual(int(schema["next_class_id"]), 3)

            self.assertTrue((masks_dir / "img1__cid_1.png").exists())
            self.assertTrue((masks_dir / "img1__cid_2.png").exists())
            self.assertFalse((masks_dir / "img1__cat.png").exists())
            self.assertFalse((masks_dir / "img1__dog.png").exists())
            self.assertTrue((masks_dir / "img2__cid_1.png").exists())
            self.assertFalse((masks_dir / "img2__cat.png").exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
