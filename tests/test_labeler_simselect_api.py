import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app import create_app
from app.services.labeling.class_schema import build_schema_from_names
from app.services.labeling.slic_cache import load_or_create_slic_cache
from app.services.job_queue import JobQueueManager


class LabelerSimilarityApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="mf_simselect_api_")
        base_dir = Path(self._tmpdir)
        self._env_backup = {
            "SEG_APP_BASE_DIR": os.environ.get("SEG_APP_BASE_DIR"),
            "SEG_APP_INSTANCE_DIR": os.environ.get("SEG_APP_INSTANCE_DIR"),
        }
        os.environ["SEG_APP_BASE_DIR"] = str(base_dir)
        os.environ["SEG_APP_INSTANCE_DIR"] = str(base_dir / "instance")

        self.app = create_app()
        self.app.testing = True
        self.app.extensions["_job_queue_started"] = True
        self.client = self.app.test_client()

        with self.app.app_context():
            storage = self.app.extensions["storage"]
            workspace_root = base_dir / "workspace" / "labeler" / "workspaces" / "api_test"
            images_dir = workspace_root / "raw_images"
            masks_dir = workspace_root / "masks"
            coco_dir = workspace_root / "coco"
            cache_dir = masks_dir / "_cache"
            images_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)
            coco_dir.mkdir(parents=True, exist_ok=True)
            cache_dir.mkdir(parents=True, exist_ok=True)

            self.workspace_id = storage.create_labeler_project(
                name="Simselect API Workspace",
                dataset_id=None,
                project_dir=str(workspace_root),
                images_dir=str(images_dir),
                masks_dir=str(masks_dir),
                coco_dir=str(coco_dir),
                cache_dir=str(cache_dir),
                categories=build_schema_from_names(["class_1"]),
                slic_algorithm="slic",
                slic_preset_name="custom",
                slic_detail_level="medium",
                slic_n_segments=150,
                slic_compactness=10.0,
                slic_sigma=1.0,
                slic_colorspace="lab",
            )
            self.image_name = "sample.png"
            image_path = images_dir / self.image_name
            rgb = np.zeros((16, 16, 3), dtype=np.uint8)
            rgb[:, :8, 0] = 230
            rgb[:, 8:, 1] = 210
            Image.fromarray(rgb, mode="RGB").save(image_path)

            segments, _ = load_or_create_slic_cache(
                image_path,
                cache_dir,
                n_segments=150,
                compactness=10.0,
                sigma=1.0,
                colorspace="lab",
                algorithm="slic",
            )
            unique_ids = sorted(int(value) for value in np.unique(segments).tolist())
            self.expected_ids = unique_ids
            self.seed_id = unique_ids[0]
            self.roi_shape = [16, 16]
            self.roi_runs = [[0, 16 * 16]]

    def tearDown(self) -> None:
        queue = self.app.extensions.get("job_queue")
        if isinstance(queue, JobQueueManager):
            queue.stop(timeout=0.1)
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_simselect_query_returns_expected_ids_for_full_roi(self) -> None:
        response = self.client.post(
            f"/workspace/{self.workspace_id}/images/api/simselect/query",
            json={
                "image_name": self.image_name,
                "seed_superpixel_id": self.seed_id,
                "roi_mask": {
                    "shape": self.roi_shape,
                    "runs": self.roi_runs,
                },
                "color_enabled": True,
                "texture_enabled": False,
                "color_threshold": 200.0,
                "feature_config": {
                    "lbp_points": 8,
                    "lbp_radius": 1,
                    "lbp_method": "uniform",
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIsInstance(payload, dict)
        matched_ids = sorted(int(value) for value in payload.get("matched_superpixel_ids", []))
        self.assertEqual(matched_ids, self.expected_ids)
        self.assertEqual(int(payload.get("matched_count", -1)), len(self.expected_ids))
        self.assertEqual(int(payload.get("candidate_count", -1)), len(self.expected_ids))
        self.assertEqual(int(payload.get("seed_superpixel_id", -1)), self.seed_id)
        self.assertTrue(bool(payload.get("matched_mask_png_base64")))
        self.assertTrue(bool(payload.get("seed_mask_png_base64")))


if __name__ == "__main__":
    unittest.main()
