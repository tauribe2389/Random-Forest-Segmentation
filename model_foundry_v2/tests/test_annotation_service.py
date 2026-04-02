import tempfile
import unittest
from pathlib import Path

import numpy as np

from model_foundry_v2.app import create_app

from model_foundry_v2.tests.support import create_test_image, make_test_config


class AnnotationServiceTests(unittest.TestCase):
    def test_save_and_reload_round_trips_without_mask_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(make_test_config(Path(tmpdir)))
            image_path = create_test_image(Path(tmpdir) / "sample.png", size=(7, 4))

            services = app.extensions["v2_services"]
            workspace_service = services["workspace"]
            dataset_service = services["dataset"]
            annotation_service = services["annotation"]

            workspace = workspace_service.create_workspace("demo")
            image = workspace_service.register_image_asset(workspace.id, str(image_path))
            dataset = dataset_service.create_draft_dataset(workspace.id, "draft", [image.id])
            dataset_service.add_class(dataset.id, "foreground", "#ff0000")
            revision = annotation_service.ensure_head_revision(
                draft_dataset_id=dataset.id,
                image_id=image.id,
                author="tester",
                operation_summary="Initialize test head",
            )

            label_map = np.zeros((image.height, image.width), dtype=np.uint16)
            label_map[1:3, 2:5] = 1
            provenance_map = np.zeros_like(label_map, dtype=np.uint8)
            provenance_map[1:3, 2:5] = 1
            protection_map = np.zeros_like(label_map, dtype=np.uint8)
            protection_map[0, 0] = 1

            saved = annotation_service.save_revision(
                revision_id=revision.id,
                label_map=label_map,
                provenance_map=provenance_map,
                protection_map=protection_map,
                author="tester",
                operation_summary="Persist exact arrays",
            )
            reloaded = annotation_service.load_revision(saved.id)

            self.assertTrue(np.array_equal(reloaded.label_map, label_map))
            self.assertTrue(np.array_equal(reloaded.provenance_map, provenance_map))
            self.assertTrue(np.array_equal(reloaded.protection_map, protection_map))

            saved_again = annotation_service.save_revision(
                revision_id=saved.id,
                label_map=label_map,
                provenance_map=provenance_map,
                protection_map=protection_map,
                author="tester",
                operation_summary="Persist identical arrays again",
            )
            self.assertEqual(saved.revision_checksum, saved_again.revision_checksum)

    def test_annotation_api_saves_and_loads_exact_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(make_test_config(Path(tmpdir)))
            image_path = create_test_image(Path(tmpdir) / "api-sample.png", size=(4, 3))

            services = app.extensions["v2_services"]
            workspace_service = services["workspace"]
            dataset_service = services["dataset"]

            workspace = workspace_service.create_workspace("api-demo")
            image = workspace_service.register_image_asset(workspace.id, str(image_path))
            dataset = dataset_service.create_draft_dataset(workspace.id, "draft", [image.id])
            dataset_service.add_class(dataset.id, "target", "#00ff00")
            dataset_service.initialize_annotation_for_image(dataset.id, image.id)

            client = app.test_client()
            payload = {
                "author": "api-user",
                "operation_summary": "API save",
                "label_map": [
                    [0, 1, 1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                ],
                "provenance_map": [
                    [0, 1, 1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                ],
                "protection_map": [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                ],
            }
            save_response = client.post(
                f"/api/datasets/{dataset.id}/images/{image.id}/annotation",
                json=payload,
            )
            self.assertEqual(save_response.status_code, 200)

            load_response = client.get(f"/api/datasets/{dataset.id}/images/{image.id}/annotation")
            self.assertEqual(load_response.status_code, 200)
            response_json = load_response.get_json()
            self.assertEqual(response_json["label_map"], payload["label_map"])
            self.assertEqual(response_json["provenance_map"], payload["provenance_map"])
            self.assertEqual(response_json["protection_map"], payload["protection_map"])

    def test_annotation_detail_page_renders_phase2_editor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(make_test_config(Path(tmpdir)))
            image_path = create_test_image(Path(tmpdir) / "editor-sample.png", size=(4, 4))

            services = app.extensions["v2_services"]
            workspace_service = services["workspace"]
            dataset_service = services["dataset"]

            workspace = workspace_service.create_workspace("editor-demo")
            image = workspace_service.register_image_asset(workspace.id, str(image_path))
            dataset = dataset_service.create_draft_dataset(workspace.id, "draft", [image.id])
            dataset_service.add_class(dataset.id, "region", "#2244ff")
            dataset_service.initialize_annotation_for_image(dataset.id, image.id)

            client = app.test_client()
            response = client.get(f"/datasets/{dataset.id}/images/{image.id}/annotation")
            self.assertEqual(response.status_code, 200)
            body = response.get_data(as_text=True)
            self.assertIn("Phase 2 Minimal Editor", body)
            self.assertIn("overlay-canvas", body)
            self.assertIn("annotation_editor.js", body)
            self.assertIn("/assets/annotation_editor.js", body)

            asset_response = client.get("/assets/annotation_editor.js")
            try:
                self.assertEqual(asset_response.status_code, 200)
                self.assertIn("loadAnnotation", asset_response.get_data(as_text=True))
            finally:
                asset_response.close()
