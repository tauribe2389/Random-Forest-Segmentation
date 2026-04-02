import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from model_foundry_v2.app import create_app

from model_foundry_v2.tests.support import create_test_image, make_test_config


class SnapshotServiceTests(unittest.TestCase):
    def test_registered_snapshot_locks_revision_and_preserves_original_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(make_test_config(Path(tmpdir)))
            image_path = create_test_image(Path(tmpdir) / "snapshot-sample.png", size=(5, 5))

            services = app.extensions["v2_services"]
            repositories = app.extensions["v2_repositories"]
            workspace_service = services["workspace"]
            dataset_service = services["dataset"]
            annotation_service = services["annotation"]
            snapshot_service = services["snapshot"]
            snapshot_repository = repositories["snapshot"]

            workspace = workspace_service.create_workspace("snapshot-demo")
            image = workspace_service.register_image_asset(workspace.id, str(image_path))
            dataset = dataset_service.create_draft_dataset(workspace.id, "draft", [image.id])
            dataset_service.add_class(dataset.id, "region", "#ffaa00")
            revision = annotation_service.ensure_head_revision(
                draft_dataset_id=dataset.id,
                image_id=image.id,
                author="tester",
                operation_summary="Init head",
            )

            original_label_map = np.zeros((image.height, image.width), dtype=np.uint16)
            original_label_map[1:4, 1:4] = 1
            empty_aux = np.zeros((image.height, image.width), dtype=np.uint8)
            saved = annotation_service.save_revision(
                revision_id=revision.id,
                label_map=original_label_map,
                provenance_map=empty_aux,
                protection_map=empty_aux,
                author="tester",
                operation_summary="Save original truth",
            )

            snapshot = snapshot_service.register_snapshot(dataset.id, name="baseline")
            snapshot_items = snapshot_repository.list_snapshot_items(snapshot.id)
            self.assertEqual(len(snapshot_items), 1)
            self.assertEqual(snapshot_items[0].annotation_revision_id, saved.id)

            locked_revision = annotation_service.load_revision(saved.id).revision
            self.assertTrue(locked_revision.is_locked)

            forked = annotation_service.fork_head_revision(
                draft_dataset_id=dataset.id,
                image_id=image.id,
                author="tester",
                operation_summary="Fork after snapshot",
            )
            fork_label_map = np.zeros((image.height, image.width), dtype=np.uint16)
            fork_label_map[0:2, 0:2] = 1
            annotation_service.save_revision(
                revision_id=forked.id,
                label_map=fork_label_map,
                provenance_map=empty_aux,
                protection_map=empty_aux,
                author="tester",
                operation_summary="Change head after snapshot",
            )

            original_after_fork = annotation_service.load_revision(saved.id)
            self.assertTrue(np.array_equal(original_after_fork.label_map, original_label_map))

            snapshot_schema = json.loads(snapshot.class_schema_json)
            self.assertEqual(snapshot_schema[0]["class_index"], 0)
            self.assertEqual(snapshot_schema[1]["name"], "region")

    def test_locked_head_returns_conflict_on_annotation_api_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(make_test_config(Path(tmpdir)))
            image_path = create_test_image(Path(tmpdir) / "locked-api-sample.png", size=(3, 3))

            services = app.extensions["v2_services"]
            workspace_service = services["workspace"]
            dataset_service = services["dataset"]
            annotation_service = services["annotation"]
            snapshot_service = services["snapshot"]

            workspace = workspace_service.create_workspace("locked-api-demo")
            image = workspace_service.register_image_asset(workspace.id, str(image_path))
            dataset = dataset_service.create_draft_dataset(workspace.id, "draft", [image.id])
            dataset_service.add_class(dataset.id, "region", "#ffaa00")
            revision = annotation_service.ensure_head_revision(
                draft_dataset_id=dataset.id,
                image_id=image.id,
                author="tester",
                operation_summary="Init head",
            )
            zeros = np.zeros((image.height, image.width), dtype=np.uint8)
            annotation_service.save_revision(
                revision_id=revision.id,
                label_map=np.zeros((image.height, image.width), dtype=np.uint16),
                provenance_map=zeros,
                protection_map=zeros,
                author="tester",
                operation_summary="Baseline",
            )
            snapshot_service.register_snapshot(dataset.id, name="baseline")

            client = app.test_client()
            response = client.post(
                f"/api/datasets/{dataset.id}/images/{image.id}/annotation",
                json={
                    "label_map": [
                        [1, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    "provenance_map": [
                        [1, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    "protection_map": [
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                },
            )
            self.assertEqual(response.status_code, 409)
            self.assertIn("Locked annotation revisions cannot be modified", response.get_json()["error"])

    def test_dataset_page_shows_stateful_head_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(make_test_config(Path(tmpdir)))
            image_path = create_test_image(Path(tmpdir) / "head-actions.png", size=(3, 3))

            services = app.extensions["v2_services"]
            workspace_service = services["workspace"]
            dataset_service = services["dataset"]
            snapshot_service = services["snapshot"]

            workspace = workspace_service.create_workspace("head-actions-demo")
            image = workspace_service.register_image_asset(workspace.id, str(image_path))
            dataset = dataset_service.create_draft_dataset(workspace.id, "draft", [image.id])
            dataset_service.add_class(dataset.id, "region", "#ffaa00")

            client = app.test_client()
            initial_response = client.get(f"/datasets/{dataset.id}")
            initial_body = initial_response.get_data(as_text=True)
            self.assertIn("Init head", initial_body)
            self.assertNotIn("View annotation", initial_body)

            dataset_service.initialize_annotation_for_image(dataset.id, image.id)
            initialized_response = client.get(f"/datasets/{dataset.id}")
            initialized_body = initialized_response.get_data(as_text=True)
            self.assertNotIn("Init head", initialized_body)
            self.assertIn("View annotation", initialized_body)
            self.assertNotIn("Fork head", initialized_body)

            snapshot_service.register_snapshot(dataset.id, name="lock-head")
            locked_response = client.get(f"/datasets/{dataset.id}")
            locked_body = locked_response.get_data(as_text=True)
            self.assertNotIn("Init head", locked_body)
            self.assertIn("View annotation", locked_body)
            self.assertIn("Fork head", locked_body)
