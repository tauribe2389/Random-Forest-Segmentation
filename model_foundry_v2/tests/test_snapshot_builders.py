import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from model_foundry_v2.app import create_app

from model_foundry_v2.tests.support import create_test_image, make_test_config


class SnapshotBuilderTests(unittest.TestCase):
    def _build_snapshot_fixture(self, tmpdir: str):
        app = create_app(make_test_config(Path(tmpdir)))
        services = app.extensions["v2_services"]
        workspace_service = services["workspace"]
        dataset_service = services["dataset"]
        annotation_service = services["annotation"]
        snapshot_service = services["snapshot"]

        image_path = create_test_image(Path(tmpdir) / "builder-sample.png", size=(6, 4))
        workspace = workspace_service.create_workspace("builder-demo")
        image = workspace_service.register_image_asset(workspace.id, str(image_path))
        dataset = dataset_service.create_draft_dataset(workspace.id, "draft", [image.id])
        dataset_service.add_class(dataset.id, "region", "#00aa44")
        revision = annotation_service.ensure_head_revision(
            draft_dataset_id=dataset.id,
            image_id=image.id,
            author="tester",
            operation_summary="Init head",
        )
        label_map = np.zeros((image.height, image.width), dtype=np.uint16)
        label_map[1:3, 2:5] = 1
        provenance = np.zeros_like(label_map, dtype=np.uint8)
        provenance[1:3, 2:5] = 1
        protection = np.zeros_like(label_map, dtype=np.uint8)
        protection[0, 0] = 1
        annotation_service.save_revision(
            revision_id=revision.id,
            label_map=label_map,
            provenance_map=provenance,
            protection_map=protection,
            author="tester",
            operation_summary="Save truth for builders",
        )
        snapshot = snapshot_service.register_snapshot(dataset.id, name="builder-snapshot")
        return app, snapshot, label_map, image

    def test_canonical_snapshot_builder_writes_manifest_of_immutable_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app, snapshot, label_map, image = self._build_snapshot_fixture(tmpdir)
            registry = app.extensions["v2_snapshot_builders"]
            artifact_service = app.extensions["v2_services"]["artifact"]

            result = registry.get("canonical_semantic_dataset").build(snapshot)
            manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))

            self.assertEqual(manifest["builder_id"], "canonical_semantic_dataset")
            self.assertEqual(manifest["snapshot_id"], snapshot.id)
            self.assertEqual(manifest["image_count"], 1)
            self.assertEqual(manifest["items"][0]["image_id"], image.id)
            self.assertTrue(manifest["items"][0]["label_map_path"].endswith("label_map.npy"))

            artifacts = artifact_service.list_artifacts("registered_snapshot", snapshot.id)
            self.assertEqual(len(artifacts), 1)
            self.assertIn("canonical_semantic_dataset", artifacts[0].metadata_json)

    def test_raster_mask_export_builder_writes_png_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app, snapshot, label_map, image = self._build_snapshot_fixture(tmpdir)
            registry = app.extensions["v2_snapshot_builders"]

            result = registry.get("raster_mask_export").build(snapshot)
            manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
            item = manifest["items"][0]
            mask_path = Path(item["mask_path"])
            metadata_path = Path(item["metadata_path"])

            self.assertTrue(mask_path.exists())
            self.assertTrue(metadata_path.exists())

            exported_mask = np.array(Image.open(mask_path), dtype=np.uint16)
            self.assertTrue(np.array_equal(exported_mask, label_map))

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["source_artifact_type"], "canonical_label_map")
            self.assertEqual(metadata["image_id"], image.id)
            self.assertEqual(metadata["class_schema_version"], snapshot.class_schema_version)

    def test_coco_placeholder_builder_marks_coco_as_derived_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app, snapshot, _, _ = self._build_snapshot_fixture(tmpdir)
            registry = app.extensions["v2_snapshot_builders"]

            result = registry.get("coco_export_placeholder").build(snapshot)
            manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))

            self.assertEqual(manifest["status"], "placeholder")
            self.assertEqual(manifest["canonical_dataset_identity"], "registered_snapshot")
            self.assertIn("derived export", manifest["note"])

    def test_snapshot_detail_page_lists_builders_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app, snapshot, _, _ = self._build_snapshot_fixture(tmpdir)
            registry = app.extensions["v2_snapshot_builders"]
            registry.get("canonical_semantic_dataset").build(snapshot)

            client = app.test_client()
            response = client.get(f"/snapshots/{snapshot.id}")
            self.assertEqual(response.status_code, 200)
            body = response.get_data(as_text=True)
            self.assertIn("Registered Snapshot", body)
            self.assertIn("canonical_semantic_dataset", body)
            self.assertIn("raster_mask_export", body)
            self.assertIn("manifest.json", body)
