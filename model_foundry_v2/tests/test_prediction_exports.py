import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from model_foundry_v2.app import create_app

from model_foundry_v2.tests.support import create_test_image, make_test_config


class PredictionExportTests(unittest.TestCase):
    def _build_prediction_fixture(self, tmpdir: str):
        app = create_app(make_test_config(Path(tmpdir)))
        services = app.extensions["v2_services"]
        workspace_service = services["workspace"]
        dataset_service = services["dataset"]
        annotation_service = services["annotation"]
        snapshot_service = services["snapshot"]
        model_service = services["model"]
        prediction_service = services["prediction"]

        image_path = create_test_image(Path(tmpdir) / "prediction-sample.png", size=(5, 4))
        workspace = workspace_service.create_workspace("prediction-demo")
        image = workspace_service.register_image_asset(workspace.id, str(image_path))
        dataset = dataset_service.create_draft_dataset(workspace.id, "draft", [image.id])
        dataset_service.add_class(dataset.id, "target", "#ff6600")
        revision = annotation_service.ensure_head_revision(
            draft_dataset_id=dataset.id,
            image_id=image.id,
            author="tester",
            operation_summary="Init head",
        )
        label_map = np.zeros((image.height, image.width), dtype=np.uint16)
        label_map[:, 1:4] = 1
        aux = np.zeros((image.height, image.width), dtype=np.uint8)
        annotation_service.save_revision(
            revision_id=revision.id,
            label_map=label_map,
            provenance_map=aux,
            protection_map=aux,
            author="tester",
            operation_summary="Save canonical truth",
        )
        snapshot = snapshot_service.register_snapshot(dataset.id, name="analysis-snapshot")
        model = model_service.create_model_definition(
            workspace_id=workspace.id,
            name="baseline-model",
            family_id="classical.semantic_segmentation",
            config={},
        )
        prediction_run = prediction_service.create_prediction_run(
            snapshot_id=snapshot.id,
            model_definition_id=model.id,
        )
        return app, workspace, snapshot, model, prediction_run, image, label_map

    def test_prediction_run_writes_raw_and_refined_prediction_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app, _, snapshot, model, prediction_run, image, label_map = self._build_prediction_fixture(tmpdir)
            services = app.extensions["v2_services"]
            prediction_service = services["prediction"]
            artifacts = prediction_service.list_prediction_artifacts(prediction_run.id)
            artifact_service = services["artifact"]

            self.assertEqual(prediction_run.status, "completed")
            self.assertTrue(any(artifact.artifact_kind == "prediction_raw" for artifact in artifacts))
            self.assertTrue(any(artifact.artifact_kind == "prediction_refined" for artifact in artifacts))
            self.assertIn('"dataset_builder_id": "classical.semantic_segmentation.dataset"', prediction_run.config_json)

            raw_array = prediction_service.load_prediction_array(prediction_run.id, image.id, "raw")
            refined_array = prediction_service.load_prediction_array(prediction_run.id, image.id, "refined")
            self.assertTrue(np.array_equal(raw_array, label_map))
            self.assertTrue(np.array_equal(refined_array, label_map))
            self.assertEqual(model.family_id, "classical.semantic_segmentation")
            self.assertEqual(snapshot.id, prediction_run.registered_snapshot_id)
            snapshot_artifacts = artifact_service.list_artifacts("registered_snapshot", snapshot.id)
            self.assertTrue(any(artifact.artifact_kind == "dataset_build" for artifact in snapshot_artifacts))

    def test_export_service_writes_class_index_masks_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app, _, snapshot, model, prediction_run, image, label_map = self._build_prediction_fixture(tmpdir)
            export_service = app.extensions["v2_services"]["export"]
            artifact_service = app.extensions["v2_services"]["artifact"]

            result = export_service.export_prediction_run(
                prediction_run_id=prediction_run.id,
                export_config={
                    "source_type": "refined",
                    "threshold_settings": {"mode": "none"},
                    "refinement_settings": {"mode": "baseline-copy"},
                },
            )

            manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["prediction_run_id"], prediction_run.id)
            self.assertEqual(manifest["source_artifact_type"], "refined")
            self.assertEqual(manifest["model_family"], model.family_id)

            item = manifest["items"][0]
            mask = np.array(Image.open(item["mask_path"]), dtype=np.uint16)
            self.assertTrue(np.array_equal(mask, label_map))

            metadata = json.loads(Path(item["metadata_path"]).read_text(encoding="utf-8"))
            self.assertEqual(metadata["image_id"], image.id)
            self.assertEqual(metadata["registered_snapshot_id"], snapshot.id)
            self.assertEqual(metadata["source_artifact_type"], "refined")
            self.assertEqual(metadata["model_family"], model.family_id)

            artifacts = artifact_service.list_artifacts("prediction_run", prediction_run.id)
            self.assertTrue(any(artifact.artifact_kind == "prediction_export" for artifact in artifacts))

    def test_prediction_run_and_snapshot_pages_render_phase4_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app, workspace, snapshot, model, prediction_run, _, _ = self._build_prediction_fixture(tmpdir)
            client = app.test_client()

            workspace_response = client.get("/workspaces/%s" % workspace.id)
            self.assertEqual(workspace_response.status_code, 200)
            workspace_body = workspace_response.get_data(as_text=True)
            self.assertIn("Workspace navigation", workspace_body)
            self.assertIn("Open Models", workspace_body)

            snapshot_response = client.get("/snapshots/%s" % snapshot.id)
            self.assertEqual(snapshot_response.status_code, 200)
            snapshot_body = snapshot_response.get_data(as_text=True)
            self.assertIn("Prediction Runs", snapshot_body)
            self.assertIn("Create prediction run", snapshot_body)
            self.assertIn(model.name, snapshot_body)

            prediction_response = client.get("/prediction-runs/%s" % prediction_run.id)
            self.assertEqual(prediction_response.status_code, 200)
            prediction_body = prediction_response.get_data(as_text=True)
            self.assertIn("Export Prediction Artifacts", prediction_body)
            self.assertIn("prediction_raw", prediction_body)
            self.assertIn("Promote refined prediction", prediction_body)

    def test_export_post_preserves_selected_source_type_in_ui(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app, _, _, _, prediction_run, _, _ = self._build_prediction_fixture(tmpdir)
            client = app.test_client()

            response = client.post(
                f"/prediction-runs/{prediction_run.id}/exports",
                data={"source_type": "raw"},
                follow_redirects=True,
            )
            self.assertEqual(response.status_code, 200)
            body = response.get_data(as_text=True)
            self.assertIn("Exported raw prediction artifacts.", body)
            self.assertIn('<option value="raw" selected>', body)
