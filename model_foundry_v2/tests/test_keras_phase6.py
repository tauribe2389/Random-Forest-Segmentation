import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from model_foundry_v2.app import create_app

from model_foundry_v2.tests.support import create_test_image, make_test_config


class KerasPhase6Tests(unittest.TestCase):
    def test_keras_training_prediction_export_and_promotion_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(make_test_config(Path(tmpdir)))
            services = app.extensions["v2_services"]
            workspace_service = services["workspace"]
            dataset_service = services["dataset"]
            annotation_service = services["annotation"]
            snapshot_service = services["snapshot"]
            model_service = services["model"]
            training_service = services["training"]
            prediction_service = services["prediction"]
            export_service = services["export"]
            promotion_service = services["promotion"]

            image_path = create_test_image(Path(tmpdir) / "keras-phase6.png", size=(32, 32))
            workspace = workspace_service.create_workspace("keras-phase6")
            image = workspace_service.register_image_asset(workspace.id, str(image_path))
            dataset = dataset_service.create_draft_dataset(workspace.id, "draft", [image.id])
            dataset_service.add_class(dataset.id, "target", "#00ffaa")
            revision = annotation_service.ensure_head_revision(
                draft_dataset_id=dataset.id,
                image_id=image.id,
                author="tester",
                operation_summary="Init head",
            )

            label_map = np.zeros((image.height, image.width), dtype=np.uint16)
            label_map[8:24, 10:22] = 1
            provenance = (label_map > 0).astype(np.uint8)
            protection = np.zeros_like(provenance)
            annotation_service.save_revision(
                revision_id=revision.id,
                label_map=label_map,
                provenance_map=provenance,
                protection_map=protection,
                author="tester",
                operation_summary="Seed canonical truth",
            )

            snapshot = snapshot_service.register_snapshot(dataset.id, name="keras-snapshot")
            model = model_service.create_model_definition(
                workspace_id=workspace.id,
                name="keras-model",
                family_id="keras.unet.semantic_segmentation",
                config={
                    "epochs": 1,
                    "batch_size": 1,
                    "target_height": 32,
                    "target_width": 32,
                    "base_filters": 4,
                },
            )

            training_run = training_service.start_training_run(
                snapshot_id=snapshot.id,
                model_definition_id=model.id,
            )
            self.assertEqual(training_run.status, "completed")

            trained_model = model_service.get_model_definition(model.id)
            trained_config = json.loads(trained_model.config_json)
            self.assertEqual(trained_config["trained_family_id"], "keras.unet.semantic_segmentation")
            self.assertTrue(Path(trained_config["trained_artifact_manifest_path"]).exists())

            prediction_run = prediction_service.create_prediction_run(
                snapshot_id=snapshot.id,
                model_definition_id=model.id,
            )
            self.assertEqual(prediction_run.status, "completed")

            refined_prediction = prediction_service.load_prediction_array(
                prediction_run_id=prediction_run.id,
                image_id=image.id,
                source_type="refined",
            )
            self.assertEqual(refined_prediction.shape, label_map.shape)
            self.assertEqual(refined_prediction.dtype, np.uint16)

            export_result = export_service.export_prediction_run(
                prediction_run_id=prediction_run.id,
                export_config={"source_type": "refined"},
            )
            self.assertTrue(Path(export_result["manifest_path"]).exists())

            promoted_dataset = promotion_service.promote_prediction_run(
                prediction_run_id=prediction_run.id,
                source_type="refined",
            )
            self.assertNotEqual(promoted_dataset.id, dataset.id)
            self.assertIn("refined", promoted_dataset.name)
            promoted_head = annotation_service.load_head_revision(promoted_dataset.id, image.id)
            self.assertIsNotNone(promoted_head)
            self.assertEqual(promoted_head.label_map.shape, label_map.shape)
            self.assertTrue(np.array_equal(promoted_head.label_map, refined_prediction))

            promoted_dataset_2 = promotion_service.promote_prediction_run(
                prediction_run_id=prediction_run.id,
                source_type="refined",
            )
            self.assertNotEqual(promoted_dataset_2.name, promoted_dataset.name)
