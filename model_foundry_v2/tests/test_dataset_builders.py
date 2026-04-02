import tempfile
import unittest
from pathlib import Path

from model_foundry_v2.app import create_app

from model_foundry_v2.tests.support import create_test_image, make_test_config


class DatasetBuilderTests(unittest.TestCase):
    def _build_snapshot_fixture(self, tmpdir: str):
        app = create_app(make_test_config(Path(tmpdir)))
        services = app.extensions["v2_services"]
        workspace_service = services["workspace"]
        dataset_service = services["dataset"]
        annotation_service = services["annotation"]
        snapshot_service = services["snapshot"]

        image_path = create_test_image(Path(tmpdir) / "dataset-builder.png", size=(4, 4))
        workspace = workspace_service.create_workspace("builder-demo")
        image = workspace_service.register_image_asset(workspace.id, str(image_path))
        dataset = dataset_service.create_draft_dataset(workspace.id, "draft", [image.id])
        dataset_service.add_class(dataset.id, "target", "#3366ff")
        revision = annotation_service.ensure_head_revision(
            draft_dataset_id=dataset.id,
            image_id=image.id,
            author="tester",
            operation_summary="Init head",
        )
        annotation_service.save_revision(
            revision_id=revision.id,
            label_map=[[0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            provenance_map=[[0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            protection_map=[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            author="tester",
            operation_summary="Seed canonical truth",
        )
        snapshot = snapshot_service.register_snapshot(dataset.id, name="phase5-snapshot")
        return app, snapshot

    def test_dataset_builder_registry_builds_classical_and_keras_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app, snapshot = self._build_snapshot_fixture(tmpdir)
            registry = app.extensions["v2_dataset_builders"]
            artifact_service = app.extensions["v2_services"]["artifact"]

            classical_result = registry.get("classical.semantic_segmentation.dataset").build(snapshot)
            keras_result = registry.get("keras.unet.semantic_segmentation.dataset").build(snapshot)

            self.assertEqual(classical_result.manifest["model_family"], "classical.semantic_segmentation")
            self.assertEqual(keras_result.manifest["model_family"], "keras.unet.semantic_segmentation")
            self.assertEqual(keras_result.manifest["input_contract"], "image-mask-pairs")

            artifacts = artifact_service.list_artifacts("registered_snapshot", snapshot.id)
            self.assertTrue(any(artifact.artifact_kind == "dataset_build" for artifact in artifacts))
