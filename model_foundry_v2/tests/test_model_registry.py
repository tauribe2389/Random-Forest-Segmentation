import unittest

from model_foundry_v2.app.domain.entities import RegisteredSnapshot
from model_foundry_v2.app.services.model_registry import build_default_model_registry


class ModelRegistryTests(unittest.TestCase):
    def test_default_registry_exposes_classical_and_keras_families(self) -> None:
        registry = build_default_model_registry()
        self.assertEqual(
            registry.list_family_ids(),
            [
                "classical.semantic_segmentation",
                "keras.unet.semantic_segmentation",
            ],
        )
        classical = registry.get("classical.semantic_segmentation")
        keras = registry.get("keras.unet.semantic_segmentation")
        self.assertIsNotNone(classical)
        self.assertIsNotNone(keras)
        snapshot = RegisteredSnapshot(
            id=1,
            workspace_id=1,
            draft_dataset_id=1,
            name="demo",
            class_schema_json="[]",
            class_schema_version=1,
            image_count=1,
            created_at="2026-03-25T00:00:00+00:00",
        )
        self.assertEqual(classical.dataset_builder_id, "classical.semantic_segmentation.dataset")
        self.assertEqual(keras.dataset_builder_id, "keras.unet.semantic_segmentation.dataset")
        self.assertTrue(classical.validate_snapshot(snapshot, {}).is_valid)
        self.assertTrue(keras.validate_snapshot(snapshot, {}).is_valid)
