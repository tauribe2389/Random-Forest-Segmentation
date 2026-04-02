"""Model-family registry and adapter contracts for Model Foundry V2."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..domain.entities import ModelDefinition, RegisteredSnapshot
from .dataset_builders import DatasetBuildResult, DatasetBuilderRegistry
from .keras_unet_runtime import predict_unet_from_manifest, train_unet_from_manifest


@dataclass(frozen=True)
class SnapshotValidationResult:
    is_valid: bool
    errors: tuple[str, ...]


@dataclass(frozen=True)
class PredictionBatchItem:
    image_id: int
    annotation_revision_id: int
    raw_prediction: np.ndarray
    refined_prediction: np.ndarray
    raw_metadata: dict
    refined_metadata: dict


@dataclass(frozen=True)
class PredictionBatch:
    family_id: str
    execution_mode: str
    dataset_builder_id: str
    items: tuple[PredictionBatchItem, ...]
    metadata: dict


class ModelFamilyAdapter(ABC):
    """Base contract for model families."""

    family_id: str
    display_name: str
    task_type: str
    dataset_builder_id: str

    @abstractmethod
    def validate_snapshot(
        self,
        snapshot: RegisteredSnapshot,
        model_config: dict,
    ) -> SnapshotValidationResult:
        raise NotImplementedError

    def build_training_artifacts(
        self,
        snapshot: RegisteredSnapshot,
        dataset_builder_registry: DatasetBuilderRegistry,
        model_config: dict,
    ) -> DatasetBuildResult:
        builder = dataset_builder_registry.get(self.dataset_builder_id)
        if builder is None:
            raise ValueError(
                "Dataset builder '%s' is not registered for family '%s'."
                % (self.dataset_builder_id, self.family_id)
            )
        return builder.build(snapshot=snapshot, config=model_config)

    @abstractmethod
    def train(self, training_manifest: dict, model_config: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        model_definition: ModelDefinition,
        snapshot: RegisteredSnapshot,
        dataset_build: DatasetBuildResult,
        model_config: dict,
    ) -> PredictionBatch:
        raise NotImplementedError


class ClassicalSemanticSegmentationAdapter(ModelFamilyAdapter):
    family_id = "classical.semantic_segmentation"
    display_name = "Classical Semantic Segmentation"
    task_type = "semantic_segmentation"
    dataset_builder_id = "classical.semantic_segmentation.dataset"

    def validate_snapshot(
        self,
        snapshot: RegisteredSnapshot,
        model_config: dict,
    ) -> SnapshotValidationResult:
        if snapshot.image_count <= 0:
            return SnapshotValidationResult(
                is_valid=False,
                errors=("Registered snapshot must contain at least one image.",),
            )
        return SnapshotValidationResult(is_valid=True, errors=())

    def train(self, training_manifest: dict, model_config: dict) -> dict:
        return {
            "family_id": self.family_id,
            "status": "planned",
            "builder_id": self.dataset_builder_id,
            "training_manifest": training_manifest,
        }

    def predict(
        self,
        model_definition: ModelDefinition,
        snapshot: RegisteredSnapshot,
        dataset_build: DatasetBuildResult,
        model_config: dict,
    ) -> PredictionBatch:
        items = []
        for entry in dataset_build.manifest["items"]:
            label_map = np.load(entry["label_map_path"], allow_pickle=False)
            raw_prediction = np.asarray(label_map, dtype=np.uint16)
            refined_prediction = np.asarray(label_map, dtype=np.uint16)
            items.append(
                PredictionBatchItem(
                    image_id=int(entry["image_id"]),
                    annotation_revision_id=int(entry["annotation_revision_id"]),
                    raw_prediction=raw_prediction,
                    refined_prediction=refined_prediction,
                    raw_metadata={
                        "builder_id": dataset_build.builder_id,
                        "source_artifact_type": "canonical_label_map",
                        "mode": "baseline-copy",
                    },
                    refined_metadata={
                        "builder_id": dataset_build.builder_id,
                        "source_artifact_type": "canonical_label_map",
                        "mode": "baseline-copy",
                    },
                )
            )
        return PredictionBatch(
            family_id=self.family_id,
            execution_mode="baseline-copy",
            dataset_builder_id=dataset_build.builder_id,
            items=tuple(items),
            metadata={
                "model_definition_id": model_definition.id,
                "registered_snapshot_id": snapshot.id,
            },
        )


class KerasUnetSemanticSegmentationAdapter(ModelFamilyAdapter):
    family_id = "keras.unet.semantic_segmentation"
    display_name = "Keras U-Net Semantic Segmentation"
    task_type = "semantic_segmentation"
    dataset_builder_id = "keras.unet.semantic_segmentation.dataset"

    def validate_snapshot(
        self,
        snapshot: RegisteredSnapshot,
        model_config: dict,
    ) -> SnapshotValidationResult:
        if snapshot.image_count <= 0:
            return SnapshotValidationResult(
                is_valid=False,
                errors=("Registered snapshot must contain at least one image.",),
            )
        return SnapshotValidationResult(is_valid=True, errors=())

    def train(self, training_manifest: dict, model_config: dict) -> dict:
        output_dir = model_config.get("output_dir")
        if not output_dir:
            raise ValueError("Keras U-Net training requires an output_dir in model_config.")
        return train_unet_from_manifest(
            training_manifest=training_manifest,
            output_dir=Path(output_dir),
            model_config=model_config,
        )

    def predict(
        self,
        model_definition: ModelDefinition,
        snapshot: RegisteredSnapshot,
        dataset_build: DatasetBuildResult,
        model_config: dict,
    ) -> PredictionBatch:
        artifact_manifest_path = model_config.get("trained_artifact_manifest_path")
        if not artifact_manifest_path:
            raise RuntimeError(
                "Keras U-Net prediction requires a trained artifact manifest. Train the model first."
            )
        runtime_predictions = predict_unet_from_manifest(
            prediction_manifest=dataset_build.manifest,
            artifact_manifest_path=Path(artifact_manifest_path),
        )
        items = []
        for item in runtime_predictions:
            items.append(
                PredictionBatchItem(
                    image_id=int(item["image_id"]),
                    annotation_revision_id=int(item["annotation_revision_id"]),
                    raw_prediction=np.asarray(item["raw_prediction"], dtype=np.uint16),
                    refined_prediction=np.asarray(item["refined_prediction"], dtype=np.uint16),
                    raw_metadata={
                        "builder_id": dataset_build.builder_id,
                        "source_artifact_type": "keras_unet_raw_argmax",
                        "mode": "keras-unet",
                    },
                    refined_metadata={
                        "builder_id": dataset_build.builder_id,
                        "source_artifact_type": "keras_unet_refined_argmax",
                        "mode": "keras-unet",
                    },
                )
            )
        return PredictionBatch(
            family_id=self.family_id,
            execution_mode="keras-unet",
            dataset_builder_id=dataset_build.builder_id,
            items=tuple(items),
            metadata={
                "model_definition_id": model_definition.id,
                "registered_snapshot_id": snapshot.id,
                "runtime": "tensorflow.keras",
            },
        )


class ModelRegistry:
    """Registry for model-family adapters."""

    def __init__(self, adapters: list[ModelFamilyAdapter]) -> None:
        self._adapters = {adapter.family_id: adapter for adapter in adapters}

    def get(self, family_id: str) -> ModelFamilyAdapter | None:
        return self._adapters.get(family_id)

    def list_family_ids(self) -> list[str]:
        return sorted(self._adapters.keys())

    def list_adapters(self) -> list[ModelFamilyAdapter]:
        return [self._adapters[key] for key in sorted(self._adapters.keys())]


def build_default_model_registry() -> ModelRegistry:
    return ModelRegistry(
        adapters=[
            ClassicalSemanticSegmentationAdapter(),
            KerasUnetSemanticSegmentationAdapter(),
        ]
    )
