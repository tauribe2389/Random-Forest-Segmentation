"""Dataset builders for model-family training and prediction contracts."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain.entities import ArtifactRecord, RegisteredSnapshot
from ..domain.enums import ArtifactKind, ArtifactOwnerType
from ..repositories.snapshot_repository import SnapshotRepository
from ..repositories.workspace_repository import WorkspaceRepository
from .annotation_service import AnnotationService
from .artifact_service import ArtifactService


@dataclass(frozen=True)
class DatasetBuildResult:
    builder_id: str
    snapshot_id: int
    manifest_path: str
    artifact_record: ArtifactRecord
    manifest: dict[str, Any]


class DatasetBuilder(ABC):
    """Base contract for snapshot-derived training and prediction datasets."""

    builder_id: str
    display_name: str
    description: str

    @abstractmethod
    def build(self, snapshot: RegisteredSnapshot, config: dict | None = None) -> DatasetBuildResult:
        raise NotImplementedError


class DatasetBuilderRegistry:
    """Registry and dispatcher for dataset builders."""

    def __init__(self, builders: list[DatasetBuilder]) -> None:
        self._builders = {builder.builder_id: builder for builder in builders}

    def get(self, builder_id: str) -> DatasetBuilder | None:
        return self._builders.get(builder_id)

    def list_builders(self) -> list[DatasetBuilder]:
        return [self._builders[key] for key in sorted(self._builders.keys())]


class _BaseDatasetBuilder(DatasetBuilder):
    def __init__(
        self,
        snapshot_repository: SnapshotRepository,
        workspace_repository: WorkspaceRepository,
        annotation_service: AnnotationService,
        artifact_service: ArtifactService,
        artifacts_root: Path,
    ) -> None:
        self.snapshot_repository = snapshot_repository
        self.workspace_repository = workspace_repository
        self.annotation_service = annotation_service
        self.artifact_service = artifact_service
        self.artifacts_root = artifacts_root

    def _builder_dir(self, snapshot_id: int, builder_folder: str) -> Path:
        path = self.artifacts_root / "snapshots" / str(snapshot_id) / "dataset_builds" / builder_folder
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> str:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _collect_base_items(self, snapshot: RegisteredSnapshot) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for item in self.snapshot_repository.list_snapshot_items(snapshot.id):
            image = self.workspace_repository.get_image_asset(item.image_id)
            revision_data = self.annotation_service.load_revision(item.annotation_revision_id)
            items.append(
                {
                    "image_id": image.id,
                    "source_path": image.source_path,
                    "width": image.width,
                    "height": image.height,
                    "annotation_revision_id": revision_data.revision.id,
                    "revision_checksum": revision_data.revision.revision_checksum,
                    "label_map_path": revision_data.revision.label_map_path,
                    "provenance_map_path": revision_data.revision.provenance_map_path,
                    "protection_map_path": revision_data.revision.protection_map_path,
                }
            )
        return items

    def _record_dataset_build_artifact(
        self,
        snapshot_id: int,
        manifest_path: Path,
        manifest_checksum: str,
        metadata: dict[str, Any],
    ) -> ArtifactRecord:
        return self.artifact_service.record_artifact(
            owner_type=ArtifactOwnerType.REGISTERED_SNAPSHOT.value,
            owner_id=snapshot_id,
            artifact_kind=ArtifactKind.DATASET_BUILD,
            artifact_path=manifest_path,
            checksum_sha256=manifest_checksum,
            metadata=metadata,
        )


class ClassicalSegmentationDatasetBuilder(_BaseDatasetBuilder):
    builder_id = "classical.semantic_segmentation.dataset"
    display_name = "Classical Segmentation Dataset Builder"
    description = "Builds a manifest for classical semantic segmentation workflows."

    def build(self, snapshot: RegisteredSnapshot, config: dict | None = None) -> DatasetBuildResult:
        output_dir = self._builder_dir(snapshot.id, "classical_semantic_segmentation")
        items = self._collect_base_items(snapshot)
        manifest = {
            "builder_id": self.builder_id,
            "snapshot_id": snapshot.id,
            "snapshot_name": snapshot.name,
            "task_type": "semantic_segmentation",
            "model_family": "classical.semantic_segmentation",
            "feature_mode": (config or {}).get("feature_mode", "baseline-copy"),
            "class_schema_version": snapshot.class_schema_version,
            "class_schema": json.loads(snapshot.class_schema_json),
            "items": items,
        }
        manifest_path = output_dir / "manifest.json"
        manifest_checksum = self._write_json(manifest_path, manifest)
        artifact = self._record_dataset_build_artifact(
            snapshot_id=snapshot.id,
            manifest_path=manifest_path,
            manifest_checksum=manifest_checksum,
            metadata={
                "builder_id": self.builder_id,
                "task_type": "semantic_segmentation",
                "item_count": len(items),
            },
        )
        return DatasetBuildResult(
            builder_id=self.builder_id,
            snapshot_id=snapshot.id,
            manifest_path=str(manifest_path),
            artifact_record=artifact,
            manifest=manifest,
        )


class KerasUnetDatasetBuilder(_BaseDatasetBuilder):
    builder_id = "keras.unet.semantic_segmentation.dataset"
    display_name = "Keras U-Net Dataset Builder"
    description = "Builds image and canonical label-map pair manifests for Keras U-Net workflows."

    def build(self, snapshot: RegisteredSnapshot, config: dict | None = None) -> DatasetBuildResult:
        output_dir = self._builder_dir(snapshot.id, "keras_unet_semantic_segmentation")
        items = self._collect_base_items(snapshot)
        manifest = {
            "builder_id": self.builder_id,
            "snapshot_id": snapshot.id,
            "snapshot_name": snapshot.name,
            "task_type": "semantic_segmentation",
            "model_family": "keras.unet.semantic_segmentation",
            "input_contract": "image-mask-pairs",
            "tile_mode": (config or {}).get("tile_mode", "full-image"),
            "class_schema_version": snapshot.class_schema_version,
            "class_schema": json.loads(snapshot.class_schema_json),
            "items": items,
        }
        manifest_path = output_dir / "manifest.json"
        manifest_checksum = self._write_json(manifest_path, manifest)
        artifact = self._record_dataset_build_artifact(
            snapshot_id=snapshot.id,
            manifest_path=manifest_path,
            manifest_checksum=manifest_checksum,
            metadata={
                "builder_id": self.builder_id,
                "task_type": "semantic_segmentation",
                "input_contract": "image-mask-pairs",
                "item_count": len(items),
            },
        )
        return DatasetBuildResult(
            builder_id=self.builder_id,
            snapshot_id=snapshot.id,
            manifest_path=str(manifest_path),
            artifact_record=artifact,
            manifest=manifest,
        )


def build_default_dataset_builder_registry(
    snapshot_repository: SnapshotRepository,
    workspace_repository: WorkspaceRepository,
    annotation_service: AnnotationService,
    artifact_service: ArtifactService,
    artifacts_root: Path,
) -> DatasetBuilderRegistry:
    shared_kwargs = {
        "snapshot_repository": snapshot_repository,
        "workspace_repository": workspace_repository,
        "annotation_service": annotation_service,
        "artifact_service": artifact_service,
        "artifacts_root": artifacts_root,
    }
    return DatasetBuilderRegistry(
        builders=[
            ClassicalSegmentationDatasetBuilder(**shared_kwargs),
            KerasUnetDatasetBuilder(**shared_kwargs),
        ]
    )
