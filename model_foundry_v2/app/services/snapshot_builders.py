"""Snapshot builders for Model Foundry V2."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..domain.entities import ArtifactRecord, RegisteredSnapshot
from ..domain.enums import ArtifactKind, ArtifactOwnerType
from ..repositories.snapshot_repository import SnapshotRepository
from ..repositories.workspace_repository import WorkspaceRepository
from .annotation_service import AnnotationService
from .artifact_service import ArtifactService


@dataclass(frozen=True)
class SnapshotBuildResult:
    builder_id: str
    snapshot_id: int
    manifest_path: str
    artifact_record: ArtifactRecord


class SnapshotBuilder(ABC):
    """Base contract for immutable snapshot builders."""

    builder_id: str
    display_name: str
    description: str

    @abstractmethod
    def build(self, snapshot: RegisteredSnapshot) -> SnapshotBuildResult:
        raise NotImplementedError


class SnapshotBuilderRegistry:
    """Registry and dispatcher for snapshot builders."""

    def __init__(self, builders: list[SnapshotBuilder]) -> None:
        self._builders = {builder.builder_id: builder for builder in builders}

    def get(self, builder_id: str) -> SnapshotBuilder | None:
        return self._builders.get(builder_id)

    def list_builders(self) -> list[SnapshotBuilder]:
        return [self._builders[key] for key in sorted(self._builders.keys())]


class _BaseSnapshotBuilder(SnapshotBuilder):
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

    def _snapshot_dir(self, snapshot_id: int, builder_folder: str) -> Path:
        path = self.artifacts_root / "snapshots" / str(snapshot_id) / builder_folder
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> str:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return digest

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _record_manifest_artifact(
        self,
        snapshot_id: int,
        manifest_path: Path,
        manifest_checksum: str,
        metadata: dict[str, Any],
    ) -> ArtifactRecord:
        return self.artifact_service.record_artifact(
            owner_type=ArtifactOwnerType.REGISTERED_SNAPSHOT.value,
            owner_id=snapshot_id,
            artifact_kind=ArtifactKind.DATASET_EXPORT,
            artifact_path=manifest_path,
            checksum_sha256=manifest_checksum,
            metadata=metadata,
        )


class CanonicalSemanticDatasetBuilder(_BaseSnapshotBuilder):
    builder_id = "canonical_semantic_dataset"
    display_name = "Canonical Semantic Dataset View"
    description = "Manifest-backed immutable view of image assets and canonical annotation revisions."

    def build(self, snapshot: RegisteredSnapshot) -> SnapshotBuildResult:
        output_dir = self._snapshot_dir(snapshot.id, "canonical_semantic_dataset")
        items = self.snapshot_repository.list_snapshot_items(snapshot.id)
        manifest_items: list[dict[str, Any]] = []
        for item in items:
            image = self.workspace_repository.get_image_asset(item.image_id)
            revision_data = self.annotation_service.load_revision(item.annotation_revision_id)
            manifest_items.append(
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
                    "label_checksum": revision_data.revision.label_checksum,
                }
            )
        manifest = {
            "builder_id": self.builder_id,
            "snapshot_id": snapshot.id,
            "snapshot_name": snapshot.name,
            "class_schema_version": snapshot.class_schema_version,
            "class_schema": json.loads(snapshot.class_schema_json),
            "image_count": snapshot.image_count,
            "items": manifest_items,
        }
        manifest_path = output_dir / "manifest.json"
        manifest_checksum = self._write_json(manifest_path, manifest)
        artifact = self._record_manifest_artifact(
            snapshot_id=snapshot.id,
            manifest_path=manifest_path,
            manifest_checksum=manifest_checksum,
            metadata={
                "builder_id": self.builder_id,
                "format": "canonical-semantic-dataset",
                "item_count": len(manifest_items),
            },
        )
        return SnapshotBuildResult(
            builder_id=self.builder_id,
            snapshot_id=snapshot.id,
            manifest_path=str(manifest_path),
            artifact_record=artifact,
        )


class RasterMaskExportBuilder(_BaseSnapshotBuilder):
    builder_id = "raster_mask_export"
    display_name = "Raster Mask Export"
    description = "Exports class-index PNG masks and per-image JSON metadata from canonical revisions."

    def build(self, snapshot: RegisteredSnapshot) -> SnapshotBuildResult:
        output_dir = self._snapshot_dir(snapshot.id, "raster_mask_export")
        masks_dir = output_dir / "masks"
        metadata_dir = output_dir / "metadata"
        masks_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        items = self.snapshot_repository.list_snapshot_items(snapshot.id)
        class_schema = json.loads(snapshot.class_schema_json)
        manifest_items: list[dict[str, Any]] = []

        for item in items:
            image = self.workspace_repository.get_image_asset(item.image_id)
            revision_data = self.annotation_service.load_revision(item.annotation_revision_id)
            label_map = np.asarray(revision_data.label_map, dtype=np.uint16)
            if int(label_map.max()) > 65535:
                raise ValueError("Raster export only supports class indices up to uint16.")

            mask_path = masks_dir / ("image_%s_class_index.png" % image.id)
            Image.fromarray(label_map, mode="I;16").save(mask_path)
            mask_checksum = self._sha256_file(mask_path)

            metadata_payload = {
                "snapshot_id": snapshot.id,
                "image_id": image.id,
                "annotation_revision_id": revision_data.revision.id,
                "revision_checksum": revision_data.revision.revision_checksum,
                "width": image.width,
                "height": image.height,
                "mask_format": "png-class-index",
                "mask_path": str(mask_path),
                "mask_checksum": mask_checksum,
                "class_schema_version": snapshot.class_schema_version,
                "class_schema": class_schema,
                "source_artifact_type": "canonical_label_map",
            }
            metadata_path = metadata_dir / ("image_%s.json" % image.id)
            metadata_checksum = self._write_json(metadata_path, metadata_payload)
            manifest_items.append(
                {
                    "image_id": image.id,
                    "mask_path": str(mask_path),
                    "mask_checksum": mask_checksum,
                    "metadata_path": str(metadata_path),
                    "metadata_checksum": metadata_checksum,
                }
            )

        manifest = {
            "builder_id": self.builder_id,
            "snapshot_id": snapshot.id,
            "snapshot_name": snapshot.name,
            "class_schema_version": snapshot.class_schema_version,
            "class_schema": class_schema,
            "image_count": snapshot.image_count,
            "items": manifest_items,
        }
        manifest_path = output_dir / "manifest.json"
        manifest_checksum = self._write_json(manifest_path, manifest)
        artifact = self._record_manifest_artifact(
            snapshot_id=snapshot.id,
            manifest_path=manifest_path,
            manifest_checksum=manifest_checksum,
            metadata={
                "builder_id": self.builder_id,
                "format": "class-index-png-export",
                "item_count": len(manifest_items),
            },
        )
        return SnapshotBuildResult(
            builder_id=self.builder_id,
            snapshot_id=snapshot.id,
            manifest_path=str(manifest_path),
            artifact_record=artifact,
        )


class CocoExportPlaceholderBuilder(_BaseSnapshotBuilder):
    builder_id = "coco_export_placeholder"
    display_name = "COCO Export Placeholder"
    description = "Writes a placeholder manifest that reserves COCO as a derived export only."

    def build(self, snapshot: RegisteredSnapshot) -> SnapshotBuildResult:
        output_dir = self._snapshot_dir(snapshot.id, "coco_export_placeholder")
        manifest = {
            "builder_id": self.builder_id,
            "snapshot_id": snapshot.id,
            "snapshot_name": snapshot.name,
            "status": "placeholder",
            "canonical_dataset_identity": "registered_snapshot",
            "note": "COCO remains a derived export and is not the canonical dataset definition in v2.",
            "class_schema_version": snapshot.class_schema_version,
            "class_schema": json.loads(snapshot.class_schema_json),
            "image_count": snapshot.image_count,
        }
        manifest_path = output_dir / "manifest.json"
        manifest_checksum = self._write_json(manifest_path, manifest)
        artifact = self._record_manifest_artifact(
            snapshot_id=snapshot.id,
            manifest_path=manifest_path,
            manifest_checksum=manifest_checksum,
            metadata={
                "builder_id": self.builder_id,
                "format": "placeholder-manifest",
            },
        )
        return SnapshotBuildResult(
            builder_id=self.builder_id,
            snapshot_id=snapshot.id,
            manifest_path=str(manifest_path),
            artifact_record=artifact,
        )


def build_default_snapshot_builder_registry(
    snapshot_repository: SnapshotRepository,
    workspace_repository: WorkspaceRepository,
    annotation_service: AnnotationService,
    artifact_service: ArtifactService,
    artifacts_root: Path,
) -> SnapshotBuilderRegistry:
    shared_kwargs = {
        "snapshot_repository": snapshot_repository,
        "workspace_repository": workspace_repository,
        "annotation_service": annotation_service,
        "artifact_service": artifact_service,
        "artifacts_root": artifacts_root,
    }
    return SnapshotBuilderRegistry(
        builders=[
            CanonicalSemanticDatasetBuilder(**shared_kwargs),
            RasterMaskExportBuilder(**shared_kwargs),
            CocoExportPlaceholderBuilder(**shared_kwargs),
        ]
    )
