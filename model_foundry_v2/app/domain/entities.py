"""Typed domain entities for Model Foundry V2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Workspace:
    id: int
    name: str
    description: str | None
    created_at: str


@dataclass
class ImageAsset:
    id: int
    workspace_id: int
    source_path: str
    checksum_sha256: str
    width: int
    height: int
    created_at: str


@dataclass
class DraftDataset:
    id: int
    workspace_id: int
    name: str
    description: str | None
    schema_version: int
    created_at: str
    updated_at: str


@dataclass
class DatasetClass:
    id: int
    draft_dataset_id: int
    class_index: int
    name: str
    color: str
    created_at: str


@dataclass
class AnnotationRevision:
    id: int
    draft_dataset_id: int
    image_id: int
    parent_revision_id: int | None
    width: int
    height: int
    label_map_path: str
    provenance_map_path: str
    protection_map_path: str
    label_checksum: str
    provenance_checksum: str
    protection_checksum: str
    revision_checksum: str
    author: str
    operation_summary: str
    is_locked: bool
    created_at: str
    updated_at: str


@dataclass
class AnnotationRevisionData:
    revision: AnnotationRevision
    label_map: np.ndarray
    provenance_map: np.ndarray
    protection_map: np.ndarray


@dataclass
class RegisteredSnapshot:
    id: int
    workspace_id: int
    draft_dataset_id: int
    name: str
    class_schema_json: str
    class_schema_version: int
    image_count: int
    created_at: str


@dataclass
class RegisteredSnapshotItem:
    snapshot_id: int
    image_id: int
    annotation_revision_id: int


@dataclass
class ArtifactRecord:
    id: int
    owner_type: str
    owner_id: int
    artifact_kind: str
    relative_path: str
    checksum_sha256: str
    metadata_json: str | None
    created_at: str
    updated_at: str


@dataclass
class ModelDefinition:
    id: int
    workspace_id: int
    name: str
    family_id: str
    config_json: str
    created_at: str


@dataclass
class TrainingRun:
    id: int
    model_definition_id: int
    registered_snapshot_id: int
    status: str
    created_at: str
    updated_at: str


@dataclass
class PredictionRun:
    id: int
    model_definition_id: int
    registered_snapshot_id: int
    status: str
    source_prediction_run_id: int | None
    config_json: str
    created_at: str
    updated_at: str
