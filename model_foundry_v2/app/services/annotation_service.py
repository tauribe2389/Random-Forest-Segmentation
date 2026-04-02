"""Annotation persistence service for Model Foundry V2."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..domain.entities import AnnotationRevision, AnnotationRevisionData
from ..domain.enums import ArtifactKind, ArtifactOwnerType
from ..repositories.annotation_repository import AnnotationRepository
from ..repositories.dataset_repository import DatasetRepository
from ..repositories.workspace_repository import WorkspaceRepository
from .artifact_service import ArtifactService


class AnnotationService:
    """Owns canonical annotation revision creation, load, save, and fork behavior."""

    def __init__(
        self,
        repository: AnnotationRepository,
        dataset_repository: DatasetRepository,
        workspace_repository: WorkspaceRepository,
        artifact_service: ArtifactService,
        annotations_root: Path,
    ) -> None:
        self.repository = repository
        self.dataset_repository = dataset_repository
        self.workspace_repository = workspace_repository
        self.artifact_service = artifact_service
        self.annotations_root = annotations_root

    def create_revision(
        self,
        draft_dataset_id: int,
        image_id: int,
        author: str,
        operation_summary: str,
        parent_revision_id: int | None = None,
    ) -> AnnotationRevision:
        if not self.dataset_repository.dataset_contains_image(draft_dataset_id, image_id):
            raise ValueError("Image is not part of the draft dataset.")
        image_asset = self.workspace_repository.get_image_asset(image_id)
        if image_asset is None:
            raise ValueError(f"Unknown image asset: {image_id}")
        revision = self.repository.create_revision_stub(
            draft_dataset_id=draft_dataset_id,
            image_id=image_id,
            parent_revision_id=parent_revision_id,
            width=image_asset.width,
            height=image_asset.height,
            author=author,
            operation_summary=operation_summary,
        )
        empty_label_map = np.zeros((revision.height, revision.width), dtype=np.uint16)
        empty_provenance_map = np.zeros((revision.height, revision.width), dtype=np.uint8)
        empty_protection_map = np.zeros((revision.height, revision.width), dtype=np.uint8)
        return self.save_revision(
            revision_id=revision.id,
            label_map=empty_label_map,
            provenance_map=empty_provenance_map,
            protection_map=empty_protection_map,
            author=author,
            operation_summary=operation_summary,
        )

    def ensure_head_revision(
        self,
        draft_dataset_id: int,
        image_id: int,
        author: str,
        operation_summary: str,
    ) -> AnnotationRevision:
        head = self.repository.get_head_revision(draft_dataset_id, image_id)
        if head is not None:
            return head
        revision = self.create_revision(
            draft_dataset_id=draft_dataset_id,
            image_id=image_id,
            author=author,
            operation_summary=operation_summary,
        )
        self.repository.set_head(draft_dataset_id, image_id, revision.id)
        return revision

    def fork_head_revision(
        self,
        draft_dataset_id: int,
        image_id: int,
        author: str,
        operation_summary: str,
    ) -> AnnotationRevision:
        current = self.repository.get_head_revision(draft_dataset_id, image_id)
        if current is None:
            raise ValueError("Cannot fork a missing annotation head.")
        current_data = self.load_revision(current.id)
        child = self.create_revision(
            draft_dataset_id=draft_dataset_id,
            image_id=image_id,
            parent_revision_id=current.id,
            author=author,
            operation_summary=operation_summary,
        )
        child = self.save_revision(
            revision_id=child.id,
            label_map=current_data.label_map,
            provenance_map=current_data.provenance_map,
            protection_map=current_data.protection_map,
            author=author,
            operation_summary=operation_summary,
        )
        self.repository.set_head(draft_dataset_id, image_id, child.id)
        return child

    def save_revision(
        self,
        revision_id: int,
        label_map: np.ndarray,
        provenance_map: np.ndarray,
        protection_map: np.ndarray,
        author: str,
        operation_summary: str,
    ) -> AnnotationRevision:
        revision = self.repository.get_revision(revision_id)
        if revision is None:
            raise ValueError(f"Unknown annotation revision: {revision_id}")
        if revision.is_locked:
            raise ValueError("Locked annotation revisions cannot be modified in place.")

        normalized_label_map = np.asarray(label_map, dtype=np.uint16)
        normalized_provenance_map = np.asarray(provenance_map, dtype=np.uint8)
        normalized_protection_map = np.asarray(protection_map, dtype=np.uint8)
        expected_shape = (revision.height, revision.width)
        for array in (
            normalized_label_map,
            normalized_provenance_map,
            normalized_protection_map,
        ):
            if array.shape != expected_shape:
                raise ValueError(
                    f"Revision arrays must match expected shape {expected_shape}, got {array.shape}"
                )

        revision_dir = self.annotations_root / str(revision.id)
        revision_dir.mkdir(parents=True, exist_ok=True)
        label_path = revision_dir / "label_map.npy"
        provenance_path = revision_dir / "provenance_map.npy"
        protection_path = revision_dir / "protection_map.npy"

        np.save(label_path, normalized_label_map, allow_pickle=False)
        np.save(provenance_path, normalized_provenance_map, allow_pickle=False)
        np.save(protection_path, normalized_protection_map, allow_pickle=False)

        label_checksum = self._sha256_file(label_path)
        provenance_checksum = self._sha256_file(provenance_path)
        protection_checksum = self._sha256_file(protection_path)
        revision_checksum = self._combine_checksums(
            label_checksum,
            provenance_checksum,
            protection_checksum,
        )

        updated = self.repository.update_revision_storage(
            revision_id=revision_id,
            label_map_path=str(label_path),
            provenance_map_path=str(provenance_path),
            protection_map_path=str(protection_path),
            label_checksum=label_checksum,
            provenance_checksum=provenance_checksum,
            protection_checksum=protection_checksum,
            revision_checksum=revision_checksum,
            author=author,
            operation_summary=operation_summary,
        )

        self.artifact_service.record_artifact(
            owner_type=ArtifactOwnerType.ANNOTATION_REVISION.value,
            owner_id=updated.id,
            artifact_kind=ArtifactKind.CANONICAL_LABEL_MAP,
            artifact_path=label_path,
            checksum_sha256=label_checksum,
            metadata={"dtype": "uint16", "shape": list(expected_shape)},
        )
        self.artifact_service.record_artifact(
            owner_type=ArtifactOwnerType.ANNOTATION_REVISION.value,
            owner_id=updated.id,
            artifact_kind=ArtifactKind.PROVENANCE_MAP,
            artifact_path=provenance_path,
            checksum_sha256=provenance_checksum,
            metadata={"dtype": "uint8", "shape": list(expected_shape)},
        )
        self.artifact_service.record_artifact(
            owner_type=ArtifactOwnerType.ANNOTATION_REVISION.value,
            owner_id=updated.id,
            artifact_kind=ArtifactKind.PROTECTION_MAP,
            artifact_path=protection_path,
            checksum_sha256=protection_checksum,
            metadata={"dtype": "uint8", "shape": list(expected_shape)},
        )
        return updated

    def load_revision(self, revision_id: int) -> AnnotationRevisionData:
        revision = self.repository.get_revision(revision_id)
        if revision is None:
            raise ValueError(f"Unknown annotation revision: {revision_id}")
        label_map = np.load(revision.label_map_path, allow_pickle=False)
        provenance_map = np.load(revision.provenance_map_path, allow_pickle=False)
        protection_map = np.load(revision.protection_map_path, allow_pickle=False)
        if self._sha256_file(Path(revision.label_map_path)) != revision.label_checksum:
            raise ValueError("Label map checksum mismatch on load.")
        if self._sha256_file(Path(revision.provenance_map_path)) != revision.provenance_checksum:
            raise ValueError("Provenance map checksum mismatch on load.")
        if self._sha256_file(Path(revision.protection_map_path)) != revision.protection_checksum:
            raise ValueError("Protection map checksum mismatch on load.")
        return AnnotationRevisionData(
            revision=revision,
            label_map=label_map,
            provenance_map=provenance_map,
            protection_map=protection_map,
        )

    def load_head_revision(self, draft_dataset_id: int, image_id: int) -> AnnotationRevisionData | None:
        head = self.repository.get_head_revision(draft_dataset_id, image_id)
        if head is None:
            return None
        return self.load_revision(head.id)

    def promote_prediction(self, *args: object, **kwargs: object) -> AnnotationRevision:
        raise NotImplementedError("Prediction promotion is planned for a later implementation phase.")

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _combine_checksums(*checksums: str) -> str:
        digest = hashlib.sha256()
        for checksum in checksums:
            digest.update(checksum.encode("utf-8"))
        return digest.hexdigest()
