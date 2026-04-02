"""Registered snapshot service for Model Foundry V2."""

from __future__ import annotations

import json

from ..domain.entities import RegisteredSnapshot
from ..repositories.annotation_repository import AnnotationRepository
from ..repositories.dataset_repository import DatasetRepository
from ..repositories.snapshot_repository import SnapshotRepository


class SnapshotService:
    """Service for registering immutable dataset snapshots."""

    def __init__(
        self,
        repository: SnapshotRepository,
        dataset_repository: DatasetRepository,
        annotation_repository: AnnotationRepository,
    ) -> None:
        self.repository = repository
        self.dataset_repository = dataset_repository
        self.annotation_repository = annotation_repository

    def register_snapshot(
        self,
        draft_dataset_id: int,
        revision_heads: dict[int, int] | None = None,
        name: str | None = None,
    ) -> RegisteredSnapshot:
        dataset = self.dataset_repository.get_draft_dataset(draft_dataset_id)
        if dataset is None:
            raise ValueError(f"Unknown draft dataset: {draft_dataset_id}")

        dataset_images = self.dataset_repository.list_dataset_images(draft_dataset_id)
        head_mapping = revision_heads or self.annotation_repository.list_head_revision_ids(draft_dataset_id)
        image_ids = {image.id for image in dataset_images}
        if set(head_mapping.keys()) != image_ids:
            missing = sorted(image_ids - set(head_mapping.keys()))
            extras = sorted(set(head_mapping.keys()) - image_ids)
            raise ValueError(
                f"Snapshot registration requires a complete head mapping. Missing={missing}, extra={extras}"
            )

        classes = self.dataset_repository.list_dataset_classes(draft_dataset_id)
        class_schema_json = json.dumps(
            [
                {
                    "class_index": dataset_class.class_index,
                    "name": dataset_class.name,
                    "color": dataset_class.color,
                }
                for dataset_class in classes
            ],
            sort_keys=True,
        )
        snapshot = self.repository.create_snapshot(
            workspace_id=dataset.workspace_id,
            draft_dataset_id=dataset.id,
            name=name or f"{dataset.name} snapshot",
            class_schema_json=class_schema_json,
            class_schema_version=dataset.schema_version,
            image_count=len(dataset_images),
        )
        self.repository.add_snapshot_items(
            snapshot_id=snapshot.id,
            items=[(image_id, revision_id) for image_id, revision_id in head_mapping.items()],
        )
        for revision_id in head_mapping.values():
            self.annotation_repository.lock_revision(revision_id)
        return snapshot

    def list_snapshots_for_dataset(self, draft_dataset_id: int) -> list[RegisteredSnapshot]:
        return self.repository.list_snapshots_for_dataset(draft_dataset_id)

    def get_snapshot(self, snapshot_id: int) -> RegisteredSnapshot | None:
        return self.repository.get_snapshot(snapshot_id)

    def list_snapshot_items(self, snapshot_id: int):
        return self.repository.list_snapshot_items(snapshot_id)
