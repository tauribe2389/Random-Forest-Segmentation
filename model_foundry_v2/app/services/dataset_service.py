"""Draft dataset service for Model Foundry V2."""

from __future__ import annotations

from ..domain.entities import DatasetClass, DraftDataset, ImageAsset
from ..repositories.dataset_repository import DatasetRepository
from ..repositories.workspace_repository import WorkspaceRepository
from .annotation_service import AnnotationService


class DatasetService:
    """Service for mutable datasets and their class schemas."""

    def __init__(
        self,
        repository: DatasetRepository,
        workspace_repository: WorkspaceRepository,
        annotation_service: AnnotationService,
    ) -> None:
        self.repository = repository
        self.workspace_repository = workspace_repository
        self.annotation_service = annotation_service

    def create_draft_dataset(
        self,
        workspace_id: int,
        name: str,
        image_ids: list[int],
        description: str | None = None,
    ) -> DraftDataset:
        if self.workspace_repository.get_workspace(workspace_id) is None:
            raise ValueError(f"Unknown workspace: {workspace_id}")
        known_image_ids = {
            image.id for image in self.workspace_repository.list_image_assets(workspace_id)
        }
        if not image_ids:
            raise ValueError("A draft dataset requires at least one image.")
        invalid_ids = [image_id for image_id in image_ids if image_id not in known_image_ids]
        if invalid_ids:
            raise ValueError(f"Images do not belong to workspace {workspace_id}: {invalid_ids}")
        return self.repository.create_draft_dataset(
            workspace_id=workspace_id,
            name=name.strip(),
            image_ids=image_ids,
            description=description,
        )

    def get_draft_dataset(self, dataset_id: int) -> DraftDataset | None:
        return self.repository.get_draft_dataset(dataset_id)

    def list_draft_datasets(self, workspace_id: int) -> list[DraftDataset]:
        return self.repository.list_draft_datasets(workspace_id)

    def list_dataset_images(self, dataset_id: int) -> list[ImageAsset]:
        return self.repository.list_dataset_images(dataset_id)

    def add_class(self, dataset_id: int, name: str, color: str) -> DatasetClass:
        dataset = self.repository.get_draft_dataset(dataset_id)
        if dataset is None:
            raise ValueError(f"Unknown draft dataset: {dataset_id}")
        return self.repository.add_class(dataset_id=dataset_id, name=name.strip(), color=color)

    def list_dataset_classes(self, dataset_id: int) -> list[DatasetClass]:
        return self.repository.list_dataset_classes(dataset_id)

    def initialize_annotation_for_image(self, dataset_id: int, image_id: int) -> int:
        revision = self.annotation_service.ensure_head_revision(
            draft_dataset_id=dataset_id,
            image_id=image_id,
            author="system",
            operation_summary="Initialize empty annotation revision",
        )
        return revision.id
