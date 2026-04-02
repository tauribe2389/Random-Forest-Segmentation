"""Prediction promotion service for Model Foundry V2."""

from __future__ import annotations

from ..repositories.dataset_repository import DatasetRepository
from ..repositories.snapshot_repository import SnapshotRepository
from .annotation_service import AnnotationService
from .dataset_service import DatasetService
from .prediction_service import PredictionService


class PromotionService:
    """Promotes prediction artifacts into a new draft dataset branch explicitly."""

    def __init__(
        self,
        dataset_repository: DatasetRepository,
        snapshot_repository: SnapshotRepository,
        dataset_service: DatasetService,
        annotation_service: AnnotationService,
        prediction_service: PredictionService,
    ) -> None:
        self.dataset_repository = dataset_repository
        self.snapshot_repository = snapshot_repository
        self.dataset_service = dataset_service
        self.annotation_service = annotation_service
        self.prediction_service = prediction_service

    def promote_prediction_run(
        self,
        prediction_run_id: int,
        source_type: str = "refined",
        new_dataset_name: str | None = None,
    ):
        prediction_run = self.prediction_service.get_prediction_run(prediction_run_id)
        if prediction_run is None:
            raise ValueError(f"Unknown prediction run: {prediction_run_id}")
        snapshot = self.snapshot_repository.get_snapshot(prediction_run.registered_snapshot_id)
        if snapshot is None:
            raise ValueError(f"Unknown registered snapshot: {prediction_run.registered_snapshot_id}")
        source_dataset = self.dataset_repository.get_draft_dataset(snapshot.draft_dataset_id)
        if source_dataset is None:
            raise ValueError(f"Unknown source draft dataset: {snapshot.draft_dataset_id}")

        source_images = self.dataset_repository.list_dataset_images(source_dataset.id)
        promoted_dataset = self.dataset_service.create_draft_dataset(
            workspace_id=source_dataset.workspace_id,
            name=new_dataset_name
            or self._build_default_dataset_name(
                workspace_id=source_dataset.workspace_id,
                source_dataset_name=source_dataset.name,
                prediction_run_id=prediction_run_id,
                source_type=source_type,
            ),
            image_ids=[image.id for image in source_images],
            description="Promoted from prediction run %s using %s artifacts." % (prediction_run_id, source_type),
        )

        source_classes = self.dataset_repository.list_dataset_classes(source_dataset.id)
        for dataset_class in source_classes:
            if dataset_class.class_index == 0:
                continue
            self.dataset_service.add_class(
                dataset_id=promoted_dataset.id,
                name=dataset_class.name,
                color=dataset_class.color,
            )

        for image in source_images:
            revision = self.annotation_service.ensure_head_revision(
                draft_dataset_id=promoted_dataset.id,
                image_id=image.id,
                author="promotion",
                operation_summary="Initialize promoted revision",
            )
            prediction_array = self.prediction_service.load_prediction_array(
                prediction_run_id=prediction_run_id,
                image_id=image.id,
                source_type=source_type,
            )
            provenance_map = (prediction_array > 0).astype("uint8") * 2
            protection_map = (prediction_array > 0).astype("uint8")
            self.annotation_service.save_revision(
                revision_id=revision.id,
                label_map=prediction_array,
                provenance_map=provenance_map,
                protection_map=protection_map,
                author="promotion",
                operation_summary=(
                    "Promoted %s prediction from run %s into draft truth seed"
                    % (source_type, prediction_run_id)
                ),
            )
        return promoted_dataset

    def _build_default_dataset_name(
        self,
        *,
        workspace_id: int,
        source_dataset_name: str,
        prediction_run_id: int,
        source_type: str,
    ) -> str:
        base_name = "%s promoted %s from run %s" % (
            source_dataset_name,
            source_type,
            prediction_run_id,
        )
        existing_names = {
            dataset.name.lower()
            for dataset in self.dataset_repository.list_draft_datasets(workspace_id)
        }
        if base_name.lower() not in existing_names:
            return base_name
        suffix = 2
        while True:
            candidate = "%s (%s)" % (base_name, suffix)
            if candidate.lower() not in existing_names:
                return candidate
            suffix += 1
