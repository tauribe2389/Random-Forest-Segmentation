"""Prediction export service contracts for Model Foundry V2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..domain.enums import ArtifactKind, ArtifactOwnerType
from ..repositories.model_repository import ModelRepository
from ..repositories.prediction_repository import PredictionRepository
from ..repositories.snapshot_repository import SnapshotRepository
from ..repositories.workspace_repository import WorkspaceRepository
from .artifact_service import ArtifactService
from .prediction_service import PredictionService


class ExportService:
    """Exports machine-readable prediction artifacts for downstream processing."""

    def __init__(
        self,
        prediction_repository: PredictionRepository,
        snapshot_repository: SnapshotRepository,
        workspace_repository: WorkspaceRepository,
        model_repository: ModelRepository,
        prediction_service: PredictionService,
        artifact_service: ArtifactService,
        exports_root: Path,
    ) -> None:
        self.prediction_repository = prediction_repository
        self.snapshot_repository = snapshot_repository
        self.workspace_repository = workspace_repository
        self.model_repository = model_repository
        self.prediction_service = prediction_service
        self.artifact_service = artifact_service
        self.exports_root = exports_root

    def export_prediction_run(self, prediction_run_id: int, export_config: dict) -> dict:
        prediction_run = self.prediction_repository.get_prediction_run(prediction_run_id)
        if prediction_run is None:
            raise ValueError(f"Unknown prediction run: {prediction_run_id}")
        snapshot = self.snapshot_repository.get_snapshot(prediction_run.registered_snapshot_id)
        if snapshot is None:
            raise ValueError(f"Unknown registered snapshot: {prediction_run.registered_snapshot_id}")
        model_definition = self.model_repository.get_model_definition(prediction_run.model_definition_id)
        if model_definition is None:
            raise ValueError(f"Unknown model definition: {prediction_run.model_definition_id}")

        source_type = str(export_config.get("source_type") or "refined")
        if source_type not in {"raw", "refined"}:
            raise ValueError("source_type must be 'raw' or 'refined'.")

        run_dir = self.exports_root / "prediction_runs" / str(prediction_run_id) / source_type
        masks_dir = run_dir / "masks"
        metadata_dir = run_dir / "metadata"
        masks_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        class_schema = json.loads(snapshot.class_schema_json)
        manifest_items: list[dict[str, Any]] = []
        for item in self.snapshot_repository.list_snapshot_items(snapshot.id):
            image = self.workspace_repository.get_image_asset(item.image_id)
            prediction_array = np.asarray(
                self.prediction_service.load_prediction_array(
                    prediction_run_id=prediction_run_id,
                    image_id=item.image_id,
                    source_type=source_type,
                ),
                dtype=np.uint16,
            )
            mask_path = masks_dir / ("image_%s_%s.png" % (item.image_id, source_type))
            Image.fromarray(prediction_array, mode="I;16").save(mask_path)
            mask_checksum = self._sha256_file(mask_path)

            item_metadata = {
                "prediction_run_id": prediction_run_id,
                "registered_snapshot_id": snapshot.id,
                "image_id": item.image_id,
                "width": image.width,
                "height": image.height,
                "source_artifact_type": source_type,
                "class_schema_version": snapshot.class_schema_version,
                "class_schema": class_schema,
                "model_definition_id": model_definition.id,
                "model_family": model_definition.family_id,
                "prediction_run_status": prediction_run.status,
                "export_format": "png-class-index",
                "mask_path": str(mask_path),
                "mask_checksum": mask_checksum,
                "threshold_settings": export_config.get("threshold_settings") or {},
                "refinement_settings": export_config.get("refinement_settings") or {},
            }
            metadata_path = metadata_dir / ("image_%s.json" % item.image_id)
            metadata_checksum = self._write_json(metadata_path, item_metadata)
            manifest_items.append(
                {
                    "image_id": item.image_id,
                    "mask_path": str(mask_path),
                    "mask_checksum": mask_checksum,
                    "metadata_path": str(metadata_path),
                    "metadata_checksum": metadata_checksum,
                }
            )

        manifest = {
            "prediction_run_id": prediction_run_id,
            "registered_snapshot_id": snapshot.id,
            "source_artifact_type": source_type,
            "class_schema_version": snapshot.class_schema_version,
            "class_schema": class_schema,
            "model_definition_id": model_definition.id,
            "model_family": model_definition.family_id,
            "export_format": "png-class-index",
            "item_count": len(manifest_items),
            "items": manifest_items,
        }
        manifest_path = run_dir / "manifest.json"
        manifest_checksum = self._write_json(manifest_path, manifest)
        artifact_record = self.artifact_service.record_artifact(
            owner_type=ArtifactOwnerType.PREDICTION_RUN.value,
            owner_id=prediction_run_id,
            artifact_kind=ArtifactKind.PREDICTION_EXPORT,
            artifact_path=manifest_path,
            checksum_sha256=manifest_checksum,
            metadata={
                "source_artifact_type": source_type,
                "format": "png-class-index",
                "item_count": len(manifest_items),
            },
        )
        return {
            "prediction_run_id": prediction_run_id,
            "manifest_path": str(manifest_path),
            "artifact_record_id": artifact_record.id,
            "source_artifact_type": source_type,
        }

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> str:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()
