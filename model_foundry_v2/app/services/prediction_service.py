"""Prediction-run service for Model Foundry V2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..domain.entities import ModelDefinition, PredictionRun
from ..domain.enums import ArtifactKind, ArtifactOwnerType
from ..repositories.model_repository import ModelRepository
from ..repositories.prediction_repository import PredictionRepository
from ..repositories.snapshot_repository import SnapshotRepository
from .annotation_service import AnnotationService
from .artifact_service import ArtifactService
from .dataset_builders import DatasetBuilderRegistry
from .model_registry import ModelRegistry


class PredictionService:
    """Creates prediction runs and stores raw/refined prediction artifacts."""

    def __init__(
        self,
        repository: PredictionRepository,
        snapshot_repository: SnapshotRepository,
        model_repository: ModelRepository,
        model_registry: ModelRegistry,
        dataset_builder_registry: DatasetBuilderRegistry,
        annotation_service: AnnotationService,
        artifact_service: ArtifactService,
        artifacts_root: Path,
    ) -> None:
        self.repository = repository
        self.snapshot_repository = snapshot_repository
        self.model_repository = model_repository
        self.model_registry = model_registry
        self.dataset_builder_registry = dataset_builder_registry
        self.annotation_service = annotation_service
        self.artifact_service = artifact_service
        self.artifacts_root = artifacts_root

    def create_prediction_run(
        self,
        snapshot_id: int,
        model_definition_id: int,
        config: dict | None = None,
    ) -> PredictionRun:
        snapshot = self.snapshot_repository.get_snapshot(snapshot_id)
        if snapshot is None:
            raise ValueError(f"Unknown snapshot: {snapshot_id}")
        model_definition = self.model_repository.get_model_definition(model_definition_id)
        if model_definition is None:
            raise ValueError(f"Unknown model definition: {model_definition_id}")
        adapter = self.model_registry.get(model_definition.family_id)
        if adapter is None:
            raise ValueError("No model-family adapter is registered for '%s'." % model_definition.family_id)

        model_config = json.loads(model_definition.config_json or "{}")
        config_payload = {"source_artifact_type": "adapter-produced"}
        config_payload.update(model_config)
        if config:
            config_payload.update(config)
        validation = adapter.validate_snapshot(snapshot=snapshot, model_config=config_payload)
        if not validation.is_valid:
            raise ValueError("; ".join(validation.errors))
        dataset_build = adapter.build_training_artifacts(
            snapshot=snapshot,
            dataset_builder_registry=self.dataset_builder_registry,
            model_config=config_payload,
        )
        prediction_batch = adapter.predict(
            model_definition=model_definition,
            snapshot=snapshot,
            dataset_build=dataset_build,
            model_config=config_payload,
        )
        config_payload.update(
            {
                "execution_mode": prediction_batch.execution_mode,
                "dataset_builder_id": prediction_batch.dataset_builder_id,
            }
        )

        prediction_run = self.repository.create_prediction_run(
            model_definition_id=model_definition_id,
            registered_snapshot_id=snapshot_id,
            status="running",
            config_json=json.dumps(config_payload, sort_keys=True),
        )
        run_dir = self._run_dir(prediction_run.id)
        raw_dir = run_dir / "raw"
        refined_dir = run_dir / "refined"
        raw_dir.mkdir(parents=True, exist_ok=True)
        refined_dir.mkdir(parents=True, exist_ok=True)

        manifest_items: list[dict[str, Any]] = []
        for item in prediction_batch.items:
            raw_prediction = np.asarray(item.raw_prediction, dtype=np.uint16)
            refined_prediction = np.asarray(item.refined_prediction, dtype=np.uint16)

            raw_path = raw_dir / ("image_%s.npy" % item.image_id)
            refined_path = refined_dir / ("image_%s.npy" % item.image_id)
            np.save(raw_path, raw_prediction, allow_pickle=False)
            np.save(refined_path, refined_prediction, allow_pickle=False)

            raw_checksum = self._sha256_file(raw_path)
            refined_checksum = self._sha256_file(refined_path)

            self.artifact_service.record_artifact(
                owner_type=ArtifactOwnerType.PREDICTION_RUN.value,
                owner_id=prediction_run.id,
                artifact_kind=ArtifactKind.PREDICTION_RAW,
                artifact_path=raw_path,
                checksum_sha256=raw_checksum,
                metadata=dict(
                    {
                        "image_id": item.image_id,
                        "dtype": "uint16",
                        "shape": list(raw_prediction.shape),
                        "annotation_revision_id": item.annotation_revision_id,
                    },
                    **item.raw_metadata,
                ),
            )
            self.artifact_service.record_artifact(
                owner_type=ArtifactOwnerType.PREDICTION_RUN.value,
                owner_id=prediction_run.id,
                artifact_kind=ArtifactKind.PREDICTION_REFINED,
                artifact_path=refined_path,
                checksum_sha256=refined_checksum,
                metadata=dict(
                    {
                        "image_id": item.image_id,
                        "dtype": "uint16",
                        "shape": list(refined_prediction.shape),
                        "annotation_revision_id": item.annotation_revision_id,
                    },
                    **item.refined_metadata,
                ),
            )
            manifest_items.append(
                {
                    "image_id": item.image_id,
                    "annotation_revision_id": item.annotation_revision_id,
                    "raw_prediction_path": str(raw_path),
                    "raw_prediction_checksum": raw_checksum,
                    "refined_prediction_path": str(refined_path),
                    "refined_prediction_checksum": refined_checksum,
                }
            )

        manifest = {
            "prediction_run_id": prediction_run.id,
            "registered_snapshot_id": snapshot_id,
            "model_definition_id": model_definition.id,
            "model_family": model_definition.family_id,
            "dataset_builder_id": dataset_build.builder_id,
            "status": "completed",
            "config": config_payload,
            "prediction_metadata": prediction_batch.metadata,
            "items": manifest_items,
        }
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        manifest_checksum = self._sha256_file(manifest_path)
        self.artifact_service.record_artifact(
            owner_type=ArtifactOwnerType.PREDICTION_RUN.value,
            owner_id=prediction_run.id,
            artifact_kind=ArtifactKind.PREDICTION_RAW,
            artifact_path=manifest_path,
            checksum_sha256=manifest_checksum,
            metadata={
                "file_type": "manifest",
                "prediction_run_id": prediction_run.id,
                "item_count": len(manifest_items),
            },
        )
        return self.repository.update_prediction_run_status(prediction_run.id, "completed")

    def get_prediction_run(self, prediction_run_id: int) -> PredictionRun | None:
        return self.repository.get_prediction_run(prediction_run_id)

    def list_prediction_runs_for_snapshot(self, snapshot_id: int) -> list[PredictionRun]:
        return self.repository.list_prediction_runs_for_snapshot(snapshot_id)

    def get_model_definition_for_run(self, prediction_run_id: int) -> ModelDefinition | None:
        run = self.get_prediction_run(prediction_run_id)
        if run is None:
            return None
        return self.model_repository.get_model_definition(run.model_definition_id)

    def load_prediction_array(self, prediction_run_id: int, image_id: int, source_type: str) -> np.ndarray:
        artifact_kind = ArtifactKind.PREDICTION_REFINED if source_type == "refined" else ArtifactKind.PREDICTION_RAW
        artifacts = self.artifact_service.list_artifacts(
            owner_type=ArtifactOwnerType.PREDICTION_RUN.value,
            owner_id=prediction_run_id,
        )
        for artifact in artifacts:
            if artifact.artifact_kind != artifact_kind.value:
                continue
            metadata = json.loads(artifact.metadata_json or "{}")
            if metadata.get("image_id") != image_id:
                continue
            artifact_path = self.artifacts_root / artifact.relative_path
            return np.load(artifact_path, allow_pickle=False)
        raise ValueError(
            "Prediction artifact not found for run %s, image %s, source %s"
            % (prediction_run_id, image_id, source_type)
        )

    def list_prediction_artifacts(self, prediction_run_id: int):
        return self.artifact_service.list_artifacts(
            owner_type=ArtifactOwnerType.PREDICTION_RUN.value,
            owner_id=prediction_run_id,
        )

    def _run_dir(self, prediction_run_id: int) -> Path:
        path = self.artifacts_root / "predictions" / str(prediction_run_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
