"""Training-run service for Model Foundry V2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..domain.entities import TrainingRun
from ..domain.enums import ArtifactKind, ArtifactOwnerType
from ..repositories.model_repository import ModelRepository
from ..repositories.snapshot_repository import SnapshotRepository
from ..repositories.training_repository import TrainingRepository
from .artifact_service import ArtifactService
from .dataset_builders import DatasetBuilderRegistry
from .model_registry import ModelRegistry


class TrainingService:
    """Delegates model-family training to adapters and stores resulting artifacts."""

    def __init__(
        self,
        repository: TrainingRepository,
        snapshot_repository: SnapshotRepository,
        model_repository: ModelRepository,
        model_registry: ModelRegistry,
        dataset_builder_registry: DatasetBuilderRegistry,
        artifact_service: ArtifactService,
        artifacts_root: Path,
    ) -> None:
        self.repository = repository
        self.snapshot_repository = snapshot_repository
        self.model_repository = model_repository
        self.model_registry = model_registry
        self.dataset_builder_registry = dataset_builder_registry
        self.artifact_service = artifact_service
        self.artifacts_root = artifacts_root

    def start_training_run(
        self,
        snapshot_id: int,
        model_definition_id: int,
        config: dict | None = None,
    ) -> TrainingRun:
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
        if config:
            model_config.update(config)

        validation = adapter.validate_snapshot(snapshot=snapshot, model_config=model_config)
        if not validation.is_valid:
            raise ValueError("; ".join(validation.errors))

        training_run = self.repository.create_training_run(
            model_definition_id=model_definition_id,
            registered_snapshot_id=snapshot_id,
            status="running",
        )
        try:
            dataset_build = adapter.build_training_artifacts(
                snapshot=snapshot,
                dataset_builder_registry=self.dataset_builder_registry,
                model_config=model_config,
            )
            output_dir = self._run_dir(training_run.id)
            training_result = adapter.train(
                training_manifest=dataset_build.manifest,
                model_config=dict(model_config, output_dir=str(output_dir), training_run_id=training_run.id),
            )
            artifact_manifest_path = training_result.get("artifact_manifest_path")
            if artifact_manifest_path:
                artifact_path = Path(artifact_manifest_path)
                checksum = self._sha256_file(artifact_path)
                self.artifact_service.record_artifact(
                    owner_type=ArtifactOwnerType.TRAINING_RUN.value,
                    owner_id=training_run.id,
                    artifact_kind=ArtifactKind.MODEL_ARTIFACT,
                    artifact_path=artifact_path,
                    checksum_sha256=checksum,
                    metadata={
                        "runtime": training_result.get("runtime"),
                        "family_id": model_definition.family_id,
                        "model_definition_id": model_definition.id,
                    },
                )
            updated_model_config = dict(model_config)
            updated_model_config.update(
                {
                    "latest_training_run_id": training_run.id,
                    "trained_family_id": model_definition.family_id,
                    "trained_snapshot_id": snapshot.id,
                    "training_runtime": training_result.get("runtime"),
                    "trained_artifact_manifest_path": training_result.get("artifact_manifest_path"),
                }
            )
            self.model_repository.update_model_definition_config(
                model_definition_id=model_definition.id,
                config_json=json.dumps(updated_model_config, sort_keys=True),
            )
            return self.repository.update_training_run_status(training_run.id, "completed")
        except Exception:
            self.repository.update_training_run_status(training_run.id, "failed")
            raise

    def get_training_run(self, training_run_id: int) -> TrainingRun | None:
        return self.repository.get_training_run(training_run_id)

    def list_training_runs_for_snapshot(self, snapshot_id: int) -> list[TrainingRun]:
        return self.repository.list_training_runs_for_snapshot(snapshot_id)

    def list_training_artifacts(self, training_run_id: int):
        return self.artifact_service.list_artifacts(
            owner_type=ArtifactOwnerType.TRAINING_RUN.value,
            owner_id=training_run_id,
        )

    def _run_dir(self, training_run_id: int) -> Path:
        path = self.artifacts_root / "training_runs" / str(training_run_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
