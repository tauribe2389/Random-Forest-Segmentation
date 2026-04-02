"""Flask application factory for Model Foundry V2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask

from .config import build_default_config, ensure_runtime_directories
from .repositories.annotation_repository import AnnotationRepository
from .repositories.artifact_repository import ArtifactRepository
from .repositories.database import Database
from .repositories.dataset_repository import DatasetRepository
from .repositories.model_repository import ModelRepository
from .repositories.prediction_repository import PredictionRepository
from .repositories.snapshot_repository import SnapshotRepository
from .repositories.training_repository import TrainingRepository
from .repositories.workspace_repository import WorkspaceRepository
from .services.annotation_service import AnnotationService
from .services.artifact_service import ArtifactService
from .services.dataset_builders import build_default_dataset_builder_registry
from .services.dataset_service import DatasetService
from .services.model_service import ModelService
from .services.promotion_service import PromotionService
from .services.export_service import ExportService
from .services.model_registry import build_default_model_registry
from .services.snapshot_builders import build_default_snapshot_builder_registry
from .services.prediction_service import PredictionService
from .services.snapshot_service import SnapshotService
from .services.training_service import TrainingService
from .services.workspace_service import WorkspaceService
from .web.navigation import build_navigation_context


def create_app(config_overrides: dict[str, Any] | None = None) -> Flask:
    """Create and configure the V2 Flask application."""
    config = build_default_config()
    if config_overrides:
        config.update(config_overrides)
    ensure_runtime_directories(config)

    app = Flask(__name__, static_folder=None)
    app.config.update(config)

    database = Database(Path(app.config["DB_PATH"]))
    database.initialize()

    workspace_repository = WorkspaceRepository(database)
    dataset_repository = DatasetRepository(database)
    annotation_repository = AnnotationRepository(database)
    artifact_repository = ArtifactRepository(database)
    snapshot_repository = SnapshotRepository(database)
    model_repository = ModelRepository(database)
    prediction_repository = PredictionRepository(database)
    training_repository = TrainingRepository(database)

    artifact_service = ArtifactService(
        repository=artifact_repository,
        artifacts_root=Path(app.config["ARTIFACTS_DIR"]),
    )
    annotation_service = AnnotationService(
        repository=annotation_repository,
        dataset_repository=dataset_repository,
        workspace_repository=workspace_repository,
        artifact_service=artifact_service,
        annotations_root=Path(app.config["ANNOTATIONS_DIR"]),
    )
    workspace_service = WorkspaceService(repository=workspace_repository)
    dataset_service = DatasetService(
        repository=dataset_repository,
        workspace_repository=workspace_repository,
        annotation_service=annotation_service,
    )
    snapshot_service = SnapshotService(
        repository=snapshot_repository,
        dataset_repository=dataset_repository,
        annotation_repository=annotation_repository,
    )
    model_registry = build_default_model_registry()
    dataset_builder_registry = build_default_dataset_builder_registry(
        snapshot_repository=snapshot_repository,
        workspace_repository=workspace_repository,
        annotation_service=annotation_service,
        artifact_service=artifact_service,
        artifacts_root=Path(app.config["ARTIFACTS_DIR"]),
    )
    model_service = ModelService(
        repository=model_repository,
        workspace_repository=workspace_repository,
        model_registry=model_registry,
    )
    prediction_service = PredictionService(
        repository=prediction_repository,
        snapshot_repository=snapshot_repository,
        model_repository=model_repository,
        model_registry=model_registry,
        dataset_builder_registry=dataset_builder_registry,
        annotation_service=annotation_service,
        artifact_service=artifact_service,
        artifacts_root=Path(app.config["ARTIFACTS_DIR"]),
    )
    training_service = TrainingService(
        repository=training_repository,
        snapshot_repository=snapshot_repository,
        model_repository=model_repository,
        model_registry=model_registry,
        dataset_builder_registry=dataset_builder_registry,
        artifact_service=artifact_service,
        artifacts_root=Path(app.config["ARTIFACTS_DIR"]),
    )
    export_service = ExportService(
        prediction_repository=prediction_repository,
        snapshot_repository=snapshot_repository,
        workspace_repository=workspace_repository,
        model_repository=model_repository,
        prediction_service=prediction_service,
        artifact_service=artifact_service,
        exports_root=Path(app.config["EXPORTS_DIR"]),
    )
    promotion_service = PromotionService(
        dataset_repository=dataset_repository,
        snapshot_repository=snapshot_repository,
        dataset_service=dataset_service,
        annotation_service=annotation_service,
        prediction_service=prediction_service,
    )
    snapshot_builder_registry = build_default_snapshot_builder_registry(
        snapshot_repository=snapshot_repository,
        workspace_repository=workspace_repository,
        annotation_service=annotation_service,
        artifact_service=artifact_service,
        artifacts_root=Path(app.config["ARTIFACTS_DIR"]),
    )

    app.extensions["v2_database"] = database
    app.extensions["v2_services"] = {
        "workspace": workspace_service,
        "dataset": dataset_service,
        "annotation": annotation_service,
        "artifact": artifact_service,
        "snapshot": snapshot_service,
        "model": model_service,
        "training": training_service,
        "prediction": prediction_service,
        "export": export_service,
        "promotion": promotion_service,
    }
    app.extensions["v2_model_registry"] = model_registry
    app.extensions["v2_dataset_builders"] = dataset_builder_registry
    app.extensions["v2_snapshot_builders"] = snapshot_builder_registry
    app.extensions["v2_repositories"] = {
        "workspace": workspace_repository,
        "dataset": dataset_repository,
        "annotation": annotation_repository,
        "artifact": artifact_repository,
        "snapshot": snapshot_repository,
        "model": model_repository,
        "prediction": prediction_repository,
        "training": training_repository,
    }

    from .web.routes import bp

    app.register_blueprint(bp)

    @app.context_processor
    def inject_workspace_navigation() -> dict[str, object]:
        return build_navigation_context(
            services=app.extensions["v2_services"],
            repositories=app.extensions["v2_repositories"],
            app_name=str(app.config["APP_NAME"]),
        )

    return app
