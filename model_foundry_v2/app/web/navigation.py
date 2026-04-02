"""Workspace shell navigation helpers for Model Foundry V2."""

from __future__ import annotations

from dataclasses import dataclass

from flask import request, session, url_for


@dataclass(frozen=True)
class Crumb:
    label: str
    url: str | None = None


def build_navigation_context(
    services: dict[str, object],
    repositories: dict[str, object],
    app_name: str,
) -> dict[str, object]:
    """Build shared workspace navigation state for page templates."""
    workspace_service = services["workspace"]
    dataset_service = services["dataset"]
    snapshot_service = services["snapshot"]
    model_service = services["model"]
    training_service = services["training"]
    prediction_service = services["prediction"]

    endpoint = request.endpoint or ""
    view_args = request.view_args or {}
    workspaces = workspace_service.list_workspaces()
    active_workspace = _resolve_active_workspace(
        workspace_service=workspace_service,
        dataset_service=dataset_service,
        snapshot_service=snapshot_service,
        model_service=model_service,
        training_service=training_service,
        prediction_service=prediction_service,
        view_args=view_args,
        endpoint=endpoint,
    )

    sidebar_items: list[dict[str, object]] = []
    active_section = ""
    breadcrumbs: list[dict[str, object]] = []
    if active_workspace is not None:
        workspace_id = int(active_workspace.id)
        session["v2_workspace_id"] = workspace_id
        sidebar_items = [
            {
                "key": "overview",
                "label": "Overview",
                "url": url_for("v2.workspace_detail", workspace_id=workspace_id),
            },
            {
                "key": "images",
                "label": "Images",
                "url": url_for("v2.workspace_images_page", workspace_id=workspace_id),
            },
            {
                "key": "datasets",
                "label": "Draft Datasets",
                "url": url_for("v2.workspace_datasets_page", workspace_id=workspace_id),
            },
            {
                "key": "snapshots",
                "label": "Registered Snapshots",
                "url": url_for("v2.workspace_snapshots_page", workspace_id=workspace_id),
            },
            {
                "key": "models",
                "label": "Models",
                "url": url_for("v2.workspace_models_page", workspace_id=workspace_id),
            },
            {
                "key": "analysis",
                "label": "Analysis",
                "url": url_for("v2.workspace_analysis_page", workspace_id=workspace_id),
            },
        ]
        active_section = _active_section_for_endpoint(endpoint)
        breadcrumbs = [
            crumb.__dict__
            for crumb in _build_breadcrumbs(
                workspace_service=workspace_service,
                dataset_service=dataset_service,
                snapshot_service=snapshot_service,
                model_service=model_service,
                training_service=training_service,
                prediction_service=prediction_service,
                active_workspace=active_workspace,
                endpoint=endpoint,
                view_args=view_args,
            )
        ]

    return {
        "app_name": app_name,
        "workspaces": workspaces,
        "active_workspace": active_workspace,
        "sidebar_items": sidebar_items,
        "active_section": active_section,
        "breadcrumbs": breadcrumbs,
    }


def _resolve_active_workspace(
    *,
    workspace_service,
    dataset_service,
    snapshot_service,
    model_service,
    training_service,
    prediction_service,
    view_args: dict[str, object],
    endpoint: str,
):
    if endpoint in {
        "v2.index",
        "v2.healthz",
        "v2.favicon",
        "v2.asset_file",
    }:
        return None

    workspace_id = view_args.get("workspace_id")
    if isinstance(workspace_id, int):
        workspace = workspace_service.get_workspace(workspace_id)
        if workspace is not None:
            return workspace

    dataset_id = view_args.get("dataset_id")
    if isinstance(dataset_id, int):
        dataset = dataset_service.get_draft_dataset(dataset_id)
        if dataset is not None:
            return workspace_service.get_workspace(dataset.workspace_id)

    snapshot_id = view_args.get("snapshot_id")
    if isinstance(snapshot_id, int):
        snapshot = snapshot_service.get_snapshot(snapshot_id)
        if snapshot is not None:
            return workspace_service.get_workspace(snapshot.workspace_id)

    model_definition_id = view_args.get("model_definition_id")
    if isinstance(model_definition_id, int):
        model_definition = model_service.get_model_definition(model_definition_id)
        if model_definition is not None:
            return workspace_service.get_workspace(model_definition.workspace_id)

    training_run_id = view_args.get("training_run_id")
    if isinstance(training_run_id, int):
        training_run = training_service.get_training_run(training_run_id)
        if training_run is not None:
            snapshot = snapshot_service.get_snapshot(training_run.registered_snapshot_id)
            if snapshot is not None:
                return workspace_service.get_workspace(snapshot.workspace_id)

    prediction_run_id = view_args.get("prediction_run_id")
    if isinstance(prediction_run_id, int):
        prediction_run = prediction_service.get_prediction_run(prediction_run_id)
        if prediction_run is not None:
            snapshot = snapshot_service.get_snapshot(prediction_run.registered_snapshot_id)
            if snapshot is not None:
                return workspace_service.get_workspace(snapshot.workspace_id)

    session_workspace_id = session.get("v2_workspace_id")
    if isinstance(session_workspace_id, int):
        return workspace_service.get_workspace(session_workspace_id)
    return None


def _active_section_for_endpoint(endpoint: str) -> str:
    if endpoint in {"v2.workspace_detail"}:
        return "overview"
    if endpoint in {"v2.workspace_images_page"}:
        return "images"
    if endpoint in {
        "v2.workspace_datasets_page",
        "v2.dataset_detail",
        "v2.annotation_detail",
    }:
        return "datasets"
    if endpoint in {"v2.workspace_snapshots_page", "v2.snapshot_detail"}:
        return "snapshots"
    if endpoint in {"v2.workspace_models_page", "v2.model_detail"}:
        return "models"
    if endpoint in {
        "v2.workspace_analysis_page",
        "v2.training_run_detail",
        "v2.prediction_run_detail",
    }:
        return "analysis"
    return ""


def _build_breadcrumbs(
    *,
    workspace_service,
    dataset_service,
    snapshot_service,
    model_service,
    training_service,
    prediction_service,
    active_workspace,
    endpoint: str,
    view_args: dict[str, object],
) -> list[Crumb]:
    workspace_id = int(active_workspace.id)
    crumbs = [
        Crumb("Workspaces", url_for("v2.index")),
        Crumb(active_workspace.name, url_for("v2.workspace_detail", workspace_id=workspace_id)),
    ]

    if endpoint == "v2.workspace_images_page":
        crumbs.append(Crumb("Images"))
    elif endpoint == "v2.workspace_datasets_page":
        crumbs.append(Crumb("Draft Datasets"))
    elif endpoint == "v2.workspace_snapshots_page":
        crumbs.append(Crumb("Registered Snapshots"))
    elif endpoint == "v2.workspace_models_page":
        crumbs.append(Crumb("Models"))
    elif endpoint == "v2.workspace_analysis_page":
        crumbs.append(Crumb("Analysis"))
    elif endpoint == "v2.dataset_detail":
        dataset = dataset_service.get_draft_dataset(int(view_args["dataset_id"]))
        crumbs.append(Crumb("Draft Datasets", url_for("v2.workspace_datasets_page", workspace_id=workspace_id)))
        crumbs.append(Crumb(dataset.name if dataset is not None else f"Dataset {view_args['dataset_id']}"))
    elif endpoint == "v2.annotation_detail":
        dataset = dataset_service.get_draft_dataset(int(view_args["dataset_id"]))
        image_id = int(view_args["image_id"])
        crumbs.append(Crumb("Draft Datasets", url_for("v2.workspace_datasets_page", workspace_id=workspace_id)))
        crumbs.append(
            Crumb(
                dataset.name if dataset is not None else f"Dataset {view_args['dataset_id']}",
                url_for("v2.dataset_detail", dataset_id=int(view_args["dataset_id"])),
            )
        )
        crumbs.append(Crumb(f"Image {image_id} Annotation"))
    elif endpoint == "v2.snapshot_detail":
        snapshot = snapshot_service.get_snapshot(int(view_args["snapshot_id"]))
        crumbs.append(Crumb("Registered Snapshots", url_for("v2.workspace_snapshots_page", workspace_id=workspace_id)))
        crumbs.append(Crumb(snapshot.name if snapshot is not None else f"Snapshot {view_args['snapshot_id']}"))
    elif endpoint == "v2.model_detail":
        model_definition = model_service.get_model_definition(int(view_args["model_definition_id"]))
        crumbs.append(Crumb("Models", url_for("v2.workspace_models_page", workspace_id=workspace_id)))
        crumbs.append(Crumb(model_definition.name if model_definition is not None else f"Model {view_args['model_definition_id']}"))
    elif endpoint == "v2.training_run_detail":
        training_run = training_service.get_training_run(int(view_args["training_run_id"]))
        snapshot = snapshot_service.get_snapshot(training_run.registered_snapshot_id) if training_run is not None else None
        crumbs.append(Crumb("Analysis", url_for("v2.workspace_analysis_page", workspace_id=workspace_id)))
        if snapshot is not None:
            crumbs.append(Crumb(snapshot.name, url_for("v2.snapshot_detail", snapshot_id=snapshot.id)))
        crumbs.append(Crumb(f"Training Run {view_args['training_run_id']}"))
    elif endpoint == "v2.prediction_run_detail":
        prediction_run = prediction_service.get_prediction_run(int(view_args["prediction_run_id"]))
        snapshot = snapshot_service.get_snapshot(prediction_run.registered_snapshot_id) if prediction_run is not None else None
        crumbs.append(Crumb("Analysis", url_for("v2.workspace_analysis_page", workspace_id=workspace_id)))
        if snapshot is not None:
            crumbs.append(Crumb(snapshot.name, url_for("v2.snapshot_detail", snapshot_id=snapshot.id)))
        crumbs.append(Crumb(f"Prediction Run {view_args['prediction_run_id']}"))
    return crumbs
