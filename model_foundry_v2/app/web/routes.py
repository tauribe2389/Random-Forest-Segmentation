"""Web routes for Model Foundry V2."""

from __future__ import annotations

import json
import re
from pathlib import Path

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    session,
    url_for,
)

bp = Blueprint("v2", __name__, template_folder="templates", static_folder="static")


def _services() -> dict[str, object]:
    return current_app.extensions["v2_services"]


def _repositories() -> dict[str, object]:
    return current_app.extensions["v2_repositories"]


def _workspace_service():
    return _services()["workspace"]


def _dataset_service():
    return _services()["dataset"]


def _annotation_service():
    return _services()["annotation"]


def _snapshot_service():
    return _services()["snapshot"]


def _artifact_service():
    return _services()["artifact"]


def _model_service():
    return _services()["model"]


def _prediction_service():
    return _services()["prediction"]


def _training_service():
    return _services()["training"]


def _export_service():
    return _services()["export"]


def _promotion_service():
    return _services()["promotion"]


def _snapshot_builders():
    return current_app.extensions["v2_snapshot_builders"]


def _model_registry():
    return current_app.extensions["v2_model_registry"]


def _status_badge_class(status_value: str) -> str:
    normalized = str(status_value or "").strip().lower()
    if normalized in {"completed", "trained", "success"}:
        return "completed"
    if normalized in {"queued", "running"}:
        return normalized
    if normalized in {"failed", "error"}:
        return "failed"
    return "queued"


def _workspace_summary(workspace_id: int) -> dict[str, int]:
    images = _workspace_service().list_image_assets(workspace_id)
    datasets = _dataset_service().list_draft_datasets(workspace_id)
    snapshots = _workspace_snapshot_rows(workspace_id)
    analysis = _workspace_analysis_rows(workspace_id)
    return {
        "images": len(images),
        "datasets": len(datasets),
        "snapshots": len(snapshots),
        "analysis_runs": len(analysis["prediction_runs"]),
    }


def _workspace_snapshot_rows(workspace_id: int) -> list[dict[str, object]]:
    dataset_service = _dataset_service()
    snapshot_service = _snapshot_service()
    rows: list[dict[str, object]] = []
    for dataset in dataset_service.list_draft_datasets(workspace_id):
        for snapshot in snapshot_service.list_snapshots_for_dataset(dataset.id):
            rows.append({"snapshot": snapshot, "dataset": dataset})
    rows.sort(key=lambda item: (item["snapshot"].created_at, item["snapshot"].id), reverse=True)
    return rows


def _workspace_analysis_rows(workspace_id: int) -> dict[str, list[dict[str, object]]]:
    model_service = _model_service()
    training_service = _training_service()
    prediction_service = _prediction_service()
    model_lookup = {model.id: model for model in model_service.list_model_definitions(workspace_id)}
    snapshots = _workspace_snapshot_rows(workspace_id)
    training_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for row in snapshots:
        snapshot = row["snapshot"]
        dataset = row["dataset"]
        for training_run in training_service.list_training_runs_for_snapshot(snapshot.id):
            training_rows.append(
                {
                    "run": training_run,
                    "snapshot": snapshot,
                    "dataset": dataset,
                    "model": model_lookup.get(training_run.model_definition_id),
                }
            )
        for prediction_run in prediction_service.list_prediction_runs_for_snapshot(snapshot.id):
            prediction_rows.append(
                {
                    "run": prediction_run,
                    "snapshot": snapshot,
                    "dataset": dataset,
                    "model": model_lookup.get(prediction_run.model_definition_id),
                }
            )
    training_rows.sort(key=lambda item: (item["run"].updated_at, item["run"].id), reverse=True)
    prediction_rows.sort(key=lambda item: (item["run"].updated_at, item["run"].id), reverse=True)
    return {"training_runs": training_rows, "prediction_runs": prediction_rows}


def _artifact_file_absolute_path(relative_path: str) -> Path:
    artifacts_root = Path(current_app.config["ARTIFACTS_DIR"]).resolve()
    target = (artifacts_root / relative_path).resolve()
    try:
        target.relative_to(artifacts_root)
    except ValueError:
        abort(404)
    if not target.exists() or not target.is_file():
        abort(404)
    return target


def _safe_workspace_redirect(workspace_id: int, next_url: str | None) -> str:
    candidate = str(next_url or "").strip()
    if candidate.startswith("/") and not candidate.startswith("//"):
        pattern = re.compile(
            r"^/workspaces/\d+/(images/library|datasets/library|snapshots/library|models/library|analysis)$"
        )
        if pattern.match(candidate):
            return re.sub(r"^/workspaces/\d+", "/workspaces/%s" % workspace_id, candidate, count=1)
        if re.match(r"^/workspaces/\d+$", candidate):
            return re.sub(r"^/workspaces/\d+", "/workspaces/%s" % workspace_id, candidate, count=1)
    return url_for("v2.workspace_detail", workspace_id=workspace_id)


def _model_config_preview(model_definition) -> list[tuple[str, object]]:
    config = _model_service().get_model_config(model_definition.id)
    preview_keys = [
        "epochs",
        "batch_size",
        "target_height",
        "target_width",
        "base_filters",
        "trained_family_id",
        "training_runtime",
        "latest_training_run_id",
    ]
    preview: list[tuple[str, object]] = []
    for key in preview_keys:
        if key in config:
            preview.append((key, config[key]))
    return preview


@bp.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok", "app": current_app.config["APP_NAME"]})


@bp.route("/favicon.ico", methods=["GET"])
def favicon():
    return make_response("", 204)


@bp.route("/assets/<path:filename>", methods=["GET"])
def asset_file(filename: str):
    return send_from_directory(Path(__file__).parent / "static", filename)


@bp.route("/artifacts/files/<path:relative_path>", methods=["GET"])
def artifact_file(relative_path: str):
    return send_file(_artifact_file_absolute_path(relative_path))


@bp.route("/", methods=["GET"])
def index():
    workspace_service = _workspace_service()
    return render_template(
        "index.html",
        workspaces=workspace_service.list_workspaces(),
    )


@bp.route("/workspaces/switch", methods=["POST"])
def switch_workspace():
    workspace_service = _workspace_service()
    workspace_id = int(request.form.get("workspace_id") or "0")
    workspace = workspace_service.get_workspace(workspace_id)
    if workspace is None:
        flash("Unknown workspace selection.", "error")
        return redirect(url_for("v2.index"))
    session["v2_workspace_id"] = workspace_id
    return redirect(_safe_workspace_redirect(workspace_id, request.form.get("next_url")))


@bp.route("/workspaces", methods=["POST"])
def create_workspace():
    workspace_service = _workspace_service()
    name = (request.form.get("name") or "").strip()
    description = (request.form.get("description") or "").strip() or None
    if not name:
        flash("Workspace name is required.", "error")
        return redirect(url_for("v2.index"))
    try:
        workspace = workspace_service.create_workspace(name=name, description=description)
    except Exception as exc:
        flash(str(exc), "error")
        return redirect(url_for("v2.index"))
    flash(f"Created workspace '{workspace.name}'.", "success")
    return redirect(url_for("v2.workspace_detail", workspace_id=workspace.id))


@bp.route("/workspaces/<int:workspace_id>", methods=["GET"])
def workspace_detail(workspace_id: int):
    workspace_service = _workspace_service()
    dataset_service = _dataset_service()
    model_service = _model_service()
    workspace = workspace_service.get_workspace(workspace_id)
    if workspace is None:
        abort(404)
    images = workspace_service.list_image_assets(workspace_id)
    datasets = dataset_service.list_draft_datasets(workspace_id)
    return render_template(
        "workspace_detail.html",
        workspace=workspace,
        summary=_workspace_summary(workspace_id),
        recent_images=images[:5],
        recent_datasets=datasets[:5],
        recent_models=model_service.list_model_definitions(workspace_id)[:5],
        recent_snapshots=_workspace_snapshot_rows(workspace_id)[:5],
        recent_predictions=_workspace_analysis_rows(workspace_id)["prediction_runs"][:5],
    )


@bp.route("/workspaces/<int:workspace_id>/images/library", methods=["GET"])
def workspace_images_page(workspace_id: int):
    workspace = _workspace_service().get_workspace(workspace_id)
    if workspace is None:
        abort(404)
    return render_template(
        "workspace_images.html",
        workspace=workspace,
        images=_workspace_service().list_image_assets(workspace_id),
    )


@bp.route("/workspaces/<int:workspace_id>/datasets/library", methods=["GET"])
def workspace_datasets_page(workspace_id: int):
    workspace = _workspace_service().get_workspace(workspace_id)
    if workspace is None:
        abort(404)
    images = _workspace_service().list_image_assets(workspace_id)
    return render_template(
        "workspace_datasets.html",
        workspace=workspace,
        datasets=_dataset_service().list_draft_datasets(workspace_id),
        images=images,
    )


@bp.route("/workspaces/<int:workspace_id>/snapshots/library", methods=["GET"])
def workspace_snapshots_page(workspace_id: int):
    workspace = _workspace_service().get_workspace(workspace_id)
    if workspace is None:
        abort(404)
    return render_template(
        "workspace_snapshots.html",
        workspace=workspace,
        snapshot_rows=_workspace_snapshot_rows(workspace_id),
    )


@bp.route("/workspaces/<int:workspace_id>/models/library", methods=["GET"])
def workspace_models_page(workspace_id: int):
    workspace = _workspace_service().get_workspace(workspace_id)
    if workspace is None:
        abort(404)
    models = _model_service().list_model_definitions(workspace_id)
    return render_template(
        "workspace_models.html",
        workspace=workspace,
        models=models,
        model_family_ids=_model_registry().list_family_ids(),
        model_previews={model.id: _model_config_preview(model) for model in models},
    )


@bp.route("/workspaces/<int:workspace_id>/analysis", methods=["GET"])
def workspace_analysis_page(workspace_id: int):
    workspace = _workspace_service().get_workspace(workspace_id)
    if workspace is None:
        abort(404)
    analysis_rows = _workspace_analysis_rows(workspace_id)
    return render_template(
        "workspace_analysis.html",
        workspace=workspace,
        training_rows=analysis_rows["training_runs"],
        prediction_rows=analysis_rows["prediction_runs"],
    )


@bp.route("/workspaces/<int:workspace_id>/images", methods=["POST"])
def register_image_asset(workspace_id: int):
    workspace_service = _workspace_service()
    source_path = (request.form.get("source_path") or "").strip()
    if not source_path:
        flash("Image path is required.", "error")
        return redirect(url_for("v2.workspace_detail", workspace_id=workspace_id))
    try:
        image = workspace_service.register_image_asset(workspace_id=workspace_id, source_path=source_path)
    except Exception as exc:
        flash(str(exc), "error")
        return redirect(url_for("v2.workspace_images_page", workspace_id=workspace_id))
    flash(f"Registered image asset #{image.id}.", "success")
    return redirect(url_for("v2.workspace_images_page", workspace_id=workspace_id))


@bp.route("/workspaces/<int:workspace_id>/datasets", methods=["POST"])
def create_draft_dataset(workspace_id: int):
    dataset_service = _dataset_service()
    name = (request.form.get("name") or "").strip()
    description = (request.form.get("description") or "").strip() or None
    image_ids = [int(value) for value in request.form.getlist("image_ids")]
    if not name:
        flash("Dataset name is required.", "error")
        return redirect(url_for("v2.workspace_datasets_page", workspace_id=workspace_id))
    try:
        dataset = dataset_service.create_draft_dataset(
            workspace_id=workspace_id,
            name=name,
            description=description,
            image_ids=image_ids,
        )
    except Exception as exc:
        flash(str(exc), "error")
        return redirect(url_for("v2.workspace_datasets_page", workspace_id=workspace_id))
    flash(f"Created draft dataset '{dataset.name}'.", "success")
    return redirect(url_for("v2.dataset_detail", dataset_id=dataset.id))


@bp.route("/workspaces/<int:workspace_id>/models", methods=["POST"])
def create_model_definition(workspace_id: int):
    model_service = _model_service()
    name = (request.form.get("name") or "").strip()
    family_id = (request.form.get("family_id") or "").strip()
    if not name or not family_id:
        flash("Model name and family are required.", "error")
        return redirect(url_for("v2.workspace_models_page", workspace_id=workspace_id))
    try:
        model = model_service.create_model_definition(
            workspace_id=workspace_id,
            name=name,
            family_id=family_id,
            config={},
        )
    except Exception as exc:
        flash(str(exc), "error")
        return redirect(url_for("v2.workspace_models_page", workspace_id=workspace_id))
    flash(f"Created model definition '{model.name}'.", "success")
    return redirect(url_for("v2.model_detail", model_definition_id=model.id))


@bp.route("/models/<int:model_definition_id>", methods=["GET"])
def model_detail(model_definition_id: int):
    model_definition = _model_service().get_model_definition(model_definition_id)
    if model_definition is None:
        abort(404)
    model_config = _model_service().get_model_config(model_definition_id)
    latest_training_run = None
    latest_training_run_id = model_config.get("latest_training_run_id")
    if isinstance(latest_training_run_id, int):
        latest_training_run = _training_service().get_training_run(latest_training_run_id)
    return render_template(
        "model_detail.html",
        model_definition=model_definition,
        model_config=model_config,
        model_config_json=json.dumps(model_config, indent=2, sort_keys=True),
        latest_training_run=latest_training_run,
    )


@bp.route("/models/<int:model_definition_id>/config", methods=["POST"])
def update_model_definition_config(model_definition_id: int):
    raw_config = (request.form.get("config_json") or "").strip()
    try:
        parsed = json.loads(raw_config or "{}")
        if not isinstance(parsed, dict):
            raise ValueError("Model config JSON must decode to an object.")
        _model_service().update_model_config(model_definition_id, parsed)
    except Exception as exc:
        flash(str(exc), "error")
        return redirect(url_for("v2.model_detail", model_definition_id=model_definition_id))
    flash("Updated model definition config.", "success")
    return redirect(url_for("v2.model_detail", model_definition_id=model_definition_id))


@bp.route("/datasets/<int:dataset_id>", methods=["GET"])
def dataset_detail(dataset_id: int):
    dataset_service = _dataset_service()
    annotation_service = _annotation_service()
    snapshot_service = _snapshot_service()
    dataset = dataset_service.get_draft_dataset(dataset_id)
    if dataset is None:
        abort(404)
    classes = dataset_service.list_dataset_classes(dataset_id)
    images = dataset_service.list_dataset_images(dataset_id)
    image_rows: list[dict[str, object]] = []
    for image in images:
        head = annotation_service.load_head_revision(dataset_id, image.id)
        image_rows.append(
            {
                "image": image,
                "head_revision": head.revision if head is not None else None,
            }
        )
    return render_template(
        "dataset_detail.html",
        dataset=dataset,
        classes=classes,
        image_rows=image_rows,
        snapshots=snapshot_service.list_snapshots_for_dataset(dataset_id),
        app_name=current_app.config["APP_NAME"],
    )


@bp.route("/datasets/<int:dataset_id>/classes", methods=["POST"])
def add_dataset_class(dataset_id: int):
    dataset_service = _dataset_service()
    name = (request.form.get("name") or "").strip()
    color = (request.form.get("color") or "#ff0000").strip()
    if not name:
        flash("Class name is required.", "error")
        return redirect(url_for("v2.dataset_detail", dataset_id=dataset_id))
    try:
        created = dataset_service.add_class(dataset_id=dataset_id, name=name, color=color)
    except Exception as exc:
        flash(str(exc), "error")
        return redirect(url_for("v2.dataset_detail", dataset_id=dataset_id))
    flash(f"Added class '{created.name}' with class index {created.class_index}.", "success")
    return redirect(url_for("v2.dataset_detail", dataset_id=dataset_id))


@bp.route("/datasets/<int:dataset_id>/images/<int:image_id>/initialize-revision", methods=["POST"])
def initialize_revision(dataset_id: int, image_id: int):
    dataset_service = _dataset_service()
    try:
        revision_id = dataset_service.initialize_annotation_for_image(dataset_id, image_id)
    except Exception as exc:
        flash(str(exc), "error")
        return redirect(url_for("v2.dataset_detail", dataset_id=dataset_id))
    flash(f"Annotation head is ready at revision {revision_id}.", "success")
    return redirect(url_for("v2.annotation_detail", dataset_id=dataset_id, image_id=image_id))


@bp.route("/datasets/<int:dataset_id>/images/<int:image_id>/fork-head", methods=["POST"])
def fork_head_revision(dataset_id: int, image_id: int):
    annotation_service = _annotation_service()
    try:
        revision = annotation_service.fork_head_revision(
            draft_dataset_id=dataset_id,
            image_id=image_id,
            author="ui",
            operation_summary="Fork annotation head after snapshot lock",
        )
    except Exception as exc:
        flash(str(exc), "error")
        return redirect(url_for("v2.dataset_detail", dataset_id=dataset_id))
    flash(f"Forked new editable head revision {revision.id}.", "success")
    return redirect(url_for("v2.annotation_detail", dataset_id=dataset_id, image_id=image_id))


@bp.route("/datasets/<int:dataset_id>/snapshots", methods=["POST"])
def register_snapshot(dataset_id: int):
    snapshot_service = _snapshot_service()
    name = (request.form.get("name") or "").strip() or None
    try:
        snapshot = snapshot_service.register_snapshot(draft_dataset_id=dataset_id, name=name)
    except Exception as exc:
        flash(str(exc), "error")
        return redirect(url_for("v2.dataset_detail", dataset_id=dataset_id))
    flash(f"Registered immutable snapshot #{snapshot.id}.", "success")
    return redirect(url_for("v2.dataset_detail", dataset_id=dataset_id))


@bp.route("/snapshots/<int:snapshot_id>", methods=["GET"])
def snapshot_detail(snapshot_id: int):
    snapshot_service = _snapshot_service()
    artifact_service = _artifact_service()
    workspace_service = _workspace_service()
    model_service = _model_service()
    training_service = _training_service()
    prediction_service = _prediction_service()
    snapshot = snapshot_service.get_snapshot(snapshot_id)
    if snapshot is None:
        abort(404)
    items = []
    for item in snapshot_service.list_snapshot_items(snapshot_id):
        image = workspace_service.get_image_asset(item.image_id)
        items.append({"item": item, "image": image})
    artifacts = artifact_service.list_artifacts("registered_snapshot", snapshot_id)
    builder_registry = _snapshot_builders()
    source_dataset = _dataset_service().get_draft_dataset(snapshot.draft_dataset_id)
    model_lookup = {model.id: model for model in model_service.list_model_definitions(snapshot.workspace_id)}
    training_rows = [
        {"run": run, "model": model_lookup.get(run.model_definition_id)}
        for run in training_service.list_training_runs_for_snapshot(snapshot_id)
    ]
    prediction_rows = [
        {"run": run, "model": model_lookup.get(run.model_definition_id)}
        for run in prediction_service.list_prediction_runs_for_snapshot(snapshot_id)
    ]
    return render_template(
        "snapshot_detail.html",
        snapshot=snapshot,
        source_dataset=source_dataset,
        snapshot_items=items,
        artifacts=artifacts,
        builders=builder_registry.list_builders(),
        model_definitions=list(model_lookup.values()),
        training_rows=training_rows,
        prediction_rows=prediction_rows,
        class_schema=_safe_json_loads(snapshot.class_schema_json),
    )


@bp.route("/snapshots/<int:snapshot_id>/builds", methods=["POST"])
def build_snapshot_artifact(snapshot_id: int):
    snapshot_service = _snapshot_service()
    snapshot = snapshot_service.get_snapshot(snapshot_id)
    if snapshot is None:
        abort(404)
    builder_id = (request.form.get("builder_id") or "").strip()
    builder = _snapshot_builders().get(builder_id)
    if builder is None:
        flash("Unknown snapshot builder.", "error")
        return redirect(url_for("v2.snapshot_detail", snapshot_id=snapshot_id))
    try:
        result = builder.build(snapshot)
    except Exception as exc:
        flash(str(exc), "error")
        return redirect(url_for("v2.snapshot_detail", snapshot_id=snapshot_id))
    flash(f"Built snapshot artifact via '{result.builder_id}'.", "success")
    return redirect(url_for("v2.snapshot_detail", snapshot_id=snapshot_id))


@bp.route("/snapshots/<int:snapshot_id>/prediction-runs", methods=["POST"])
def create_prediction_run(snapshot_id: int):
    prediction_service = _prediction_service()
    model_definition_id = int(request.form.get("model_definition_id") or "0")
    try:
        prediction_run = prediction_service.create_prediction_run(
            snapshot_id=snapshot_id,
            model_definition_id=model_definition_id,
        )
    except Exception as exc:
        flash(str(exc), "error")
        return redirect(url_for("v2.snapshot_detail", snapshot_id=snapshot_id))
    flash(f"Created prediction run #{prediction_run.id}.", "success")
    return redirect(url_for("v2.prediction_run_detail", prediction_run_id=prediction_run.id))


@bp.route("/snapshots/<int:snapshot_id>/training-runs", methods=["POST"])
def create_training_run(snapshot_id: int):
    training_service = _training_service()
    model_definition_id = int(request.form.get("model_definition_id") or "0")
    try:
        training_run = training_service.start_training_run(
            snapshot_id=snapshot_id,
            model_definition_id=model_definition_id,
        )
    except Exception as exc:
        flash(str(exc), "error")
        return redirect(url_for("v2.snapshot_detail", snapshot_id=snapshot_id))
    flash(f"Created training run #{training_run.id}.", "success")
    return redirect(url_for("v2.training_run_detail", training_run_id=training_run.id))


@bp.route("/training-runs/<int:training_run_id>", methods=["GET"])
def training_run_detail(training_run_id: int):
    training_service = _training_service()
    snapshot_service = _snapshot_service()
    model_service = _model_service()
    training_run = training_service.get_training_run(training_run_id)
    if training_run is None:
        abort(404)
    snapshot = snapshot_service.get_snapshot(training_run.registered_snapshot_id)
    model_definition = model_service.get_model_definition(training_run.model_definition_id)
    artifacts = training_service.list_training_artifacts(training_run_id)
    return render_template(
        "training_run_detail.html",
        training_run=training_run,
        snapshot=snapshot,
        model_definition=model_definition,
        model_config=_model_service().get_model_config(model_definition.id) if model_definition is not None else {},
        artifacts=artifacts,
    )


@bp.route("/prediction-runs/<int:prediction_run_id>", methods=["GET"])
def prediction_run_detail(prediction_run_id: int):
    prediction_service = _prediction_service()
    snapshot_service = _snapshot_service()
    artifact_service = _artifact_service()
    prediction_run = prediction_service.get_prediction_run(prediction_run_id)
    if prediction_run is None:
        abort(404)
    snapshot = snapshot_service.get_snapshot(prediction_run.registered_snapshot_id)
    model_definition = prediction_service.get_model_definition_for_run(prediction_run_id)
    artifacts = artifact_service.list_artifacts("prediction_run", prediction_run_id)
    selected_export_source_type = (request.args.get("source_type") or "refined").strip().lower()
    if selected_export_source_type not in {"raw", "refined"}:
        selected_export_source_type = "refined"
    return render_template(
        "prediction_run_detail.html",
        prediction_run=prediction_run,
        snapshot=snapshot,
        model_definition=model_definition,
        artifacts=artifacts,
        selected_export_source_type=selected_export_source_type,
        promote_refined_url=url_for("v2.promote_prediction_run", prediction_run_id=prediction_run.id, source_type="refined"),
        promote_raw_url=url_for("v2.promote_prediction_run", prediction_run_id=prediction_run.id, source_type="raw"),
    )


@bp.route("/prediction-runs/<int:prediction_run_id>/exports", methods=["POST"])
def export_prediction_run(prediction_run_id: int):
    export_service = _export_service()
    source_type = (request.form.get("source_type") or "refined").strip()
    try:
        export_service.export_prediction_run(
            prediction_run_id=prediction_run_id,
            export_config={
                "source_type": source_type,
                "threshold_settings": {},
                "refinement_settings": {},
            },
        )
    except Exception as exc:
        flash(str(exc), "error")
        return redirect(url_for("v2.prediction_run_detail", prediction_run_id=prediction_run_id))
    flash(f"Exported {source_type} prediction artifacts.", "success")
    return redirect(url_for("v2.prediction_run_detail", prediction_run_id=prediction_run_id, source_type=source_type))


@bp.route("/prediction-runs/<int:prediction_run_id>/promote/<source_type>", methods=["POST"])
def promote_prediction_run(prediction_run_id: int, source_type: str):
    promotion_service = _promotion_service()
    try:
        dataset = promotion_service.promote_prediction_run(
            prediction_run_id=prediction_run_id,
            source_type=source_type,
        )
    except Exception as exc:
        flash(str(exc), "error")
        return redirect(url_for("v2.prediction_run_detail", prediction_run_id=prediction_run_id))
    flash(f"Promoted {source_type} prediction into draft dataset '{dataset.name}'.", "success")
    return redirect(url_for("v2.dataset_detail", dataset_id=dataset.id))


@bp.route("/datasets/<int:dataset_id>/images/<int:image_id>/annotation", methods=["GET"])
def annotation_detail(dataset_id: int, image_id: int):
    dataset_service = _dataset_service()
    annotation_service = _annotation_service()
    dataset = dataset_service.get_draft_dataset(dataset_id)
    image = _workspace_service().get_image_asset(image_id)
    if dataset is None or image is None:
        abort(404)
    revision_data = annotation_service.load_head_revision(dataset_id, image_id)
    if revision_data is None:
        flash("Initialize an annotation head before viewing annotation details.", "error")
        return redirect(url_for("v2.dataset_detail", dataset_id=dataset_id))
    class_payload = [
        {
            "id": dataset_class.id,
            "class_index": dataset_class.class_index,
            "name": dataset_class.name,
            "color": dataset_class.color,
        }
        for dataset_class in dataset_service.list_dataset_classes(dataset_id)
    ]
    return render_template(
        "annotation_detail.html",
        dataset=dataset,
        image=image,
        classes=class_payload,
        revision=revision_data.revision,
        api_url=url_for("v2.annotation_api", dataset_id=dataset_id, image_id=image_id),
        image_url=url_for("v2.image_source", image_id=image_id),
        fork_url=url_for("v2.fork_head_revision", dataset_id=dataset_id, image_id=image_id),
        app_name=current_app.config["APP_NAME"],
    )


@bp.route("/images/<int:image_id>/source", methods=["GET"])
def image_source(image_id: int):
    image = _workspace_service().get_image_asset(image_id)
    if image is None:
        abort(404)
    source_path = Path(image.source_path)
    if not source_path.exists():
        abort(404)
    return send_file(source_path)


@bp.route("/api/datasets/<int:dataset_id>/images/<int:image_id>/annotation", methods=["GET"])
def annotation_api(dataset_id: int, image_id: int):
    annotation_service = _annotation_service()
    revision_data = annotation_service.load_head_revision(dataset_id, image_id)
    if revision_data is None:
        return jsonify({"error": "annotation head not initialized"}), 404
    return jsonify(_revision_payload(revision_data))


@bp.route("/api/datasets/<int:dataset_id>/images/<int:image_id>/annotation", methods=["POST"])
def save_annotation_api(dataset_id: int, image_id: int):
    annotation_service = _annotation_service()
    payload = request.get_json(silent=True) or {}
    revision_data = annotation_service.load_head_revision(dataset_id, image_id)
    if revision_data is None:
        revision = annotation_service.ensure_head_revision(
            draft_dataset_id=dataset_id,
            image_id=image_id,
            author=str(payload.get("author") or "api"),
            operation_summary="Initialize empty annotation revision via API",
        )
        revision_data = annotation_service.load_revision(revision.id)
    try:
        saved = annotation_service.save_revision(
            revision_id=revision_data.revision.id,
            label_map=payload.get("label_map", revision_data.label_map),
            provenance_map=payload.get("provenance_map", revision_data.provenance_map),
            protection_map=payload.get("protection_map", revision_data.protection_map),
            author=str(payload.get("author") or "api"),
            operation_summary=str(payload.get("operation_summary") or "Saved via annotation API"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify(_revision_payload(annotation_service.load_revision(saved.id)))


def _revision_payload(revision_data: object) -> dict[str, object]:
    revision = revision_data.revision
    return {
        "revision": {
            "id": revision.id,
            "draft_dataset_id": revision.draft_dataset_id,
            "image_id": revision.image_id,
            "parent_revision_id": revision.parent_revision_id,
            "width": revision.width,
            "height": revision.height,
            "label_checksum": revision.label_checksum,
            "provenance_checksum": revision.provenance_checksum,
            "protection_checksum": revision.protection_checksum,
            "revision_checksum": revision.revision_checksum,
            "author": revision.author,
            "operation_summary": revision.operation_summary,
            "is_locked": revision.is_locked,
            "created_at": revision.created_at,
            "updated_at": revision.updated_at,
        },
        "label_map": revision_data.label_map.tolist(),
        "provenance_map": revision_data.provenance_map.tolist(),
        "protection_map": revision_data.protection_map.tolist(),
    }


def _safe_json_loads(raw: str) -> object:
    return json.loads(raw)
