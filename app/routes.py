"""Flask routes for dashboard, datasets, training, and analysis runs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from .services.coco import compute_sha256, load_coco, parse_categories
from .services.inference import run_analysis
from .services.schemas import DatasetSpec, FeatureConfig, TrainConfig
from .services.storage import Storage
from .services.training import train_model

bp = Blueprint("main", __name__)


def _storage() -> Storage:
    return current_app.extensions["storage"]


def _base_dir() -> Path:
    return Path(current_app.config["BASE_DIR"]).resolve()


def _to_relative_under_base(path_value: str) -> str | None:
    base = _base_dir()
    path = Path(path_value)
    if not path.is_absolute():
        return str(path).replace("\\", "/")
    try:
        return path.resolve().relative_to(base).as_posix()
    except ValueError:
        return None


def _collect_input_images(form: dict[str, Any]) -> list[str]:
    allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    collected: list[str] = []

    image_paths_raw = str(form.get("image_paths", "")).strip()
    if image_paths_raw:
        for line in image_paths_raw.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            collected.append(str(Path(candidate).expanduser()))

    image_folder = str(form.get("image_folder", "")).strip()
    glob_pattern = str(form.get("glob_pattern", "*.png")).strip() or "*.png"
    if image_folder:
        folder = Path(image_folder).expanduser()
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"Image folder not found: {folder}")
        for candidate in sorted(folder.glob(glob_pattern)):
            if candidate.is_file() and candidate.suffix.lower() in allowed_extensions:
                collected.append(str(candidate))

    deduped: list[str] = []
    seen: set[str] = set()
    for path in collected:
        resolved = str(Path(path).resolve())
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)
    return deduped


@bp.app_template_filter("prettyjson")
def pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


@bp.app_context_processor
def inject_template_helpers() -> dict[str, Any]:
    def file_url(path_value: str | None) -> str | None:
        if not path_value:
            return None
        relative = _to_relative_under_base(path_value)
        if relative is None:
            return None
        return url_for("main.serve_file", relative_path=relative)

    return {"file_url": file_url}


@bp.route("/", methods=["GET"])
def dashboard() -> str:
    storage = _storage()
    models = storage.list_models()
    runs = storage.list_runs(limit=50)
    datasets = storage.list_datasets()
    return render_template(
        "dashboard.html",
        models=models,
        runs=runs,
        datasets=datasets,
        default_seed=current_app.config["RANDOM_SEED"],
    )


@bp.route("/datasets/new", methods=["GET", "POST"])
def register_dataset() -> str:
    if request.method == "GET":
        return render_template("dataset_new.html")

    name = str(request.form.get("name", "")).strip()
    dataset_path_raw = str(request.form.get("dataset_path", "")).strip()
    image_root = str(request.form.get("image_root", "images")).strip() or "images"
    coco_json_raw = str(request.form.get("coco_json_path", "annotations.json")).strip()
    coco_json_value = coco_json_raw or "annotations.json"

    if not name or not dataset_path_raw:
        flash("Dataset name and folder path are required.", "error")
        return render_template("dataset_new.html"), 400

    dataset_path = Path(dataset_path_raw).expanduser().resolve()
    if not dataset_path.exists() or not dataset_path.is_dir():
        flash(f"Dataset folder not found: {dataset_path}", "error")
        return render_template("dataset_new.html"), 400

    coco_path = Path(coco_json_value)
    if not coco_path.is_absolute():
        coco_path = (dataset_path / coco_path).resolve()
    if not coco_path.exists() or not coco_path.is_file():
        flash(f"COCO annotations file not found: {coco_path}", "error")
        return render_template("dataset_new.html"), 400

    try:
        coco = load_coco(coco_path)
        categories = parse_categories(coco)
        checksum = compute_sha256(coco_path)
    except Exception as exc:
        flash(f"Failed to parse COCO file: {exc}", "error")
        return render_template("dataset_new.html"), 400

    spec = DatasetSpec(
        name=name,
        dataset_path=str(dataset_path),
        image_root=image_root,
        coco_json_path=str(coco_path),
    )
    try:
        dataset_id = _storage().create_dataset(
            spec,
            coco_checksum=checksum,
            categories=categories,
        )
    except sqlite3.IntegrityError:
        flash(f"A dataset named '{name}' already exists.", "error")
        return render_template("dataset_new.html"), 400

    flash(
        f"Dataset '{name}' registered (id={dataset_id}, categories={len(categories)}).",
        "success",
    )
    return redirect(url_for("main.dashboard"))


@bp.route("/models/train", methods=["POST"])
def train_new_model() -> str:
    storage = _storage()
    model_name = str(request.form.get("model_name", "")).strip()
    dataset_id_raw = str(request.form.get("dataset_id", "")).strip()
    if not model_name or not dataset_id_raw:
        flash("Model name and dataset are required.", "error")
        return redirect(url_for("main.dashboard"))

    try:
        dataset_id = int(dataset_id_raw)
    except ValueError:
        flash("Invalid dataset selection.", "error")
        return redirect(url_for("main.dashboard"))

    dataset = storage.get_dataset(dataset_id)
    if dataset is None:
        flash("Selected dataset does not exist.", "error")
        return redirect(url_for("main.dashboard"))

    logs: list[str] = []

    def log(message: str) -> None:
        logs.append(message)

    try:
        feature_config = FeatureConfig.from_form(request.form)
        train_config = TrainConfig.from_form(
            request.form,
            default_seed=int(current_app.config["RANDOM_SEED"]),
        )
        result = train_model(
            model_name=model_name,
            dataset=dataset,
            feature_config=feature_config,
            train_config=train_config,
            models_dir=Path(current_app.config["MODELS_DIR"]),
            code_version=str(current_app.config["CODE_VERSION"]),
            log_fn=log,
        )
        model_id = storage.create_model(
            name=model_name,
            dataset_id=dataset_id,
            classes=result.classes,
            feature_config=result.feature_config,
            hyperparams=result.hyperparams,
            metrics=result.metrics,
            artifact_dir=result.artifact_dir,
            model_path=result.model_path,
            metadata_path=result.metadata_path,
            status="trained",
            error_message=None,
        )
    except Exception as exc:
        logs.append(f"Training failed: {exc}")
        flash(f"Training failed: {exc}", "error")
        return render_template(
            "train_status.html",
            success=False,
            model_id=None,
            logs=logs,
            warnings=[],
            error=str(exc),
        ), 500

    if result.warnings:
        flash(f"Training completed with {len(result.warnings)} warning(s).", "warning")
    else:
        flash("Training completed successfully.", "success")
    return render_template(
        "train_status.html",
        success=True,
        model_id=model_id,
        logs=logs,
        warnings=result.warnings,
        error=None,
    )


@bp.route("/models/<int:model_id>", methods=["GET"])
def model_details(model_id: int) -> str:
    model = _storage().get_model(model_id)
    if model is None:
        abort(404, description=f"Model id={model_id} not found.")
    return render_template("model_detail.html", model=model)


@bp.route("/analysis/new", methods=["GET", "POST"])
def new_analysis() -> str:
    storage = _storage()
    models = storage.list_models()

    if request.method == "GET":
        selected_model_id = request.args.get("model_id", type=int)
        return render_template(
            "analysis_new.html",
            models=models,
            selected_model_id=selected_model_id,
        )

    model_id_raw = str(request.form.get("model_id", "")).strip()
    try:
        model_id = int(model_id_raw)
    except ValueError:
        flash("Please select a valid trained model.", "error")
        return render_template("analysis_new.html", models=models, selected_model_id=None), 400

    model = storage.get_model(model_id)
    if model is None:
        flash("Selected model does not exist.", "error")
        return render_template("analysis_new.html", models=models, selected_model_id=None), 400

    try:
        input_images = _collect_input_images(request.form)
    except Exception as exc:
        flash(f"Failed to collect input images: {exc}", "error")
        return render_template(
            "analysis_new.html",
            models=models,
            selected_model_id=model_id,
        ), 400
    if not input_images:
        flash("Provide at least one image path or folder+glob pattern.", "error")
        return render_template(
            "analysis_new.html",
            models=models,
            selected_model_id=model_id,
        ), 400

    run_id = storage.create_analysis_run(model_id=model_id, input_images=input_images)
    logs: list[str] = []

    def log(message: str) -> None:
        logs.append(message)

    try:
        result = run_analysis(
            run_id=run_id,
            model_record=model,
            image_paths=input_images,
            runs_dir=Path(current_app.config["RUNS_DIR"]),
            base_dir=_base_dir(),
            random_seed=int(current_app.config["RANDOM_SEED"]),
            log_fn=log,
        )
        for item in result.items:
            storage.add_analysis_item(
                run_id=run_id,
                input_image=item.input_image,
                mask_path=item.mask_path,
                overlay_path=item.overlay_path,
                summary=item.summary,
            )
        storage.update_analysis_run(
            run_id,
            output_dir=result.output_dir,
            summary=result.summary,
            status="completed",
            error_message=None,
        )
    except Exception as exc:
        storage.update_analysis_run(
            run_id,
            output_dir="",
            summary={"logs": logs},
            status="failed",
            error_message=str(exc),
        )
        flash(f"Analysis failed: {exc}", "error")
        return redirect(url_for("main.new_analysis", model_id=model_id))

    if result.warnings:
        flash(f"Analysis completed with {len(result.warnings)} warning(s).", "warning")
    else:
        flash("Analysis completed successfully.", "success")
    return redirect(url_for("main.analysis_details", run_id=run_id))


@bp.route("/analysis/<int:run_id>", methods=["GET"])
def analysis_details(run_id: int) -> str:
    storage = _storage()
    run = storage.get_analysis_run(run_id)
    if run is None:
        abort(404, description=f"Analysis run id={run_id} not found.")

    items = storage.get_analysis_items(run_id)
    model = storage.get_model(int(run["model_id"]))
    return render_template("analysis_detail.html", run=run, items=items, model=model)


@bp.route("/files/<path:relative_path>", methods=["GET"])
def serve_file(relative_path: str):
    base = _base_dir()
    target = (base / relative_path).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        abort(404)

    if not target.exists() or not target.is_file():
        abort(404)
    return send_file(target)
