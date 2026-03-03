"""Flask application factory for the local segmentation app."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, request, session, url_for

from .services.job_queue import JobQueueManager
from .services.storage import Storage


def create_app() -> Flask:
    """Create and configure the Flask application."""
    package_root = Path(__file__).resolve().parent
    base_dir = Path(os.getenv("SEG_APP_BASE_DIR", package_root.parent)).resolve()
    instance_dir = Path(
        os.getenv("SEG_APP_INSTANCE_DIR", str(base_dir / "instance"))
    ).resolve()
    models_dir = Path(os.getenv("SEG_APP_MODELS_DIR", str(base_dir / "models"))).resolve()
    runs_dir = Path(os.getenv("SEG_APP_RUNS_DIR", str(base_dir / "runs"))).resolve()
    labeler_dir = Path(
        os.getenv("SEG_APP_LABELER_DIR", str(base_dir / "workspace" / "labeler"))
    ).resolve()
    labeler_images_dir = Path(
        os.getenv("SEG_APP_LABELER_IMAGES_DIR", str(labeler_dir / "images"))
    ).resolve()
    labeler_masks_dir = Path(
        os.getenv("SEG_APP_LABELER_MASKS_DIR", str(labeler_dir / "masks"))
    ).resolve()
    labeler_coco_dir = Path(
        os.getenv("SEG_APP_LABELER_COCO_DIR", str(labeler_dir / "coco"))
    ).resolve()
    labeler_cache_dir = Path(
        os.getenv("SEG_APP_LABELER_CACHE_DIR", str(labeler_masks_dir / "_cache"))
    ).resolve()
    db_path = Path(
        os.getenv("SEG_APP_DB_PATH", str(instance_dir / "segmentation.sqlite3"))
    ).resolve()
    labeler_categories = [
        item.strip()
        for item in os.getenv("SEG_APP_LABELER_CATEGORIES", "class_1,class_2,class_3").split(",")
        if item.strip()
    ]
    if not labeler_categories:
        labeler_categories = ["class_1", "class_2", "class_3"]

    for path in (
        instance_dir,
        models_dir,
        runs_dir,
        labeler_images_dir,
        labeler_masks_dir,
        labeler_coco_dir,
        labeler_cache_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    app = Flask(
        __name__,
        instance_path=str(instance_dir),
        instance_relative_config=False,
    )
    app.config.update(
        SECRET_KEY=os.getenv("FLASK_SECRET_KEY", "local-dev-secret"),
        BASE_DIR=str(base_dir),
        INSTANCE_DIR=str(instance_dir),
        MODELS_DIR=str(models_dir),
        RUNS_DIR=str(runs_dir),
        DB_PATH=str(db_path),
        APP_NAME="Model Foundry",
        CODE_VERSION=os.getenv("SEG_APP_CODE_VERSION", "0.1.0"),
        RANDOM_SEED=int(os.getenv("SEG_APP_RANDOM_SEED", "42")),
        LABELER_DIR=str(labeler_dir),
        LABELER_IMAGES_DIR=str(labeler_images_dir),
        LABELER_MASKS_DIR=str(labeler_masks_dir),
        LABELER_COCO_DIR=str(labeler_coco_dir),
        LABELER_CACHE_DIR=str(labeler_cache_dir),
        LABELER_CATEGORIES=labeler_categories,
        LABELER_SLIC_N_SEGMENTS=int(os.getenv("SEG_APP_LABELER_SLIC_N_SEGMENTS", "1200")),
        LABELER_SLIC_COMPACTNESS=float(os.getenv("SEG_APP_LABELER_SLIC_COMPACTNESS", "10")),
        LABELER_SLIC_SIGMA=float(os.getenv("SEG_APP_LABELER_SLIC_SIGMA", "1")),
        LABELER_MIN_COCO_AREA=int(os.getenv("SEG_APP_LABELER_MIN_COCO_AREA", "50")),
        JOB_WORKER_COUNT=int(os.getenv("SEG_APP_JOB_WORKER_COUNT", "1")),
        JOB_QUEUE_POLL_INTERVAL=float(os.getenv("SEG_APP_JOB_QUEUE_POLL_INTERVAL", "1.0")),
        JOB_HEARTBEAT_INTERVAL=float(os.getenv("SEG_APP_JOB_HEARTBEAT_INTERVAL", "4.0")),
        JOB_STALE_HEARTBEAT_SECONDS=int(os.getenv("SEG_APP_JOB_STALE_HEARTBEAT_SECONDS", "180")),
    )

    storage = Storage(db_path=db_path)
    storage.init_db()
    storage.migrate_labeler_class_schema_and_masks(fallback_names=labeler_categories)
    app.extensions["storage"] = storage
    job_queue = JobQueueManager(
        storage=storage,
        db_path=db_path,
        base_dir=base_dir,
        models_dir=models_dir,
        runs_dir=runs_dir,
        code_version=str(app.config["CODE_VERSION"]),
        random_seed=int(app.config["RANDOM_SEED"]),
        worker_count=int(app.config["JOB_WORKER_COUNT"]),
        poll_interval_seconds=float(app.config["JOB_QUEUE_POLL_INTERVAL"]),
        heartbeat_interval_seconds=float(app.config["JOB_HEARTBEAT_INTERVAL"]),
        stale_heartbeat_seconds=int(app.config["JOB_STALE_HEARTBEAT_SECONDS"]),
    )
    app.extensions["job_queue"] = job_queue

    from .labeler_routes import bp as labeler_bp
    from .routes import bp

    app.register_blueprint(bp)
    app.register_blueprint(labeler_bp)

    @app.before_request
    def _start_background_queue_once() -> None:
        queue = app.extensions.get("job_queue")
        if not isinstance(queue, JobQueueManager):
            return
        if app.extensions.get("_job_queue_started"):
            return
        queue.start()
        app.extensions["_job_queue_started"] = True

    @app.context_processor
    def inject_workspace_navigation() -> dict[str, object]:
        storage = app.extensions.get("storage")
        workspaces = storage.list_workspaces() if storage is not None else []
        endpoint = request.endpoint or ""
        is_workspace_landing = endpoint == "main.dashboard_root"
        active_workspace = None
        view_workspace_id = None
        if (
            not is_workspace_landing
            and request.view_args
            and isinstance(request.view_args.get("workspace_id"), int)
        ):
            view_workspace_id = int(request.view_args["workspace_id"])
        if not is_workspace_landing and view_workspace_id is not None and storage is not None:
            active_workspace = storage.get_workspace(view_workspace_id)
            if active_workspace is not None:
                session["workspace_id"] = int(active_workspace["id"])
        if not is_workspace_landing and active_workspace is None and storage is not None:
            session_workspace_id = session.get("workspace_id")
            if isinstance(session_workspace_id, int):
                active_workspace = storage.get_workspace(session_workspace_id)
        if not is_workspace_landing and active_workspace is None and workspaces:
            active_workspace = workspaces[0]
            session["workspace_id"] = int(active_workspace["id"])

        sidebar_items: list[dict[str, object]] = []
        breadcrumbs: list[dict[str, object]] = []
        active_section = ""

        if active_workspace is not None:
            workspace_id = int(active_workspace["id"])
            sidebar_items = [
                {
                    "key": "overview",
                    "label": "Overview",
                    "url": url_for("main.workspace_dashboard", workspace_id=workspace_id),
                },
                {
                    "key": "images",
                    "label": "Images",
                    "url": url_for("labeler.workspace_images", workspace_id=workspace_id),
                },
                {
                    "key": "datasets",
                    "label": "Datasets",
                    "url": url_for("main.workspace_datasets", workspace_id=workspace_id),
                },
                {
                    "key": "models",
                    "label": "Models",
                    "url": url_for("main.workspace_models", workspace_id=workspace_id),
                },
                {
                    "key": "analysis",
                    "label": "Analysis",
                    "url": url_for("main.workspace_analysis", workspace_id=workspace_id),
                },
                {
                    "key": "jobs",
                    "label": "Jobs",
                    "url": url_for("main.workspace_jobs", workspace_id=workspace_id),
                },
                {
                    "key": "settings",
                    "label": "Settings",
                    "url": url_for("main.workspace_settings", workspace_id=workspace_id),
                },
            ]

            if endpoint in {"main.workspace_dashboard"}:
                active_section = "overview"
            elif endpoint in {"labeler.dataset_page"}:
                active_section = "datasets"
            elif endpoint.startswith("labeler."):
                active_section = "images"
            elif endpoint in {
                "main.workspace_datasets",
                "main.register_dataset",
                "main.workspace_dataset_detail",
                "main.workspace_registered_dataset_detail",
                "main.register_workspace_dataset",
                "main.augment_new",
            }:
                active_section = "datasets"
            elif endpoint in {"main.workspace_models", "main.train_new_model", "main.model_details"}:
                active_section = "models"
            elif endpoint in {
                "main.workspace_analysis",
                "main.new_analysis",
                "main.analysis_details",
                "main.promote_analysis_run",
            }:
                active_section = "analysis"
            elif endpoint in {
                "main.workspace_jobs",
                "main.workspace_jobs_poll",
                "main.workspace_job_cancel",
                "main.workspace_job_rerun",
                "main.workspace_jobs_reorder",
            }:
                active_section = "jobs"
            elif endpoint in {"main.workspace_settings"}:
                active_section = "settings"

            breadcrumbs = [
                {
                    "label": "Workspaces",
                    "url": url_for("main.dashboard_root"),
                },
                {
                    "label": str(active_workspace["name"]),
                    "url": url_for("main.workspace_dashboard", workspace_id=workspace_id),
                }
            ]

            if endpoint in {"labeler.workspace_images"}:
                breadcrumbs.append(
                    {
                        "label": "Images",
                        "url": url_for("labeler.workspace_images", workspace_id=workspace_id),
                    }
                )
            elif endpoint in {"labeler.page"}:
                breadcrumbs.append(
                    {
                        "label": "Images",
                        "url": url_for("labeler.workspace_images", workspace_id=workspace_id),
                    }
                )
                image_name = ""
                if request.view_args:
                    image_name = str(request.view_args.get("image_name", ""))
                breadcrumbs.append({"label": image_name, "url": None})
            elif endpoint in {"labeler.dataset_page"}:
                breadcrumbs.append(
                    {
                        "label": "Datasets",
                        "url": url_for("main.workspace_datasets", workspace_id=workspace_id),
                    }
                )
                dataset_id = None
                image_name = ""
                if request.view_args and isinstance(request.view_args.get("dataset_id"), int):
                    dataset_id = int(request.view_args["dataset_id"])
                if request.view_args:
                    image_name = str(request.view_args.get("image_name", ""))
                dataset_project = (
                    storage.get_workspace_dataset(workspace_id, dataset_id)
                    if dataset_id is not None
                    else None
                )
                dataset_label = (
                    str(dataset_project["name"]) if dataset_project is not None else f"Dataset_{dataset_id}"
                )
                breadcrumbs.append(
                    {
                        "label": dataset_label,
                        "url": url_for(
                            "main.workspace_dataset_detail",
                            workspace_id=workspace_id,
                            dataset_id=dataset_id,
                        ),
                    }
                )
                breadcrumbs.append({"label": image_name, "url": None})
            elif endpoint in {"main.workspace_datasets"}:
                breadcrumbs.append(
                    {
                        "label": "Datasets",
                        "url": url_for("main.workspace_datasets", workspace_id=workspace_id),
                    }
                )
            elif endpoint in {"main.register_dataset"}:
                breadcrumbs.append(
                    {
                        "label": "Datasets",
                        "url": url_for("main.workspace_datasets", workspace_id=workspace_id),
                    }
                )
                breadcrumbs.append({"label": "Create", "url": None})
            elif endpoint in {"main.augment_new"}:
                breadcrumbs.append(
                    {
                        "label": "Datasets",
                        "url": url_for("main.workspace_datasets", workspace_id=workspace_id),
                    }
                )
                breadcrumbs.append({"label": "Augment", "url": None})
            elif endpoint in {"main.workspace_dataset_detail"}:
                breadcrumbs.append(
                    {
                        "label": "Datasets",
                        "url": url_for("main.workspace_datasets", workspace_id=workspace_id),
                    }
                )
                dataset_id = None
                if request.view_args and isinstance(request.view_args.get("dataset_id"), int):
                    dataset_id = int(request.view_args["dataset_id"])
                dataset_project = (
                    storage.get_workspace_dataset(workspace_id, dataset_id)
                    if dataset_id is not None
                    else None
                )
                label = str(dataset_project["name"]) if dataset_project is not None else f"Dataset_{dataset_id}"
                breadcrumbs.append({"label": label, "url": None})
            elif endpoint in {"main.workspace_registered_dataset_detail"}:
                breadcrumbs.append(
                    {
                        "label": "Datasets",
                        "url": url_for("main.workspace_datasets", workspace_id=workspace_id),
                    }
                )
                breadcrumbs.append({"label": "Registered", "url": None})
                dataset_id = None
                if request.view_args and isinstance(request.view_args.get("dataset_id"), int):
                    dataset_id = int(request.view_args["dataset_id"])
                dataset = storage.get_dataset(dataset_id, workspace_id=workspace_id) if dataset_id is not None else None
                label = str(dataset["name"]) if dataset is not None else f"Dataset_{dataset_id}"
                breadcrumbs.append({"label": label, "url": None})
            elif endpoint in {"main.workspace_models"}:
                breadcrumbs.append(
                    {
                        "label": "Models",
                        "url": url_for("main.workspace_models", workspace_id=workspace_id),
                    }
                )
            elif endpoint in {"main.model_details"}:
                breadcrumbs.append(
                    {
                        "label": "Models",
                        "url": url_for("main.workspace_models", workspace_id=workspace_id),
                    }
                )
                model_id = None
                if request.view_args and isinstance(request.view_args.get("model_id"), int):
                    model_id = int(request.view_args["model_id"])
                model = storage.get_model(model_id, workspace_id=workspace_id) if model_id is not None else None
                label = str(model["name"]) if model is not None else f"Model_{model_id}"
                breadcrumbs.append({"label": label, "url": None})
            elif endpoint in {"main.workspace_analysis"}:
                breadcrumbs.append(
                    {
                        "label": "Analysis",
                        "url": url_for("main.workspace_analysis", workspace_id=workspace_id),
                    }
                )
            elif endpoint in {"main.new_analysis"}:
                breadcrumbs.append(
                    {
                        "label": "Analysis",
                        "url": url_for("main.workspace_analysis", workspace_id=workspace_id),
                    }
                )
                breadcrumbs.append({"label": "New", "url": None})
            elif endpoint in {"main.analysis_details"}:
                run_id = None
                if request.view_args and isinstance(request.view_args.get("run_id"), int):
                    run_id = int(request.view_args["run_id"])
                run = storage.get_analysis_run(run_id, workspace_id=workspace_id) if run_id is not None else None
                model = (
                    storage.get_model(int(run["model_id"]), workspace_id=workspace_id)
                    if run is not None
                    else None
                )
                breadcrumbs.append(
                    {
                        "label": "Models",
                        "url": url_for("main.workspace_models", workspace_id=workspace_id),
                    }
                )
                if model is not None:
                    breadcrumbs.append(
                        {
                            "label": str(model["name"]),
                            "url": url_for("main.model_details", workspace_id=workspace_id, model_id=int(model["id"])),
                        }
                    )
                breadcrumbs.append({"label": f"Analysis_{run_id}", "url": None})
            elif endpoint in {"main.promote_analysis_run"}:
                run_id = None
                if request.view_args and isinstance(request.view_args.get("run_id"), int):
                    run_id = int(request.view_args["run_id"])
                breadcrumbs.append(
                    {
                        "label": "Analysis",
                        "url": url_for("main.workspace_analysis", workspace_id=workspace_id),
                    }
                )
                if run_id is not None:
                    breadcrumbs.append(
                        {
                            "label": f"Run_{run_id}",
                            "url": url_for(
                                "main.analysis_details",
                                workspace_id=workspace_id,
                                run_id=run_id,
                            ),
                        }
                    )
                breadcrumbs.append({"label": "Promote", "url": None})
            elif endpoint in {"main.workspace_settings"}:
                breadcrumbs.append(
                    {
                        "label": "Settings",
                        "url": url_for("main.workspace_settings", workspace_id=workspace_id),
                    }
                )
            elif endpoint in {"main.workspace_jobs"}:
                breadcrumbs.append(
                    {
                        "label": "Jobs",
                        "url": url_for("main.workspace_jobs", workspace_id=workspace_id),
                    }
                )

            for crumb in breadcrumbs:
                if crumb.get("url") is None:
                    crumb["url"] = request.path

        return {
            "workspaces": workspaces,
            "active_workspace": active_workspace,
            "sidebar_items": sidebar_items,
            "active_section": active_section,
            "breadcrumbs": breadcrumbs,
        }

    return app
