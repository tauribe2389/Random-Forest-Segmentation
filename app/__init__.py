"""Flask application factory for the local segmentation app."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask

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
    db_path = Path(
        os.getenv("SEG_APP_DB_PATH", str(instance_dir / "segmentation.sqlite3"))
    ).resolve()

    for path in (instance_dir, models_dir, runs_dir):
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
        CODE_VERSION=os.getenv("SEG_APP_CODE_VERSION", "0.1.0"),
        RANDOM_SEED=int(os.getenv("SEG_APP_RANDOM_SEED", "42")),
    )

    storage = Storage(db_path=db_path)
    storage.init_db()
    app.extensions["storage"] = storage

    from .routes import bp

    app.register_blueprint(bp)
    return app

