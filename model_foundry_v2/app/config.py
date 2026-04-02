"""Configuration helpers for Model Foundry V2."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def build_default_config() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent.parent
    base_dir = Path(os.getenv("MODEL_FOUNDRY_V2_BASE_DIR", str(package_root))).resolve()
    instance_dir = Path(
        os.getenv("MODEL_FOUNDRY_V2_INSTANCE_DIR", str(base_dir / "instance"))
    ).resolve()
    data_dir = Path(os.getenv("MODEL_FOUNDRY_V2_DATA_DIR", str(base_dir / "data"))).resolve()
    artifacts_dir = Path(
        os.getenv("MODEL_FOUNDRY_V2_ARTIFACTS_DIR", str(data_dir / "artifacts"))
    ).resolve()
    annotations_dir = Path(
        os.getenv(
            "MODEL_FOUNDRY_V2_ANNOTATIONS_DIR",
            str(artifacts_dir / "annotations" / "revisions"),
        )
    ).resolve()
    exports_dir = Path(
        os.getenv("MODEL_FOUNDRY_V2_EXPORTS_DIR", str(artifacts_dir / "exports"))
    ).resolve()
    db_path = Path(
        os.getenv("MODEL_FOUNDRY_V2_DB_PATH", str(instance_dir / "model_foundry_v2.sqlite3"))
    ).resolve()
    return {
        "APP_NAME": "Model Foundry V2",
        "BASE_DIR": base_dir,
        "INSTANCE_DIR": instance_dir,
        "DATA_DIR": data_dir,
        "ARTIFACTS_DIR": artifacts_dir,
        "ANNOTATIONS_DIR": annotations_dir,
        "EXPORTS_DIR": exports_dir,
        "DB_PATH": db_path,
        "DEFAULT_PORT": int(os.getenv("MODEL_FOUNDRY_V2_PORT", "5001")),
        "SECRET_KEY": os.getenv("MODEL_FOUNDRY_V2_SECRET_KEY", "model-foundry-v2-local-dev"),
    }


def ensure_runtime_directories(config: dict[str, Any]) -> None:
    for key in (
        "INSTANCE_DIR",
        "DATA_DIR",
        "ARTIFACTS_DIR",
        "ANNOTATIONS_DIR",
        "EXPORTS_DIR",
    ):
        path = Path(config[key])
        path.mkdir(parents=True, exist_ok=True)
