"""Test helpers for Model Foundry V2."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def make_test_config(base_dir: Path) -> dict[str, object]:
    instance_dir = base_dir / "instance"
    data_dir = base_dir / "data"
    artifacts_dir = data_dir / "artifacts"
    return {
        "TESTING": True,
        "BASE_DIR": base_dir,
        "INSTANCE_DIR": instance_dir,
        "DATA_DIR": data_dir,
        "ARTIFACTS_DIR": artifacts_dir,
        "ANNOTATIONS_DIR": artifacts_dir / "annotations" / "revisions",
        "EXPORTS_DIR": artifacts_dir / "exports",
        "DB_PATH": instance_dir / "model_foundry_v2.sqlite3",
        "SECRET_KEY": "test-secret",
        "DEFAULT_PORT": 5001,
    }


def create_test_image(path: Path, size: tuple[int, int] = (6, 5)) -> Path:
    image = Image.new("RGB", size, color=(120, 80, 30))
    image.save(path)
    return path
