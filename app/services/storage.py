"""SQLite persistence layer for datasets, models, and analysis runs."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import DatasetSpec


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Storage:
    """Simple SQLite registry for app entities."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init_db(self) -> None:
        """Create tables if they do not exist."""
        schema = """
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            dataset_path TEXT NOT NULL,
            image_root TEXT NOT NULL,
            coco_json_path TEXT NOT NULL,
            coco_checksum TEXT NOT NULL,
            categories_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            dataset_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            classes_json TEXT NOT NULL,
            feature_config_json TEXT NOT NULL,
            hyperparams_json TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            artifact_dir TEXT NOT NULL,
            model_path TEXT NOT NULL,
            metadata_path TEXT NOT NULL,
            status TEXT NOT NULL,
            error_message TEXT,
            FOREIGN KEY (dataset_id) REFERENCES datasets(id)
        );

        CREATE TABLE IF NOT EXISTS analysis_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            input_images_json TEXT NOT NULL,
            output_dir TEXT NOT NULL,
            summary_json TEXT NOT NULL,
            status TEXT NOT NULL,
            error_message TEXT,
            FOREIGN KEY (model_id) REFERENCES models(id)
        );

        CREATE TABLE IF NOT EXISTS analysis_run_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            input_image TEXT NOT NULL,
            mask_path TEXT NOT NULL,
            overlay_path TEXT NOT NULL,
            summary_json TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES analysis_runs(id) ON DELETE CASCADE
        );
        """
        with self._connect() as conn:
            conn.executescript(schema)
            conn.commit()

    @staticmethod
    def _decode_json_fields(row_dict: dict[str, Any]) -> dict[str, Any]:
        for key, value in list(row_dict.items()):
            if key.endswith("_json") and isinstance(value, str):
                row_dict[key] = json.loads(value)
        return row_dict

    def create_dataset(
        self,
        spec: DatasetSpec,
        *,
        coco_checksum: str,
        categories: list[dict[str, Any]],
    ) -> int:
        payload = (
            spec.name,
            spec.dataset_path,
            spec.image_root,
            spec.coco_json_path,
            coco_checksum,
            json.dumps(categories),
            _utc_now_iso(),
        )
        sql = """
        INSERT INTO datasets (
            name, dataset_path, image_root, coco_json_path,
            coco_checksum, categories_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            cursor = conn.execute(sql, payload)
            conn.commit()
            return int(cursor.lastrowid)

    def list_datasets(self) -> list[dict[str, Any]]:
        sql = """
        SELECT *
        FROM datasets
        ORDER BY created_at DESC, id DESC
        """
        with self._connect() as conn:
            rows = conn.execute(sql).fetchall()
        return [self._decode_json_fields(dict(row)) for row in rows]

    def get_dataset(self, dataset_id: int) -> dict[str, Any] | None:
        sql = "SELECT * FROM datasets WHERE id = ?"
        with self._connect() as conn:
            row = conn.execute(sql, (dataset_id,)).fetchone()
        if row is None:
            return None
        return self._decode_json_fields(dict(row))

    def create_model(
        self,
        *,
        name: str,
        dataset_id: int,
        classes: list[dict[str, Any]],
        feature_config: dict[str, Any],
        hyperparams: dict[str, Any],
        metrics: dict[str, Any],
        artifact_dir: str,
        model_path: str,
        metadata_path: str,
        status: str = "trained",
        error_message: str | None = None,
    ) -> int:
        payload = (
            name,
            dataset_id,
            _utc_now_iso(),
            json.dumps(classes),
            json.dumps(feature_config),
            json.dumps(hyperparams),
            json.dumps(metrics),
            artifact_dir,
            model_path,
            metadata_path,
            status,
            error_message,
        )
        sql = """
        INSERT INTO models (
            name, dataset_id, created_at, classes_json, feature_config_json,
            hyperparams_json, metrics_json, artifact_dir, model_path, metadata_path,
            status, error_message
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            cursor = conn.execute(sql, payload)
            conn.commit()
            return int(cursor.lastrowid)

    def list_models(self) -> list[dict[str, Any]]:
        sql = """
        SELECT
            m.*,
            d.name AS dataset_name
        FROM models m
        JOIN datasets d ON d.id = m.dataset_id
        ORDER BY m.created_at DESC, m.id DESC
        """
        with self._connect() as conn:
            rows = conn.execute(sql).fetchall()

        records: list[dict[str, Any]] = []
        for row in rows:
            item = self._decode_json_fields(dict(row))
            item["classes_count"] = max(0, len(item["classes_json"]) - 1)
            records.append(item)
        return records

    def get_model(self, model_id: int) -> dict[str, Any] | None:
        sql = """
        SELECT
            m.*,
            d.name AS dataset_name,
            d.dataset_path AS dataset_path,
            d.image_root AS dataset_image_root,
            d.coco_json_path AS dataset_coco_json_path,
            d.coco_checksum AS dataset_coco_checksum
        FROM models m
        JOIN datasets d ON d.id = m.dataset_id
        WHERE m.id = ?
        """
        with self._connect() as conn:
            row = conn.execute(sql, (model_id,)).fetchone()
        if row is None:
            return None
        item = self._decode_json_fields(dict(row))
        item["classes_count"] = max(0, len(item["classes_json"]) - 1)
        return item

    def create_analysis_run(
        self,
        *,
        model_id: int,
        input_images: list[str],
        output_dir: str = "",
        status: str = "running",
    ) -> int:
        payload = (
            model_id,
            _utc_now_iso(),
            json.dumps(input_images),
            output_dir,
            json.dumps({}),
            status,
            None,
        )
        sql = """
        INSERT INTO analysis_runs (
            model_id, created_at, input_images_json, output_dir, summary_json,
            status, error_message
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            cursor = conn.execute(sql, payload)
            conn.commit()
            return int(cursor.lastrowid)

    def update_analysis_run(
        self,
        run_id: int,
        *,
        output_dir: str,
        summary: dict[str, Any],
        status: str,
        error_message: str | None = None,
    ) -> None:
        payload = (
            output_dir,
            json.dumps(summary),
            status,
            error_message,
            run_id,
        )
        sql = """
        UPDATE analysis_runs
        SET output_dir = ?, summary_json = ?, status = ?, error_message = ?
        WHERE id = ?
        """
        with self._connect() as conn:
            conn.execute(sql, payload)
            conn.commit()

    def add_analysis_item(
        self,
        *,
        run_id: int,
        input_image: str,
        mask_path: str,
        overlay_path: str,
        summary: dict[str, Any],
    ) -> int:
        payload = (
            run_id,
            input_image,
            mask_path,
            overlay_path,
            json.dumps(summary),
        )
        sql = """
        INSERT INTO analysis_run_items (
            run_id, input_image, mask_path, overlay_path, summary_json
        )
        VALUES (?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            cursor = conn.execute(sql, payload)
            conn.commit()
            return int(cursor.lastrowid)

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        sql = """
        SELECT
            r.*,
            m.name AS model_name
        FROM analysis_runs r
        JOIN models m ON m.id = r.model_id
        ORDER BY r.created_at DESC, r.id DESC
        LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()

        records: list[dict[str, Any]] = []
        for row in rows:
            item = self._decode_json_fields(dict(row))
            item["num_images"] = len(item["input_images_json"])
            records.append(item)
        return records

    def get_analysis_run(self, run_id: int) -> dict[str, Any] | None:
        sql = """
        SELECT
            r.*,
            m.name AS model_name
        FROM analysis_runs r
        JOIN models m ON m.id = r.model_id
        WHERE r.id = ?
        """
        with self._connect() as conn:
            row = conn.execute(sql, (run_id,)).fetchone()
        if row is None:
            return None
        item = self._decode_json_fields(dict(row))
        item["num_images"] = len(item["input_images_json"])
        return item

    def get_analysis_items(self, run_id: int) -> list[dict[str, Any]]:
        sql = """
        SELECT *
        FROM analysis_run_items
        WHERE run_id = ?
        ORDER BY id ASC
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (run_id,)).fetchall()
        return [self._decode_json_fields(dict(row)) for row in rows]

