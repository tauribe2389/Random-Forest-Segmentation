"""SQLite persistence layer for datasets, models, and analysis runs."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .labeling.class_schema import normalize_class_schema
from .labeling.image_io import mask_filename_for_class_id, sanitize_class_name
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

    @staticmethod
    def _column_names(conn: sqlite3.Connection, table_name: str) -> set[str]:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {str(row[1]) for row in rows}

    @classmethod
    def _ensure_column(
        cls,
        conn: sqlite3.Connection,
        *,
        table_name: str,
        column_name: str,
        definition: str,
    ) -> None:
        if column_name in cls._column_names(conn, table_name):
            return
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")

    @staticmethod
    def _normalize_labeler_categories_payload(
        payload: Any,
        *,
        fallback_names: list[str] | None = None,
    ) -> dict[str, Any]:
        return normalize_class_schema(payload, fallback_names=fallback_names)

    def init_db(self) -> None:
        """Create tables if they do not exist."""
        schema = """
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id INTEGER,
            name TEXT NOT NULL UNIQUE,
            dataset_path TEXT NOT NULL,
            image_root TEXT NOT NULL,
            coco_json_path TEXT NOT NULL,
            coco_checksum TEXT NOT NULL,
            categories_json TEXT NOT NULL,
            source_draft_project_id INTEGER,
            source_draft_version INTEGER,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id INTEGER,
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
            workspace_id INTEGER,
            model_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            input_images_json TEXT NOT NULL,
            output_dir TEXT NOT NULL,
            summary_json TEXT NOT NULL,
            postprocess_enabled INTEGER NOT NULL DEFAULT 0,
            postprocess_config_json TEXT,
            flip_stats_json TEXT,
            area_raw_json TEXT,
            area_refined_json TEXT,
            area_delta_json TEXT,
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
            raw_mask_path TEXT,
            conf_path TEXT,
            raw_overlay_path TEXT,
            refined_mask_path TEXT,
            refined_overlay_path TEXT,
            summary_json TEXT NOT NULL,
            flip_stats_json TEXT,
            area_raw_json TEXT,
            area_refined_json TEXT,
            area_delta_json TEXT,
            postprocess_applied INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (run_id) REFERENCES analysis_runs(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS labeler_projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            dataset_id INTEGER,
            kind TEXT NOT NULL DEFAULT 'workspace',
            parent_workspace_id INTEGER,
            augmentation_seed INTEGER,
            draft_version INTEGER NOT NULL DEFAULT 1,
            origin_type TEXT NOT NULL DEFAULT 'manual',
            origin_draft_project_id INTEGER,
            origin_draft_version INTEGER,
            origin_registered_dataset_id INTEGER,
            project_dir TEXT NOT NULL,
            images_dir TEXT NOT NULL,
            masks_dir TEXT NOT NULL,
            coco_dir TEXT NOT NULL,
            cache_dir TEXT NOT NULL,
            slic_algorithm TEXT NOT NULL DEFAULT 'slic',
            slic_preset_name TEXT NOT NULL DEFAULT 'medium',
            slic_detail_level TEXT NOT NULL DEFAULT 'medium',
            slic_n_segments INTEGER NOT NULL DEFAULT 1200,
            slic_compactness REAL NOT NULL DEFAULT 10.0,
            slic_sigma REAL NOT NULL DEFAULT 1.0,
            slic_colorspace TEXT NOT NULL DEFAULT 'lab',
            categories_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (dataset_id) REFERENCES datasets(id)
        );

        CREATE TABLE IF NOT EXISTS labeler_image_slic_overrides (
            project_id INTEGER NOT NULL,
            image_name TEXT NOT NULL,
            slic_algorithm TEXT NOT NULL DEFAULT 'slic',
            preset_name TEXT NOT NULL,
            detail_level TEXT NOT NULL,
            n_segments INTEGER NOT NULL,
            compactness REAL NOT NULL,
            sigma REAL NOT NULL,
            colorspace TEXT NOT NULL DEFAULT 'lab',
            updated_at TEXT NOT NULL,
            PRIMARY KEY (project_id, image_name),
            FOREIGN KEY (project_id) REFERENCES labeler_projects(id) ON DELETE CASCADE
        );
        """
        with self._connect() as conn:
            conn.executescript(schema)
            self._ensure_column(
                conn,
                table_name="datasets",
                column_name="workspace_id",
                definition="INTEGER",
            )
            self._ensure_column(
                conn,
                table_name="models",
                column_name="workspace_id",
                definition="INTEGER",
            )
            self._ensure_column(
                conn,
                table_name="analysis_runs",
                column_name="workspace_id",
                definition="INTEGER",
            )
            self._ensure_column(
                conn,
                table_name="analysis_runs",
                column_name="postprocess_enabled",
                definition="INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                conn,
                table_name="analysis_runs",
                column_name="postprocess_config_json",
                definition="TEXT",
            )
            self._ensure_column(
                conn,
                table_name="analysis_runs",
                column_name="flip_stats_json",
                definition="TEXT",
            )
            self._ensure_column(
                conn,
                table_name="analysis_runs",
                column_name="area_raw_json",
                definition="TEXT",
            )
            self._ensure_column(
                conn,
                table_name="analysis_runs",
                column_name="area_refined_json",
                definition="TEXT",
            )
            self._ensure_column(
                conn,
                table_name="analysis_runs",
                column_name="area_delta_json",
                definition="TEXT",
            )
            self._ensure_column(
                conn,
                table_name="analysis_run_items",
                column_name="raw_mask_path",
                definition="TEXT",
            )
            self._ensure_column(
                conn,
                table_name="analysis_run_items",
                column_name="raw_overlay_path",
                definition="TEXT",
            )
            self._ensure_column(
                conn,
                table_name="analysis_run_items",
                column_name="conf_path",
                definition="TEXT",
            )
            self._ensure_column(
                conn,
                table_name="analysis_run_items",
                column_name="refined_mask_path",
                definition="TEXT",
            )
            self._ensure_column(
                conn,
                table_name="analysis_run_items",
                column_name="refined_overlay_path",
                definition="TEXT",
            )
            self._ensure_column(
                conn,
                table_name="analysis_run_items",
                column_name="flip_stats_json",
                definition="TEXT",
            )
            self._ensure_column(
                conn,
                table_name="analysis_run_items",
                column_name="area_raw_json",
                definition="TEXT",
            )
            self._ensure_column(
                conn,
                table_name="analysis_run_items",
                column_name="area_refined_json",
                definition="TEXT",
            )
            self._ensure_column(
                conn,
                table_name="analysis_run_items",
                column_name="area_delta_json",
                definition="TEXT",
            )
            self._ensure_column(
                conn,
                table_name="analysis_run_items",
                column_name="postprocess_applied",
                definition="INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="kind",
                definition="TEXT NOT NULL DEFAULT 'workspace'",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="parent_workspace_id",
                definition="INTEGER",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="augmentation_seed",
                definition="INTEGER",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="draft_version",
                definition="INTEGER NOT NULL DEFAULT 1",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="origin_type",
                definition="TEXT NOT NULL DEFAULT 'manual'",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="origin_draft_project_id",
                definition="INTEGER",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="origin_draft_version",
                definition="INTEGER",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="origin_registered_dataset_id",
                definition="INTEGER",
            )
            self._ensure_column(
                conn,
                table_name="datasets",
                column_name="source_draft_project_id",
                definition="INTEGER",
            )
            self._ensure_column(
                conn,
                table_name="datasets",
                column_name="source_draft_version",
                definition="INTEGER",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="slic_algorithm",
                definition="TEXT NOT NULL DEFAULT 'slic'",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="slic_preset_name",
                definition="TEXT NOT NULL DEFAULT 'medium'",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="slic_detail_level",
                definition="TEXT NOT NULL DEFAULT 'medium'",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="slic_n_segments",
                definition="INTEGER NOT NULL DEFAULT 1200",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="slic_compactness",
                definition="REAL NOT NULL DEFAULT 10.0",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="slic_sigma",
                definition="REAL NOT NULL DEFAULT 1.0",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="slic_colorspace",
                definition="TEXT NOT NULL DEFAULT 'lab'",
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_workspace_id ON datasets(workspace_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_models_workspace_id ON models(workspace_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analysis_runs_workspace_id ON analysis_runs(workspace_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_labeler_projects_kind ON labeler_projects(kind)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_labeler_projects_parent_workspace ON labeler_projects(parent_workspace_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_labeler_projects_origin_draft ON labeler_projects(origin_draft_project_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_datasets_source_draft ON datasets(source_draft_project_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_labeler_image_slic_overrides_project_id "
                "ON labeler_image_slic_overrides(project_id)"
            )
            conn.execute("UPDATE labeler_projects SET kind = 'workspace' WHERE kind IS NULL OR kind = ''")
            conn.execute(
                "UPDATE labeler_projects SET parent_workspace_id = id WHERE kind = 'workspace' AND parent_workspace_id IS NULL"
            )
            conn.execute(
                "UPDATE labeler_projects SET draft_version = 1 WHERE draft_version IS NULL OR draft_version <= 0"
            )
            conn.execute(
                "UPDATE labeler_projects SET origin_type = 'manual' WHERE origin_type IS NULL OR origin_type = ''"
            )
            conn.execute("UPDATE labeler_projects SET slic_algorithm = 'slic' WHERE slic_algorithm IS NULL OR slic_algorithm = ''")
            conn.execute(
                "UPDATE labeler_projects SET slic_preset_name = 'medium' WHERE slic_preset_name IS NULL OR slic_preset_name = ''"
            )
            conn.execute(
                "UPDATE labeler_projects SET slic_detail_level = 'medium' "
                "WHERE slic_detail_level IS NULL OR slic_detail_level = ''"
            )
            conn.execute(
                "UPDATE labeler_projects SET slic_n_segments = 1200 WHERE slic_n_segments IS NULL OR slic_n_segments <= 0"
            )
            conn.execute(
                "UPDATE labeler_projects SET slic_compactness = 10.0 WHERE slic_compactness IS NULL OR slic_compactness <= 0"
            )
            conn.execute(
                "UPDATE labeler_projects SET slic_sigma = 1.0 WHERE slic_sigma IS NULL OR slic_sigma < 0"
            )
            conn.execute(
                "UPDATE labeler_projects SET slic_colorspace = 'lab' WHERE slic_colorspace IS NULL OR slic_colorspace = ''"
            )
            row = conn.execute("SELECT id FROM labeler_projects ORDER BY id ASC LIMIT 1").fetchone()
            if row is not None:
                default_workspace_id = int(row[0])
                conn.execute(
                    "UPDATE datasets SET workspace_id = ? WHERE workspace_id IS NULL",
                    (default_workspace_id,),
                )
                conn.execute(
                    "UPDATE models SET workspace_id = ? WHERE workspace_id IS NULL",
                    (default_workspace_id,),
                )
                conn.execute(
                    "UPDATE analysis_runs SET workspace_id = ? WHERE workspace_id IS NULL",
                    (default_workspace_id,),
                )
            conn.execute("UPDATE analysis_runs SET postprocess_enabled = 0 WHERE postprocess_enabled IS NULL")
            conn.execute("UPDATE analysis_run_items SET postprocess_applied = 0 WHERE postprocess_applied IS NULL")
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
        workspace_id: int | None = None,
        source_draft_project_id: int | None = None,
        source_draft_version: int | None = None,
    ) -> int:
        payload = (
            workspace_id,
            spec.name,
            spec.dataset_path,
            spec.image_root,
            spec.coco_json_path,
            coco_checksum,
            json.dumps(categories),
            source_draft_project_id,
            source_draft_version,
            _utc_now_iso(),
        )
        sql = """
        INSERT INTO datasets (
            workspace_id, name, dataset_path, image_root, coco_json_path,
            coco_checksum, categories_json, source_draft_project_id, source_draft_version, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            cursor = conn.execute(sql, payload)
            conn.commit()
            return int(cursor.lastrowid)

    def list_datasets(self, workspace_id: int | None = None) -> list[dict[str, Any]]:
        if workspace_id is None:
            sql = """
            SELECT *
            FROM datasets
            ORDER BY created_at DESC, id DESC
            """
            params: tuple[Any, ...] = ()
        else:
            sql = """
            SELECT *
            FROM datasets
            WHERE workspace_id = ?
            ORDER BY created_at DESC, id DESC
            """
            params = (workspace_id,)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._decode_json_fields(dict(row)) for row in rows]

    def get_dataset(self, dataset_id: int, workspace_id: int | None = None) -> dict[str, Any] | None:
        if workspace_id is None:
            sql = "SELECT * FROM datasets WHERE id = ?"
            params: tuple[Any, ...] = (dataset_id,)
        else:
            sql = "SELECT * FROM datasets WHERE id = ? AND workspace_id = ?"
            params = (dataset_id, workspace_id)
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        if row is None:
            return None
        return self._decode_json_fields(dict(row))

    def list_datasets_for_draft(
        self,
        source_draft_project_id: int,
        *,
        workspace_id: int | None = None,
    ) -> list[dict[str, Any]]:
        if workspace_id is None:
            sql = """
            SELECT *
            FROM datasets
            WHERE source_draft_project_id = ?
            ORDER BY created_at DESC, id DESC
            """
            params: tuple[Any, ...] = (source_draft_project_id,)
        else:
            sql = """
            SELECT *
            FROM datasets
            WHERE source_draft_project_id = ? AND workspace_id = ?
            ORDER BY created_at DESC, id DESC
            """
            params = (source_draft_project_id, workspace_id)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._decode_json_fields(dict(row)) for row in rows]

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
        workspace_id: int | None = None,
    ) -> int:
        payload = (
            workspace_id,
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
            workspace_id, name, dataset_id, created_at, classes_json, feature_config_json,
            hyperparams_json, metrics_json, artifact_dir, model_path, metadata_path,
            status, error_message
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            cursor = conn.execute(sql, payload)
            conn.commit()
            return int(cursor.lastrowid)

    def list_models(self, workspace_id: int | None = None) -> list[dict[str, Any]]:
        if workspace_id is None:
            sql = """
            SELECT
                m.*,
                d.name AS dataset_name
            FROM models m
            JOIN datasets d ON d.id = m.dataset_id
            ORDER BY m.created_at DESC, m.id DESC
            """
            params: tuple[Any, ...] = ()
        else:
            sql = """
            SELECT
                m.*,
                d.name AS dataset_name
            FROM models m
            JOIN datasets d ON d.id = m.dataset_id
            WHERE m.workspace_id = ?
            ORDER BY m.created_at DESC, m.id DESC
            """
            params = (workspace_id,)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        records: list[dict[str, Any]] = []
        for row in rows:
            item = self._decode_json_fields(dict(row))
            item["classes_count"] = max(0, len(item["classes_json"]) - 1)
            records.append(item)
        return records

    def get_model(self, model_id: int, workspace_id: int | None = None) -> dict[str, Any] | None:
        if workspace_id is None:
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
            params: tuple[Any, ...] = (model_id,)
        else:
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
            WHERE m.id = ? AND m.workspace_id = ?
            """
            params = (model_id, workspace_id)
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
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
        workspace_id: int | None = None,
        postprocess_enabled: bool = False,
        postprocess_config: dict[str, Any] | None = None,
    ) -> int:
        payload = (
            workspace_id,
            model_id,
            _utc_now_iso(),
            json.dumps(input_images),
            output_dir,
            json.dumps({}),
            1 if postprocess_enabled else 0,
            json.dumps(postprocess_config) if postprocess_config is not None else None,
            None,
            None,
            None,
            None,
            status,
            None,
        )
        sql = """
        INSERT INTO analysis_runs (
            workspace_id, model_id, created_at, input_images_json, output_dir, summary_json,
            postprocess_enabled, postprocess_config_json, flip_stats_json, area_raw_json,
            area_refined_json, area_delta_json, status, error_message
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        postprocess_enabled: bool | None = None,
        postprocess_config: dict[str, Any] | None = None,
        flip_stats: dict[str, Any] | None = None,
        area_raw: dict[str, Any] | None = None,
        area_refined: dict[str, Any] | None = None,
        area_delta: dict[str, Any] | None = None,
    ) -> None:
        assignments = [
            "output_dir = ?",
            "summary_json = ?",
            "status = ?",
            "error_message = ?",
        ]
        payload: list[Any] = [
            output_dir,
            json.dumps(summary),
            status,
            error_message,
        ]
        if postprocess_enabled is not None:
            assignments.append("postprocess_enabled = ?")
            payload.append(1 if postprocess_enabled else 0)
        if postprocess_config is not None:
            assignments.append("postprocess_config_json = ?")
            payload.append(json.dumps(postprocess_config))
        if flip_stats is not None:
            assignments.append("flip_stats_json = ?")
            payload.append(json.dumps(flip_stats))
        if area_raw is not None:
            assignments.append("area_raw_json = ?")
            payload.append(json.dumps(area_raw))
        if area_refined is not None:
            assignments.append("area_refined_json = ?")
            payload.append(json.dumps(area_refined))
        if area_delta is not None:
            assignments.append("area_delta_json = ?")
            payload.append(json.dumps(area_delta))
        payload.append(run_id)
        sql = f"""
        UPDATE analysis_runs
        SET {", ".join(assignments)}
        WHERE id = ?
        """
        with self._connect() as conn:
            conn.execute(sql, tuple(payload))
            conn.commit()

    def add_analysis_item(
        self,
        *,
        run_id: int,
        input_image: str,
        mask_path: str,
        overlay_path: str,
        summary: dict[str, Any],
        raw_mask_path: str | None = None,
        conf_path: str | None = None,
        raw_overlay_path: str | None = None,
        refined_mask_path: str | None = None,
        refined_overlay_path: str | None = None,
        flip_stats: dict[str, Any] | None = None,
        area_raw: dict[str, Any] | None = None,
        area_refined: dict[str, Any] | None = None,
        area_delta: dict[str, Any] | None = None,
        postprocess_applied: bool = False,
    ) -> int:
        payload = (
            run_id,
            input_image,
            mask_path,
            overlay_path,
            raw_mask_path,
            conf_path,
            raw_overlay_path,
            refined_mask_path,
            refined_overlay_path,
            json.dumps(summary),
            json.dumps(flip_stats) if flip_stats is not None else None,
            json.dumps(area_raw) if area_raw is not None else None,
            json.dumps(area_refined) if area_refined is not None else None,
            json.dumps(area_delta) if area_delta is not None else None,
            1 if postprocess_applied else 0,
        )
        sql = """
        INSERT INTO analysis_run_items (
            run_id, input_image, mask_path, overlay_path, raw_mask_path, conf_path, raw_overlay_path,
            refined_mask_path, refined_overlay_path, summary_json, flip_stats_json, area_raw_json,
            area_refined_json, area_delta_json, postprocess_applied
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            cursor = conn.execute(sql, payload)
            conn.commit()
            return int(cursor.lastrowid)

    def list_runs(self, workspace_id: int | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if workspace_id is None:
            sql = """
            SELECT
                r.*,
                m.name AS model_name
            FROM analysis_runs r
            JOIN models m ON m.id = r.model_id
            ORDER BY r.created_at DESC, r.id DESC
            LIMIT ?
            """
            params: tuple[Any, ...] = (limit,)
        else:
            sql = """
            SELECT
                r.*,
                m.name AS model_name
            FROM analysis_runs r
            JOIN models m ON m.id = r.model_id
            WHERE r.workspace_id = ?
            ORDER BY r.created_at DESC, r.id DESC
            LIMIT ?
            """
            params = (workspace_id, limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        records: list[dict[str, Any]] = []
        for row in rows:
            item = self._decode_json_fields(dict(row))
            item["num_images"] = len(item["input_images_json"])
            records.append(item)
        return records

    def get_analysis_run(self, run_id: int, workspace_id: int | None = None) -> dict[str, Any] | None:
        if workspace_id is None:
            sql = """
            SELECT
                r.*,
                m.name AS model_name
            FROM analysis_runs r
            JOIN models m ON m.id = r.model_id
            WHERE r.id = ?
            """
            params: tuple[Any, ...] = (run_id,)
        else:
            sql = """
            SELECT
                r.*,
                m.name AS model_name
            FROM analysis_runs r
            JOIN models m ON m.id = r.model_id
            WHERE r.id = ? AND r.workspace_id = ?
            """
            params = (run_id, workspace_id)
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
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

    def create_labeler_project(
        self,
        *,
        name: str,
        dataset_id: int | None,
        project_dir: str,
        images_dir: str,
        masks_dir: str,
        coco_dir: str,
        cache_dir: str,
        categories: Any,
        kind: str = "workspace",
        parent_workspace_id: int | None = None,
        draft_version: int = 1,
        augmentation_seed: int | None = None,
        origin_type: str = "manual",
        origin_draft_project_id: int | None = None,
        origin_draft_version: int | None = None,
        origin_registered_dataset_id: int | None = None,
        slic_algorithm: str = "slic",
        slic_preset_name: str = "medium",
        slic_detail_level: str = "medium",
        slic_n_segments: int = 1200,
        slic_compactness: float = 10.0,
        slic_sigma: float = 1.0,
        slic_colorspace: str = "lab",
    ) -> int:
        now = _utc_now_iso()
        normalized_kind = str(kind or "workspace").strip().lower()
        if normalized_kind not in {"workspace", "dataset"}:
            normalized_kind = "workspace"
        if parent_workspace_id is None and normalized_kind == "workspace":
            parent_workspace_id = 0
        normalized_origin = str(origin_type or "manual").strip().lower() or "manual"
        if normalized_origin not in {"manual", "branched_from_draft", "augmented_from_registered"}:
            normalized_origin = "manual"
        if normalized_kind == "workspace":
            normalized_origin = "manual"
            origin_draft_project_id = None
            origin_draft_version = None
            origin_registered_dataset_id = None
        normalized_categories = self._normalize_labeler_categories_payload(categories)
        normalized_seed: int | None = None
        if augmentation_seed is not None:
            normalized_seed = int(augmentation_seed)
        payload = (
            name,
            dataset_id,
            normalized_kind,
            parent_workspace_id,
            normalized_seed,
            max(1, int(draft_version)),
            normalized_origin,
            origin_draft_project_id,
            origin_draft_version,
            origin_registered_dataset_id,
            project_dir,
            images_dir,
            masks_dir,
            coco_dir,
            cache_dir,
            str(slic_algorithm or "slic").strip().lower() or "slic",
            str(slic_preset_name or "medium").strip().lower() or "medium",
            str(slic_detail_level or "medium").strip().lower() or "medium",
            max(50, int(slic_n_segments)),
            max(0.01, float(slic_compactness)),
            max(0.0, float(slic_sigma)),
            str(slic_colorspace or "lab").strip().lower() or "lab",
            json.dumps(normalized_categories),
            now,
            now,
        )
        sql = """
        INSERT INTO labeler_projects (
            name, dataset_id, kind, parent_workspace_id, augmentation_seed, draft_version, origin_type, origin_draft_project_id,
            origin_draft_version, origin_registered_dataset_id, project_dir, images_dir, masks_dir, coco_dir, cache_dir,
            slic_algorithm, slic_preset_name, slic_detail_level, slic_n_segments, slic_compactness, slic_sigma, slic_colorspace,
            categories_json, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            cursor = conn.execute(sql, payload)
            project_id = int(cursor.lastrowid)
            if normalized_kind == "workspace":
                conn.execute(
                    "UPDATE labeler_projects SET parent_workspace_id = ? WHERE id = ?",
                    (project_id, project_id),
                )
            conn.commit()
            return project_id

    def list_labeler_projects(self) -> list[dict[str, Any]]:
        sql = """
        SELECT
            p.*,
            d.name AS dataset_name
        FROM labeler_projects p
        LEFT JOIN datasets d ON d.id = p.dataset_id
        ORDER BY p.updated_at DESC, p.id DESC
        """
        with self._connect() as conn:
            rows = conn.execute(sql).fetchall()
        return [self._decode_json_fields(dict(row)) for row in rows]

    def list_workspaces(self) -> list[dict[str, Any]]:
        sql = """
        SELECT
            p.*
        FROM labeler_projects p
        WHERE p.kind = 'workspace'
        ORDER BY p.updated_at DESC, p.id DESC
        """
        with self._connect() as conn:
            rows = conn.execute(sql).fetchall()
        return [self._decode_json_fields(dict(row)) for row in rows]

    def get_labeler_project(self, project_id: int) -> dict[str, Any] | None:
        sql = """
        SELECT
            p.*,
            d.name AS dataset_name
        FROM labeler_projects p
        LEFT JOIN datasets d ON d.id = p.dataset_id
        WHERE p.id = ?
        """
        with self._connect() as conn:
            row = conn.execute(sql, (project_id,)).fetchone()
        if row is None:
            return None
        return self._decode_json_fields(dict(row))

    def get_workspace(self, workspace_id: int) -> dict[str, Any] | None:
        sql = """
        SELECT
            p.*
        FROM labeler_projects p
        WHERE p.id = ? AND p.kind = 'workspace'
        """
        with self._connect() as conn:
            row = conn.execute(sql, (workspace_id,)).fetchone()
        if row is None:
            return None
        return self._decode_json_fields(dict(row))

    def create_workspace_dataset(
        self,
        *,
        workspace_id: int,
        name: str,
        project_dir: str,
        images_dir: str,
        masks_dir: str,
        coco_dir: str,
        cache_dir: str,
        categories: Any,
        slic_algorithm: str = "slic",
        slic_preset_name: str = "medium",
        slic_detail_level: str = "medium",
        slic_n_segments: int = 1200,
        slic_compactness: float = 10.0,
        slic_sigma: float = 1.0,
        slic_colorspace: str = "lab",
        draft_version: int = 1,
        origin_type: str = "manual",
        origin_draft_project_id: int | None = None,
        origin_draft_version: int | None = None,
        origin_registered_dataset_id: int | None = None,
    ) -> int:
        return self.create_labeler_project(
            name=name,
            dataset_id=None,
            kind="dataset",
            parent_workspace_id=workspace_id,
            draft_version=draft_version,
            origin_type=origin_type,
            origin_draft_project_id=origin_draft_project_id,
            origin_draft_version=origin_draft_version,
            origin_registered_dataset_id=origin_registered_dataset_id,
            project_dir=project_dir,
            images_dir=images_dir,
            masks_dir=masks_dir,
            coco_dir=coco_dir,
            cache_dir=cache_dir,
            categories=categories,
            slic_algorithm=slic_algorithm,
            slic_preset_name=slic_preset_name,
            slic_detail_level=slic_detail_level,
            slic_n_segments=slic_n_segments,
            slic_compactness=slic_compactness,
            slic_sigma=slic_sigma,
            slic_colorspace=slic_colorspace,
        )

    def list_workspace_datasets(self, workspace_id: int) -> list[dict[str, Any]]:
        sql = """
        SELECT
            p.*
        FROM labeler_projects p
        WHERE p.kind = 'dataset' AND p.parent_workspace_id = ?
        ORDER BY p.updated_at DESC, p.id DESC
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (workspace_id,)).fetchall()
        return [self._decode_json_fields(dict(row)) for row in rows]

    def get_workspace_dataset(self, workspace_id: int, dataset_id: int) -> dict[str, Any] | None:
        sql = """
        SELECT
            p.*
        FROM labeler_projects p
        WHERE p.id = ? AND p.kind = 'dataset' AND p.parent_workspace_id = ?
        """
        with self._connect() as conn:
            row = conn.execute(sql, (dataset_id, workspace_id)).fetchone()
        if row is None:
            return None
        return self._decode_json_fields(dict(row))

    def link_labeler_project_dataset(self, project_id: int, dataset_id: int) -> None:
        payload = (dataset_id, _utc_now_iso(), project_id)
        sql = """
        UPDATE labeler_projects
        SET dataset_id = ?, updated_at = ?
        WHERE id = ?
        """
        with self._connect() as conn:
            conn.execute(sql, payload)
            conn.commit()

    def update_project_slic_defaults(
        self,
        project_id: int,
        *,
        slic_algorithm: str,
        preset_name: str,
        detail_level: str,
        n_segments: int,
        compactness: float,
        sigma: float,
        colorspace: str,
    ) -> None:
        payload = (
            str(slic_algorithm or "slic").strip().lower() or "slic",
            str(preset_name or "medium").strip().lower() or "medium",
            str(detail_level or "medium").strip().lower() or "medium",
            max(50, int(n_segments)),
            max(0.01, float(compactness)),
            max(0.0, float(sigma)),
            str(colorspace or "lab").strip().lower() or "lab",
            _utc_now_iso(),
            project_id,
        )
        sql = """
        UPDATE labeler_projects
        SET
            slic_algorithm = ?,
            slic_preset_name = ?,
            slic_detail_level = ?,
            slic_n_segments = ?,
            slic_compactness = ?,
            slic_sigma = ?,
            slic_colorspace = ?,
            updated_at = ?
        WHERE id = ?
        """
        with self._connect() as conn:
            conn.execute(sql, payload)
            conn.commit()

    def get_image_slic_override(self, project_id: int, image_name: str) -> dict[str, Any] | None:
        normalized_name = Path(str(image_name)).name
        sql = """
        SELECT *
        FROM labeler_image_slic_overrides
        WHERE project_id = ? AND image_name = ?
        """
        with self._connect() as conn:
            row = conn.execute(sql, (project_id, normalized_name)).fetchone()
        if row is None:
            return None
        return dict(row)

    def upsert_image_slic_override(
        self,
        project_id: int,
        image_name: str,
        *,
        slic_algorithm: str,
        preset_name: str,
        detail_level: str,
        n_segments: int,
        compactness: float,
        sigma: float,
        colorspace: str,
    ) -> None:
        normalized_name = Path(str(image_name)).name
        payload = (
            project_id,
            normalized_name,
            str(slic_algorithm or "slic").strip().lower() or "slic",
            str(preset_name or "medium").strip().lower() or "medium",
            str(detail_level or "medium").strip().lower() or "medium",
            max(50, int(n_segments)),
            max(0.01, float(compactness)),
            max(0.0, float(sigma)),
            str(colorspace or "lab").strip().lower() or "lab",
            _utc_now_iso(),
        )
        sql = """
        INSERT INTO labeler_image_slic_overrides (
            project_id, image_name, slic_algorithm, preset_name, detail_level,
            n_segments, compactness, sigma, colorspace, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(project_id, image_name) DO UPDATE SET
            slic_algorithm = excluded.slic_algorithm,
            preset_name = excluded.preset_name,
            detail_level = excluded.detail_level,
            n_segments = excluded.n_segments,
            compactness = excluded.compactness,
            sigma = excluded.sigma,
            colorspace = excluded.colorspace,
            updated_at = excluded.updated_at
        """
        with self._connect() as conn:
            conn.execute(sql, payload)
            conn.commit()

    def delete_image_slic_override(self, project_id: int, image_name: str) -> None:
        normalized_name = Path(str(image_name)).name
        sql = """
        DELETE FROM labeler_image_slic_overrides
        WHERE project_id = ? AND image_name = ?
        """
        with self._connect() as conn:
            conn.execute(sql, (project_id, normalized_name))
            conn.commit()

    def delete_image_slic_overrides_for_images(self, project_id: int, image_names: list[str]) -> None:
        normalized = [Path(str(name)).name for name in image_names if str(name).strip()]
        if not normalized:
            return
        placeholders = ", ".join("?" for _ in normalized)
        sql = (
            "DELETE FROM labeler_image_slic_overrides "
            f"WHERE project_id = ? AND image_name IN ({placeholders})"
        )
        params: tuple[Any, ...] = (project_id, *normalized)
        with self._connect() as conn:
            conn.execute(sql, params)
            conn.commit()

    def update_labeler_project_categories(
        self,
        project_id: int,
        categories: Any,
        *,
        bump_version: bool = False,
    ) -> None:
        normalized_categories = self._normalize_labeler_categories_payload(categories)
        encoded = json.dumps(normalized_categories)
        if bump_version:
            payload = (encoded, _utc_now_iso(), project_id)
            sql = """
            UPDATE labeler_projects
            SET categories_json = ?, draft_version = draft_version + 1, updated_at = ?
            WHERE id = ?
            """
        else:
            payload = (encoded, _utc_now_iso(), project_id)
            sql = """
            UPDATE labeler_projects
            SET categories_json = ?, updated_at = ?
            WHERE id = ?
            """
        with self._connect() as conn:
            conn.execute(sql, payload)
            conn.commit()

    def migrate_labeler_class_schema_and_masks(
        self,
        *,
        fallback_names: list[str] | None = None,
    ) -> dict[str, int]:
        stats = {
            "projects_scanned": 0,
            "projects_updated": 0,
            "masks_renamed": 0,
            "legacy_masks_removed": 0,
            "rename_errors": 0,
        }
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, categories_json, masks_dir
                FROM labeler_projects
                ORDER BY id ASC
                """
            ).fetchall()
            for row in rows:
                stats["projects_scanned"] += 1
                project_id = int(row["id"])
                raw_categories: Any = row["categories_json"]
                parsed_categories: Any = raw_categories
                if isinstance(raw_categories, str):
                    try:
                        parsed_categories = json.loads(raw_categories)
                    except json.JSONDecodeError:
                        parsed_categories = fallback_names or []
                normalized = self._normalize_labeler_categories_payload(
                    parsed_categories,
                    fallback_names=fallback_names,
                )
                encoded = json.dumps(normalized)
                if raw_categories != encoded:
                    conn.execute(
                        "UPDATE labeler_projects SET categories_json = ?, updated_at = ? WHERE id = ?",
                        (encoded, _utc_now_iso(), project_id),
                    )
                    stats["projects_updated"] += 1

                masks_dir = Path(str(row["masks_dir"] or "")).resolve()
                if not masks_dir.exists() or not masks_dir.is_dir():
                    continue

                for class_entry in normalized.get("classes", []):
                    try:
                        class_id = int(class_entry.get("id"))
                    except (TypeError, ValueError):
                        continue
                    if class_id <= 0:
                        continue
                    class_name = str(class_entry.get("name", "")).strip()
                    if not class_name:
                        continue
                    sanitized = sanitize_class_name(class_name)
                    if not sanitized:
                        continue
                    for legacy_path in masks_dir.glob(f"*__{sanitized}.png"):
                        if not legacy_path.exists() or not legacy_path.is_file():
                            continue
                        image_stem = str(legacy_path.stem).split("__", 1)[0].strip()
                        if not image_stem:
                            continue
                        target_path = masks_dir / mask_filename_for_class_id(image_stem, class_id)
                        try:
                            if target_path.exists():
                                legacy_path.unlink()
                                stats["legacy_masks_removed"] += 1
                            else:
                                legacy_path.rename(target_path)
                                stats["masks_renamed"] += 1
                        except OSError:
                            stats["rename_errors"] += 1
            conn.commit()
        return stats

    def touch_labeler_project(self, project_id: int) -> None:
        payload = (_utc_now_iso(), project_id)
        sql = """
        UPDATE labeler_projects
        SET updated_at = ?
        WHERE id = ?
        """
        with self._connect() as conn:
            conn.execute(sql, payload)
            conn.commit()

    def update_workspace_augmentation_seed(self, workspace_id: int, augmentation_seed: int) -> None:
        payload = (int(augmentation_seed), _utc_now_iso(), workspace_id)
        sql = """
        UPDATE labeler_projects
        SET augmentation_seed = ?, updated_at = ?
        WHERE id = ? AND kind = 'workspace'
        """
        with self._connect() as conn:
            conn.execute(sql, payload)
            conn.commit()

    def bump_labeler_project_version(self, project_id: int) -> None:
        payload = (_utc_now_iso(), project_id)
        sql = """
        UPDATE labeler_projects
        SET draft_version = draft_version + 1, updated_at = ?
        WHERE id = ? AND kind = 'dataset'
        """
        with self._connect() as conn:
            conn.execute(sql, payload)
            conn.commit()

    def backfill_workspace_links(self, workspace_id: int) -> None:
        payload = (workspace_id,)
        with self._connect() as conn:
            conn.execute("UPDATE datasets SET workspace_id = ? WHERE workspace_id IS NULL", payload)
            conn.execute("UPDATE models SET workspace_id = ? WHERE workspace_id IS NULL", payload)
            conn.execute("UPDATE analysis_runs SET workspace_id = ? WHERE workspace_id IS NULL", payload)
            conn.commit()
