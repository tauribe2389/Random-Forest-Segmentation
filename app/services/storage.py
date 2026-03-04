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


_UNSET = object()


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

        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id INTEGER NOT NULL,
            job_type TEXT NOT NULL,
            status TEXT NOT NULL,
            stage TEXT NOT NULL,
            progress REAL NOT NULL DEFAULT 0.0,
            logs_json TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            dedupe_key TEXT,
            priority INTEGER NOT NULL DEFAULT 0,
            cancel_requested INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            heartbeat_at TEXT,
            worker_id TEXT,
            error_message TEXT,
            entity_type TEXT,
            entity_id INTEGER,
            result_ref_type TEXT,
            result_ref_id INTEGER,
            rerun_of_job_id INTEGER
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
            quickshift_ratio REAL NOT NULL DEFAULT 1.0,
            quickshift_kernel_size INTEGER NOT NULL DEFAULT 5,
            quickshift_max_dist REAL NOT NULL DEFAULT 10.0,
            quickshift_sigma REAL NOT NULL DEFAULT 0.0,
            felzenszwalb_scale REAL NOT NULL DEFAULT 100.0,
            felzenszwalb_sigma REAL NOT NULL DEFAULT 0.8,
            felzenszwalb_min_size INTEGER NOT NULL DEFAULT 50,
            texture_enabled INTEGER NOT NULL DEFAULT 0,
            texture_mode TEXT NOT NULL DEFAULT 'append_to_color',
            texture_lbp_enabled INTEGER NOT NULL DEFAULT 0,
            texture_lbp_points INTEGER NOT NULL DEFAULT 8,
            texture_lbp_radii_json TEXT NOT NULL DEFAULT '[1]',
            texture_lbp_method TEXT NOT NULL DEFAULT 'uniform',
            texture_lbp_normalize INTEGER NOT NULL DEFAULT 1,
            texture_gabor_enabled INTEGER NOT NULL DEFAULT 0,
            texture_gabor_frequencies_json TEXT NOT NULL DEFAULT '[0.1, 0.2]',
            texture_gabor_thetas_json TEXT NOT NULL DEFAULT '[0.0, 45.0, 90.0, 135.0]',
            texture_gabor_bandwidth REAL NOT NULL DEFAULT 1.0,
            texture_gabor_include_real INTEGER NOT NULL DEFAULT 0,
            texture_gabor_include_imag INTEGER NOT NULL DEFAULT 0,
            texture_gabor_include_magnitude INTEGER NOT NULL DEFAULT 1,
            texture_gabor_normalize INTEGER NOT NULL DEFAULT 1,
            texture_weight_color REAL NOT NULL DEFAULT 1.0,
            texture_weight_lbp REAL NOT NULL DEFAULT 0.25,
            texture_weight_gabor REAL NOT NULL DEFAULT 0.25,
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
            quickshift_ratio REAL NOT NULL DEFAULT 1.0,
            quickshift_kernel_size INTEGER NOT NULL DEFAULT 5,
            quickshift_max_dist REAL NOT NULL DEFAULT 10.0,
            quickshift_sigma REAL NOT NULL DEFAULT 0.0,
            felzenszwalb_scale REAL NOT NULL DEFAULT 100.0,
            felzenszwalb_sigma REAL NOT NULL DEFAULT 0.8,
            felzenszwalb_min_size INTEGER NOT NULL DEFAULT 50,
            texture_enabled INTEGER NOT NULL DEFAULT 0,
            texture_mode TEXT NOT NULL DEFAULT 'append_to_color',
            texture_lbp_enabled INTEGER NOT NULL DEFAULT 0,
            texture_lbp_points INTEGER NOT NULL DEFAULT 8,
            texture_lbp_radii_json TEXT NOT NULL DEFAULT '[1]',
            texture_lbp_method TEXT NOT NULL DEFAULT 'uniform',
            texture_lbp_normalize INTEGER NOT NULL DEFAULT 1,
            texture_gabor_enabled INTEGER NOT NULL DEFAULT 0,
            texture_gabor_frequencies_json TEXT NOT NULL DEFAULT '[0.1, 0.2]',
            texture_gabor_thetas_json TEXT NOT NULL DEFAULT '[0.0, 45.0, 90.0, 135.0]',
            texture_gabor_bandwidth REAL NOT NULL DEFAULT 1.0,
            texture_gabor_include_real INTEGER NOT NULL DEFAULT 0,
            texture_gabor_include_imag INTEGER NOT NULL DEFAULT 0,
            texture_gabor_include_magnitude INTEGER NOT NULL DEFAULT 1,
            texture_gabor_normalize INTEGER NOT NULL DEFAULT 1,
            texture_weight_color REAL NOT NULL DEFAULT 1.0,
            texture_weight_lbp REAL NOT NULL DEFAULT 0.25,
            texture_weight_gabor REAL NOT NULL DEFAULT 0.25,
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
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="quickshift_ratio",
                definition="REAL NOT NULL DEFAULT 1.0",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="quickshift_kernel_size",
                definition="INTEGER NOT NULL DEFAULT 5",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="quickshift_max_dist",
                definition="REAL NOT NULL DEFAULT 10.0",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="quickshift_sigma",
                definition="REAL NOT NULL DEFAULT 0.0",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="felzenszwalb_scale",
                definition="REAL NOT NULL DEFAULT 100.0",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="felzenszwalb_sigma",
                definition="REAL NOT NULL DEFAULT 0.8",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="felzenszwalb_min_size",
                definition="INTEGER NOT NULL DEFAULT 50",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="texture_enabled",
                definition="INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="texture_mode",
                definition="TEXT NOT NULL DEFAULT 'append_to_color'",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="texture_lbp_enabled",
                definition="INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="texture_lbp_points",
                definition="INTEGER NOT NULL DEFAULT 8",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="texture_lbp_radii_json",
                definition="TEXT NOT NULL DEFAULT '[1]'",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="texture_lbp_method",
                definition="TEXT NOT NULL DEFAULT 'uniform'",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="texture_lbp_normalize",
                definition="INTEGER NOT NULL DEFAULT 1",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="texture_gabor_enabled",
                definition="INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="texture_gabor_frequencies_json",
                definition="TEXT NOT NULL DEFAULT '[0.1, 0.2]'",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="texture_gabor_thetas_json",
                definition="TEXT NOT NULL DEFAULT '[0.0, 45.0, 90.0, 135.0]'",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="texture_gabor_bandwidth",
                definition="REAL NOT NULL DEFAULT 1.0",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="texture_gabor_include_real",
                definition="INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="texture_gabor_include_imag",
                definition="INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="texture_gabor_include_magnitude",
                definition="INTEGER NOT NULL DEFAULT 1",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="texture_gabor_normalize",
                definition="INTEGER NOT NULL DEFAULT 1",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="texture_weight_color",
                definition="REAL NOT NULL DEFAULT 1.0",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="texture_weight_lbp",
                definition="REAL NOT NULL DEFAULT 0.25",
            )
            self._ensure_column(
                conn,
                table_name="labeler_projects",
                column_name="texture_weight_gabor",
                definition="REAL NOT NULL DEFAULT 0.25",
            )
            self._ensure_column(
                conn,
                table_name="labeler_image_slic_overrides",
                column_name="quickshift_ratio",
                definition="REAL NOT NULL DEFAULT 1.0",
            )
            self._ensure_column(
                conn,
                table_name="labeler_image_slic_overrides",
                column_name="quickshift_kernel_size",
                definition="INTEGER NOT NULL DEFAULT 5",
            )
            self._ensure_column(
                conn,
                table_name="labeler_image_slic_overrides",
                column_name="quickshift_max_dist",
                definition="REAL NOT NULL DEFAULT 10.0",
            )
            self._ensure_column(
                conn,
                table_name="labeler_image_slic_overrides",
                column_name="quickshift_sigma",
                definition="REAL NOT NULL DEFAULT 0.0",
            )
            self._ensure_column(
                conn,
                table_name="labeler_image_slic_overrides",
                column_name="felzenszwalb_scale",
                definition="REAL NOT NULL DEFAULT 100.0",
            )
            self._ensure_column(
                conn,
                table_name="labeler_image_slic_overrides",
                column_name="felzenszwalb_sigma",
                definition="REAL NOT NULL DEFAULT 0.8",
            )
            self._ensure_column(
                conn,
                table_name="labeler_image_slic_overrides",
                column_name="felzenszwalb_min_size",
                definition="INTEGER NOT NULL DEFAULT 50",
            )
            self._ensure_column(
                conn,
                table_name="labeler_image_slic_overrides",
                column_name="texture_enabled",
                definition="INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                conn,
                table_name="labeler_image_slic_overrides",
                column_name="texture_mode",
                definition="TEXT NOT NULL DEFAULT 'append_to_color'",
            )
            self._ensure_column(
                conn,
                table_name="labeler_image_slic_overrides",
                column_name="texture_lbp_enabled",
                definition="INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                conn,
                table_name="labeler_image_slic_overrides",
                column_name="texture_lbp_points",
                definition="INTEGER NOT NULL DEFAULT 8",
            )
            self._ensure_column(
                conn,
                table_name="labeler_image_slic_overrides",
                column_name="texture_lbp_radii_json",
                definition="TEXT NOT NULL DEFAULT '[1]'",
            )
            self._ensure_column(
                conn,
                table_name="labeler_image_slic_overrides",
                column_name="texture_lbp_method",
                definition="TEXT NOT NULL DEFAULT 'uniform'",
            )
            self._ensure_column(
                conn,
                table_name="labeler_image_slic_overrides",
                column_name="texture_lbp_normalize",
                definition="INTEGER NOT NULL DEFAULT 1",
            )
            self._ensure_column(
                conn,
                table_name="labeler_image_slic_overrides",
                column_name="texture_gabor_enabled",
                definition="INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                conn,
                table_name="labeler_image_slic_overrides",
                column_name="texture_gabor_frequencies_json",
                definition="TEXT NOT NULL DEFAULT '[0.1, 0.2]'",
            )
            self._ensure_column(
                conn,
                table_name="labeler_image_slic_overrides",
                column_name="texture_gabor_thetas_json",
                definition="TEXT NOT NULL DEFAULT '[0.0, 45.0, 90.0, 135.0]'",
            )
            self._ensure_column(
                conn,
                table_name="labeler_image_slic_overrides",
                column_name="texture_gabor_bandwidth",
                definition="REAL NOT NULL DEFAULT 1.0",
            )
            self._ensure_column(
                conn,
                table_name="labeler_image_slic_overrides",
                column_name="texture_gabor_include_real",
                definition="INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                conn,
                table_name="labeler_image_slic_overrides",
                column_name="texture_gabor_include_imag",
                definition="INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                conn,
                table_name="labeler_image_slic_overrides",
                column_name="texture_gabor_include_magnitude",
                definition="INTEGER NOT NULL DEFAULT 1",
            )
            self._ensure_column(
                conn,
                table_name="labeler_image_slic_overrides",
                column_name="texture_gabor_normalize",
                definition="INTEGER NOT NULL DEFAULT 1",
            )
            self._ensure_column(
                conn,
                table_name="labeler_image_slic_overrides",
                column_name="texture_weight_color",
                definition="REAL NOT NULL DEFAULT 1.0",
            )
            self._ensure_column(
                conn,
                table_name="labeler_image_slic_overrides",
                column_name="texture_weight_lbp",
                definition="REAL NOT NULL DEFAULT 0.25",
            )
            self._ensure_column(
                conn,
                table_name="labeler_image_slic_overrides",
                column_name="texture_weight_gabor",
                definition="REAL NOT NULL DEFAULT 0.25",
            )
            self._ensure_column(
                conn,
                table_name="jobs",
                column_name="workspace_id",
                definition="INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                conn,
                table_name="jobs",
                column_name="job_type",
                definition="TEXT NOT NULL DEFAULT 'training'",
            )
            self._ensure_column(
                conn,
                table_name="jobs",
                column_name="status",
                definition="TEXT NOT NULL DEFAULT 'queued'",
            )
            self._ensure_column(
                conn,
                table_name="jobs",
                column_name="stage",
                definition="TEXT NOT NULL DEFAULT 'queued'",
            )
            self._ensure_column(
                conn,
                table_name="jobs",
                column_name="progress",
                definition="REAL NOT NULL DEFAULT 0.0",
            )
            self._ensure_column(
                conn,
                table_name="jobs",
                column_name="logs_json",
                definition="TEXT NOT NULL DEFAULT '[]'",
            )
            self._ensure_column(
                conn,
                table_name="jobs",
                column_name="payload_json",
                definition="TEXT NOT NULL DEFAULT '{}'",
            )
            self._ensure_column(
                conn,
                table_name="jobs",
                column_name="dedupe_key",
                definition="TEXT",
            )
            self._ensure_column(
                conn,
                table_name="jobs",
                column_name="priority",
                definition="INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                conn,
                table_name="jobs",
                column_name="cancel_requested",
                definition="INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                conn,
                table_name="jobs",
                column_name="created_at",
                definition="TEXT",
            )
            self._ensure_column(
                conn,
                table_name="jobs",
                column_name="started_at",
                definition="TEXT",
            )
            self._ensure_column(
                conn,
                table_name="jobs",
                column_name="finished_at",
                definition="TEXT",
            )
            self._ensure_column(
                conn,
                table_name="jobs",
                column_name="heartbeat_at",
                definition="TEXT",
            )
            self._ensure_column(
                conn,
                table_name="jobs",
                column_name="worker_id",
                definition="TEXT",
            )
            self._ensure_column(
                conn,
                table_name="jobs",
                column_name="error_message",
                definition="TEXT",
            )
            self._ensure_column(
                conn,
                table_name="jobs",
                column_name="entity_type",
                definition="TEXT",
            )
            self._ensure_column(
                conn,
                table_name="jobs",
                column_name="entity_id",
                definition="INTEGER",
            )
            self._ensure_column(
                conn,
                table_name="jobs",
                column_name="result_ref_type",
                definition="TEXT",
            )
            self._ensure_column(
                conn,
                table_name="jobs",
                column_name="result_ref_id",
                definition="INTEGER",
            )
            self._ensure_column(
                conn,
                table_name="jobs",
                column_name="rerun_of_job_id",
                definition="INTEGER",
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_workspace_id ON datasets(workspace_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_models_workspace_id ON models(workspace_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analysis_runs_workspace_id ON analysis_runs(workspace_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_workspace_id ON jobs(workspace_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status_priority ON jobs(status, priority, id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at, id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_dedupe_status ON jobs(dedupe_key, status)")
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
            conn.execute(
                "UPDATE labeler_projects SET quickshift_ratio = 1.0 WHERE quickshift_ratio IS NULL OR quickshift_ratio <= 0"
            )
            conn.execute(
                "UPDATE labeler_projects SET quickshift_kernel_size = 5 "
                "WHERE quickshift_kernel_size IS NULL OR quickshift_kernel_size <= 0"
            )
            conn.execute(
                "UPDATE labeler_projects SET quickshift_max_dist = 10.0 "
                "WHERE quickshift_max_dist IS NULL OR quickshift_max_dist <= 0"
            )
            conn.execute(
                "UPDATE labeler_projects SET quickshift_sigma = 0.0 WHERE quickshift_sigma IS NULL OR quickshift_sigma < 0"
            )
            conn.execute(
                "UPDATE labeler_projects SET felzenszwalb_scale = 100.0 "
                "WHERE felzenszwalb_scale IS NULL OR felzenszwalb_scale <= 0"
            )
            conn.execute(
                "UPDATE labeler_projects SET felzenszwalb_sigma = 0.8 "
                "WHERE felzenszwalb_sigma IS NULL OR felzenszwalb_sigma < 0"
            )
            conn.execute(
                "UPDATE labeler_projects SET felzenszwalb_min_size = 50 "
                "WHERE felzenszwalb_min_size IS NULL OR felzenszwalb_min_size <= 1"
            )
            conn.execute(
                "UPDATE labeler_projects SET texture_enabled = 0 WHERE texture_enabled IS NULL"
            )
            conn.execute(
                "UPDATE labeler_projects SET texture_mode = 'append_to_color' "
                "WHERE texture_mode IS NULL OR texture_mode = ''"
            )
            conn.execute(
                "UPDATE labeler_projects SET texture_lbp_enabled = 0 WHERE texture_lbp_enabled IS NULL"
            )
            conn.execute(
                "UPDATE labeler_projects SET texture_lbp_points = 8 "
                "WHERE texture_lbp_points IS NULL OR texture_lbp_points < 1"
            )
            conn.execute(
                "UPDATE labeler_projects SET texture_lbp_radii_json = '[1]' "
                "WHERE texture_lbp_radii_json IS NULL OR texture_lbp_radii_json = ''"
            )
            conn.execute(
                "UPDATE labeler_projects SET texture_lbp_method = 'uniform' "
                "WHERE texture_lbp_method IS NULL OR texture_lbp_method = ''"
            )
            conn.execute(
                "UPDATE labeler_projects SET texture_lbp_normalize = 1 WHERE texture_lbp_normalize IS NULL"
            )
            conn.execute(
                "UPDATE labeler_projects SET texture_gabor_enabled = 0 WHERE texture_gabor_enabled IS NULL"
            )
            conn.execute(
                "UPDATE labeler_projects SET texture_gabor_frequencies_json = '[0.1, 0.2]' "
                "WHERE texture_gabor_frequencies_json IS NULL OR texture_gabor_frequencies_json = ''"
            )
            conn.execute(
                "UPDATE labeler_projects SET texture_gabor_thetas_json = '[0.0, 45.0, 90.0, 135.0]' "
                "WHERE texture_gabor_thetas_json IS NULL OR texture_gabor_thetas_json = ''"
            )
            conn.execute(
                "UPDATE labeler_projects SET texture_gabor_bandwidth = 1.0 "
                "WHERE texture_gabor_bandwidth IS NULL OR texture_gabor_bandwidth <= 0"
            )
            conn.execute(
                "UPDATE labeler_projects SET texture_gabor_include_real = 0 WHERE texture_gabor_include_real IS NULL"
            )
            conn.execute(
                "UPDATE labeler_projects SET texture_gabor_include_imag = 0 WHERE texture_gabor_include_imag IS NULL"
            )
            conn.execute(
                "UPDATE labeler_projects SET texture_gabor_include_magnitude = 1 "
                "WHERE texture_gabor_include_magnitude IS NULL"
            )
            conn.execute(
                "UPDATE labeler_projects SET texture_gabor_normalize = 1 WHERE texture_gabor_normalize IS NULL"
            )
            conn.execute(
                "UPDATE labeler_projects SET texture_weight_color = 1.0 "
                "WHERE texture_weight_color IS NULL OR texture_weight_color <= 0"
            )
            conn.execute(
                "UPDATE labeler_projects SET texture_weight_lbp = 0.25 WHERE texture_weight_lbp IS NULL OR texture_weight_lbp < 0"
            )
            conn.execute(
                "UPDATE labeler_projects SET texture_weight_gabor = 0.25 "
                "WHERE texture_weight_gabor IS NULL OR texture_weight_gabor < 0"
            )
            conn.execute(
                "UPDATE labeler_image_slic_overrides SET quickshift_ratio = 1.0 "
                "WHERE quickshift_ratio IS NULL OR quickshift_ratio <= 0"
            )
            conn.execute(
                "UPDATE labeler_image_slic_overrides SET quickshift_kernel_size = 5 "
                "WHERE quickshift_kernel_size IS NULL OR quickshift_kernel_size <= 0"
            )
            conn.execute(
                "UPDATE labeler_image_slic_overrides SET quickshift_max_dist = 10.0 "
                "WHERE quickshift_max_dist IS NULL OR quickshift_max_dist <= 0"
            )
            conn.execute(
                "UPDATE labeler_image_slic_overrides SET quickshift_sigma = 0.0 "
                "WHERE quickshift_sigma IS NULL OR quickshift_sigma < 0"
            )
            conn.execute(
                "UPDATE labeler_image_slic_overrides SET felzenszwalb_scale = 100.0 "
                "WHERE felzenszwalb_scale IS NULL OR felzenszwalb_scale <= 0"
            )
            conn.execute(
                "UPDATE labeler_image_slic_overrides SET felzenszwalb_sigma = 0.8 "
                "WHERE felzenszwalb_sigma IS NULL OR felzenszwalb_sigma < 0"
            )
            conn.execute(
                "UPDATE labeler_image_slic_overrides SET felzenszwalb_min_size = 50 "
                "WHERE felzenszwalb_min_size IS NULL OR felzenszwalb_min_size <= 1"
            )
            conn.execute(
                "UPDATE labeler_image_slic_overrides SET texture_enabled = 0 WHERE texture_enabled IS NULL"
            )
            conn.execute(
                "UPDATE labeler_image_slic_overrides SET texture_mode = 'append_to_color' "
                "WHERE texture_mode IS NULL OR texture_mode = ''"
            )
            conn.execute(
                "UPDATE labeler_image_slic_overrides SET texture_lbp_enabled = 0 WHERE texture_lbp_enabled IS NULL"
            )
            conn.execute(
                "UPDATE labeler_image_slic_overrides SET texture_lbp_points = 8 "
                "WHERE texture_lbp_points IS NULL OR texture_lbp_points < 1"
            )
            conn.execute(
                "UPDATE labeler_image_slic_overrides SET texture_lbp_radii_json = '[1]' "
                "WHERE texture_lbp_radii_json IS NULL OR texture_lbp_radii_json = ''"
            )
            conn.execute(
                "UPDATE labeler_image_slic_overrides SET texture_lbp_method = 'uniform' "
                "WHERE texture_lbp_method IS NULL OR texture_lbp_method = ''"
            )
            conn.execute(
                "UPDATE labeler_image_slic_overrides SET texture_lbp_normalize = 1 "
                "WHERE texture_lbp_normalize IS NULL"
            )
            conn.execute(
                "UPDATE labeler_image_slic_overrides SET texture_gabor_enabled = 0 WHERE texture_gabor_enabled IS NULL"
            )
            conn.execute(
                "UPDATE labeler_image_slic_overrides SET texture_gabor_frequencies_json = '[0.1, 0.2]' "
                "WHERE texture_gabor_frequencies_json IS NULL OR texture_gabor_frequencies_json = ''"
            )
            conn.execute(
                "UPDATE labeler_image_slic_overrides SET texture_gabor_thetas_json = '[0.0, 45.0, 90.0, 135.0]' "
                "WHERE texture_gabor_thetas_json IS NULL OR texture_gabor_thetas_json = ''"
            )
            conn.execute(
                "UPDATE labeler_image_slic_overrides SET texture_gabor_bandwidth = 1.0 "
                "WHERE texture_gabor_bandwidth IS NULL OR texture_gabor_bandwidth <= 0"
            )
            conn.execute(
                "UPDATE labeler_image_slic_overrides SET texture_gabor_include_real = 0 "
                "WHERE texture_gabor_include_real IS NULL"
            )
            conn.execute(
                "UPDATE labeler_image_slic_overrides SET texture_gabor_include_imag = 0 "
                "WHERE texture_gabor_include_imag IS NULL"
            )
            conn.execute(
                "UPDATE labeler_image_slic_overrides SET texture_gabor_include_magnitude = 1 "
                "WHERE texture_gabor_include_magnitude IS NULL"
            )
            conn.execute(
                "UPDATE labeler_image_slic_overrides SET texture_gabor_normalize = 1 "
                "WHERE texture_gabor_normalize IS NULL"
            )
            conn.execute(
                "UPDATE labeler_image_slic_overrides SET texture_weight_color = 1.0 "
                "WHERE texture_weight_color IS NULL OR texture_weight_color <= 0"
            )
            conn.execute(
                "UPDATE labeler_image_slic_overrides SET texture_weight_lbp = 0.25 "
                "WHERE texture_weight_lbp IS NULL OR texture_weight_lbp < 0"
            )
            conn.execute(
                "UPDATE labeler_image_slic_overrides SET texture_weight_gabor = 0.25 "
                "WHERE texture_weight_gabor IS NULL OR texture_weight_gabor < 0"
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
            conn.execute("UPDATE jobs SET status = 'queued' WHERE status IS NULL OR status = ''")
            conn.execute("UPDATE jobs SET stage = status WHERE stage IS NULL OR stage = ''")
            conn.execute("UPDATE jobs SET progress = 0.0 WHERE progress IS NULL")
            conn.execute("UPDATE jobs SET logs_json = '[]' WHERE logs_json IS NULL OR logs_json = ''")
            conn.execute("UPDATE jobs SET payload_json = '{}' WHERE payload_json IS NULL OR payload_json = ''")
            conn.execute("UPDATE jobs SET cancel_requested = 0 WHERE cancel_requested IS NULL")
            conn.execute("UPDATE jobs SET created_at = ? WHERE created_at IS NULL OR created_at = ''", (_utc_now_iso(),))
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

    def update_model(
        self,
        model_id: int,
        *,
        status: str | None = None,
        error_message: str | None | object = _UNSET,
        classes: list[dict[str, Any]] | None = None,
        feature_config: dict[str, Any] | None = None,
        hyperparams: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        artifact_dir: str | None = None,
        model_path: str | None = None,
        metadata_path: str | None = None,
        workspace_id: int | None = None,
    ) -> None:
        assignments: list[str] = []
        payload: list[Any] = []
        if status is not None:
            assignments.append("status = ?")
            payload.append(status)
        if error_message is not _UNSET:
            assignments.append("error_message = ?")
            payload.append(error_message)
        if classes is not None:
            assignments.append("classes_json = ?")
            payload.append(json.dumps(classes))
        if feature_config is not None:
            assignments.append("feature_config_json = ?")
            payload.append(json.dumps(feature_config))
        if hyperparams is not None:
            assignments.append("hyperparams_json = ?")
            payload.append(json.dumps(hyperparams))
        if metrics is not None:
            assignments.append("metrics_json = ?")
            payload.append(json.dumps(metrics))
        if artifact_dir is not None:
            assignments.append("artifact_dir = ?")
            payload.append(artifact_dir)
        if model_path is not None:
            assignments.append("model_path = ?")
            payload.append(model_path)
        if metadata_path is not None:
            assignments.append("metadata_path = ?")
            payload.append(metadata_path)
        if not assignments:
            return

        if workspace_id is None:
            sql = f"""
            UPDATE models
            SET {", ".join(assignments)}
            WHERE id = ?
            """
            payload.append(model_id)
        else:
            sql = f"""
            UPDATE models
            SET {", ".join(assignments)}
            WHERE id = ? AND workspace_id = ?
            """
            payload.extend((model_id, workspace_id))
        with self._connect() as conn:
            conn.execute(sql, tuple(payload))
            conn.commit()

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

    def update_analysis_run_state(
        self,
        run_id: int,
        *,
        status: str,
        error_message: str | None | object = _UNSET,
        workspace_id: int | None = None,
    ) -> None:
        assignments = ["status = ?"]
        payload: list[Any] = [status]
        if error_message is not _UNSET:
            assignments.append("error_message = ?")
            payload.append(error_message)
        if workspace_id is None:
            sql = f"""
            UPDATE analysis_runs
            SET {", ".join(assignments)}
            WHERE id = ?
            """
            payload.append(run_id)
        else:
            sql = f"""
            UPDATE analysis_runs
            SET {", ".join(assignments)}
            WHERE id = ? AND workspace_id = ?
            """
            payload.extend((run_id, workspace_id))
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

    @staticmethod
    def _clamp_progress(value: float | int | None) -> float:
        if value is None:
            return 0.0
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        if numeric < 0:
            return 0.0
        if numeric > 1:
            return 1.0
        return numeric

    @staticmethod
    def _next_queue_priority(conn: sqlite3.Connection) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(priority), 0) AS max_priority FROM jobs WHERE status = 'queued'"
        ).fetchone()
        max_priority = int(row["max_priority"]) if row is not None and row["max_priority"] is not None else 0
        return max_priority + 10

    def enqueue_job(
        self,
        *,
        workspace_id: int,
        job_type: str,
        payload: dict[str, Any],
        dedupe_key: str | None = None,
        entity_type: str | None = None,
        entity_id: int | None = None,
        rerun_of_job_id: int | None = None,
    ) -> tuple[int, bool]:
        created_at = _utc_now_iso()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if dedupe_key:
                existing = conn.execute(
                    """
                    SELECT id
                    FROM jobs
                    WHERE dedupe_key = ? AND status IN ('queued', 'running')
                    ORDER BY id ASC
                    LIMIT 1
                    """,
                    (dedupe_key,),
                ).fetchone()
                if existing is not None:
                    conn.commit()
                    return int(existing["id"]), False
            priority = self._next_queue_priority(conn)
            cursor = conn.execute(
                """
                INSERT INTO jobs (
                    workspace_id, job_type, status, stage, progress, logs_json, payload_json,
                    dedupe_key, priority, cancel_requested, created_at, started_at, finished_at,
                    heartbeat_at, worker_id, error_message, entity_type, entity_id, result_ref_type,
                    result_ref_id, rerun_of_job_id
                )
                VALUES (?, ?, 'queued', 'queued', ?, ?, ?, ?, ?, 0, ?, NULL, NULL, ?, NULL, NULL, ?, ?, NULL, NULL, ?)
                """,
                (
                    workspace_id,
                    str(job_type or "").strip().lower(),
                    0.0,
                    json.dumps([]),
                    json.dumps(payload),
                    dedupe_key,
                    priority,
                    created_at,
                    created_at,
                    entity_type,
                    entity_id,
                    rerun_of_job_id,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid), True

    def find_active_job_by_dedupe(
        self,
        *,
        dedupe_key: str,
        workspace_id: int | None = None,
        job_type: str | None = None,
    ) -> dict[str, Any] | None:
        where = ["dedupe_key = ?", "status IN ('queued', 'running')"]
        params: list[Any] = [dedupe_key]
        if workspace_id is not None:
            where.append("workspace_id = ?")
            params.append(workspace_id)
        if job_type is not None:
            where.append("job_type = ?")
            params.append(str(job_type).strip().lower())
        sql = f"""
        SELECT *
        FROM jobs
        WHERE {" AND ".join(where)}
        ORDER BY id ASC
        LIMIT 1
        """
        with self._connect() as conn:
            row = conn.execute(sql, tuple(params)).fetchone()
        if row is None:
            return None
        return self._decode_json_fields(dict(row))

    def get_job(self, job_id: int, *, workspace_id: int | None = None) -> dict[str, Any] | None:
        if workspace_id is None:
            sql = "SELECT * FROM jobs WHERE id = ?"
            params: tuple[Any, ...] = (job_id,)
        else:
            sql = "SELECT * FROM jobs WHERE id = ? AND workspace_id = ?"
            params = (job_id, workspace_id)
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        if row is None:
            return None
        return self._decode_json_fields(dict(row))

    def list_jobs(
        self,
        *,
        workspace_id: int | None = None,
        limit: int = 200,
        job_type: str | None = None,
        statuses: list[str] | tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if workspace_id is not None:
            where.append("workspace_id = ?")
            params.append(workspace_id)
        if job_type is not None:
            where.append("job_type = ?")
            params.append(str(job_type).strip().lower())
        if statuses:
            normalized = [str(item).strip().lower() for item in statuses if str(item).strip()]
            if normalized:
                placeholders = ", ".join("?" for _ in normalized)
                where.append(f"status IN ({placeholders})")
                params.extend(normalized)
        sql = "SELECT * FROM jobs"
        if where:
            sql += f" WHERE {' AND '.join(where)}"
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._decode_json_fields(dict(row)) for row in rows]

    def list_queued_jobs(self, *, limit: int = 500) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM jobs
                WHERE status = 'queued'
                ORDER BY
                    CASE
                        WHEN job_type IN ('training', 'analysis') THEN 0
                        WHEN job_type = 'slic_warmup' THEN 1
                        ELSE 2
                    END ASC,
                    priority ASC,
                    created_at ASC,
                    id ASC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [self._decode_json_fields(dict(row)) for row in rows]

    def list_running_jobs(self, *, limit: int = 500) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM jobs
                WHERE status = 'running'
                ORDER BY started_at ASC, id ASC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [self._decode_json_fields(dict(row)) for row in rows]

    def claim_next_queued_job(self, *, worker_id: str) -> dict[str, Any] | None:
        now = _utc_now_iso()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT id
                FROM jobs
                WHERE status = 'queued' AND cancel_requested = 0
                ORDER BY
                    CASE
                        WHEN job_type IN ('training', 'analysis') THEN 0
                        WHEN job_type = 'slic_warmup' THEN 1
                        ELSE 2
                    END ASC,
                    priority ASC,
                    created_at ASC,
                    id ASC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            job_id = int(row["id"])
            conn.execute(
                """
                UPDATE jobs
                SET
                    status = 'running',
                    stage = CASE
                        WHEN stage IS NULL OR stage = '' OR stage = 'queued' THEN 'starting'
                        ELSE stage
                    END,
                    progress = CASE
                        WHEN progress < 0.01 THEN 0.01
                        ELSE progress
                    END,
                    started_at = COALESCE(started_at, ?),
                    heartbeat_at = ?,
                    worker_id = ?,
                    error_message = NULL
                WHERE id = ? AND status = 'queued'
                """,
                (now, now, worker_id, job_id),
            )
            claimed = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            conn.commit()
        if claimed is None:
            return None
        return self._decode_json_fields(dict(claimed))

    def update_job(
        self,
        job_id: int,
        *,
        status: str | None = None,
        stage: str | None = None,
        progress: float | None = None,
        error_message: str | None | object = _UNSET,
        cancel_requested: bool | None = None,
        worker_id: str | None = None,
        result_ref_type: str | None = None,
        result_ref_id: int | None = None,
        finished: bool = False,
        heartbeat: bool = False,
        workspace_id: int | None = None,
    ) -> None:
        assignments: list[str] = []
        payload: list[Any] = []
        now = _utc_now_iso()
        if status is not None:
            assignments.append("status = ?")
            payload.append(str(status).strip().lower())
        if stage is not None:
            assignments.append("stage = ?")
            payload.append(str(stage).strip().lower())
        if progress is not None:
            assignments.append("progress = ?")
            payload.append(self._clamp_progress(progress))
        if error_message is not _UNSET:
            assignments.append("error_message = ?")
            payload.append(error_message)
        if cancel_requested is not None:
            assignments.append("cancel_requested = ?")
            payload.append(1 if cancel_requested else 0)
        if worker_id is not None:
            assignments.append("worker_id = ?")
            payload.append(worker_id)
        if result_ref_type is not None:
            assignments.append("result_ref_type = ?")
            payload.append(result_ref_type)
        if result_ref_id is not None:
            assignments.append("result_ref_id = ?")
            payload.append(int(result_ref_id))
        if finished:
            assignments.append("finished_at = ?")
            payload.append(now)
        if heartbeat:
            assignments.append("heartbeat_at = ?")
            payload.append(now)
        if not assignments:
            return
        if workspace_id is None:
            sql = f"""
            UPDATE jobs
            SET {", ".join(assignments)}
            WHERE id = ?
            """
            payload.append(job_id)
        else:
            sql = f"""
            UPDATE jobs
            SET {", ".join(assignments)}
            WHERE id = ? AND workspace_id = ?
            """
            payload.extend((job_id, workspace_id))
        with self._connect() as conn:
            conn.execute(sql, tuple(payload))
            conn.commit()

    def append_job_log(
        self,
        job_id: int,
        *,
        message: str,
        stage: str | None = None,
        progress: float | None = None,
        workspace_id: int | None = None,
    ) -> None:
        now = _utc_now_iso()
        if workspace_id is None:
            select_sql = "SELECT logs_json FROM jobs WHERE id = ?"
            select_params: tuple[Any, ...] = (job_id,)
            where_clause = "id = ?"
            where_params: tuple[Any, ...] = (job_id,)
        else:
            select_sql = "SELECT logs_json FROM jobs WHERE id = ? AND workspace_id = ?"
            select_params = (job_id, workspace_id)
            where_clause = "id = ? AND workspace_id = ?"
            where_params = (job_id, workspace_id)
        with self._connect() as conn:
            row = conn.execute(select_sql, select_params).fetchone()
            if row is None:
                return
            raw_logs = row["logs_json"]
            logs: list[dict[str, Any]]
            if isinstance(raw_logs, str):
                try:
                    parsed = json.loads(raw_logs)
                    logs = parsed if isinstance(parsed, list) else []
                except json.JSONDecodeError:
                    logs = []
            elif isinstance(raw_logs, list):
                logs = raw_logs
            else:
                logs = []
            logs.append({"at": now, "message": str(message)})
            logs = logs[-500:]
            assignments = [
                "logs_json = ?",
                "heartbeat_at = ?",
            ]
            payload: list[Any] = [json.dumps(logs), now]
            if stage is not None:
                assignments.append("stage = ?")
                payload.append(str(stage).strip().lower())
            if progress is not None:
                assignments.append("progress = ?")
                payload.append(self._clamp_progress(progress))
            payload.extend(where_params)
            conn.execute(
                f"""
                UPDATE jobs
                SET {", ".join(assignments)}
                WHERE {where_clause}
                """,
                tuple(payload),
            )
            conn.commit()

    def is_job_cancel_requested(self, job_id: int) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT cancel_requested FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return False
        return bool(int(row["cancel_requested"]))

    def request_job_cancel(
        self,
        job_id: int,
        *,
        workspace_id: int | None = None,
    ) -> dict[str, Any] | None:
        now = _utc_now_iso()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if workspace_id is None:
                row = conn.execute(
                    "SELECT id, status FROM jobs WHERE id = ?",
                    (job_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT id, status FROM jobs WHERE id = ? AND workspace_id = ?",
                    (job_id, workspace_id),
                ).fetchone()
            if row is None:
                conn.commit()
                return None
            status = str(row["status"]).strip().lower()
            if status == "queued":
                if workspace_id is None:
                    conn.execute(
                        """
                        UPDATE jobs
                        SET
                            status = 'canceled',
                            stage = 'canceled',
                            cancel_requested = 1,
                            error_message = 'Canceled by user.',
                            finished_at = ?,
                            heartbeat_at = ?
                        WHERE id = ?
                        """,
                        (now, now, job_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE jobs
                        SET
                            status = 'canceled',
                            stage = 'canceled',
                            cancel_requested = 1,
                            error_message = 'Canceled by user.',
                            finished_at = ?,
                            heartbeat_at = ?
                        WHERE id = ? AND workspace_id = ?
                        """,
                        (now, now, job_id, workspace_id),
                    )
                conn.commit()
                return {"status": "canceled", "immediate": True}
            if status == "running":
                if workspace_id is None:
                    conn.execute(
                        """
                        UPDATE jobs
                        SET
                            cancel_requested = 1,
                            stage = 'cancel_requested',
                            heartbeat_at = ?
                        WHERE id = ?
                        """,
                        (now, job_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE jobs
                        SET
                            cancel_requested = 1,
                            stage = 'cancel_requested',
                            heartbeat_at = ?
                        WHERE id = ? AND workspace_id = ?
                        """,
                        (now, job_id, workspace_id),
                    )
                conn.commit()
                return {"status": "running", "immediate": False}
            conn.commit()
            return {"status": status, "immediate": False}

    def mark_job_completed(
        self,
        job_id: int,
        *,
        result_ref_type: str | None = None,
        result_ref_id: int | None = None,
        stage: str = "completed",
    ) -> None:
        self.update_job(
            job_id,
            status="completed",
            stage=stage,
            progress=1.0,
            error_message=None,
            cancel_requested=False,
            result_ref_type=result_ref_type,
            result_ref_id=result_ref_id,
            finished=True,
            heartbeat=True,
        )

    def mark_job_failed(self, job_id: int, *, error_message: str) -> None:
        self.update_job(
            job_id,
            status="failed",
            stage="failed",
            error_message=error_message,
            finished=True,
            heartbeat=True,
        )

    def mark_job_canceled(self, job_id: int, *, error_message: str = "Canceled by user.") -> None:
        self.update_job(
            job_id,
            status="canceled",
            stage="canceled",
            error_message=error_message,
            cancel_requested=True,
            finished=True,
            heartbeat=True,
        )

    def reorder_workspace_queued_jobs(
        self,
        workspace_id: int,
        ordered_job_ids: list[int],
    ) -> bool:
        desired_ids = [int(item) for item in ordered_job_ids]
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            global_rows = conn.execute(
                """
                SELECT id, workspace_id
                FROM jobs
                WHERE status = 'queued'
                ORDER BY
                    CASE
                        WHEN job_type IN ('training', 'analysis') THEN 0
                        WHEN job_type = 'slic_warmup' THEN 1
                        ELSE 2
                    END ASC,
                    priority ASC,
                    created_at ASC,
                    id ASC
                """
            ).fetchall()
            if not global_rows:
                conn.commit()
                return False
            global_ids = [int(row["id"]) for row in global_rows]
            workspace_queued_ids = [
                int(row["id"])
                for row in global_rows
                if int(row["workspace_id"]) == int(workspace_id)
            ]
            if len(workspace_queued_ids) <= 1:
                conn.commit()
                return False
            ordered_filtered = [job_id for job_id in desired_ids if job_id in workspace_queued_ids]
            for job_id in workspace_queued_ids:
                if job_id not in ordered_filtered:
                    ordered_filtered.append(job_id)
            if ordered_filtered == workspace_queued_ids:
                conn.commit()
                return False
            original_positions = [global_ids.index(job_id) for job_id in workspace_queued_ids]
            insert_at = min(original_positions) if original_positions else len(global_ids)
            remainder = [job_id for job_id in global_ids if job_id not in workspace_queued_ids]
            reordered_global = remainder[:insert_at] + ordered_filtered + remainder[insert_at:]
            for index, job_id in enumerate(reordered_global, start=1):
                conn.execute(
                    "UPDATE jobs SET priority = ? WHERE id = ?",
                    (index * 10, job_id),
                )
            conn.commit()
            return True

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
        quickshift_ratio: float = 1.0,
        quickshift_kernel_size: int = 5,
        quickshift_max_dist: float = 10.0,
        quickshift_sigma: float = 0.0,
        felzenszwalb_scale: float = 100.0,
        felzenszwalb_sigma: float = 0.8,
        felzenszwalb_min_size: int = 50,
        texture_enabled: bool = False,
        texture_mode: str = "append_to_color",
        texture_lbp_enabled: bool = False,
        texture_lbp_points: int = 8,
        texture_lbp_radii: list[int] | tuple[int, ...] = (1,),
        texture_lbp_method: str = "uniform",
        texture_lbp_normalize: bool = True,
        texture_gabor_enabled: bool = False,
        texture_gabor_frequencies: list[float] | tuple[float, ...] = (0.1, 0.2),
        texture_gabor_thetas: list[float] | tuple[float, ...] = (0.0, 45.0, 90.0, 135.0),
        texture_gabor_bandwidth: float = 1.0,
        texture_gabor_include_real: bool = False,
        texture_gabor_include_imag: bool = False,
        texture_gabor_include_magnitude: bool = True,
        texture_gabor_normalize: bool = True,
        texture_weight_color: float = 1.0,
        texture_weight_lbp: float = 0.25,
        texture_weight_gabor: float = 0.25,
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
        normalized_texture_mode = str(texture_mode or "append_to_color").strip().lower() or "append_to_color"
        if normalized_texture_mode not in {"append_to_color"}:
            normalized_texture_mode = "append_to_color"
        normalized_texture_lbp_method = str(texture_lbp_method or "uniform").strip().lower() or "uniform"
        if normalized_texture_lbp_method not in {"uniform", "ror", "default"}:
            normalized_texture_lbp_method = "uniform"
        normalized_texture_lbp_radii = [int(value) for value in texture_lbp_radii if int(value) > 0]
        if not normalized_texture_lbp_radii:
            normalized_texture_lbp_radii = [1]
        normalized_texture_gabor_frequencies = [float(value) for value in texture_gabor_frequencies if float(value) > 0]
        if not normalized_texture_gabor_frequencies:
            normalized_texture_gabor_frequencies = [0.1, 0.2]
        normalized_texture_gabor_thetas = [float(value) for value in texture_gabor_thetas]
        if not normalized_texture_gabor_thetas:
            normalized_texture_gabor_thetas = [0.0, 45.0, 90.0, 135.0]
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
            max(0.01, float(quickshift_ratio)),
            max(1, int(quickshift_kernel_size)),
            max(0.01, float(quickshift_max_dist)),
            max(0.0, float(quickshift_sigma)),
            max(0.01, float(felzenszwalb_scale)),
            max(0.0, float(felzenszwalb_sigma)),
            max(2, int(felzenszwalb_min_size)),
            1 if bool(texture_enabled) else 0,
            normalized_texture_mode,
            1 if bool(texture_lbp_enabled) else 0,
            max(1, int(texture_lbp_points)),
            json.dumps(normalized_texture_lbp_radii),
            normalized_texture_lbp_method,
            1 if bool(texture_lbp_normalize) else 0,
            1 if bool(texture_gabor_enabled) else 0,
            json.dumps(normalized_texture_gabor_frequencies),
            json.dumps(normalized_texture_gabor_thetas),
            max(0.01, float(texture_gabor_bandwidth)),
            1 if bool(texture_gabor_include_real) else 0,
            1 if bool(texture_gabor_include_imag) else 0,
            1 if bool(texture_gabor_include_magnitude) else 0,
            1 if bool(texture_gabor_normalize) else 0,
            max(0.01, float(texture_weight_color)),
            max(0.0, float(texture_weight_lbp)),
            max(0.0, float(texture_weight_gabor)),
            json.dumps(normalized_categories),
            now,
            now,
        )
        sql = """
        INSERT INTO labeler_projects (
            name, dataset_id, kind, parent_workspace_id, augmentation_seed, draft_version, origin_type, origin_draft_project_id,
            origin_draft_version, origin_registered_dataset_id, project_dir, images_dir, masks_dir, coco_dir, cache_dir,
            slic_algorithm, slic_preset_name, slic_detail_level, slic_n_segments, slic_compactness, slic_sigma, slic_colorspace,
            quickshift_ratio, quickshift_kernel_size, quickshift_max_dist, quickshift_sigma,
            felzenszwalb_scale, felzenszwalb_sigma, felzenszwalb_min_size,
            texture_enabled, texture_mode,
            texture_lbp_enabled, texture_lbp_points, texture_lbp_radii_json, texture_lbp_method, texture_lbp_normalize,
            texture_gabor_enabled, texture_gabor_frequencies_json, texture_gabor_thetas_json, texture_gabor_bandwidth,
            texture_gabor_include_real, texture_gabor_include_imag, texture_gabor_include_magnitude, texture_gabor_normalize,
            texture_weight_color, texture_weight_lbp, texture_weight_gabor,
            categories_json, created_at, updated_at
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
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
        quickshift_ratio: float = 1.0,
        quickshift_kernel_size: int = 5,
        quickshift_max_dist: float = 10.0,
        quickshift_sigma: float = 0.0,
        felzenszwalb_scale: float = 100.0,
        felzenszwalb_sigma: float = 0.8,
        felzenszwalb_min_size: int = 50,
        texture_enabled: bool = False,
        texture_mode: str = "append_to_color",
        texture_lbp_enabled: bool = False,
        texture_lbp_points: int = 8,
        texture_lbp_radii: list[int] | tuple[int, ...] = (1,),
        texture_lbp_method: str = "uniform",
        texture_lbp_normalize: bool = True,
        texture_gabor_enabled: bool = False,
        texture_gabor_frequencies: list[float] | tuple[float, ...] = (0.1, 0.2),
        texture_gabor_thetas: list[float] | tuple[float, ...] = (0.0, 45.0, 90.0, 135.0),
        texture_gabor_bandwidth: float = 1.0,
        texture_gabor_include_real: bool = False,
        texture_gabor_include_imag: bool = False,
        texture_gabor_include_magnitude: bool = True,
        texture_gabor_normalize: bool = True,
        texture_weight_color: float = 1.0,
        texture_weight_lbp: float = 0.25,
        texture_weight_gabor: float = 0.25,
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
            quickshift_ratio=quickshift_ratio,
            quickshift_kernel_size=quickshift_kernel_size,
            quickshift_max_dist=quickshift_max_dist,
            quickshift_sigma=quickshift_sigma,
            felzenszwalb_scale=felzenszwalb_scale,
            felzenszwalb_sigma=felzenszwalb_sigma,
            felzenszwalb_min_size=felzenszwalb_min_size,
            texture_enabled=texture_enabled,
            texture_mode=texture_mode,
            texture_lbp_enabled=texture_lbp_enabled,
            texture_lbp_points=texture_lbp_points,
            texture_lbp_radii=texture_lbp_radii,
            texture_lbp_method=texture_lbp_method,
            texture_lbp_normalize=texture_lbp_normalize,
            texture_gabor_enabled=texture_gabor_enabled,
            texture_gabor_frequencies=texture_gabor_frequencies,
            texture_gabor_thetas=texture_gabor_thetas,
            texture_gabor_bandwidth=texture_gabor_bandwidth,
            texture_gabor_include_real=texture_gabor_include_real,
            texture_gabor_include_imag=texture_gabor_include_imag,
            texture_gabor_include_magnitude=texture_gabor_include_magnitude,
            texture_gabor_normalize=texture_gabor_normalize,
            texture_weight_color=texture_weight_color,
            texture_weight_lbp=texture_weight_lbp,
            texture_weight_gabor=texture_weight_gabor,
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
        quickshift_ratio: float,
        quickshift_kernel_size: int,
        quickshift_max_dist: float,
        quickshift_sigma: float,
        felzenszwalb_scale: float,
        felzenszwalb_sigma: float,
        felzenszwalb_min_size: int,
        texture_enabled: bool,
        texture_mode: str,
        texture_lbp_enabled: bool,
        texture_lbp_points: int,
        texture_lbp_radii: list[int] | tuple[int, ...],
        texture_lbp_method: str,
        texture_lbp_normalize: bool,
        texture_gabor_enabled: bool,
        texture_gabor_frequencies: list[float] | tuple[float, ...],
        texture_gabor_thetas: list[float] | tuple[float, ...],
        texture_gabor_bandwidth: float,
        texture_gabor_include_real: bool,
        texture_gabor_include_imag: bool,
        texture_gabor_include_magnitude: bool,
        texture_gabor_normalize: bool,
        texture_weight_color: float,
        texture_weight_lbp: float,
        texture_weight_gabor: float,
    ) -> None:
        normalized_texture_mode = str(texture_mode or "append_to_color").strip().lower() or "append_to_color"
        if normalized_texture_mode not in {"append_to_color"}:
            normalized_texture_mode = "append_to_color"
        normalized_texture_lbp_method = str(texture_lbp_method or "uniform").strip().lower() or "uniform"
        if normalized_texture_lbp_method not in {"uniform", "ror", "default"}:
            normalized_texture_lbp_method = "uniform"
        normalized_texture_lbp_radii = [int(value) for value in texture_lbp_radii if int(value) > 0]
        if not normalized_texture_lbp_radii:
            normalized_texture_lbp_radii = [1]
        normalized_texture_gabor_frequencies = [float(value) for value in texture_gabor_frequencies if float(value) > 0]
        if not normalized_texture_gabor_frequencies:
            normalized_texture_gabor_frequencies = [0.1, 0.2]
        normalized_texture_gabor_thetas = [float(value) for value in texture_gabor_thetas]
        if not normalized_texture_gabor_thetas:
            normalized_texture_gabor_thetas = [0.0, 45.0, 90.0, 135.0]
        payload = (
            str(slic_algorithm or "slic").strip().lower() or "slic",
            str(preset_name or "medium").strip().lower() or "medium",
            str(detail_level or "medium").strip().lower() or "medium",
            max(50, int(n_segments)),
            max(0.01, float(compactness)),
            max(0.0, float(sigma)),
            str(colorspace or "lab").strip().lower() or "lab",
            max(0.01, float(quickshift_ratio)),
            max(1, int(quickshift_kernel_size)),
            max(0.01, float(quickshift_max_dist)),
            max(0.0, float(quickshift_sigma)),
            max(0.01, float(felzenszwalb_scale)),
            max(0.0, float(felzenszwalb_sigma)),
            max(2, int(felzenszwalb_min_size)),
            1 if bool(texture_enabled) else 0,
            normalized_texture_mode,
            1 if bool(texture_lbp_enabled) else 0,
            max(1, int(texture_lbp_points)),
            json.dumps(normalized_texture_lbp_radii),
            normalized_texture_lbp_method,
            1 if bool(texture_lbp_normalize) else 0,
            1 if bool(texture_gabor_enabled) else 0,
            json.dumps(normalized_texture_gabor_frequencies),
            json.dumps(normalized_texture_gabor_thetas),
            max(0.01, float(texture_gabor_bandwidth)),
            1 if bool(texture_gabor_include_real) else 0,
            1 if bool(texture_gabor_include_imag) else 0,
            1 if bool(texture_gabor_include_magnitude) else 0,
            1 if bool(texture_gabor_normalize) else 0,
            max(0.01, float(texture_weight_color)),
            max(0.0, float(texture_weight_lbp)),
            max(0.0, float(texture_weight_gabor)),
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
            quickshift_ratio = ?,
            quickshift_kernel_size = ?,
            quickshift_max_dist = ?,
            quickshift_sigma = ?,
            felzenszwalb_scale = ?,
            felzenszwalb_sigma = ?,
            felzenszwalb_min_size = ?,
            texture_enabled = ?,
            texture_mode = ?,
            texture_lbp_enabled = ?,
            texture_lbp_points = ?,
            texture_lbp_radii_json = ?,
            texture_lbp_method = ?,
            texture_lbp_normalize = ?,
            texture_gabor_enabled = ?,
            texture_gabor_frequencies_json = ?,
            texture_gabor_thetas_json = ?,
            texture_gabor_bandwidth = ?,
            texture_gabor_include_real = ?,
            texture_gabor_include_imag = ?,
            texture_gabor_include_magnitude = ?,
            texture_gabor_normalize = ?,
            texture_weight_color = ?,
            texture_weight_lbp = ?,
            texture_weight_gabor = ?,
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
        return self._decode_json_fields(dict(row))

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
        quickshift_ratio: float,
        quickshift_kernel_size: int,
        quickshift_max_dist: float,
        quickshift_sigma: float,
        felzenszwalb_scale: float,
        felzenszwalb_sigma: float,
        felzenszwalb_min_size: int,
        texture_enabled: bool,
        texture_mode: str,
        texture_lbp_enabled: bool,
        texture_lbp_points: int,
        texture_lbp_radii: list[int] | tuple[int, ...],
        texture_lbp_method: str,
        texture_lbp_normalize: bool,
        texture_gabor_enabled: bool,
        texture_gabor_frequencies: list[float] | tuple[float, ...],
        texture_gabor_thetas: list[float] | tuple[float, ...],
        texture_gabor_bandwidth: float,
        texture_gabor_include_real: bool,
        texture_gabor_include_imag: bool,
        texture_gabor_include_magnitude: bool,
        texture_gabor_normalize: bool,
        texture_weight_color: float,
        texture_weight_lbp: float,
        texture_weight_gabor: float,
    ) -> None:
        normalized_name = Path(str(image_name)).name
        normalized_texture_mode = str(texture_mode or "append_to_color").strip().lower() or "append_to_color"
        if normalized_texture_mode not in {"append_to_color"}:
            normalized_texture_mode = "append_to_color"
        normalized_texture_lbp_method = str(texture_lbp_method or "uniform").strip().lower() or "uniform"
        if normalized_texture_lbp_method not in {"uniform", "ror", "default"}:
            normalized_texture_lbp_method = "uniform"
        normalized_texture_lbp_radii = [int(value) for value in texture_lbp_radii if int(value) > 0]
        if not normalized_texture_lbp_radii:
            normalized_texture_lbp_radii = [1]
        normalized_texture_gabor_frequencies = [float(value) for value in texture_gabor_frequencies if float(value) > 0]
        if not normalized_texture_gabor_frequencies:
            normalized_texture_gabor_frequencies = [0.1, 0.2]
        normalized_texture_gabor_thetas = [float(value) for value in texture_gabor_thetas]
        if not normalized_texture_gabor_thetas:
            normalized_texture_gabor_thetas = [0.0, 45.0, 90.0, 135.0]
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
            max(0.01, float(quickshift_ratio)),
            max(1, int(quickshift_kernel_size)),
            max(0.01, float(quickshift_max_dist)),
            max(0.0, float(quickshift_sigma)),
            max(0.01, float(felzenszwalb_scale)),
            max(0.0, float(felzenszwalb_sigma)),
            max(2, int(felzenszwalb_min_size)),
            1 if bool(texture_enabled) else 0,
            normalized_texture_mode,
            1 if bool(texture_lbp_enabled) else 0,
            max(1, int(texture_lbp_points)),
            json.dumps(normalized_texture_lbp_radii),
            normalized_texture_lbp_method,
            1 if bool(texture_lbp_normalize) else 0,
            1 if bool(texture_gabor_enabled) else 0,
            json.dumps(normalized_texture_gabor_frequencies),
            json.dumps(normalized_texture_gabor_thetas),
            max(0.01, float(texture_gabor_bandwidth)),
            1 if bool(texture_gabor_include_real) else 0,
            1 if bool(texture_gabor_include_imag) else 0,
            1 if bool(texture_gabor_include_magnitude) else 0,
            1 if bool(texture_gabor_normalize) else 0,
            max(0.01, float(texture_weight_color)),
            max(0.0, float(texture_weight_lbp)),
            max(0.0, float(texture_weight_gabor)),
            _utc_now_iso(),
        )
        sql = """
        INSERT INTO labeler_image_slic_overrides (
            project_id, image_name, slic_algorithm, preset_name, detail_level,
            n_segments, compactness, sigma, colorspace,
            quickshift_ratio, quickshift_kernel_size, quickshift_max_dist, quickshift_sigma,
            felzenszwalb_scale, felzenszwalb_sigma, felzenszwalb_min_size,
            texture_enabled, texture_mode,
            texture_lbp_enabled, texture_lbp_points, texture_lbp_radii_json, texture_lbp_method, texture_lbp_normalize,
            texture_gabor_enabled, texture_gabor_frequencies_json, texture_gabor_thetas_json, texture_gabor_bandwidth,
            texture_gabor_include_real, texture_gabor_include_imag, texture_gabor_include_magnitude, texture_gabor_normalize,
            texture_weight_color, texture_weight_lbp, texture_weight_gabor,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(project_id, image_name) DO UPDATE SET
            slic_algorithm = excluded.slic_algorithm,
            preset_name = excluded.preset_name,
            detail_level = excluded.detail_level,
            n_segments = excluded.n_segments,
            compactness = excluded.compactness,
            sigma = excluded.sigma,
            colorspace = excluded.colorspace,
            quickshift_ratio = excluded.quickshift_ratio,
            quickshift_kernel_size = excluded.quickshift_kernel_size,
            quickshift_max_dist = excluded.quickshift_max_dist,
            quickshift_sigma = excluded.quickshift_sigma,
            felzenszwalb_scale = excluded.felzenszwalb_scale,
            felzenszwalb_sigma = excluded.felzenszwalb_sigma,
            felzenszwalb_min_size = excluded.felzenszwalb_min_size,
            texture_enabled = excluded.texture_enabled,
            texture_mode = excluded.texture_mode,
            texture_lbp_enabled = excluded.texture_lbp_enabled,
            texture_lbp_points = excluded.texture_lbp_points,
            texture_lbp_radii_json = excluded.texture_lbp_radii_json,
            texture_lbp_method = excluded.texture_lbp_method,
            texture_lbp_normalize = excluded.texture_lbp_normalize,
            texture_gabor_enabled = excluded.texture_gabor_enabled,
            texture_gabor_frequencies_json = excluded.texture_gabor_frequencies_json,
            texture_gabor_thetas_json = excluded.texture_gabor_thetas_json,
            texture_gabor_bandwidth = excluded.texture_gabor_bandwidth,
            texture_gabor_include_real = excluded.texture_gabor_include_real,
            texture_gabor_include_imag = excluded.texture_gabor_include_imag,
            texture_gabor_include_magnitude = excluded.texture_gabor_include_magnitude,
            texture_gabor_normalize = excluded.texture_gabor_normalize,
            texture_weight_color = excluded.texture_weight_color,
            texture_weight_lbp = excluded.texture_weight_lbp,
            texture_weight_gabor = excluded.texture_weight_gabor,
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
