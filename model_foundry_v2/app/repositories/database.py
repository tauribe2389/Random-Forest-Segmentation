"""SQLite database helpers for Model Foundry V2."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS workspaces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        description TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS image_assets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workspace_id INTEGER NOT NULL,
        source_path TEXT NOT NULL,
        checksum_sha256 TEXT NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS draft_datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workspace_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        description TEXT,
        schema_version INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS draft_dataset_items (
        draft_dataset_id INTEGER NOT NULL,
        image_id INTEGER NOT NULL,
        PRIMARY KEY (draft_dataset_id, image_id),
        FOREIGN KEY (draft_dataset_id) REFERENCES draft_datasets(id) ON DELETE CASCADE,
        FOREIGN KEY (image_id) REFERENCES image_assets(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dataset_classes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        draft_dataset_id INTEGER NOT NULL,
        class_index INTEGER NOT NULL,
        name TEXT NOT NULL,
        color TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (draft_dataset_id, class_index),
        FOREIGN KEY (draft_dataset_id) REFERENCES draft_datasets(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS annotation_revisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        draft_dataset_id INTEGER NOT NULL,
        image_id INTEGER NOT NULL,
        parent_revision_id INTEGER,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        label_map_path TEXT NOT NULL,
        provenance_map_path TEXT NOT NULL,
        protection_map_path TEXT NOT NULL,
        label_checksum TEXT NOT NULL,
        provenance_checksum TEXT NOT NULL,
        protection_checksum TEXT NOT NULL,
        revision_checksum TEXT NOT NULL,
        author TEXT NOT NULL,
        operation_summary TEXT NOT NULL,
        is_locked INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (draft_dataset_id) REFERENCES draft_datasets(id) ON DELETE CASCADE,
        FOREIGN KEY (image_id) REFERENCES image_assets(id) ON DELETE CASCADE,
        FOREIGN KEY (parent_revision_id) REFERENCES annotation_revisions(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS annotation_heads (
        draft_dataset_id INTEGER NOT NULL,
        image_id INTEGER NOT NULL,
        revision_id INTEGER NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (draft_dataset_id, image_id),
        FOREIGN KEY (draft_dataset_id) REFERENCES draft_datasets(id) ON DELETE CASCADE,
        FOREIGN KEY (image_id) REFERENCES image_assets(id) ON DELETE CASCADE,
        FOREIGN KEY (revision_id) REFERENCES annotation_revisions(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS registered_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workspace_id INTEGER NOT NULL,
        draft_dataset_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        class_schema_json TEXT NOT NULL,
        class_schema_version INTEGER NOT NULL,
        image_count INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
        FOREIGN KEY (draft_dataset_id) REFERENCES draft_datasets(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS registered_snapshot_items (
        snapshot_id INTEGER NOT NULL,
        image_id INTEGER NOT NULL,
        annotation_revision_id INTEGER NOT NULL,
        PRIMARY KEY (snapshot_id, image_id),
        FOREIGN KEY (snapshot_id) REFERENCES registered_snapshots(id) ON DELETE CASCADE,
        FOREIGN KEY (image_id) REFERENCES image_assets(id) ON DELETE CASCADE,
        FOREIGN KEY (annotation_revision_id) REFERENCES annotation_revisions(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS artifact_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        owner_type TEXT NOT NULL,
        owner_id INTEGER NOT NULL,
        artifact_kind TEXT NOT NULL,
        relative_path TEXT NOT NULL,
        checksum_sha256 TEXT NOT NULL,
        metadata_json TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (owner_type, owner_id, artifact_kind, relative_path)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_definitions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workspace_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        family_id TEXT NOT NULL,
        config_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS training_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_definition_id INTEGER NOT NULL,
        registered_snapshot_id INTEGER NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (model_definition_id) REFERENCES model_definitions(id) ON DELETE CASCADE,
        FOREIGN KEY (registered_snapshot_id) REFERENCES registered_snapshots(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prediction_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_definition_id INTEGER NOT NULL,
        registered_snapshot_id INTEGER NOT NULL,
        status TEXT NOT NULL,
        source_prediction_run_id INTEGER,
        config_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (model_definition_id) REFERENCES model_definitions(id) ON DELETE CASCADE,
        FOREIGN KEY (registered_snapshot_id) REFERENCES registered_snapshots(id) ON DELETE CASCADE,
        FOREIGN KEY (source_prediction_run_id) REFERENCES prediction_runs(id)
    )
    """,
)


class Database:
    """Thin SQLite wrapper with schema bootstrap support."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
            for statement in SCHEMA_STATEMENTS:
                conn.execute(statement)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
