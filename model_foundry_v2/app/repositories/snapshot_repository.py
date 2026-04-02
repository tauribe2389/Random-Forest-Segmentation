"""Registered snapshot persistence for Model Foundry V2."""

from __future__ import annotations

from datetime import datetime, timezone

from ..domain.entities import RegisteredSnapshot, RegisteredSnapshotItem
from .database import Database


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SnapshotRepository:
    """Repository for immutable registered snapshots."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def create_snapshot(
        self,
        workspace_id: int,
        draft_dataset_id: int,
        name: str,
        class_schema_json: str,
        class_schema_version: int,
        image_count: int,
    ) -> RegisteredSnapshot:
        created_at = _utc_now()
        with self.database.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO registered_snapshots (
                    workspace_id,
                    draft_dataset_id,
                    name,
                    class_schema_json,
                    class_schema_version,
                    image_count,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workspace_id,
                    draft_dataset_id,
                    name,
                    class_schema_json,
                    class_schema_version,
                    image_count,
                    created_at,
                ),
            )
            snapshot_id = int(cursor.lastrowid)
            conn.commit()
        return self.get_snapshot(snapshot_id)

    def add_snapshot_items(self, snapshot_id: int, items: list[tuple[int, int]]) -> None:
        with self.database.connection() as conn:
            for image_id, annotation_revision_id in items:
                conn.execute(
                    """
                    INSERT INTO registered_snapshot_items (
                        snapshot_id, image_id, annotation_revision_id
                    ) VALUES (?, ?, ?)
                    """,
                    (snapshot_id, image_id, annotation_revision_id),
                )

    def get_snapshot(self, snapshot_id: int) -> RegisteredSnapshot | None:
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT * FROM registered_snapshots WHERE id = ?",
                (snapshot_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_snapshot(row)

    def list_snapshots_for_dataset(self, draft_dataset_id: int) -> list[RegisteredSnapshot]:
        with self.database.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM registered_snapshots
                WHERE draft_dataset_id = ?
                ORDER BY created_at DESC, id DESC
                """,
                (draft_dataset_id,),
            ).fetchall()
        return [self._row_to_snapshot(row) for row in rows]

    def list_snapshot_items(self, snapshot_id: int) -> list[RegisteredSnapshotItem]:
        with self.database.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM registered_snapshot_items
                WHERE snapshot_id = ?
                ORDER BY image_id ASC
                """,
                (snapshot_id,),
            ).fetchall()
        return [
            RegisteredSnapshotItem(
                snapshot_id=int(row["snapshot_id"]),
                image_id=int(row["image_id"]),
                annotation_revision_id=int(row["annotation_revision_id"]),
            )
            for row in rows
        ]

    @staticmethod
    def _row_to_snapshot(row: object) -> RegisteredSnapshot:
        return RegisteredSnapshot(
            id=int(row["id"]),
            workspace_id=int(row["workspace_id"]),
            draft_dataset_id=int(row["draft_dataset_id"]),
            name=str(row["name"]),
            class_schema_json=str(row["class_schema_json"]),
            class_schema_version=int(row["class_schema_version"]),
            image_count=int(row["image_count"]),
            created_at=str(row["created_at"]),
        )
