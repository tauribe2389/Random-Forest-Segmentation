"""Annotation revision persistence for Model Foundry V2."""

from __future__ import annotations

from datetime import datetime, timezone

from ..domain.entities import AnnotationRevision
from .database import Database


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AnnotationRepository:
    """Repository for annotation revision metadata and dataset heads."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def create_revision_stub(
        self,
        draft_dataset_id: int,
        image_id: int,
        parent_revision_id: int | None,
        width: int,
        height: int,
        author: str,
        operation_summary: str,
    ) -> AnnotationRevision:
        timestamp = _utc_now()
        with self.database.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO annotation_revisions (
                    draft_dataset_id,
                    image_id,
                    parent_revision_id,
                    width,
                    height,
                    label_map_path,
                    provenance_map_path,
                    protection_map_path,
                    label_checksum,
                    provenance_checksum,
                    protection_checksum,
                    revision_checksum,
                    author,
                    operation_summary,
                    is_locked,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, '', '', '', '', '', '', '', ?, ?, 0, ?, ?)
                """,
                (
                    draft_dataset_id,
                    image_id,
                    parent_revision_id,
                    width,
                    height,
                    author,
                    operation_summary,
                    timestamp,
                    timestamp,
                ),
            )
            revision_id = int(cursor.lastrowid)
            conn.commit()
        return self.get_revision(revision_id)

    def update_revision_storage(
        self,
        revision_id: int,
        label_map_path: str,
        provenance_map_path: str,
        protection_map_path: str,
        label_checksum: str,
        provenance_checksum: str,
        protection_checksum: str,
        revision_checksum: str,
        author: str,
        operation_summary: str,
    ) -> AnnotationRevision:
        updated_at = _utc_now()
        with self.database.connection() as conn:
            conn.execute(
                """
                UPDATE annotation_revisions
                SET label_map_path = ?,
                    provenance_map_path = ?,
                    protection_map_path = ?,
                    label_checksum = ?,
                    provenance_checksum = ?,
                    protection_checksum = ?,
                    revision_checksum = ?,
                    author = ?,
                    operation_summary = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    label_map_path,
                    provenance_map_path,
                    protection_map_path,
                    label_checksum,
                    provenance_checksum,
                    protection_checksum,
                    revision_checksum,
                    author,
                    operation_summary,
                    updated_at,
                    revision_id,
                ),
            )
            conn.commit()
        return self.get_revision(revision_id)

    def get_revision(self, revision_id: int) -> AnnotationRevision | None:
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT * FROM annotation_revisions WHERE id = ?",
                (revision_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_revision(row)

    def set_head(self, draft_dataset_id: int, image_id: int, revision_id: int) -> None:
        updated_at = _utc_now()
        with self.database.connection() as conn:
            conn.execute(
                """
                INSERT INTO annotation_heads (draft_dataset_id, image_id, revision_id, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(draft_dataset_id, image_id)
                DO UPDATE SET revision_id = excluded.revision_id, updated_at = excluded.updated_at
                """,
                (draft_dataset_id, image_id, revision_id, updated_at),
            )

    def get_head_revision_id(self, draft_dataset_id: int, image_id: int) -> int | None:
        with self.database.connection() as conn:
            row = conn.execute(
                """
                SELECT revision_id
                FROM annotation_heads
                WHERE draft_dataset_id = ? AND image_id = ?
                """,
                (draft_dataset_id, image_id),
            ).fetchone()
        if row is None:
            return None
        return int(row["revision_id"])

    def get_head_revision(self, draft_dataset_id: int, image_id: int) -> AnnotationRevision | None:
        revision_id = self.get_head_revision_id(draft_dataset_id, image_id)
        if revision_id is None:
            return None
        return self.get_revision(revision_id)

    def list_head_revision_ids(self, draft_dataset_id: int) -> dict[int, int]:
        with self.database.connection() as conn:
            rows = conn.execute(
                """
                SELECT image_id, revision_id
                FROM annotation_heads
                WHERE draft_dataset_id = ?
                """,
                (draft_dataset_id,),
            ).fetchall()
        return {int(row["image_id"]): int(row["revision_id"]) for row in rows}

    def lock_revision(self, revision_id: int) -> AnnotationRevision:
        with self.database.connection() as conn:
            conn.execute(
                "UPDATE annotation_revisions SET is_locked = 1 WHERE id = ?",
                (revision_id,),
            )
            conn.commit()
        return self.get_revision(revision_id)

    @staticmethod
    def _row_to_revision(row: object) -> AnnotationRevision:
        return AnnotationRevision(
            id=int(row["id"]),
            draft_dataset_id=int(row["draft_dataset_id"]),
            image_id=int(row["image_id"]),
            parent_revision_id=(
                int(row["parent_revision_id"]) if row["parent_revision_id"] is not None else None
            ),
            width=int(row["width"]),
            height=int(row["height"]),
            label_map_path=str(row["label_map_path"]),
            provenance_map_path=str(row["provenance_map_path"]),
            protection_map_path=str(row["protection_map_path"]),
            label_checksum=str(row["label_checksum"]),
            provenance_checksum=str(row["provenance_checksum"]),
            protection_checksum=str(row["protection_checksum"]),
            revision_checksum=str(row["revision_checksum"]),
            author=str(row["author"]),
            operation_summary=str(row["operation_summary"]),
            is_locked=bool(row["is_locked"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
