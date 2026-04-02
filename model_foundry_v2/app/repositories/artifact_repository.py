"""Artifact record persistence for Model Foundry V2."""

from __future__ import annotations

from datetime import datetime, timezone

from ..domain.entities import ArtifactRecord
from .database import Database


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ArtifactRepository:
    """Repository for metadata about derived and canonical artifact files."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def record_artifact(
        self,
        owner_type: str,
        owner_id: int,
        artifact_kind: str,
        relative_path: str,
        checksum_sha256: str,
        metadata_json: str | None,
    ) -> ArtifactRecord:
        timestamp = _utc_now()
        with self.database.connection() as conn:
            conn.execute(
                """
                INSERT INTO artifact_records (
                    owner_type,
                    owner_id,
                    artifact_kind,
                    relative_path,
                    checksum_sha256,
                    metadata_json,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(owner_type, owner_id, artifact_kind, relative_path)
                DO UPDATE SET
                    checksum_sha256 = excluded.checksum_sha256,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    owner_type,
                    owner_id,
                    artifact_kind,
                    relative_path,
                    checksum_sha256,
                    metadata_json,
                    timestamp,
                    timestamp,
                ),
            )
            row = conn.execute(
                """
                SELECT * FROM artifact_records
                WHERE owner_type = ? AND owner_id = ? AND artifact_kind = ? AND relative_path = ?
                """,
                (owner_type, owner_id, artifact_kind, relative_path),
            ).fetchone()
        return self._row_to_artifact(row)

    def list_artifacts(self, owner_type: str, owner_id: int) -> list[ArtifactRecord]:
        with self.database.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM artifact_records
                WHERE owner_type = ? AND owner_id = ?
                ORDER BY artifact_kind ASC, relative_path ASC
                """,
                (owner_type, owner_id),
            ).fetchall()
        return [self._row_to_artifact(row) for row in rows]

    @staticmethod
    def _row_to_artifact(row: object) -> ArtifactRecord:
        return ArtifactRecord(
            id=int(row["id"]),
            owner_type=str(row["owner_type"]),
            owner_id=int(row["owner_id"]),
            artifact_kind=str(row["artifact_kind"]),
            relative_path=str(row["relative_path"]),
            checksum_sha256=str(row["checksum_sha256"]),
            metadata_json=row["metadata_json"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
