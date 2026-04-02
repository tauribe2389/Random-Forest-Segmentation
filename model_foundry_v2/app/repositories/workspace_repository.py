"""Workspace and image persistence for Model Foundry V2."""

from __future__ import annotations

from datetime import datetime, timezone

from ..domain.entities import ImageAsset, Workspace
from .database import Database


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkspaceRepository:
    """Repository for workspaces and source image assets."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def create_workspace(self, name: str, description: str | None = None) -> Workspace:
        created_at = _utc_now()
        with self.database.connection() as conn:
            cursor = conn.execute(
                "INSERT INTO workspaces (name, description, created_at) VALUES (?, ?, ?)",
                (name, description, created_at),
            )
            conn.commit()
            return self.get_workspace(int(cursor.lastrowid))

    def list_workspaces(self) -> list[Workspace]:
        with self.database.connection() as conn:
            rows = conn.execute("SELECT * FROM workspaces ORDER BY created_at DESC").fetchall()
        return [self._row_to_workspace(row) for row in rows]

    def get_workspace(self, workspace_id: int) -> Workspace | None:
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT * FROM workspaces WHERE id = ?",
                (workspace_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_workspace(row)

    def create_image_asset(
        self,
        workspace_id: int,
        source_path: str,
        checksum_sha256: str,
        width: int,
        height: int,
    ) -> ImageAsset:
        created_at = _utc_now()
        with self.database.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO image_assets (
                    workspace_id, source_path, checksum_sha256, width, height, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (workspace_id, source_path, checksum_sha256, width, height, created_at),
            )
            conn.commit()
            return self.get_image_asset(int(cursor.lastrowid))

    def list_image_assets(self, workspace_id: int) -> list[ImageAsset]:
        with self.database.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM image_assets
                WHERE workspace_id = ?
                ORDER BY created_at DESC, id DESC
                """,
                (workspace_id,),
            ).fetchall()
        return [self._row_to_image_asset(row) for row in rows]

    def get_image_asset(self, image_id: int) -> ImageAsset | None:
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT * FROM image_assets WHERE id = ?",
                (image_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_image_asset(row)

    @staticmethod
    def _row_to_workspace(row: object) -> Workspace:
        return Workspace(
            id=int(row["id"]),
            name=str(row["name"]),
            description=row["description"],
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _row_to_image_asset(row: object) -> ImageAsset:
        return ImageAsset(
            id=int(row["id"]),
            workspace_id=int(row["workspace_id"]),
            source_path=str(row["source_path"]),
            checksum_sha256=str(row["checksum_sha256"]),
            width=int(row["width"]),
            height=int(row["height"]),
            created_at=str(row["created_at"]),
        )
