"""Draft dataset and class persistence for Model Foundry V2."""

from __future__ import annotations

from datetime import datetime, timezone

from ..domain.entities import DatasetClass, DraftDataset, ImageAsset
from .database import Database


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DatasetRepository:
    """Repository for mutable draft datasets and class schema."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def create_draft_dataset(
        self,
        workspace_id: int,
        name: str,
        image_ids: list[int],
        description: str | None = None,
    ) -> DraftDataset:
        timestamp = _utc_now()
        with self.database.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO draft_datasets (
                    workspace_id, name, description, schema_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (workspace_id, name, description, 1, timestamp, timestamp),
            )
            dataset_id = int(cursor.lastrowid)
            conn.execute(
                """
                INSERT INTO dataset_classes (
                    draft_dataset_id, class_index, name, color, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (dataset_id, 0, "background", "#000000", timestamp),
            )
            for image_id in image_ids:
                conn.execute(
                    """
                    INSERT INTO draft_dataset_items (draft_dataset_id, image_id)
                    VALUES (?, ?)
                    """,
                    (dataset_id, image_id),
                )
            conn.commit()
        return self.get_draft_dataset(dataset_id)

    def get_draft_dataset(self, dataset_id: int) -> DraftDataset | None:
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT * FROM draft_datasets WHERE id = ?",
                (dataset_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_draft_dataset(row)

    def list_draft_datasets(self, workspace_id: int) -> list[DraftDataset]:
        with self.database.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM draft_datasets
                WHERE workspace_id = ?
                ORDER BY created_at DESC, id DESC
                """,
                (workspace_id,),
            ).fetchall()
        return [self._row_to_draft_dataset(row) for row in rows]

    def list_dataset_images(self, dataset_id: int) -> list[ImageAsset]:
        with self.database.connection() as conn:
            rows = conn.execute(
                """
                SELECT image_assets.*
                FROM image_assets
                INNER JOIN draft_dataset_items
                    ON draft_dataset_items.image_id = image_assets.id
                WHERE draft_dataset_items.draft_dataset_id = ?
                ORDER BY image_assets.id ASC
                """,
                (dataset_id,),
            ).fetchall()
        return [
            ImageAsset(
                id=int(row["id"]),
                workspace_id=int(row["workspace_id"]),
                source_path=str(row["source_path"]),
                checksum_sha256=str(row["checksum_sha256"]),
                width=int(row["width"]),
                height=int(row["height"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def dataset_contains_image(self, dataset_id: int, image_id: int) -> bool:
        with self.database.connection() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM draft_dataset_items
                WHERE draft_dataset_id = ? AND image_id = ?
                """,
                (dataset_id, image_id),
            ).fetchone()
        return row is not None

    def add_class(self, dataset_id: int, name: str, color: str) -> DatasetClass:
        timestamp = _utc_now()
        with self.database.connection() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(class_index), 0) AS max_class_index
                FROM dataset_classes
                WHERE draft_dataset_id = ?
                """,
                (dataset_id,),
            ).fetchone()
            next_index = int(row["max_class_index"]) + 1
            cursor = conn.execute(
                """
                INSERT INTO dataset_classes (
                    draft_dataset_id, class_index, name, color, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (dataset_id, next_index, name, color, timestamp),
            )
            conn.execute(
                """
                UPDATE draft_datasets
                SET schema_version = schema_version + 1, updated_at = ?
                WHERE id = ?
                """,
                (timestamp, dataset_id),
            )
            class_id = int(cursor.lastrowid)
            row = conn.execute(
                "SELECT * FROM dataset_classes WHERE id = ?",
                (class_id,),
            ).fetchone()
        return self._row_to_dataset_class(row)

    def list_dataset_classes(self, dataset_id: int) -> list[DatasetClass]:
        with self.database.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM dataset_classes
                WHERE draft_dataset_id = ?
                ORDER BY class_index ASC
                """,
                (dataset_id,),
            ).fetchall()
        return [self._row_to_dataset_class(row) for row in rows]

    @staticmethod
    def _row_to_draft_dataset(row: object) -> DraftDataset:
        return DraftDataset(
            id=int(row["id"]),
            workspace_id=int(row["workspace_id"]),
            name=str(row["name"]),
            description=row["description"],
            schema_version=int(row["schema_version"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _row_to_dataset_class(row: object) -> DatasetClass:
        return DatasetClass(
            id=int(row["id"]),
            draft_dataset_id=int(row["draft_dataset_id"]),
            class_index=int(row["class_index"]),
            name=str(row["name"]),
            color=str(row["color"]),
            created_at=str(row["created_at"]),
        )
