"""Model metadata persistence for Model Foundry V2."""

from __future__ import annotations

from datetime import datetime, timezone

from ..domain.entities import ModelDefinition
from .database import Database


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelRepository:
    """Repository for model definitions."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def create_model_definition(
        self,
        workspace_id: int,
        name: str,
        family_id: str,
        config_json: str,
    ) -> ModelDefinition:
        created_at = _utc_now()
        with self.database.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO model_definitions (workspace_id, name, family_id, config_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (workspace_id, name, family_id, config_json, created_at),
            )
            row = conn.execute(
                "SELECT * FROM model_definitions WHERE id = ?",
                (int(cursor.lastrowid),),
            ).fetchone()
        return self._row_to_model_definition(row)

    def get_model_definition(self, model_definition_id: int) -> ModelDefinition | None:
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT * FROM model_definitions WHERE id = ?",
                (model_definition_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_model_definition(row)

    def update_model_definition_config(self, model_definition_id: int, config_json: str) -> ModelDefinition:
        with self.database.connection() as conn:
            conn.execute(
                """
                UPDATE model_definitions
                SET config_json = ?
                WHERE id = ?
                """,
                (config_json, model_definition_id),
            )
            conn.commit()
        return self.get_model_definition(model_definition_id)

    def list_model_definitions(self, workspace_id: int) -> list[ModelDefinition]:
        with self.database.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM model_definitions
                WHERE workspace_id = ?
                ORDER BY created_at DESC, id DESC
                """,
                (workspace_id,),
            ).fetchall()
        return [self._row_to_model_definition(row) for row in rows]

    @staticmethod
    def _row_to_model_definition(row: object) -> ModelDefinition:
        return ModelDefinition(
            id=int(row["id"]),
            workspace_id=int(row["workspace_id"]),
            name=str(row["name"]),
            family_id=str(row["family_id"]),
            config_json=str(row["config_json"]),
            created_at=str(row["created_at"]),
        )
