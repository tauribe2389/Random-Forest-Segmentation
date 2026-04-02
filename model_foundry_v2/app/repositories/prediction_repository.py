"""Prediction run persistence for Model Foundry V2."""

from __future__ import annotations

from datetime import datetime, timezone

from ..domain.entities import PredictionRun
from .database import Database


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PredictionRepository:
    """Repository for prediction-run metadata."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def create_prediction_run(
        self,
        model_definition_id: int,
        registered_snapshot_id: int,
        status: str,
        config_json: str,
        source_prediction_run_id: int | None = None,
    ) -> PredictionRun:
        timestamp = _utc_now()
        with self.database.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO prediction_runs (
                    model_definition_id,
                    registered_snapshot_id,
                    status,
                    source_prediction_run_id,
                    config_json,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_definition_id,
                    registered_snapshot_id,
                    status,
                    source_prediction_run_id,
                    config_json,
                    timestamp,
                    timestamp,
                ),
            )
            conn.commit()
        return self.get_prediction_run(int(cursor.lastrowid))

    def update_prediction_run_status(self, prediction_run_id: int, status: str) -> PredictionRun:
        updated_at = _utc_now()
        with self.database.connection() as conn:
            conn.execute(
                """
                UPDATE prediction_runs
                SET status = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, updated_at, prediction_run_id),
            )
            conn.commit()
        return self.get_prediction_run(prediction_run_id)

    def get_prediction_run(self, prediction_run_id: int) -> PredictionRun | None:
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT * FROM prediction_runs WHERE id = ?",
                (prediction_run_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_prediction_run(row)

    def list_prediction_runs_for_snapshot(self, snapshot_id: int) -> list[PredictionRun]:
        with self.database.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM prediction_runs
                WHERE registered_snapshot_id = ?
                ORDER BY created_at DESC, id DESC
                """,
                (snapshot_id,),
            ).fetchall()
        return [self._row_to_prediction_run(row) for row in rows]

    @staticmethod
    def _row_to_prediction_run(row: object) -> PredictionRun:
        return PredictionRun(
            id=int(row["id"]),
            model_definition_id=int(row["model_definition_id"]),
            registered_snapshot_id=int(row["registered_snapshot_id"]),
            status=str(row["status"]),
            source_prediction_run_id=(
                int(row["source_prediction_run_id"])
                if row["source_prediction_run_id"] is not None
                else None
            ),
            config_json=str(row["config_json"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
