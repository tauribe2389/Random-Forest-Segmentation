"""Training run persistence for Model Foundry V2."""

from __future__ import annotations

from datetime import datetime, timezone

from ..domain.entities import TrainingRun
from .database import Database


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrainingRepository:
    """Repository for training-run metadata."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def create_training_run(
        self,
        model_definition_id: int,
        registered_snapshot_id: int,
        status: str,
    ) -> TrainingRun:
        timestamp = _utc_now()
        with self.database.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO training_runs (
                    model_definition_id,
                    registered_snapshot_id,
                    status,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (model_definition_id, registered_snapshot_id, status, timestamp, timestamp),
            )
            conn.commit()
        return self.get_training_run(int(cursor.lastrowid))

    def update_training_run_status(self, training_run_id: int, status: str) -> TrainingRun:
        updated_at = _utc_now()
        with self.database.connection() as conn:
            conn.execute(
                """
                UPDATE training_runs
                SET status = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, updated_at, training_run_id),
            )
            conn.commit()
        return self.get_training_run(training_run_id)

    def get_training_run(self, training_run_id: int) -> TrainingRun | None:
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT * FROM training_runs WHERE id = ?",
                (training_run_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_training_run(row)

    def list_training_runs_for_snapshot(self, snapshot_id: int) -> list[TrainingRun]:
        with self.database.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM training_runs
                WHERE registered_snapshot_id = ?
                ORDER BY created_at DESC, id DESC
                """,
                (snapshot_id,),
            ).fetchall()
        return [self._row_to_training_run(row) for row in rows]

    @staticmethod
    def _row_to_training_run(row: object) -> TrainingRun:
        return TrainingRun(
            id=int(row["id"]),
            model_definition_id=int(row["model_definition_id"]),
            registered_snapshot_id=int(row["registered_snapshot_id"]),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
