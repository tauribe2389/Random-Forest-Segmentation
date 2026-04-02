"""Artifact service for Model Foundry V2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..domain.entities import ArtifactRecord
from ..domain.enums import ArtifactKind
from ..repositories.artifact_repository import ArtifactRepository


class ArtifactService:
    """Owns artifact metadata records for canonical and derived files."""

    def __init__(self, repository: ArtifactRepository, artifacts_root: Path) -> None:
        self.repository = repository
        self.artifacts_root = artifacts_root

    def record_artifact(
        self,
        owner_type: str,
        owner_id: int,
        artifact_kind: ArtifactKind,
        artifact_path: Path,
        checksum_sha256: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        relative_path = str(artifact_path.resolve().relative_to(self.artifacts_root.resolve()))
        metadata_json = json.dumps(metadata or {}, sort_keys=True)
        return self.repository.record_artifact(
            owner_type=owner_type,
            owner_id=owner_id,
            artifact_kind=artifact_kind.value,
            relative_path=relative_path,
            checksum_sha256=checksum_sha256,
            metadata_json=metadata_json,
        )

    def list_artifacts(self, owner_type: str, owner_id: int) -> list[ArtifactRecord]:
        return self.repository.list_artifacts(owner_type=owner_type, owner_id=owner_id)
