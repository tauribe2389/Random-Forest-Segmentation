"""Workspace service for Model Foundry V2."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from ..domain.entities import ImageAsset, Workspace
from ..repositories.workspace_repository import WorkspaceRepository


class WorkspaceService:
    """Service for workspaces and source image registration."""

    def __init__(self, repository: WorkspaceRepository) -> None:
        self.repository = repository

    def create_workspace(self, name: str, description: str | None = None) -> Workspace:
        return self.repository.create_workspace(name=name.strip(), description=description)

    def list_workspaces(self) -> list[Workspace]:
        return self.repository.list_workspaces()

    def get_workspace(self, workspace_id: int) -> Workspace | None:
        return self.repository.get_workspace(workspace_id)

    def register_image_asset(self, workspace_id: int, source_path: str) -> ImageAsset:
        resolved = Path(source_path).expanduser().resolve()
        if not resolved.exists() or not resolved.is_file():
            raise ValueError(f"Image path does not exist: {resolved}")
        with Image.open(resolved) as image:
            width, height = image.size
        checksum = self._sha256_file(resolved)
        return self.repository.create_image_asset(
            workspace_id=workspace_id,
            source_path=str(resolved),
            checksum_sha256=checksum,
            width=width,
            height=height,
        )

    def list_image_assets(self, workspace_id: int) -> list[ImageAsset]:
        return self.repository.list_image_assets(workspace_id)

    def get_image_asset(self, image_id: int) -> ImageAsset | None:
        return self.repository.get_image_asset(image_id)

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
