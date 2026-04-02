"""Model definition service for Model Foundry V2."""

from __future__ import annotations

import json

from ..domain.entities import ModelDefinition
from ..repositories.model_repository import ModelRepository
from ..repositories.workspace_repository import WorkspaceRepository
from .model_registry import ModelRegistry


class ModelService:
    """Service for model definitions tied to known model families."""

    def __init__(
        self,
        repository: ModelRepository,
        workspace_repository: WorkspaceRepository,
        model_registry: ModelRegistry,
    ) -> None:
        self.repository = repository
        self.workspace_repository = workspace_repository
        self.model_registry = model_registry

    def create_model_definition(
        self,
        workspace_id: int,
        name: str,
        family_id: str,
        config: dict | None = None,
    ) -> ModelDefinition:
        if self.workspace_repository.get_workspace(workspace_id) is None:
            raise ValueError(f"Unknown workspace: {workspace_id}")
        if self.model_registry.get(family_id) is None:
            raise ValueError(f"Unknown model family: {family_id}")
        return self.repository.create_model_definition(
            workspace_id=workspace_id,
            name=name.strip(),
            family_id=family_id,
            config_json=json.dumps(config or {}, sort_keys=True),
        )

    def get_model_definition(self, model_definition_id: int) -> ModelDefinition | None:
        return self.repository.get_model_definition(model_definition_id)

    def list_model_definitions(self, workspace_id: int) -> list[ModelDefinition]:
        return self.repository.list_model_definitions(workspace_id)

    def get_model_config(self, model_definition_id: int) -> dict:
        model_definition = self.get_model_definition(model_definition_id)
        if model_definition is None:
            raise ValueError(f"Unknown model definition: {model_definition_id}")
        return json.loads(model_definition.config_json or "{}")

    def update_model_config(self, model_definition_id: int, config: dict) -> ModelDefinition:
        model_definition = self.get_model_definition(model_definition_id)
        if model_definition is None:
            raise ValueError(f"Unknown model definition: {model_definition_id}")
        return self.repository.update_model_definition_config(
            model_definition_id=model_definition_id,
            config_json=json.dumps(config, indent=2, sort_keys=True),
        )
