# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.cli.repositories.model_repository import ModelConfigRegistry, ModelRepository
from data_designer.config.models import ModelConfig


class ModelService:
    """Business logic for model management."""

    def __init__(self, repository: ModelRepository):
        self.repository = repository

    def list_all(self) -> list[ModelConfig]:
        """Get all configured models."""
        registry = self.repository.load()
        return list(registry.model_configs) if registry else []

    def get_by_alias(self, alias: str) -> ModelConfig | None:
        """Get a model by alias."""
        models = self.list_all()
        return next((m for m in models if m.alias == alias), None)

    def add(self, model: ModelConfig) -> None:
        """Add a new model.

        Raises:
            ValueError: If model alias already exists
        """
        registry = self.repository.load() or ModelConfigRegistry(model_configs=[])

        # Business rule: No duplicate aliases
        if any(m.alias == model.alias for m in registry.model_configs):
            raise ValueError(f"Model alias '{model.alias}' already exists")

        registry.model_configs.append(model)
        self.repository.save(registry)

    def update(self, original_alias: str, updated_model: ModelConfig) -> None:
        """Update an existing model.

        Raises:
            ValueError: If model not found or new alias already exists
        """
        registry = self.repository.load()
        if not registry:
            raise ValueError("No models configured")

        # Find model index
        index = next(
            (i for i, m in enumerate(registry.model_configs) if m.alias == original_alias),
            None,
        )
        if index is None:
            raise ValueError(f"Model '{original_alias}' not found")

        # Business rule: Alias change must not conflict
        if updated_model.alias != original_alias:
            if any(m.alias == updated_model.alias for m in registry.model_configs):
                raise ValueError(f"Model alias '{updated_model.alias}' already exists")

        # Update
        registry.model_configs[index] = updated_model
        self.repository.save(registry)

    def delete(self, alias: str) -> None:
        """Delete a model.

        Raises:
            ValueError: If model not found
        """
        registry = self.repository.load()
        if not registry:
            raise ValueError("No models configured")

        if not any(m.alias == alias for m in registry.model_configs):
            raise ValueError(f"Model '{alias}' not found")

        # Remove model
        registry.model_configs = [m for m in registry.model_configs if m.alias != alias]

        # Business rule: Delete file if no models left
        if registry.model_configs:
            self.repository.save(registry)
        else:
            self.repository.delete()
