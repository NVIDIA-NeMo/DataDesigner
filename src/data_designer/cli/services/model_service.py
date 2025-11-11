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

    def find_by_provider(self, provider_name: str) -> list[ModelConfig]:
        """Find all models associated with a provider.

        Args:
            provider_name: Name of the provider to search for

        Returns:
            List of models associated with the provider
        """
        models = self.list_all()
        return [m for m in models if m.provider == provider_name]

    def add(self, model: ModelConfig) -> None:
        """Add a new model.

        Raises:
            ValueError: If model alias already exists
        """
        registry = self.repository.load() or ModelConfigRegistry(model_configs=[])

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

        if updated_model.alias != original_alias:
            if any(m.alias == updated_model.alias for m in registry.model_configs):
                raise ValueError(f"Model alias '{updated_model.alias}' already exists")

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

        registry.model_configs = [m for m in registry.model_configs if m.alias != alias]

        if registry.model_configs:
            self.repository.save(registry)
        else:
            self.repository.delete()

    def delete_by_aliases(self, aliases: list[str]) -> None:
        """Delete multiple models by their aliases.

        Args:
            aliases: List of model aliases to delete

        Raises:
            ValueError: If no models configured
        """
        if not aliases:
            return

        registry = self.repository.load()
        if not registry:
            raise ValueError("No models configured")

        registry.model_configs = [m for m in registry.model_configs if m.alias not in aliases]

        if registry.model_configs:
            self.repository.save(registry)
        else:
            self.repository.delete()
