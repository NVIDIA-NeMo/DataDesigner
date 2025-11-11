# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.cli.repositories.provider_repository import ProviderRepository
from data_designer.engine.model_provider import ModelProvider, ModelProviderRegistry


class ProviderService:
    """Business logic for provider management."""

    def __init__(self, repository: ProviderRepository):
        self.repository = repository

    def list_all(self) -> list[ModelProvider]:
        """Get all configured providers."""
        registry = self.repository.load()
        return list(registry.providers) if registry else []

    def get_by_name(self, name: str) -> ModelProvider | None:
        """Get a provider by name."""
        providers = self.list_all()
        return next((p for p in providers if p.name == name), None)

    def add(self, provider: ModelProvider) -> None:
        """Add a new provider.

        Raises:
            ValueError: If provider name already exists
        """
        registry = self.repository.load()

        if registry:
            # Business rule: No duplicate names
            if any(p.name == provider.name for p in registry.providers):
                raise ValueError(f"Provider '{provider.name}' already exists")

            registry.providers.append(provider)
        else:
            # Create new registry with first provider
            registry = ModelProviderRegistry(providers=[provider], default=provider.name)

        # Business rule: First provider is default (for existing registries adding first provider)
        if len(registry.providers) == 1 and registry.default is None:
            registry.default = provider.name

        self.repository.save(registry)

    def update(self, original_name: str, updated_provider: ModelProvider) -> None:
        """Update an existing provider.

        Raises:
            ValueError: If provider not found or new name already exists
        """
        registry = self.repository.load()
        if not registry:
            raise ValueError("No providers configured")

        # Find provider index
        index = next(
            (i for i, p in enumerate(registry.providers) if p.name == original_name),
            None,
        )
        if index is None:
            raise ValueError(f"Provider '{original_name}' not found")

        # Business rule: Name change must not conflict
        if updated_provider.name != original_name:
            if any(p.name == updated_provider.name for p in registry.providers):
                raise ValueError(f"Provider name '{updated_provider.name}' already exists")

        # Update
        registry.providers[index] = updated_provider

        # Business rule: Update default if name changed
        if registry.default == original_name and updated_provider.name != original_name:
            registry.default = updated_provider.name

        self.repository.save(registry)

    def delete(self, name: str) -> None:
        """Delete a provider.

        Raises:
            ValueError: If provider not found
        """
        registry = self.repository.load()
        if not registry:
            raise ValueError("No providers configured")

        if not any(p.name == name for p in registry.providers):
            raise ValueError(f"Provider '{name}' not found")

        # Remove provider
        registry.providers = [p for p in registry.providers if p.name != name]

        # Business rule: Update default if deleted
        if registry.default == name:
            registry.default = registry.providers[0].name if registry.providers else None

        # Business rule: Delete file if no providers left
        if registry.providers:
            self.repository.save(registry)
        else:
            self.repository.delete()

    def set_default(self, name: str) -> None:
        """Set the default provider.

        Raises:
            ValueError: If provider not found
        """
        registry = self.repository.load()
        if not registry:
            raise ValueError("No providers configured")

        if not any(p.name == name for p in registry.providers):
            raise ValueError(f"Provider '{name}' not found")

        registry.default = name
        self.repository.save(registry)

    def get_default(self) -> str | None:
        """Get the default provider name."""
        registry = self.repository.load()
        if not registry:
            return None
        return registry.default or (registry.providers[0].name if registry.providers else None)
