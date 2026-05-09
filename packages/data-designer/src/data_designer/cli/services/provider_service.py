# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.cli.repositories.provider_repository import ModelProviderRegistry, ProviderRepository
from data_designer.config.models import ModelProvider


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
        registry = self.repository.load() or ModelProviderRegistry(providers=[])

        if any(p.name == provider.name for p in registry.providers):
            raise ValueError(f"Provider '{provider.name}' already exists")

        registry.providers.append(provider)
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

        if updated_provider.name != original_name:
            if any(p.name == updated_provider.name for p in registry.providers):
                raise ValueError(f"Provider name '{updated_provider.name}' already exists")

        registry.providers[index] = updated_provider

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

        registry.providers = [p for p in registry.providers if p.name != name]

        if registry.providers:
            self.repository.save(registry)
        else:
            self.repository.delete()
