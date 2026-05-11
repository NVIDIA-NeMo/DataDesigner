# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import cached_property

from pydantic import BaseModel, field_validator

from data_designer.config.mcp import MCPProviderT
from data_designer.config.models import ModelProvider
from data_designer.engine.errors import NoModelProvidersError, UnknownProviderError


class ModelProviderRegistry(BaseModel):
    providers: list[ModelProvider]

    @field_validator("providers", mode="after")
    @classmethod
    def validate_providers_not_empty(cls, v: list[ModelProvider]) -> list[ModelProvider]:
        if len(v) == 0:
            raise ValueError("At least one model provider must be defined")
        return v

    @field_validator("providers", mode="after")
    @classmethod
    def validate_providers_have_unique_names(cls, v: list[ModelProvider]) -> list[ModelProvider]:
        names = set()
        dupes = set()
        for provider in v:
            if provider.name in names:
                dupes.add(provider.name)
            names.add(provider.name)

        if len(dupes) > 0:
            raise ValueError(f"Model providers must have unique names, found duplicates: {dupes}")
        return v

    @cached_property
    def _providers_dict(self) -> dict[str, ModelProvider]:
        return {p.name: p for p in self.providers}

    def get_provider(self, name: str) -> ModelProvider:
        try:
            return self._providers_dict[name]
        except KeyError:
            raise UnknownProviderError(f"No provider named {name!r} registered")


def resolve_model_provider_registry(model_providers: list[ModelProvider]) -> ModelProviderRegistry:
    if len(model_providers) == 0:
        raise NoModelProvidersError("At least one model provider must be defined")
    return ModelProviderRegistry(providers=model_providers)


class MCPProviderRegistry(BaseModel):
    """Registry for MCP providers.

    Unlike ModelProviderRegistry, MCPProviderRegistry can be empty since MCP providers
    are optional. Users only need to register MCP providers if they want to use MCP tools
    for generation.

    Attributes:
        providers: List of MCP providers (both MCPProvider and LocalStdioMCPProvider).
    """

    providers: list[MCPProviderT] = []

    @field_validator("providers", mode="after")
    @classmethod
    def validate_providers_have_unique_names(cls, v: list[MCPProviderT]) -> list[MCPProviderT]:
        names = set()
        dupes = set()
        for provider in v:
            if provider.name in names:
                dupes.add(provider.name)
            names.add(provider.name)

        if len(dupes) > 0:
            raise ValueError(f"MCP providers must have unique names, found duplicates: {dupes}")
        return v

    @cached_property
    def _providers_dict(self) -> dict[str, MCPProviderT]:
        return {p.name: p for p in self.providers}

    def get_provider(self, name: str) -> MCPProviderT:
        """Get an MCP provider by name.

        Args:
            name: The name of the MCP provider.

        Returns:
            The MCP provider with the given name.

        Raises:
            UnknownProviderError: If no provider with the given name is registered.
        """
        try:
            return self._providers_dict[name]
        except KeyError:
            raise UnknownProviderError(f"No MCP provider named {name!r} registered")

    def is_empty(self) -> bool:
        """Check if the registry has no providers."""
        return len(self.providers) == 0


def resolve_mcp_provider_registry(
    mcp_providers: list[MCPProviderT] | None = None,
) -> MCPProviderRegistry:
    """Create an MCPProviderRegistry from a list of MCP providers.

    Args:
        mcp_providers: Optional list of MCP providers. If None or empty, returns an empty registry.

    Returns:
        An MCPProviderRegistry containing the provided MCP providers.
    """
    return MCPProviderRegistry(providers=mcp_providers or [])
