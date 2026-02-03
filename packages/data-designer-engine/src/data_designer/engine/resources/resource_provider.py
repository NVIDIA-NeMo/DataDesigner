# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.base import ConfigBase
from data_designer.config.dataset_metadata import DatasetMetadata
from data_designer.config.mcp import MCPProviderT, ToolConfig
from data_designer.config.models import ModelConfig
from data_designer.config.run_config import RunConfig
from data_designer.config.seed_source import SeedSource
from data_designer.config.utils.type_helpers import StrEnum
from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage
from data_designer.engine.mcp.factory import create_mcp_registry
from data_designer.engine.mcp.registry import MCPRegistry
from data_designer.engine.model_provider import (
    ModelProviderRegistry,
    resolve_mcp_provider_registry,
)
from data_designer.engine.models.factory import create_model_registry
from data_designer.engine.models.registry import ModelRegistry
from data_designer.engine.resources.managed_storage import ManagedBlobStorage
from data_designer.engine.resources.seed_reader import SeedReader, SeedReaderRegistry
from data_designer.engine.secret_resolver import SecretResolver


class ResourceType(StrEnum):
    BLOB_STORAGE = "blob_storage"
    MODEL_REGISTRY = "model_registry"
    SEED_READER = "seed_reader"


class ResourceProvider(ConfigBase):
    artifact_storage: ArtifactStorage
    blob_storage: ManagedBlobStorage | None = None
    model_registry: ModelRegistry | None = None
    mcp_registry: MCPRegistry | None = None
    run_config: RunConfig = RunConfig()
    seed_reader: SeedReader | None = None

    def get_dataset_metadata(self) -> DatasetMetadata:
        """Get metadata about the dataset being generated.

        Returns:
            DatasetMetadata with seed column names and other metadata.
        """
        seed_column_names = []
        if self.seed_reader is not None:
            seed_column_names = self.seed_reader.get_column_names()
        return DatasetMetadata(seed_column_names=seed_column_names)


def _validate_tool_configs_against_providers(
    tool_configs: list[ToolConfig],
    mcp_providers: list[MCPProviderT],
) -> None:
    """Validate that all providers referenced in tool configs exist.

    Args:
        tool_configs: List of tool configurations to validate.
        mcp_providers: List of available MCP provider configurations.

    Raises:
        ValueError: If a tool config references a provider that doesn't exist.
    """
    available_providers = {p.name for p in mcp_providers}

    for tc in tool_configs:
        missing_providers = [p for p in tc.providers if p not in available_providers]
        if missing_providers:
            available_list = sorted(available_providers) if available_providers else ["(none configured)"]
            raise ValueError(
                f"ToolConfig '{tc.tool_alias}' references provider(s) {missing_providers!r} "
                f"which are not registered. Available providers: {available_list}"
            )


def create_resource_provider(
    *,
    artifact_storage: ArtifactStorage,
    model_configs: list[ModelConfig],
    secret_resolver: SecretResolver,
    model_provider_registry: ModelProviderRegistry,
    seed_reader_registry: SeedReaderRegistry,
    blob_storage: ManagedBlobStorage | None = None,
    seed_dataset_source: SeedSource | None = None,
    run_config: RunConfig | None = None,
    mcp_providers: list[MCPProviderT] | None = None,
    tool_configs: list[ToolConfig] | None = None,
) -> ResourceProvider:
    """Factory function for creating a ResourceProvider instance.

    This function triggers lazy loading of heavy dependencies like litellm.
    The creation order is:
    1. MCPProviderRegistry (can be empty)
    2. MCPRegistry with tool_configs
    3. ModelRegistry with mcp_registry

    Args:
        artifact_storage: Storage for build artifacts.
        model_configs: List of model configurations.
        secret_resolver: Resolver for secrets.
        model_provider_registry: Registry of model providers.
        seed_reader_registry: Registry of seed readers.
        blob_storage: Optional blob storage for large files.
        seed_dataset_source: Optional source for seed datasets.
        run_config: Optional runtime configuration.
        mcp_providers: Optional list of MCP provider configurations.
        tool_configs: Optional list of tool configurations.

    Returns:
        A configured ResourceProvider instance.
    """
    seed_reader = None
    if seed_dataset_source:
        seed_reader = seed_reader_registry.get_reader(
            seed_dataset_source,
            secret_resolver,
        )

    # Create MCPProviderRegistry first (can be empty)
    mcp_provider_registry = resolve_mcp_provider_registry(mcp_providers)

    # Create MCPRegistry with tool configs (only if tool_configs provided)
    # Tool validation is performed during dataset builder health checks.
    mcp_registry = None
    if tool_configs:
        mcp_registry = create_mcp_registry(
            tool_configs=tool_configs,
            secret_resolver=secret_resolver,
            mcp_provider_registry=mcp_provider_registry,
        )

    return ResourceProvider(
        artifact_storage=artifact_storage,
        model_registry=create_model_registry(
            model_configs=model_configs,
            secret_resolver=secret_resolver,
            model_provider_registry=model_provider_registry,
            mcp_registry=mcp_registry,
        ),
        blob_storage=blob_storage,
        mcp_registry=mcp_registry,
        seed_reader=seed_reader,
        run_config=run_config or RunConfig(),
    )
