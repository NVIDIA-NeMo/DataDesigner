# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from data_designer.config.mcp import LocalStdioMCPProvider, MCPProvider, ToolConfig
from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage
from data_designer.engine.resources.resource_provider import (
    ResourceProvider,
    _validate_tool_configs_against_providers,
    create_resource_provider,
)


def test_resource_provider_artifact_storage_required():
    with pytest.raises(ValueError, match="Field required"):
        ResourceProvider()


@pytest.mark.parametrize(
    "test_case,expected_error",
    [
        ("model_registry_creation_error", "Model registry creation failed"),
    ],
)
def test_create_resource_provider_error_cases(test_case, expected_error, tmp_path):
    artifact_storage = ArtifactStorage(artifact_path=str(tmp_path), dataset_name="test")
    mock_model_configs = [Mock(), Mock()]
    mock_secret_resolver = Mock()
    mock_model_provider_registry = Mock()
    mock_seed_reader_registry = Mock()

    with patch("data_designer.engine.resources.resource_provider.create_model_registry") as mock_create_model_registry:
        mock_create_model_registry.side_effect = Exception(expected_error)

        with pytest.raises(Exception, match=expected_error):
            create_resource_provider(
                artifact_storage=artifact_storage,
                model_configs=mock_model_configs,
                secret_resolver=mock_secret_resolver,
                model_provider_registry=mock_model_provider_registry,
                seed_reader_registry=mock_seed_reader_registry,
            )


class TestToolConfigValidation:
    """Tests for ToolConfig validation against MCP providers."""

    def test_valid_tool_config_with_existing_providers(self) -> None:
        """Valid tool config passes when all providers exist."""
        providers = [
            MCPProvider(name="mcp-1", endpoint="http://localhost:8080/sse"),
            LocalStdioMCPProvider(name="mcp-2", command="python", args=["-m", "server"]),
        ]
        tool_configs = [
            ToolConfig(tool_alias="tools-1", providers=["mcp-1"]),
            ToolConfig(tool_alias="tools-2", providers=["mcp-1", "mcp-2"]),
        ]

        # Should not raise
        _validate_tool_configs_against_providers(tool_configs, providers)

    def test_tool_config_with_missing_provider_raises_error(self) -> None:
        """Tool config referencing non-existent provider raises ValueError."""
        providers = [
            MCPProvider(name="mcp-1", endpoint="http://localhost:8080/sse"),
        ]
        tool_configs = [
            ToolConfig(tool_alias="search-tools", providers=["mcp-1", "nonexistent-mcp"]),
        ]

        with pytest.raises(ValueError, match="ToolConfig 'search-tools' references provider"):
            _validate_tool_configs_against_providers(tool_configs, providers)

    def test_tool_config_with_no_providers_available(self) -> None:
        """Tool config fails when no MCP providers are configured."""
        tool_configs = [
            ToolConfig(tool_alias="search-tools", providers=["some-mcp"]),
        ]

        with pytest.raises(ValueError, match="not registered.*none configured"):
            _validate_tool_configs_against_providers(tool_configs, [])

    def test_empty_tool_configs_passes(self) -> None:
        """Empty tool configs list passes validation."""
        providers = [MCPProvider(name="mcp-1", endpoint="http://localhost:8080/sse")]

        # Should not raise
        _validate_tool_configs_against_providers([], providers)

    def test_create_resource_provider_validates_tool_configs(self, tmp_path: str) -> None:
        """create_resource_provider validates tool configs against providers."""
        artifact_storage = ArtifactStorage(artifact_path=str(tmp_path), dataset_name="test")
        tool_configs = [ToolConfig(tool_alias="tools", providers=["nonexistent"])]

        with (
            patch("data_designer.engine.resources.resource_provider.create_model_registry"),
            pytest.raises(ValueError, match="ToolConfig 'tools' references provider"),
        ):
            create_resource_provider(
                artifact_storage=artifact_storage,
                model_configs=[],
                secret_resolver=Mock(),
                model_provider_registry=Mock(),
                seed_reader_registry=Mock(),
                tool_configs=tool_configs,
                mcp_providers=[],
            )
