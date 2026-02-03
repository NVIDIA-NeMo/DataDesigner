# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from data_designer.config.mcp import LocalStdioMCPProvider, MCPProvider, ToolConfig
from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage
from data_designer.engine.models.registry import ModelRegistry
from data_designer.engine.resources.resource_provider import (
    ResourceProvider,
    _validate_tool_configs_against_providers,
    create_resource_provider,
)


def _stub_model_registry() -> ModelRegistry:
    return ModelRegistry(
        secret_resolver=Mock(),
        model_provider_registry=Mock(),
        model_facade_factory=lambda *_args, **_kwargs: Mock(),
        model_configs=[],
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

    def test_tool_config_validation_happens_during_health_check(self, tmp_path: str) -> None:
        """Tool config validation is deferred to health checks."""
        artifact_storage = ArtifactStorage(artifact_path=str(tmp_path), dataset_name="test")
        tool_configs = [ToolConfig(tool_alias="tools", providers=["nonexistent"])]

        with patch(
            "data_designer.engine.resources.resource_provider.create_model_registry",
            return_value=_stub_model_registry(),
        ):
            resource_provider = create_resource_provider(
                artifact_storage=artifact_storage,
                model_configs=[],
                secret_resolver=Mock(),
                model_provider_registry=Mock(),
                seed_reader_registry=Mock(),
                tool_configs=tool_configs,
                mcp_providers=[],
            )

        with pytest.raises(ValueError, match="ToolConfig 'tools' references provider"):
            resource_provider.mcp_registry.run_health_check(["tools"])


class TestDuplicateToolNameValidation:
    """Tests for duplicate tool name validation at health check time."""

    def test_health_check_validates_duplicate_tool_names(self, tmp_path: str) -> None:
        """Health check validates for duplicate tool names across providers."""
        from data_designer.engine.mcp.errors import DuplicateToolNameError
        from data_designer.engine.mcp.registry import MCPToolDefinition

        artifact_storage = ArtifactStorage(artifact_path=str(tmp_path), dataset_name="test")
        providers = [
            LocalStdioMCPProvider(name="provider-1", command="python"),
            LocalStdioMCPProvider(name="provider-2", command="python"),
        ]
        # ToolConfig uses both providers which will have duplicate tool names
        tool_configs = [ToolConfig(tool_alias="tools", providers=["provider-1", "provider-2"])]

        def mock_list_tools(
            provider: LocalStdioMCPProvider, timeout_sec: float | None = None
        ) -> tuple[MCPToolDefinition, ...]:
            # Both providers return a tool named "lookup" - this is a duplicate
            return (MCPToolDefinition(name="lookup", description="Lookup", input_schema={"type": "object"}),)

        with (
            patch(
                "data_designer.engine.resources.resource_provider.create_model_registry",
                return_value=_stub_model_registry(),
            ),
            patch("data_designer.engine.mcp.io.list_tools", side_effect=mock_list_tools),
        ):
            resource_provider = create_resource_provider(
                artifact_storage=artifact_storage,
                model_configs=[],
                secret_resolver=Mock(),
                model_provider_registry=Mock(),
                seed_reader_registry=Mock(),
                tool_configs=tool_configs,
                mcp_providers=providers,
            )

            with pytest.raises(DuplicateToolNameError, match="Duplicate tool names found"):
                resource_provider.mcp_registry.run_health_check(["tools"])

    def test_duplicate_validation_happens_at_health_check(self, tmp_path: str) -> None:
        """Duplicate tool names are caught during health checks, not creation."""
        from data_designer.engine.mcp.errors import DuplicateToolNameError
        from data_designer.engine.mcp.registry import MCPToolDefinition

        artifact_storage = ArtifactStorage(artifact_path=str(tmp_path), dataset_name="test")
        providers = [
            LocalStdioMCPProvider(name="provider-1", command="python"),
            LocalStdioMCPProvider(name="provider-2", command="python"),
        ]
        tool_configs = [ToolConfig(tool_alias="tools", providers=["provider-1", "provider-2"])]

        list_tools_call_count = 0

        def mock_list_tools(
            provider: LocalStdioMCPProvider, timeout_sec: float | None = None
        ) -> tuple[MCPToolDefinition, ...]:
            nonlocal list_tools_call_count
            list_tools_call_count += 1
            # Both providers return "lookup" - duplicate
            return (MCPToolDefinition(name="lookup", description="Lookup", input_schema={"type": "object"}),)

        with (
            patch(
                "data_designer.engine.resources.resource_provider.create_model_registry",
                return_value=_stub_model_registry(),
            ),
            patch("data_designer.engine.mcp.io.list_tools", side_effect=mock_list_tools),
        ):
            resource_provider = create_resource_provider(
                artifact_storage=artifact_storage,
                model_configs=[],
                secret_resolver=Mock(),
                model_provider_registry=Mock(),
                seed_reader_registry=Mock(),
                tool_configs=tool_configs,
                mcp_providers=providers,
            )

            assert list_tools_call_count == 0
            with pytest.raises(DuplicateToolNameError):
                resource_provider.mcp_registry.run_health_check(["tools"])
            assert list_tools_call_count > 0
