# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for engine testing.

Located in src/ so it can be packaged and shared across subpackages via pytest_plugins.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from data_designer.config.mcp import LocalStdioMCPProvider, MCPProvider, ToolConfig
from data_designer.engine.mcp.facade import MCPFacade
from data_designer.engine.model_provider import MCPProviderRegistry
from data_designer.engine.models.clients.types import AssistantMessage, ChatCompletionResponse, ToolCall
from data_designer.engine.secret_resolver import SecretResolver
from data_designer.engine.testing.stubs import StubHuggingFaceSeedReader

# =============================================================================
# Seed reader fixtures
# =============================================================================


@pytest.fixture
def stub_seed_reader() -> StubHuggingFaceSeedReader:
    """Stub seed reader for testing seed dataset functionality."""
    return StubHuggingFaceSeedReader()


# =============================================================================
# MCP Provider fixtures
# =============================================================================


@pytest.fixture
def stub_mcp_provider_registry() -> MCPProviderRegistry:
    """Create a stub MCP provider registry with test providers."""
    return MCPProviderRegistry(
        providers=[
            LocalStdioMCPProvider(name="tools", command="python"),
            LocalStdioMCPProvider(name="secondary", command="python"),
        ]
    )


@pytest.fixture
def stub_mcp_provider_registry_single() -> MCPProviderRegistry:
    """Create a stub MCP provider registry with a single provider."""
    return MCPProviderRegistry(providers=[LocalStdioMCPProvider(name="tools", command="python")])


@pytest.fixture
def stub_secret_resolver() -> MagicMock:
    """Create a stub secret resolver for testing."""
    resolver = MagicMock(spec=SecretResolver)
    resolver.resolve.side_effect = lambda x: x  # Return the input as-is
    return resolver


@pytest.fixture
def stub_stdio_provider() -> LocalStdioMCPProvider:
    """Create a stub stdio MCP provider for testing."""
    return LocalStdioMCPProvider(
        name="test-stdio",
        command="python",
        args=["-m", "test_server"],
        env={"TEST_VAR": "value"},
    )


@pytest.fixture
def stub_sse_provider() -> MCPProvider:
    """Create a stub SSE MCP provider for testing."""
    return MCPProvider(
        name="test-sse",
        endpoint="http://localhost:8080/sse",
        api_key="test-key",
    )


@pytest.fixture
def stub_streamable_http_provider() -> MCPProvider:
    """Create a stub Streamable HTTP MCP provider for testing."""
    return MCPProvider(
        name="test-streamable-http",
        endpoint="https://api.example.com/mcp",
        api_key="test-key",
        provider_type="streamable_http",
    )


# =============================================================================
# Tool config fixtures
# =============================================================================


@pytest.fixture
def stub_tool_config() -> ToolConfig:
    """Create a basic tool configuration for testing."""
    return ToolConfig(
        tool_alias="test-tools",
        providers=["tools"],
        max_tool_call_turns=3,
        timeout_sec=30.0,
    )


@pytest.fixture
def stub_tool_config_with_allow_list() -> ToolConfig:
    """Create a tool configuration with an allow list."""
    return ToolConfig(
        tool_alias="test-tools",
        providers=["tools"],
        allow_tools=["lookup", "search"],
        max_tool_call_turns=3,
    )


# =============================================================================
# Facade fixtures
# =============================================================================


@pytest.fixture
def stub_mcp_facade(
    stub_tool_config: ToolConfig, stub_secret_resolver: MagicMock, stub_mcp_provider_registry: MCPProviderRegistry
) -> MCPFacade:
    """Create a stub MCPFacade for testing."""
    return MCPFacade(
        tool_config=stub_tool_config,
        secret_resolver=stub_secret_resolver,
        mcp_provider_registry=stub_mcp_provider_registry,
    )


@pytest.fixture
def stub_mcp_facade_factory() -> Any:
    """Create a stub MCP facade factory for testing."""

    def factory(
        tool_config: ToolConfig, secret_resolver: SecretResolver, provider_registry: MCPProviderRegistry
    ) -> MCPFacade:
        return MCPFacade(
            tool_config=tool_config, secret_resolver=secret_resolver, mcp_provider_registry=provider_registry
        )

    return factory


# =============================================================================
# Completion response fixtures (canonical ChatCompletionResponse)
# =============================================================================


@pytest.fixture
def mock_completion_response_no_tools() -> ChatCompletionResponse:
    """Mock LLM response with no tool calls."""
    return ChatCompletionResponse(
        message=AssistantMessage(content="Hello, I can help with that."),
    )


@pytest.fixture
def mock_completion_response_single_tool() -> ChatCompletionResponse:
    """Mock LLM response with single tool call."""
    return ChatCompletionResponse(
        message=AssistantMessage(
            content="Let me look that up.",
            tool_calls=[
                ToolCall(id="call-1", name="lookup", arguments_json='{"query": "test"}'),
            ],
        ),
    )


@pytest.fixture
def mock_completion_response_parallel_tools() -> ChatCompletionResponse:
    """Mock LLM response with multiple parallel tool calls."""
    return ChatCompletionResponse(
        message=AssistantMessage(
            content="Executing multiple tools.",
            tool_calls=[
                ToolCall(id="call-1", name="lookup", arguments_json='{"query": "first"}'),
                ToolCall(id="call-2", name="search", arguments_json='{"term": "second"}'),
                ToolCall(id="call-3", name="fetch", arguments_json='{"url": "example.com"}'),
            ],
        ),
    )


@pytest.fixture
def mock_completion_response_with_reasoning() -> ChatCompletionResponse:
    """Mock LLM response with reasoning_content."""
    return ChatCompletionResponse(
        message=AssistantMessage(
            content="  Final answer with extra spaces.  ",
            reasoning_content="  Thinking about the problem...  ",
        ),
    )


@pytest.fixture
def mock_completion_response_tool_with_reasoning() -> ChatCompletionResponse:
    """Mock LLM response with tool calls and reasoning_content."""
    return ChatCompletionResponse(
        message=AssistantMessage(
            content="  Looking it up...  ",
            tool_calls=[
                ToolCall(id="call-1", name="lookup", arguments_json='{"query": "test"}'),
            ],
            reasoning_content="  I should use the lookup tool.  ",
        ),
    )
