# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for MCP tests.

Mock classes for direct use in test functions are defined inline in fixtures below.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from data_designer.config.mcp import LocalStdioMCPProvider, MCPProvider, ToolConfig
from data_designer.engine.mcp.facade import MCPFacade
from data_designer.engine.model_provider import MCPProviderRegistry
from data_designer.engine.secret_resolver import SecretResolver

# =============================================================================
# Fake LLM response classes (used by completion response fixtures)
# =============================================================================


class _FakeMessage:
    """Fake message class for mocking LLM completion responses."""

    def __init__(
        self,
        content: str | None,
        tool_calls: list[dict] | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


class _FakeChoice:
    """Fake choice class for mocking LLM completion responses."""

    def __init__(self, message: _FakeMessage) -> None:
        self.message = message


class _FakeResponse:
    """Fake response class for mocking LLM completion responses."""

    def __init__(self, message: _FakeMessage) -> None:
        self.choices = [_FakeChoice(message)]


# =============================================================================
# Provider fixtures
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
# Completion response fixtures
# =============================================================================


@pytest.fixture
def mock_completion_response_no_tools() -> _FakeResponse:
    """Mock LLM response with no tool calls."""
    return _FakeResponse(_FakeMessage(content="Hello, I can help with that."))


@pytest.fixture
def mock_completion_response_single_tool() -> _FakeResponse:
    """Mock LLM response with single tool call."""
    tool_call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"query": "test"}'},
    }
    return _FakeResponse(_FakeMessage(content="Let me look that up.", tool_calls=[tool_call]))


@pytest.fixture
def mock_completion_response_parallel_tools() -> _FakeResponse:
    """Mock LLM response with multiple parallel tool calls."""
    tool_calls = [
        {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": '{"query": "first"}'}},
        {"id": "call-2", "type": "function", "function": {"name": "search", "arguments": '{"term": "second"}'}},
        {"id": "call-3", "type": "function", "function": {"name": "fetch", "arguments": '{"url": "example.com"}'}},
    ]
    return _FakeResponse(_FakeMessage(content="Executing multiple tools.", tool_calls=tool_calls))


@pytest.fixture
def mock_completion_response_with_reasoning() -> _FakeResponse:
    """Mock LLM response with reasoning_content."""
    return _FakeResponse(
        _FakeMessage(
            content="  Final answer with extra spaces.  ",
            reasoning_content="  Thinking about the problem...  ",
        )
    )


@pytest.fixture
def mock_completion_response_tool_with_reasoning() -> _FakeResponse:
    """Mock LLM response with tool calls and reasoning_content."""
    tool_call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"query": "test"}'},
    }
    return _FakeResponse(
        _FakeMessage(
            content="  Looking it up...  ",
            tool_calls=[tool_call],
            reasoning_content="  I should use the lookup tool.  ",
        )
    )
