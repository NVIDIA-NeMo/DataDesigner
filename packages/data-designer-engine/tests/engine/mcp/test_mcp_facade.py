# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from data_designer.config.mcp import LocalStdioMCPProvider, ToolConfig
from data_designer.engine.mcp import io as mcp_io
from data_designer.engine.mcp.errors import MCPToolError
from data_designer.engine.mcp.facade import DEFAULT_TOOL_REFUSAL_MESSAGE, MCPFacade
from data_designer.engine.mcp.registry import MCPToolDefinition, MCPToolResult
from data_designer.engine.model_provider import MCPProviderRegistry


class FakeMessage:
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


class FakeChoice:
    """Fake choice class for mocking LLM completion responses."""

    def __init__(self, message: FakeMessage) -> None:
        self.message = message


class FakeResponse:
    """Fake response class for mocking LLM completion responses."""

    def __init__(self, message: FakeMessage) -> None:
        self.choices = [FakeChoice(message)]


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
def stub_secret_resolver() -> MagicMock:
    """Create a stub secret resolver for testing."""
    resolver = MagicMock()
    resolver.resolve.side_effect = lambda x: x  # Return the input as-is
    return resolver


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
def mock_completion_response_no_tools() -> FakeResponse:
    """Mock LLM response with no tool calls."""
    return FakeResponse(FakeMessage(content="Hello, I can help with that."))


@pytest.fixture
def mock_completion_response_single_tool() -> FakeResponse:
    """Mock LLM response with single tool call."""
    tool_call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"query": "test"}'},
    }
    return FakeResponse(FakeMessage(content="Let me look that up.", tool_calls=[tool_call]))


@pytest.fixture
def mock_completion_response_parallel_tools() -> FakeResponse:
    """Mock LLM response with multiple parallel tool calls."""
    tool_calls = [
        {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": '{"query": "first"}'}},
        {"id": "call-2", "type": "function", "function": {"name": "search", "arguments": '{"term": "second"}'}},
        {"id": "call-3", "type": "function", "function": {"name": "fetch", "arguments": '{"url": "example.com"}'}},
    ]
    return FakeResponse(FakeMessage(content="Executing multiple tools.", tool_calls=tool_calls))


@pytest.fixture
def mock_completion_response_with_reasoning() -> FakeResponse:
    """Mock LLM response with reasoning_content."""
    return FakeResponse(
        FakeMessage(
            content="  Final answer with extra spaces.  ",
            reasoning_content="  Thinking about the problem...  ",
        )
    )


@pytest.fixture
def mock_completion_response_tool_with_reasoning() -> FakeResponse:
    """Mock LLM response with tool calls and reasoning_content."""
    tool_call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"query": "test"}'},
    }
    return FakeResponse(
        FakeMessage(
            content="  Looking it up...  ",
            tool_calls=[tool_call],
            reasoning_content="  I should use the lookup tool.  ",
        )
    )


# =============================================================================
# tool_call_count() tests
# =============================================================================


def test_tool_call_count_no_tools(mock_completion_response_no_tools: FakeResponse) -> None:
    """Returns 0 when response has no tool calls."""
    assert MCPFacade.tool_call_count(mock_completion_response_no_tools) == 0


def test_tool_call_count_single_tool(mock_completion_response_single_tool: FakeResponse) -> None:
    """Returns 1 for single tool call."""
    assert MCPFacade.tool_call_count(mock_completion_response_single_tool) == 1


def test_tool_call_count_parallel_tools(mock_completion_response_parallel_tools: FakeResponse) -> None:
    """Returns correct count for parallel tool calls (e.g., 3)."""
    assert MCPFacade.tool_call_count(mock_completion_response_parallel_tools) == 3


def test_tool_call_count_none_tool_calls_attribute() -> None:
    """Returns 0 when tool_calls attribute is None."""
    response = FakeResponse(FakeMessage(content="Hello", tool_calls=None))
    assert MCPFacade.tool_call_count(response) == 0


# =============================================================================
# has_tool_calls() tests
# =============================================================================


def test_has_tool_calls_true(mock_completion_response_single_tool: FakeResponse) -> None:
    """Returns True when tool calls are present."""
    assert MCPFacade.has_tool_calls(mock_completion_response_single_tool) is True


def test_has_tool_calls_false(mock_completion_response_no_tools: FakeResponse) -> None:
    """Returns False when no tool calls are present."""
    assert MCPFacade.has_tool_calls(mock_completion_response_no_tools) is False


# =============================================================================
# process_completion_response() tests
# =============================================================================


def test_process_completion_no_tool_calls(
    stub_mcp_facade: MCPFacade,
    mock_completion_response_no_tools: FakeResponse,
) -> None:
    """Returns [assistant_message] when no tool calls present."""
    messages = stub_mcp_facade.process_completion_response(mock_completion_response_no_tools)

    assert len(messages) == 1
    assert messages[0].role == "assistant"
    assert messages[0].content == "Hello, I can help with that."
    assert not messages[0].tool_calls


def test_process_completion_with_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
    stub_mcp_facade: MCPFacade,
    mock_completion_response_single_tool: FakeResponse,
) -> None:
    """Returns [assistant_msg, tool_msg] for tool calls."""

    def mock_list_tools(provider: Any) -> tuple[MCPToolDefinition, ...]:
        return (MCPToolDefinition(name="lookup", description="Lookup", input_schema={"type": "object"}),)

    def mock_call_tools_parallel(
        calls: list[tuple[Any, str, dict[str, Any]]],
        *,
        timeout_sec: float | None = None,
    ) -> list[MCPToolResult]:
        return [MCPToolResult(content="Tool result for: " + args.get("query", "")) for _, _, args in calls]

    monkeypatch.setattr(mcp_io, "list_tools", mock_list_tools)
    monkeypatch.setattr(mcp_io, "call_tools_parallel", mock_call_tools_parallel)

    messages = stub_mcp_facade.process_completion_response(mock_completion_response_single_tool)

    assert len(messages) == 2
    assert messages[0].role == "assistant"
    assert messages[0].content == "Let me look that up."
    assert len(messages[0].tool_calls) == 1
    assert messages[1].role == "tool"
    assert messages[1].content == "Tool result for: test"
    assert messages[1].tool_call_id == "call-1"


def test_process_completion_preserves_content(
    stub_mcp_facade: MCPFacade,
    mock_completion_response_no_tools: FakeResponse,
) -> None:
    """Assistant content is preserved in returned message."""
    messages = stub_mcp_facade.process_completion_response(mock_completion_response_no_tools)

    assert messages[0].content == "Hello, I can help with that."


def test_process_completion_preserves_reasoning_content(
    stub_mcp_facade: MCPFacade,
    mock_completion_response_with_reasoning: FakeResponse,
) -> None:
    """Reasoning content is preserved when present."""
    messages = stub_mcp_facade.process_completion_response(mock_completion_response_with_reasoning)

    assert len(messages) == 1
    assert messages[0].reasoning_content == "Thinking about the problem..."


def test_process_completion_strips_whitespace_with_reasoning(
    stub_mcp_facade: MCPFacade,
    mock_completion_response_with_reasoning: FakeResponse,
) -> None:
    """Content and reasoning are stripped when reasoning is present."""
    messages = stub_mcp_facade.process_completion_response(mock_completion_response_with_reasoning)

    assert messages[0].content == "Final answer with extra spaces."
    assert messages[0].reasoning_content == "Thinking about the problem..."


def test_process_completion_parallel_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
    stub_mcp_facade: MCPFacade,
    mock_completion_response_parallel_tools: FakeResponse,
) -> None:
    """All parallel tool calls are executed and messages returned."""

    def mock_list_tools(provider: Any) -> tuple[MCPToolDefinition, ...]:
        return (
            MCPToolDefinition(name="lookup", description="Lookup", input_schema={"type": "object"}),
            MCPToolDefinition(name="search", description="Search", input_schema={"type": "object"}),
            MCPToolDefinition(name="fetch", description="Fetch", input_schema={"type": "object"}),
        )

    tool_names_called: list[str] = []

    def mock_call_tools_parallel(
        calls: list[tuple[Any, str, dict[str, Any]]],
        *,
        timeout_sec: float | None = None,
    ) -> list[MCPToolResult]:
        for _, tool_name, _ in calls:
            tool_names_called.append(tool_name)
        return [MCPToolResult(content=f"Result from {tool_name}") for _, tool_name, _ in calls]

    monkeypatch.setattr(mcp_io, "list_tools", mock_list_tools)
    monkeypatch.setattr(mcp_io, "call_tools_parallel", mock_call_tools_parallel)

    messages = stub_mcp_facade.process_completion_response(mock_completion_response_parallel_tools)

    assert len(messages) == 4  # 1 assistant + 3 tool results
    assert messages[0].role == "assistant"
    assert len(messages[0].tool_calls) == 3
    assert messages[1].role == "tool"
    assert messages[1].tool_call_id == "call-1"
    assert messages[2].role == "tool"
    assert messages[2].tool_call_id == "call-2"
    assert messages[3].role == "tool"
    assert messages[3].tool_call_id == "call-3"
    assert tool_names_called == ["lookup", "search", "fetch"]


def test_process_completion_tool_not_in_allow_list(
    monkeypatch: pytest.MonkeyPatch,
    stub_secret_resolver: MagicMock,
    stub_mcp_provider_registry: MCPProviderRegistry,
    stub_tool_config_with_allow_list: ToolConfig,
) -> None:
    """Raises MCPToolError when tool not in allow_tools."""
    facade = MCPFacade(
        tool_config=stub_tool_config_with_allow_list,
        secret_resolver=stub_secret_resolver,
        mcp_provider_registry=stub_mcp_provider_registry,
    )

    # Tool "forbidden" is not in allow_tools ["lookup", "search"]
    tool_call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "forbidden", "arguments": "{}"},
    }
    response = FakeResponse(FakeMessage(content="", tool_calls=[tool_call]))

    with pytest.raises(MCPToolError, match="not permitted"):
        facade.process_completion_response(response)


def test_process_completion_empty_content(
    monkeypatch: pytest.MonkeyPatch,
    stub_mcp_facade: MCPFacade,
) -> None:
    """Handles empty/None content gracefully."""

    def mock_list_tools(provider: Any) -> tuple[MCPToolDefinition, ...]:
        return (MCPToolDefinition(name="lookup", description="Lookup", input_schema={"type": "object"}),)

    def mock_call_tools_parallel(
        calls: list[tuple[Any, str, dict[str, Any]]],
        *,
        timeout_sec: float | None = None,
    ) -> list[MCPToolResult]:
        return [MCPToolResult(content="result") for _ in calls]

    monkeypatch.setattr(mcp_io, "list_tools", mock_list_tools)
    monkeypatch.setattr(mcp_io, "call_tools_parallel", mock_call_tools_parallel)

    tool_call = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    response = FakeResponse(FakeMessage(content=None, tool_calls=[tool_call]))

    messages = stub_mcp_facade.process_completion_response(response)

    assert len(messages) == 2
    assert messages[0].role == "assistant"
    assert messages[0].content == ""


# =============================================================================
# refuse_completion_response() tests
# =============================================================================


def test_refuse_completion_no_tool_calls(
    stub_mcp_facade: MCPFacade,
    mock_completion_response_no_tools: FakeResponse,
) -> None:
    """Returns [assistant_message] when no tool calls to refuse."""
    messages = stub_mcp_facade.refuse_completion_response(mock_completion_response_no_tools)

    assert len(messages) == 1
    assert messages[0].role == "assistant"
    assert messages[0].content == "Hello, I can help with that."


def test_refuse_completion_single_tool(
    stub_mcp_facade: MCPFacade,
    mock_completion_response_single_tool: FakeResponse,
) -> None:
    """Returns assistant + refusal message for single tool call."""
    messages = stub_mcp_facade.refuse_completion_response(mock_completion_response_single_tool)

    assert len(messages) == 2
    assert messages[0].role == "assistant"
    assert len(messages[0].tool_calls) == 1
    assert messages[1].role == "tool"
    assert messages[1].content == DEFAULT_TOOL_REFUSAL_MESSAGE
    assert messages[1].tool_call_id == "call-1"


def test_refuse_completion_parallel_tools(
    stub_mcp_facade: MCPFacade,
    mock_completion_response_parallel_tools: FakeResponse,
) -> None:
    """Returns assistant + refusal for each parallel tool call."""
    messages = stub_mcp_facade.refuse_completion_response(mock_completion_response_parallel_tools)

    assert len(messages) == 4  # 1 assistant + 3 refusals
    assert messages[0].role == "assistant"
    assert len(messages[0].tool_calls) == 3
    for i, msg in enumerate(messages[1:], start=1):
        assert msg.role == "tool"
        assert msg.content == DEFAULT_TOOL_REFUSAL_MESSAGE
        assert msg.tool_call_id == f"call-{i}"


def test_refuse_completion_default_message(
    stub_mcp_facade: MCPFacade,
    mock_completion_response_single_tool: FakeResponse,
) -> None:
    """Uses default refusal message when none provided."""
    messages = stub_mcp_facade.refuse_completion_response(mock_completion_response_single_tool)

    assert messages[1].content == DEFAULT_TOOL_REFUSAL_MESSAGE


def test_refuse_completion_custom_message(
    stub_mcp_facade: MCPFacade,
    mock_completion_response_single_tool: FakeResponse,
) -> None:
    """Uses custom refusal message when provided."""
    custom_message = "Custom refusal: Budget exceeded."
    messages = stub_mcp_facade.refuse_completion_response(
        mock_completion_response_single_tool,
        refusal_message=custom_message,
    )

    assert messages[1].content == custom_message


def test_refuse_completion_preserves_tool_call_ids(
    stub_mcp_facade: MCPFacade,
    mock_completion_response_parallel_tools: FakeResponse,
) -> None:
    """Refusal messages have correct tool_call_id linkage."""
    messages = stub_mcp_facade.refuse_completion_response(mock_completion_response_parallel_tools)

    # Verify each refusal message has the correct tool_call_id
    assert messages[1].tool_call_id == "call-1"
    assert messages[2].tool_call_id == "call-2"
    assert messages[3].tool_call_id == "call-3"


def test_refuse_completion_preserves_reasoning(
    stub_mcp_facade: MCPFacade,
    mock_completion_response_tool_with_reasoning: FakeResponse,
) -> None:
    """Reasoning content preserved in refusal scenario."""
    messages = stub_mcp_facade.refuse_completion_response(mock_completion_response_tool_with_reasoning)

    assert messages[0].role == "assistant"
    assert messages[0].reasoning_content == "I should use the lookup tool."
    assert messages[0].content == "Looking it up..."


def test_refuse_does_not_call_mcp_server(
    monkeypatch: pytest.MonkeyPatch,
    stub_mcp_facade: MCPFacade,
    mock_completion_response_single_tool: FakeResponse,
) -> None:
    """Verify MCP server is NOT called during refusal."""
    call_tool_called = False

    def mock_call_tool(*args: Any, **kwargs: Any) -> MCPToolResult:
        nonlocal call_tool_called
        call_tool_called = True
        return MCPToolResult(content="should not be called")

    monkeypatch.setattr(mcp_io, "call_tool", mock_call_tool)

    stub_mcp_facade.refuse_completion_response(mock_completion_response_single_tool)

    assert call_tool_called is False


# =============================================================================
# get_tool_schemas() tests
# =============================================================================


def test_get_tool_schemas_single_provider(
    monkeypatch: pytest.MonkeyPatch,
    stub_mcp_facade: MCPFacade,
) -> None:
    """Fetches schemas from single provider."""

    def mock_list_tools(provider: Any) -> tuple[MCPToolDefinition, ...]:
        return (
            MCPToolDefinition(name="lookup", description="Lookup tool", input_schema={"type": "object"}),
            MCPToolDefinition(name="search", description="Search tool", input_schema={"type": "object"}),
        )

    monkeypatch.setattr(mcp_io, "list_tools", mock_list_tools)

    schemas = stub_mcp_facade.get_tool_schemas()

    assert len(schemas) == 2
    assert schemas[0]["function"]["name"] == "lookup"
    assert schemas[1]["function"]["name"] == "search"


def test_get_tool_schemas_multiple_providers(
    monkeypatch: pytest.MonkeyPatch,
    stub_secret_resolver: MagicMock,
    stub_mcp_provider_registry: MCPProviderRegistry,
) -> None:
    """Fetches and combines schemas from multiple providers."""
    tool_config = ToolConfig(
        tool_alias="multi-provider",
        providers=["tools", "secondary"],
        max_tool_call_turns=3,
    )
    facade = MCPFacade(
        tool_config=tool_config, secret_resolver=stub_secret_resolver, mcp_provider_registry=stub_mcp_provider_registry
    )

    def mock_list_tools(provider: Any) -> tuple[MCPToolDefinition, ...]:
        if provider.name == "tools":
            return (MCPToolDefinition(name="lookup", description="Lookup", input_schema={"type": "object"}),)
        return (MCPToolDefinition(name="fetch", description="Fetch", input_schema={"type": "object"}),)

    monkeypatch.setattr(mcp_io, "list_tools", mock_list_tools)

    schemas = facade.get_tool_schemas()

    assert len(schemas) == 2
    tool_names = {s["function"]["name"] for s in schemas}
    assert tool_names == {"lookup", "fetch"}


def test_get_tool_schemas_with_allow_tools_filter(
    monkeypatch: pytest.MonkeyPatch,
    stub_secret_resolver: MagicMock,
    stub_mcp_provider_registry: MCPProviderRegistry,
    stub_tool_config_with_allow_list: ToolConfig,
) -> None:
    """Only returns schemas for allowed tools."""
    facade = MCPFacade(
        tool_config=stub_tool_config_with_allow_list,
        secret_resolver=stub_secret_resolver,
        mcp_provider_registry=stub_mcp_provider_registry,
    )

    def mock_list_tools(provider: Any) -> tuple[MCPToolDefinition, ...]:
        return (
            MCPToolDefinition(name="lookup", description="Lookup", input_schema={"type": "object"}),
            MCPToolDefinition(name="search", description="Search", input_schema={"type": "object"}),
            MCPToolDefinition(name="forbidden", description="Forbidden", input_schema={"type": "object"}),
        )

    monkeypatch.setattr(mcp_io, "list_tools", mock_list_tools)

    schemas = facade.get_tool_schemas()

    assert len(schemas) == 2
    tool_names = {s["function"]["name"] for s in schemas}
    assert tool_names == {"lookup", "search"}


def test_get_tool_schemas_missing_allowed_tool(
    monkeypatch: pytest.MonkeyPatch,
    stub_secret_resolver: MagicMock,
    stub_mcp_provider_registry: MCPProviderRegistry,
) -> None:
    """Raises error when allowed tool not found on any provider."""
    tool_config = ToolConfig(
        tool_alias="test",
        providers=["tools"],
        allow_tools=["missing_tool"],
    )
    facade = MCPFacade(
        tool_config=tool_config, secret_resolver=stub_secret_resolver, mcp_provider_registry=stub_mcp_provider_registry
    )

    def mock_list_tools(provider: Any) -> tuple[MCPToolDefinition, ...]:
        return (MCPToolDefinition(name="lookup", description="Lookup", input_schema={"type": "object"}),)

    monkeypatch.setattr(mcp_io, "list_tools", mock_list_tools)

    from data_designer.engine.mcp.errors import MCPConfigurationError

    with pytest.raises(MCPConfigurationError, match="not found"):
        facade.get_tool_schemas()


# =============================================================================
# _normalize_tool_call() edge case tests
# =============================================================================


def test_normalize_tool_call_missing_name(stub_mcp_facade: MCPFacade) -> None:
    """Raises MCPToolError when tool call is missing a name."""
    raw_tool_call = {"id": "call-1", "function": {"arguments": "{}"}}

    with pytest.raises(MCPToolError, match="missing a tool name"):
        stub_mcp_facade._normalize_tool_call(raw_tool_call)


def test_normalize_tool_call_invalid_arguments(stub_mcp_facade: MCPFacade) -> None:
    """Raises MCPToolError when arguments are invalid JSON."""
    raw_tool_call = {"id": "call-1", "function": {"name": "lookup", "arguments": "not json"}}

    with pytest.raises(MCPToolError, match="Invalid tool arguments"):
        stub_mcp_facade._normalize_tool_call(raw_tool_call)


def test_normalize_tool_call_dict_arguments(stub_mcp_facade: MCPFacade) -> None:
    """Handles dict arguments correctly."""
    raw_tool_call = {"id": "call-1", "function": {"name": "lookup", "arguments": {"query": "test"}}}

    result = stub_mcp_facade._normalize_tool_call(raw_tool_call)

    assert result["name"] == "lookup"
    assert result["arguments"] == {"query": "test"}
    assert result["arguments_json"] == '{"query": "test"}'


def test_normalize_tool_call_empty_arguments(stub_mcp_facade: MCPFacade) -> None:
    """Handles empty/None arguments gracefully."""
    raw_tool_call = {"id": "call-1", "function": {"name": "lookup", "arguments": None}}

    result = stub_mcp_facade._normalize_tool_call(raw_tool_call)

    assert result["arguments"] == {}
    assert result["arguments_json"] == "{}"


def test_normalize_tool_call_generates_id(stub_mcp_facade: MCPFacade) -> None:
    """Generates UUID when tool call id is missing."""
    raw_tool_call = {"function": {"name": "lookup", "arguments": "{}"}}

    result = stub_mcp_facade._normalize_tool_call(raw_tool_call)

    assert result["id"] is not None
    assert len(result["id"]) == 32  # UUID hex format


def test_normalize_tool_call_object_format(stub_mcp_facade: MCPFacade) -> None:
    """Handles object format tool calls (not dict)."""

    class FakeFunction:
        name = "lookup"
        arguments = '{"query": "test"}'

    class FakeToolCall:
        id = "call-obj-1"
        function = FakeFunction()

    result = stub_mcp_facade._normalize_tool_call(FakeToolCall())

    assert result["id"] == "call-obj-1"
    assert result["name"] == "lookup"
    assert result["arguments"] == {"query": "test"}


# =============================================================================
# Properties tests
# =============================================================================


def test_tool_alias_property(stub_mcp_facade: MCPFacade, stub_tool_config: ToolConfig) -> None:
    """Tool alias property returns configured value."""
    assert stub_mcp_facade.tool_alias == stub_tool_config.tool_alias


def test_providers_property(stub_mcp_facade: MCPFacade, stub_tool_config: ToolConfig) -> None:
    """Providers property returns configured value."""
    assert stub_mcp_facade.providers == stub_tool_config.providers


def test_max_tool_call_turns_property(stub_mcp_facade: MCPFacade, stub_tool_config: ToolConfig) -> None:
    """Max tool call turns property returns configured value."""
    assert stub_mcp_facade.max_tool_call_turns == stub_tool_config.max_tool_call_turns


def test_allow_tools_property_none(stub_mcp_facade: MCPFacade) -> None:
    """Allow tools property returns None when not configured."""
    assert stub_mcp_facade.allow_tools is None


def test_allow_tools_property_with_list(
    stub_tool_config_with_allow_list: ToolConfig,
    stub_secret_resolver: MagicMock,
    stub_mcp_provider_registry: MCPProviderRegistry,
) -> None:
    """Allow tools property returns configured list."""
    facade = MCPFacade(
        tool_config=stub_tool_config_with_allow_list,
        secret_resolver=stub_secret_resolver,
        mcp_provider_registry=stub_mcp_provider_registry,
    )
    assert facade.allow_tools == ["lookup", "search"]


def test_timeout_sec_property(stub_mcp_facade: MCPFacade, stub_tool_config: ToolConfig) -> None:
    """Timeout sec property returns configured value."""
    assert stub_mcp_facade.timeout_sec == stub_tool_config.timeout_sec
