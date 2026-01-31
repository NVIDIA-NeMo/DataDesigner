# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from data_designer.config.mcp import LocalStdioMCPProvider, ToolConfig
from data_designer.engine.mcp import io as mcp_io
from data_designer.engine.mcp.errors import DuplicateToolNameError, MCPToolError
from data_designer.engine.mcp.facade import DEFAULT_TOOL_REFUSAL_MESSAGE, MCPFacade
from data_designer.engine.mcp.registry import MCPToolDefinition, MCPToolResult
from data_designer.engine.model_provider import MCPProviderRegistry


# Fake classes are used directly in tests to create custom responses
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

    def mock_list_tools(provider: Any, timeout_sec: float | None = None) -> tuple[MCPToolDefinition, ...]:
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

    def mock_list_tools(provider: Any, timeout_sec: float | None = None) -> tuple[MCPToolDefinition, ...]:
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

    def mock_list_tools(provider: Any, timeout_sec: float | None = None) -> tuple[MCPToolDefinition, ...]:
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

    def mock_list_tools(provider: Any, timeout_sec: float | None = None) -> tuple[MCPToolDefinition, ...]:
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

    def mock_list_tools(provider: Any, timeout_sec: float | None = None) -> tuple[MCPToolDefinition, ...]:
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

    def mock_list_tools(provider: Any, timeout_sec: float | None = None) -> tuple[MCPToolDefinition, ...]:
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

    def mock_list_tools(provider: Any, timeout_sec: float | None = None) -> tuple[MCPToolDefinition, ...]:
        return (MCPToolDefinition(name="lookup", description="Lookup", input_schema={"type": "object"}),)

    monkeypatch.setattr(mcp_io, "list_tools", mock_list_tools)

    from data_designer.engine.mcp.errors import MCPConfigurationError

    with pytest.raises(MCPConfigurationError, match="not found"):
        facade.get_tool_schemas()


# =============================================================================
# Tool call normalization via public API (process_completion_response)
# =============================================================================


def test_process_completion_missing_tool_name(stub_mcp_facade: MCPFacade) -> None:
    """process_completion_response raises MCPToolError when tool call has no name."""
    tool_call = {"id": "call-1", "function": {"arguments": "{}"}}  # Missing name
    response = FakeResponse(FakeMessage(content="", tool_calls=[tool_call]))

    with pytest.raises(MCPToolError, match="missing a tool name"):
        stub_mcp_facade.process_completion_response(response)


def test_process_completion_invalid_json_arguments(stub_mcp_facade: MCPFacade) -> None:
    """process_completion_response raises MCPToolError when arguments are invalid JSON."""
    tool_call = {"id": "call-1", "function": {"name": "lookup", "arguments": "not valid json"}}
    response = FakeResponse(FakeMessage(content="", tool_calls=[tool_call]))

    with pytest.raises(MCPToolError, match="Invalid tool arguments"):
        stub_mcp_facade.process_completion_response(response)


def test_process_completion_dict_arguments(
    monkeypatch: pytest.MonkeyPatch,
    stub_mcp_facade: MCPFacade,
) -> None:
    """process_completion_response handles dict arguments correctly."""

    def mock_list_tools(provider: Any, timeout_sec: float | None = None) -> tuple[MCPToolDefinition, ...]:
        return (MCPToolDefinition(name="lookup", description="Lookup", input_schema={"type": "object"}),)

    captured_args: list[dict[str, Any]] = []

    def mock_call_tools_parallel(
        calls: list[tuple[Any, str, dict[str, Any]]],
        *,
        timeout_sec: float | None = None,
    ) -> list[MCPToolResult]:
        for _, _, args in calls:
            captured_args.append(args)
        return [MCPToolResult(content="result") for _ in calls]

    monkeypatch.setattr(mcp_io, "list_tools", mock_list_tools)
    monkeypatch.setattr(mcp_io, "call_tools_parallel", mock_call_tools_parallel)

    # Pass dict arguments (not JSON string)
    tool_call = {"id": "call-1", "function": {"name": "lookup", "arguments": {"query": "test"}}}
    response = FakeResponse(FakeMessage(content="", tool_calls=[tool_call]))

    messages = stub_mcp_facade.process_completion_response(response)

    assert len(messages) == 2
    assert captured_args[0] == {"query": "test"}


def test_process_completion_empty_arguments(
    monkeypatch: pytest.MonkeyPatch,
    stub_mcp_facade: MCPFacade,
) -> None:
    """process_completion_response handles None/empty arguments gracefully."""

    def mock_list_tools(provider: Any, timeout_sec: float | None = None) -> tuple[MCPToolDefinition, ...]:
        return (MCPToolDefinition(name="lookup", description="Lookup", input_schema={"type": "object"}),)

    captured_args: list[dict[str, Any]] = []

    def mock_call_tools_parallel(
        calls: list[tuple[Any, str, dict[str, Any]]],
        *,
        timeout_sec: float | None = None,
    ) -> list[MCPToolResult]:
        for _, _, args in calls:
            captured_args.append(args)
        return [MCPToolResult(content="result") for _ in calls]

    monkeypatch.setattr(mcp_io, "list_tools", mock_list_tools)
    monkeypatch.setattr(mcp_io, "call_tools_parallel", mock_call_tools_parallel)

    tool_call = {"id": "call-1", "function": {"name": "lookup", "arguments": None}}
    response = FakeResponse(FakeMessage(content="", tool_calls=[tool_call]))

    messages = stub_mcp_facade.process_completion_response(response)

    assert len(messages) == 2
    assert captured_args[0] == {}  # Empty dict for None arguments


def test_process_completion_generates_tool_call_id(
    monkeypatch: pytest.MonkeyPatch,
    stub_mcp_facade: MCPFacade,
) -> None:
    """process_completion_response generates UUID for tool calls without ID."""

    def mock_list_tools(provider: Any, timeout_sec: float | None = None) -> tuple[MCPToolDefinition, ...]:
        return (MCPToolDefinition(name="lookup", description="Lookup", input_schema={"type": "object"}),)

    def mock_call_tools_parallel(
        calls: list[tuple[Any, str, dict[str, Any]]],
        *,
        timeout_sec: float | None = None,
    ) -> list[MCPToolResult]:
        return [MCPToolResult(content="result") for _ in calls]

    monkeypatch.setattr(mcp_io, "list_tools", mock_list_tools)
    monkeypatch.setattr(mcp_io, "call_tools_parallel", mock_call_tools_parallel)

    # Tool call without id
    tool_call = {"function": {"name": "lookup", "arguments": "{}"}}
    response = FakeResponse(FakeMessage(content="", tool_calls=[tool_call]))

    messages = stub_mcp_facade.process_completion_response(response)

    # Should have generated an ID
    assert len(messages) == 2
    assert messages[1].tool_call_id is not None
    assert len(messages[1].tool_call_id) == 32  # UUID hex format


def test_process_completion_object_format_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
    stub_mcp_facade: MCPFacade,
) -> None:
    """process_completion_response handles object format tool calls."""

    def mock_list_tools(provider: Any, timeout_sec: float | None = None) -> tuple[MCPToolDefinition, ...]:
        return (MCPToolDefinition(name="lookup", description="Lookup", input_schema={"type": "object"}),)

    captured_calls: list[tuple[str, dict[str, Any]]] = []

    def mock_call_tools_parallel(
        calls: list[tuple[Any, str, dict[str, Any]]],
        *,
        timeout_sec: float | None = None,
    ) -> list[MCPToolResult]:
        for _, tool_name, args in calls:
            captured_calls.append((tool_name, args))
        return [MCPToolResult(content="result") for _ in calls]

    monkeypatch.setattr(mcp_io, "list_tools", mock_list_tools)
    monkeypatch.setattr(mcp_io, "call_tools_parallel", mock_call_tools_parallel)

    # Create object format tool call (simulating what some LLM libraries return)
    class FakeFunction:
        name = "lookup"
        arguments = '{"query": "test"}'

    class FakeToolCall:
        id = "call-obj-1"
        function = FakeFunction()

    response = FakeResponse(FakeMessage(content="", tool_calls=[FakeToolCall()]))

    messages = stub_mcp_facade.process_completion_response(response)

    assert len(messages) == 2
    assert captured_calls[0] == ("lookup", {"query": "test"})
    assert messages[1].tool_call_id == "call-obj-1"


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


# =============================================================================
# _resolve_provider tests
# =============================================================================


def test_resolve_provider_with_api_key() -> None:
    """Test that _resolve_provider resolves api_key when present."""
    from data_designer.config.mcp import MCPProvider

    secret_resolver = MagicMock()
    secret_resolver.resolve.return_value = "resolved-secret-key"

    provider = MCPProvider(
        name="test",
        endpoint="http://localhost:8080/sse",
        api_key="API_KEY_ENV_VAR",
    )

    tool_config = ToolConfig(tool_alias="test-tools", providers=["test"])
    mcp_provider_registry = MCPProviderRegistry(providers=[provider])

    facade = MCPFacade(
        tool_config=tool_config,
        secret_resolver=secret_resolver,
        mcp_provider_registry=mcp_provider_registry,
    )

    resolved_provider = facade._resolve_provider(provider)

    secret_resolver.resolve.assert_called_with("API_KEY_ENV_VAR")
    assert resolved_provider.api_key == "resolved-secret-key"
    # Original provider should not be modified
    assert provider.api_key == "API_KEY_ENV_VAR"


def test_resolve_provider_without_api_key() -> None:
    """Test that _resolve_provider returns provider unchanged when no api_key."""
    provider = LocalStdioMCPProvider(name="test", command="python")

    tool_config = ToolConfig(tool_alias="test-tools", providers=["test"])
    mcp_provider_registry = MCPProviderRegistry(providers=[provider])
    secret_resolver = MagicMock()

    facade = MCPFacade(
        tool_config=tool_config,
        secret_resolver=secret_resolver,
        mcp_provider_registry=mcp_provider_registry,
    )

    resolved_provider = facade._resolve_provider(provider)

    # Should return the same provider without calling resolve
    assert resolved_provider is provider
    secret_resolver.resolve.assert_not_called()


# =============================================================================
# Duplicate tool name validation tests
# =============================================================================


def test_get_tool_schemas_duplicate_tool_names_raises_error(
    monkeypatch: pytest.MonkeyPatch,
    stub_secret_resolver: MagicMock,
    stub_mcp_provider_registry: MCPProviderRegistry,
) -> None:
    """Raises DuplicateToolNameError when same tool name appears in multiple providers."""
    tool_config = ToolConfig(
        tool_alias="multi-provider",
        providers=["tools", "secondary"],
        max_tool_call_turns=3,
    )
    facade = MCPFacade(
        tool_config=tool_config, secret_resolver=stub_secret_resolver, mcp_provider_registry=stub_mcp_provider_registry
    )

    def mock_list_tools(provider: Any, timeout_sec: float | None = None) -> tuple[MCPToolDefinition, ...]:
        # Both providers have a tool named "lookup"
        if provider.name == "tools":
            return (
                MCPToolDefinition(name="lookup", description="Lookup from tools", input_schema={"type": "object"}),
                MCPToolDefinition(name="unique_to_tools", description="Unique", input_schema={"type": "object"}),
            )
        return (
            MCPToolDefinition(name="lookup", description="Lookup from secondary", input_schema={"type": "object"}),
            MCPToolDefinition(name="unique_to_secondary", description="Unique", input_schema={"type": "object"}),
        )

    monkeypatch.setattr(mcp_io, "list_tools", mock_list_tools)

    with pytest.raises(DuplicateToolNameError, match="Duplicate tool names found"):
        facade.get_tool_schemas()


def test_get_tool_schemas_duplicate_tool_names_reports_all_duplicates(
    monkeypatch: pytest.MonkeyPatch,
    stub_secret_resolver: MagicMock,
    stub_mcp_provider_registry: MCPProviderRegistry,
) -> None:
    """Error message reports all duplicate tool names, not just the first."""
    tool_config = ToolConfig(
        tool_alias="multi-provider",
        providers=["tools", "secondary"],
        max_tool_call_turns=3,
    )
    facade = MCPFacade(
        tool_config=tool_config, secret_resolver=stub_secret_resolver, mcp_provider_registry=stub_mcp_provider_registry
    )

    def mock_list_tools(provider: Any, timeout_sec: float | None = None) -> tuple[MCPToolDefinition, ...]:
        # Both providers have "lookup" and "search" as duplicates
        if provider.name == "tools":
            return (
                MCPToolDefinition(name="lookup", description="Lookup", input_schema={"type": "object"}),
                MCPToolDefinition(name="search", description="Search", input_schema={"type": "object"}),
            )
        return (
            MCPToolDefinition(name="lookup", description="Lookup", input_schema={"type": "object"}),
            MCPToolDefinition(name="search", description="Search", input_schema={"type": "object"}),
        )

    monkeypatch.setattr(mcp_io, "list_tools", mock_list_tools)

    with pytest.raises(DuplicateToolNameError) as exc_info:
        facade.get_tool_schemas()

    # Both duplicates should be reported
    assert "lookup" in str(exc_info.value)
    assert "search" in str(exc_info.value)


def test_get_tool_schemas_no_duplicates_passes(
    monkeypatch: pytest.MonkeyPatch,
    stub_secret_resolver: MagicMock,
    stub_mcp_provider_registry: MCPProviderRegistry,
) -> None:
    """No error when tool names are unique across providers."""
    tool_config = ToolConfig(
        tool_alias="multi-provider",
        providers=["tools", "secondary"],
        max_tool_call_turns=3,
    )
    facade = MCPFacade(
        tool_config=tool_config, secret_resolver=stub_secret_resolver, mcp_provider_registry=stub_mcp_provider_registry
    )

    def mock_list_tools(provider: Any, timeout_sec: float | None = None) -> tuple[MCPToolDefinition, ...]:
        # Each provider has unique tool names
        if provider.name == "tools":
            return (MCPToolDefinition(name="lookup", description="Lookup", input_schema={"type": "object"}),)
        return (MCPToolDefinition(name="fetch", description="Fetch", input_schema={"type": "object"}),)

    monkeypatch.setattr(mcp_io, "list_tools", mock_list_tools)

    # Should not raise
    schemas = facade.get_tool_schemas()
    assert len(schemas) == 2


def test_get_tool_schemas_single_provider_no_duplicates(
    monkeypatch: pytest.MonkeyPatch,
    stub_mcp_facade: MCPFacade,
) -> None:
    """Single provider cannot have duplicates (each tool name unique within provider)."""

    def mock_list_tools(provider: Any, timeout_sec: float | None = None) -> tuple[MCPToolDefinition, ...]:
        return (
            MCPToolDefinition(name="lookup", description="Lookup", input_schema={"type": "object"}),
            MCPToolDefinition(name="search", description="Search", input_schema={"type": "object"}),
        )

    monkeypatch.setattr(mcp_io, "list_tools", mock_list_tools)

    # Should not raise
    schemas = stub_mcp_facade.get_tool_schemas()
    assert len(schemas) == 2
