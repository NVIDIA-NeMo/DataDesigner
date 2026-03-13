# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data_designer.engine.mcp.errors import MCPConfigurationError, MCPToolError
from data_designer.engine.models.clients.types import (
    ChatCompletionResponse,
    EmbeddingResponse,
    ImageGenerationResponse,
    ImagePayload,
    ToolCall,
)
from data_designer.engine.models.errors import ImageGenerationError, ModelGenerationValidationFailureError
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.models.parsers.errors import ParserException
from data_designer.engine.models.utils import ChatMessage
from data_designer.engine.testing import StubMCPFacade, StubMCPRegistry, make_stub_completion_response


def _make_response(content: str | None = None, **kwargs: Any) -> ChatCompletionResponse:
    """Shorthand for creating a ChatCompletionResponse in tests."""
    return make_stub_completion_response(content=content, **kwargs)


@pytest.fixture
def stub_model_facade(
    stub_model_configs: list[Any],
    stub_model_client: MagicMock,
    stub_model_provider_registry: Any,
) -> ModelFacade:
    return ModelFacade(
        model_config=stub_model_configs[0],
        model_provider_registry=stub_model_provider_registry,
        client=stub_model_client,
    )


@pytest.fixture
def stub_completion_messages() -> list[ChatMessage]:
    return [ChatMessage.as_user("test")]


@pytest.mark.parametrize(
    "max_correction_steps,max_conversation_restarts,total_calls",
    [
        (0, 0, 1),
        (1, 1, 4),
        (1, 2, 6),
        (5, 0, 6),
        (0, 5, 6),
        (3, 3, 16),
    ],
)
@patch.object(ModelFacade, "completion", autospec=True)
def test_generate(
    mock_completion: Any,
    stub_model_facade: ModelFacade,
    max_correction_steps: int,
    max_conversation_restarts: int,
    total_calls: int,
) -> None:
    bad_response = _make_response("bad response")
    mock_completion.side_effect = lambda *args, **kwargs: bad_response

    def _failing_parser(response: str) -> str:
        raise ParserException("parser exception")

    with pytest.raises(ModelGenerationValidationFailureError):
        stub_model_facade.generate(
            prompt="foo",
            system_prompt="bar",
            parser=_failing_parser,
            max_correction_steps=max_correction_steps,
            max_conversation_restarts=max_conversation_restarts,
        )
    assert mock_completion.call_count == total_calls

    with pytest.raises(ModelGenerationValidationFailureError):
        stub_model_facade.generate(
            prompt="foo",
            parser=_failing_parser,
            system_prompt="bar",
            max_correction_steps=max_correction_steps,
            max_conversation_restarts=max_conversation_restarts,
        )
    assert mock_completion.call_count == 2 * total_calls


@pytest.mark.parametrize(
    "system_prompt,expected_messages",
    [
        ("", [ChatMessage.as_user("does not matter")]),
        ("hello!", [ChatMessage.as_system("hello!"), ChatMessage.as_user("does not matter")]),
    ],
)
@patch.object(ModelFacade, "completion", autospec=True)
def test_generate_with_system_prompt(
    mock_completion: Any,
    stub_model_facade: ModelFacade,
    system_prompt: str,
    expected_messages: list[ChatMessage],
) -> None:
    captured_messages = []

    def capture_and_return(*args: Any, **kwargs: Any) -> ChatCompletionResponse:
        captured_messages.append(list(args[1]))
        return _make_response("Hello!")

    mock_completion.side_effect = capture_and_return

    stub_model_facade.generate(prompt="does not matter", system_prompt=system_prompt, parser=lambda x: x)
    assert mock_completion.call_count == 1
    assert captured_messages[0] == expected_messages


@patch.object(ModelFacade, "completion", autospec=True)
def test_generate_includes_parser_validation_detail_in_user_facing_error(
    mock_completion: Any,
    stub_model_facade: ModelFacade,
) -> None:
    mock_completion.return_value = _make_response("bad response")

    def _failing_parser(response: str) -> str:
        raise ParserException("Response doesn't match requested <response_schema>\n'name' is a required property")

    with pytest.raises(
        ModelGenerationValidationFailureError,
        match="Validation detail: Response doesn't match requested <response_schema> 'name' is a required property.",
    ):
        stub_model_facade.generate(
            prompt="foo",
            parser=_failing_parser,
            max_correction_steps=0,
            max_conversation_restarts=0,
        )


@pytest.mark.parametrize(
    "raw_content,expected",
    [
        ("\nHello world", "Hello world"),
        ("  Hello world  ", "Hello world"),
        ("\n\n  Hello world\n", "Hello world"),
        ("Hello world", "Hello world"),
    ],
)
@patch("data_designer.engine.models.facade.ModelFacade.completion", autospec=True)
def test_generate_strips_response_content(
    mock_completion: Any,
    stub_model_facade: ModelFacade,
    raw_content: str,
    expected: str,
) -> None:
    """Response content from the LLM is stripped of leading/trailing whitespace."""
    mock_completion.side_effect = lambda *args, **kwargs: _make_response(raw_content)
    result, _ = stub_model_facade.generate(prompt="test", parser=lambda x: x)
    assert result == expected


def test_model_alias_property(stub_model_facade: ModelFacade, stub_model_configs: list[Any]) -> None:
    assert stub_model_facade.model_alias == stub_model_configs[0].alias


def test_usage_stats_property(stub_model_facade: ModelFacade) -> None:
    assert stub_model_facade.usage_stats is not None
    assert hasattr(stub_model_facade.usage_stats, "model_dump")


def test_consolidate_kwargs(stub_model_configs: list[Any], stub_model_facade: ModelFacade) -> None:
    # Model config generate kwargs are used as base, and purpose is removed
    result = stub_model_facade.consolidate_kwargs(purpose="test")
    assert result == stub_model_configs[0].inference_parameters.generate_kwargs

    # kwargs overrides model config generate kwargs
    result = stub_model_facade.consolidate_kwargs(temperature=0.01, purpose="test")
    assert result == {**stub_model_configs[0].inference_parameters.generate_kwargs, "temperature": 0.01}

    # Provider extra_body overrides all other kwargs
    stub_model_facade.model_provider.extra_body = {"foo_provider": "bar_provider"}
    result = stub_model_facade.consolidate_kwargs(extra_body={"foo": "bar"}, purpose="test")
    assert result == {
        **stub_model_configs[0].inference_parameters.generate_kwargs,
        "extra_body": {"foo_provider": "bar_provider", "foo": "bar"},
    }

    # Provider extra_headers merges with caller headers (provider takes precedence)
    stub_model_facade.model_provider.extra_body = None
    stub_model_facade.model_provider.extra_headers = {"hello": "world", "hola": "mundo"}
    result = stub_model_facade.consolidate_kwargs(extra_headers={"hello": "caller", "X-Trace-ID": "abc"})
    assert result == {
        **stub_model_configs[0].inference_parameters.generate_kwargs,
        "extra_headers": {"hello": "world", "hola": "mundo", "X-Trace-ID": "abc"},
    }


@pytest.mark.parametrize(
    "skip_usage_tracking",
    [
        False,
        True,
    ],
)
def test_completion_success(
    stub_completion_messages: list[ChatMessage],
    stub_model_configs: Any,
    stub_model_facade: ModelFacade,
    stub_model_client: MagicMock,
    skip_usage_tracking: bool,
) -> None:
    expected_response = _make_response("Test response")
    stub_model_client.completion.return_value = expected_response
    result = stub_model_facade.completion(stub_completion_messages, skip_usage_tracking=skip_usage_tracking)
    assert result == expected_response
    assert stub_model_client.completion.call_count == 1


def test_completion_with_exception(
    stub_completion_messages: list[ChatMessage],
    stub_model_facade: ModelFacade,
    stub_model_client: MagicMock,
) -> None:
    stub_model_client.completion.side_effect = Exception("Router error")

    with pytest.raises(Exception, match="Router error"):
        stub_model_facade.completion(stub_completion_messages)


def test_completion_with_kwargs(
    stub_completion_messages: list[ChatMessage],
    stub_model_configs: Any,
    stub_model_facade: ModelFacade,
    stub_model_client: MagicMock,
) -> None:
    expected_response = _make_response("Test response")
    stub_model_client.completion.return_value = expected_response

    kwargs = {"temperature": 0.7, "max_tokens": 100}
    result = stub_model_facade.completion(stub_completion_messages, **kwargs)

    assert result == expected_response
    assert stub_model_client.completion.call_count == 1


def test_generate_text_embeddings_success(
    stub_model_facade: ModelFacade,
    stub_model_client: MagicMock,
) -> None:
    expected_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    stub_model_client.embeddings.return_value = EmbeddingResponse(vectors=expected_vectors)
    input_texts = ["test1", "test2"]
    result = stub_model_facade.generate_text_embeddings(input_texts)
    assert result == expected_vectors


def test_generate_text_embeddings_with_exception(
    stub_model_facade: ModelFacade,
    stub_model_client: MagicMock,
) -> None:
    stub_model_client.embeddings.side_effect = Exception("Router error")

    with pytest.raises(Exception, match="Router error"):
        stub_model_facade.generate_text_embeddings(["test1", "test2"])


def test_generate_text_embeddings_with_kwargs(
    stub_model_configs: Any,
    stub_model_facade: ModelFacade,
    stub_model_client: MagicMock,
) -> None:
    expected_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    stub_model_client.embeddings.return_value = EmbeddingResponse(vectors=expected_vectors)
    kwargs = {"temperature": 0.7, "max_tokens": 100, "input_type": "query"}
    _ = stub_model_facade.generate_text_embeddings(["test1", "test2"], **kwargs)
    assert stub_model_client.embeddings.call_count == 1


def test_generate_with_mcp_tools(
    stub_model_configs: Any,
    stub_model_client: MagicMock,
    stub_model_provider_registry: Any,
) -> None:
    tool_call = ToolCall(id="call-1", name="lookup", arguments_json='{"query": "foo"}')
    responses = [
        _make_response(content=None, tool_calls=[tool_call]),
        _make_response("final result"),
    ]
    captured_calls: list[tuple[list[ChatMessage], dict[str, Any]]] = []
    registry_calls: list[tuple[str, str, dict[str, str], None]] = []

    def process_with_tracking(completion_response: ChatCompletionResponse) -> list[ChatMessage]:
        message = completion_response.message
        if not message.tool_calls:
            return [ChatMessage.as_assistant(content=message.content or "")]
        registry_calls.append(("tools", "lookup", {"query": "foo"}, None))
        tc_dict = {
            "id": "call-1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"query": "foo"}'},
        }
        return [
            ChatMessage.as_assistant(content="", tool_calls=[tc_dict]),
            ChatMessage.as_tool(content="tool-output", tool_call_id="call-1"),
        ]

    facade = StubMCPFacade(
        tool_schemas=[
            {
                "type": "function",
                "function": {"name": "lookup", "description": "Lookup", "parameters": {"type": "object"}},
            }
        ],
        process_fn=process_with_tracking,
    )
    registry = StubMCPRegistry(facade)

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> ChatCompletionResponse:
        captured_calls.append((messages, kwargs))
        return responses.pop(0)

    model = ModelFacade(
        model_config=stub_model_configs[0],
        model_provider_registry=stub_model_provider_registry,
        client=stub_model_client,
        mcp_registry=registry,
    )

    with patch.object(ModelFacade, "completion", new=_completion):
        result, _ = model.generate(prompt="question", parser=lambda x: x, tool_alias="tools")

    assert result == "final result"
    assert len(captured_calls) == 2
    assert "tools" in captured_calls[0][1]
    assert captured_calls[0][1]["tools"][0]["function"]["name"] == "lookup"
    assert any(message.role == "tool" for message in captured_calls[1][0])
    assert registry_calls == [("tools", "lookup", {"query": "foo"}, None)]


def test_generate_with_tools_missing_registry(
    stub_model_configs: Any, stub_model_client: MagicMock, stub_model_provider_registry: Any
) -> None:
    model = ModelFacade(
        model_config=stub_model_configs[0],
        model_provider_registry=stub_model_provider_registry,
        client=stub_model_client,
        mcp_registry=None,
    )

    with pytest.raises(MCPConfigurationError):
        model.generate(prompt="question", parser=lambda x: x, tool_alias="tools")


# =============================================================================
# Tool calling integration tests
# =============================================================================


def test_generate_with_tool_alias_multiple_turns(
    stub_model_configs: Any,
    stub_model_client: MagicMock,
    stub_model_provider_registry: Any,
) -> None:
    """Multiple tool call turns before final response."""
    tool_call_1 = ToolCall(id="call-1", name="lookup", arguments_json='{"query": "foo"}')
    tool_call_2 = ToolCall(id="call-2", name="search", arguments_json='{"term": "bar"}')

    responses = [
        _make_response("First lookup", tool_calls=[tool_call_1]),
        _make_response("Second search", tool_calls=[tool_call_2]),
        _make_response("final result after two tool turns"),
    ]
    call_count = 0

    facade = StubMCPFacade(max_tool_call_turns=5)
    registry = StubMCPRegistry(facade)

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> ChatCompletionResponse:
        nonlocal call_count
        call_count += 1
        return responses.pop(0)

    model = ModelFacade(
        model_config=stub_model_configs[0],
        model_provider_registry=stub_model_provider_registry,
        client=stub_model_client,
        mcp_registry=registry,
    )

    with patch.object(ModelFacade, "completion", new=_completion):
        result, trace = model.generate(prompt="question", parser=lambda x: x, tool_alias="tools")

    assert result == "final result after two tool turns"
    assert call_count == 3  # 2 tool turns + 1 final


def test_generate_with_tools_tracks_usage_stats(
    stub_model_configs: Any,
    stub_model_client: MagicMock,
    stub_model_provider_registry: Any,
) -> None:
    """Tool usage stats are properly tracked with generations_with_tools incremented."""
    tool_call_1 = ToolCall(id="call-1", name="lookup", arguments_json='{"query": "foo"}')
    tool_call_2 = ToolCall(id="call-2", name="search", arguments_json='{"term": "bar"}')

    responses = [
        _make_response("First lookup", tool_calls=[tool_call_1]),
        _make_response("Second search", tool_calls=[tool_call_2]),
        _make_response("final result"),
    ]

    facade = StubMCPFacade(max_tool_call_turns=5)
    registry = StubMCPRegistry(facade)

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> ChatCompletionResponse:
        return responses.pop(0)

    model = ModelFacade(
        model_config=stub_model_configs[0],
        model_provider_registry=stub_model_provider_registry,
        client=stub_model_client,
        mcp_registry=registry,
    )

    # Verify initial state
    assert model.usage_stats.tool_usage.total_tool_calls == 0
    assert model.usage_stats.tool_usage.total_tool_call_turns == 0
    assert model.usage_stats.tool_usage.total_generations == 0
    assert model.usage_stats.tool_usage.generations_with_tools == 0

    with patch.object(ModelFacade, "completion", new=_completion):
        result, _ = model.generate(prompt="question", parser=lambda x: x, tool_alias="tools")

    assert result == "final result"

    # Verify tool usage stats are tracked correctly
    assert model.usage_stats.tool_usage.total_tool_calls == 2
    assert model.usage_stats.tool_usage.total_tool_call_turns == 2
    assert model.usage_stats.tool_usage.total_generations == 1
    assert model.usage_stats.tool_usage.generations_with_tools == 1


def test_generate_with_tools_tracks_multiple_generations(
    stub_model_configs: Any,
    stub_model_client: MagicMock,
    stub_model_provider_registry: Any,
) -> None:
    """Tool usage is correctly tracked across multiple generations."""
    facade = StubMCPFacade(max_tool_call_turns=10)
    registry = StubMCPRegistry(facade)

    model = ModelFacade(
        model_config=stub_model_configs[0],
        model_provider_registry=stub_model_provider_registry,
        client=stub_model_client,
        mcp_registry=registry,
    )

    # Generation 1: 2 tool calls across 1 turn
    tool_call_a = ToolCall(id="call-a", name="lookup", arguments_json='{"q": "1"}')
    tool_call_b = ToolCall(id="call-b", name="lookup", arguments_json='{"q": "2"}')
    responses_gen1 = [
        _make_response("", tool_calls=[tool_call_a, tool_call_b]),
        _make_response("result 1"),
    ]

    def _completion_gen1(self: Any, messages: list[ChatMessage], **kwargs: Any) -> ChatCompletionResponse:
        return responses_gen1.pop(0)

    with patch.object(ModelFacade, "completion", new=_completion_gen1):
        model.generate(prompt="q1", parser=lambda x: x, tool_alias="tools")

    # Generation 2: 4 tool calls across 2 turns
    tool_call_c = ToolCall(id="call-c", name="search", arguments_json='{"q": "3"}')
    tool_call_d = ToolCall(id="call-d", name="search", arguments_json='{"q": "4"}')
    responses_gen2 = [
        _make_response("", tool_calls=[tool_call_a, tool_call_b]),
        _make_response("", tool_calls=[tool_call_c, tool_call_d]),
        _make_response("result 2"),
    ]

    def _completion_gen2(self: Any, messages: list[ChatMessage], **kwargs: Any) -> ChatCompletionResponse:
        return responses_gen2.pop(0)

    with patch.object(ModelFacade, "completion", new=_completion_gen2):
        model.generate(prompt="q2", parser=lambda x: x, tool_alias="tools")

    # Generation 3: No tool calls
    responses_gen3 = [
        _make_response("result 3"),
    ]

    def _completion_gen3(self: Any, messages: list[ChatMessage], **kwargs: Any) -> ChatCompletionResponse:
        return responses_gen3.pop(0)

    with patch.object(ModelFacade, "completion", new=_completion_gen3):
        model.generate(prompt="q3", parser=lambda x: x, tool_alias="tools")

    # Verify totals: 2 + 4 + 0 = 6 calls, 1 + 2 + 0 = 3 turns, 3 total generations, 2 with tools
    assert model.usage_stats.tool_usage.total_tool_calls == 6
    assert model.usage_stats.tool_usage.total_tool_call_turns == 3
    assert model.usage_stats.tool_usage.total_generations == 3
    assert model.usage_stats.tool_usage.generations_with_tools == 2


def test_generate_tool_turn_limit_triggers_refusal(
    stub_model_configs: Any,
    stub_model_client: MagicMock,
    stub_model_provider_registry: Any,
) -> None:
    """When max_tool_call_turns exceeded, refusal is used."""
    tool_call = ToolCall(id="call-1", name="lookup", arguments_json="{}")

    responses = [
        _make_response("", tool_calls=[tool_call]),  # Turn 1
        _make_response("", tool_calls=[tool_call]),  # Turn 2 (max)
        _make_response("", tool_calls=[tool_call]),  # Turn 3 (exceeds, should refuse)
        _make_response("final answer after refusal"),
    ]
    process_calls = 0
    refuse_calls = 0

    tc_dict = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}

    def custom_process_fn(completion_response: ChatCompletionResponse) -> list[ChatMessage]:
        nonlocal process_calls
        process_calls += 1
        return [
            ChatMessage.as_assistant(content="", tool_calls=[tc_dict]),
            ChatMessage.as_tool(content="tool-result", tool_call_id="call-1"),
        ]

    def custom_refuse_fn(completion_response: ChatCompletionResponse) -> list[ChatMessage]:
        nonlocal refuse_calls
        refuse_calls += 1
        return [
            ChatMessage.as_assistant(content="", tool_calls=[tc_dict]),
            ChatMessage.as_tool(content="REFUSED: Budget exceeded", tool_call_id="call-1"),
        ]

    facade = StubMCPFacade(max_tool_call_turns=2, process_fn=custom_process_fn, refuse_fn=custom_refuse_fn)
    registry = StubMCPRegistry(facade)

    response_idx = 0

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> ChatCompletionResponse:
        nonlocal response_idx
        resp = responses[response_idx]
        response_idx += 1
        return resp

    model = ModelFacade(
        model_config=stub_model_configs[0],
        model_provider_registry=stub_model_provider_registry,
        client=stub_model_client,
        mcp_registry=registry,
    )

    with patch.object(ModelFacade, "completion", new=_completion):
        result, _ = model.generate(prompt="question", parser=lambda x: x, tool_alias="tools")

    assert result == "final answer after refusal"
    assert process_calls == 2  # Turns 1 and 2
    assert refuse_calls == 1  # Turn 3 was refused


def test_generate_tool_turn_limit_model_responds_after_refusal(
    stub_model_configs: Any,
    stub_model_client: MagicMock,
    stub_model_provider_registry: Any,
) -> None:
    """Model provides final answer after refusal message."""
    tool_call = ToolCall(id="call-1", name="lookup", arguments_json="{}")
    tc_dict = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}

    responses = [
        _make_response("", tool_calls=[tool_call]),  # Exceeds on first turn
        _make_response("I understand, here is the answer without tools"),
    ]

    def custom_refuse_fn(completion_response: ChatCompletionResponse) -> list[ChatMessage]:
        return [
            ChatMessage.as_assistant(content="", tool_calls=[tc_dict]),
            ChatMessage.as_tool(
                content="Tool call refused: You have reached the maximum number of tool-calling turns.",
                tool_call_id="call-1",
            ),
        ]

    facade = StubMCPFacade(
        max_tool_call_turns=0,
        process_fn=lambda _: [],  # Should not be called
        refuse_fn=custom_refuse_fn,
    )
    registry = StubMCPRegistry(facade)

    response_idx = 0

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> ChatCompletionResponse:
        nonlocal response_idx
        resp = responses[response_idx]
        response_idx += 1
        return resp

    model = ModelFacade(
        model_config=stub_model_configs[0],
        model_provider_registry=stub_model_provider_registry,
        client=stub_model_client,
        mcp_registry=registry,
    )

    with patch.object(ModelFacade, "completion", new=_completion):
        result, trace = model.generate(prompt="question", parser=lambda x: x, tool_alias="tools")

    assert result == "I understand, here is the answer without tools"
    # Trace should include refusal message
    assert any(msg.content and "refused" in msg.content.lower() for msg in trace if msg.role == "tool")


def test_generate_tool_alias_not_in_registry(
    stub_model_configs: Any,
    stub_model_client: MagicMock,
    stub_model_provider_registry: Any,
) -> None:
    """Raises error when tool_alias not found in MCPRegistry."""

    class _StubMCPRegistry:
        def get_mcp(self, *, tool_alias: str) -> Any:
            raise ValueError(f"No tool config with alias {tool_alias!r} found!")

    model = ModelFacade(
        model_config=stub_model_configs[0],
        model_provider_registry=stub_model_provider_registry,
        client=stub_model_client,
        mcp_registry=_StubMCPRegistry(),
    )

    with pytest.raises(MCPConfigurationError, match="not registered"):
        model.generate(prompt="question", parser=lambda x: x, tool_alias="nonexistent")


def test_generate_no_tool_alias_ignores_mcp(
    stub_model_configs: Any,
    stub_model_client: MagicMock,
    stub_model_provider_registry: Any,
) -> None:
    """When tool_alias is None, no MCP operations occur."""
    get_mcp_called = False

    class _StubMCPRegistry:
        def get_mcp(self, *, tool_alias: str) -> Any:
            nonlocal get_mcp_called
            get_mcp_called = True
            raise RuntimeError("Should not be called")

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> ChatCompletionResponse:
        assert "tools" not in kwargs  # No tools should be passed
        return _make_response("response without tools")

    model = ModelFacade(
        model_config=stub_model_configs[0],
        model_provider_registry=stub_model_provider_registry,
        client=stub_model_client,
        mcp_registry=_StubMCPRegistry(),
    )

    with patch.object(ModelFacade, "completion", new=_completion):
        result, _ = model.generate(prompt="question", parser=lambda x: x, tool_alias=None)

    assert result == "response without tools"
    assert get_mcp_called is False


def test_generate_tool_calls_with_parser_corrections(
    stub_model_configs: Any,
    stub_model_client: MagicMock,
    stub_model_provider_registry: Any,
) -> None:
    """Tool calling works correctly with parser correction steps."""
    tool_call = ToolCall(id="call-1", name="lookup", arguments_json="{}")
    parse_count = 0

    responses = [
        _make_response("", tool_calls=[tool_call]),  # Tool call
        _make_response("bad format"),  # Parser will fail
        _make_response("correct format"),  # Parser will succeed
    ]

    facade = StubMCPFacade()
    registry = StubMCPRegistry(facade)

    response_idx = 0

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> ChatCompletionResponse:
        nonlocal response_idx
        resp = responses[response_idx]
        response_idx += 1
        return resp

    def _parser(text: str) -> str:
        nonlocal parse_count
        parse_count += 1
        if text == "bad format":
            raise ParserException("Invalid format")
        return text

    model = ModelFacade(
        model_config=stub_model_configs[0],
        model_provider_registry=stub_model_provider_registry,
        client=stub_model_client,
        mcp_registry=registry,
    )

    with patch.object(ModelFacade, "completion", new=_completion):
        result, _ = model.generate(prompt="question", parser=_parser, tool_alias="tools", max_correction_steps=1)

    assert result == "correct format"
    assert parse_count == 2  # Failed once, then succeeded


def test_generate_tool_calls_with_conversation_restarts(
    stub_model_configs: Any,
    stub_model_client: MagicMock,
    stub_model_provider_registry: Any,
) -> None:
    """Tool calling works correctly with conversation restarts."""
    tool_call = ToolCall(id="call-1", name="lookup", arguments_json="{}")
    messages_at_call: list[int] = []

    responses = [
        _make_response("", tool_calls=[tool_call]),
        _make_response("still bad"),  # Fails parser, triggers restart
        _make_response("", tool_calls=[tool_call]),  # After restart
        _make_response("good result"),
    ]

    facade = StubMCPFacade()
    registry = StubMCPRegistry(facade)

    response_idx = 0

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> ChatCompletionResponse:
        nonlocal response_idx
        messages_at_call.append(len(messages))
        resp = responses[response_idx]
        response_idx += 1
        return resp

    def _parser(text: str) -> str:
        if text == "still bad":
            raise ParserException("Bad format")
        return text

    model = ModelFacade(
        model_config=stub_model_configs[0],
        model_provider_registry=stub_model_provider_registry,
        client=stub_model_client,
        mcp_registry=registry,
    )

    with patch.object(ModelFacade, "completion", new=_completion):
        result, _ = model.generate(
            prompt="question", parser=_parser, tool_alias="tools", max_correction_steps=0, max_conversation_restarts=1
        )

    assert result == "good result"
    # After restart, message count should preserve tool call history (restart from checkpoint)
    assert messages_at_call[2] == messages_at_call[1]


# =============================================================================
# Message trace tests
# =============================================================================


def test_generate_trace_includes_tool_calls(
    stub_model_configs: Any,
    stub_model_client: MagicMock,
    stub_model_provider_registry: Any,
) -> None:
    """Returned trace includes tool call messages."""
    tool_call = ToolCall(id="call-1", name="lookup", arguments_json='{"q": "test"}')

    responses = [
        _make_response("Let me look that up", tool_calls=[tool_call]),
        _make_response("Here is the answer"),
    ]

    facade = StubMCPFacade()
    registry = StubMCPRegistry(facade)

    response_idx = 0

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> ChatCompletionResponse:
        nonlocal response_idx
        resp = responses[response_idx]
        response_idx += 1
        return resp

    model = ModelFacade(
        model_config=stub_model_configs[0],
        model_provider_registry=stub_model_provider_registry,
        client=stub_model_client,
        mcp_registry=registry,
    )

    with patch.object(ModelFacade, "completion", new=_completion):
        _, trace = model.generate(prompt="question", parser=lambda x: x, tool_alias="tools")

    # Find assistant message with tool_calls
    assistant_with_tools = [msg for msg in trace if msg.role == "assistant" and msg.tool_calls]
    assert len(assistant_with_tools) >= 1
    assert assistant_with_tools[0].tool_calls[0]["function"]["name"] == "lookup"


def test_generate_trace_includes_tool_responses(
    stub_model_configs: Any,
    stub_model_client: MagicMock,
    stub_model_provider_registry: Any,
) -> None:
    """Returned trace includes tool response messages."""
    tool_call = ToolCall(id="call-1", name="lookup", arguments_json="{}")
    tc_dict = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}

    responses = [
        _make_response("", tool_calls=[tool_call]),
        _make_response("final"),
    ]

    def custom_process_fn(completion_response: ChatCompletionResponse) -> list[ChatMessage]:
        return [
            ChatMessage.as_assistant(content="", tool_calls=[tc_dict]),
            ChatMessage.as_tool(content="THE TOOL RESPONSE CONTENT", tool_call_id="call-1"),
        ]

    facade = StubMCPFacade(process_fn=custom_process_fn)
    registry = StubMCPRegistry(facade)

    response_idx = 0

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> ChatCompletionResponse:
        nonlocal response_idx
        resp = responses[response_idx]
        response_idx += 1
        return resp

    model = ModelFacade(
        model_config=stub_model_configs[0],
        model_provider_registry=stub_model_provider_registry,
        client=stub_model_client,
        mcp_registry=registry,
    )

    with patch.object(ModelFacade, "completion", new=_completion):
        _, trace = model.generate(prompt="question", parser=lambda x: x, tool_alias="tools")

    tool_messages = [msg for msg in trace if msg.role == "tool"]
    assert len(tool_messages) >= 1
    assert tool_messages[0].content == "THE TOOL RESPONSE CONTENT"
    assert tool_messages[0].tool_call_id == "call-1"


def test_generate_trace_includes_refusal_messages(
    stub_model_configs: Any,
    stub_model_client: MagicMock,
    stub_model_provider_registry: Any,
) -> None:
    """Returned trace includes refusal messages when budget exhausted."""
    tool_call = ToolCall(id="call-1", name="lookup", arguments_json="{}")
    tc_dict = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}

    responses = [
        _make_response("", tool_calls=[tool_call]),  # Will be refused (max_turns=0)
        _make_response("answer without tools"),
    ]

    def custom_refuse_fn(completion_response: ChatCompletionResponse) -> list[ChatMessage]:
        return [
            ChatMessage.as_assistant(content="", tool_calls=[tc_dict]),
            ChatMessage.as_tool(content="BUDGET_EXCEEDED_REFUSAL", tool_call_id="call-1"),
        ]

    facade = StubMCPFacade(
        max_tool_call_turns=0,
        process_fn=lambda _: [],
        refuse_fn=custom_refuse_fn,
    )
    registry = StubMCPRegistry(facade)

    response_idx = 0

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> ChatCompletionResponse:
        nonlocal response_idx
        resp = responses[response_idx]
        response_idx += 1
        return resp

    model = ModelFacade(
        model_config=stub_model_configs[0],
        model_provider_registry=stub_model_provider_registry,
        client=stub_model_client,
        mcp_registry=registry,
    )

    with patch.object(ModelFacade, "completion", new=_completion):
        _, trace = model.generate(prompt="question", parser=lambda x: x, tool_alias="tools")

    # Check for refusal message in trace
    tool_messages = [msg for msg in trace if msg.role == "tool"]
    assert any("BUDGET_EXCEEDED_REFUSAL" in msg.content for msg in tool_messages)


def test_generate_trace_preserves_reasoning_content(
    stub_model_configs: Any,
    stub_model_client: MagicMock,
    stub_model_provider_registry: Any,
) -> None:
    """Trace messages preserve reasoning_content field."""
    response = _make_response(
        "The answer is 42",
        reasoning_content="Let me think about this carefully...",
    )

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> ChatCompletionResponse:
        return response

    model = ModelFacade(
        model_config=stub_model_configs[0],
        model_provider_registry=stub_model_provider_registry,
        client=stub_model_client,
    )

    with patch.object(ModelFacade, "completion", new=_completion):
        _, trace = model.generate(prompt="question", parser=lambda x: x)

    # Find assistant message and check reasoning content
    assistant_messages = [msg for msg in trace if msg.role == "assistant"]
    assert len(assistant_messages) >= 1
    assert assistant_messages[-1].reasoning_content == "Let me think about this carefully..."


# =============================================================================
# Error handling tests
# =============================================================================


def test_generate_tool_execution_error(
    stub_model_configs: Any,
    stub_model_client: MagicMock,
    stub_model_provider_registry: Any,
) -> None:
    """Handles MCP tool execution errors appropriately."""
    tool_call = ToolCall(id="call-1", name="lookup", arguments_json="{}")

    responses = [_make_response("", tool_calls=[tool_call])]

    def error_process_fn(completion_response: ChatCompletionResponse) -> list[ChatMessage]:
        raise MCPToolError("Tool execution failed: Connection refused")

    facade = StubMCPFacade(process_fn=error_process_fn)
    registry = StubMCPRegistry(facade)

    response_idx = 0

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> ChatCompletionResponse:
        nonlocal response_idx
        resp = responses[response_idx]
        response_idx += 1
        return resp

    model = ModelFacade(
        model_config=stub_model_configs[0],
        model_provider_registry=stub_model_provider_registry,
        client=stub_model_client,
        mcp_registry=registry,
    )

    with patch.object(ModelFacade, "completion", new=_completion):
        with pytest.raises(MCPToolError, match="Connection refused"):
            model.generate(prompt="question", parser=lambda x: x, tool_alias="tools")


def test_generate_tool_invalid_arguments(
    stub_model_configs: Any,
    stub_model_client: MagicMock,
    stub_model_provider_registry: Any,
) -> None:
    """Handles invalid tool arguments from LLM."""
    tool_call = ToolCall(id="call-1", name="lookup", arguments_json="not valid json")

    responses = [_make_response("", tool_calls=[tool_call])]

    def error_process_fn(completion_response: ChatCompletionResponse) -> list[ChatMessage]:
        raise MCPToolError("Invalid tool arguments for 'lookup': not valid json")

    facade = StubMCPFacade(process_fn=error_process_fn)
    registry = StubMCPRegistry(facade)

    response_idx = 0

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> ChatCompletionResponse:
        nonlocal response_idx
        resp = responses[response_idx]
        response_idx += 1
        return resp

    model = ModelFacade(
        model_config=stub_model_configs[0],
        model_provider_registry=stub_model_provider_registry,
        client=stub_model_client,
        mcp_registry=registry,
    )

    with patch.object(ModelFacade, "completion", new=_completion):
        with pytest.raises(MCPToolError, match="Invalid tool arguments"):
            model.generate(prompt="question", parser=lambda x: x, tool_alias="tools")


# =============================================================================
# Image generation tests
# =============================================================================


def test_generate_image_diffusion_tracks_image_usage(
    stub_model_facade: ModelFacade,
    stub_model_client: MagicMock,
) -> None:
    """Test that generate_image tracks image usage for diffusion models."""
    stub_model_client.generate_image.return_value = ImageGenerationResponse(
        images=[
            ImagePayload(b64_data="image1_base64"),
            ImagePayload(b64_data="image2_base64"),
            ImagePayload(b64_data="image3_base64"),
        ]
    )

    assert stub_model_facade.usage_stats.image_usage.total_images == 0

    with patch("data_designer.engine.models.facade.is_image_diffusion_model", return_value=True):
        images = stub_model_facade.generate_image(prompt="test prompt", extra_body={"n": 3})

    assert len(images) == 3
    assert images == ["image1_base64", "image2_base64", "image3_base64"]
    assert stub_model_facade.usage_stats.image_usage.total_images == 3
    assert stub_model_facade.usage_stats.image_usage.has_usage is True


def test_generate_image_chat_completion_tracks_image_usage(
    stub_model_facade: ModelFacade,
    stub_model_client: MagicMock,
) -> None:
    """Test that generate_image tracks image usage for chat completion models."""
    stub_model_client.generate_image.return_value = ImageGenerationResponse(
        images=[
            ImagePayload(b64_data="image1"),
            ImagePayload(b64_data="image2"),
        ]
    )

    assert stub_model_facade.usage_stats.image_usage.total_images == 0

    with patch("data_designer.engine.models.facade.is_image_diffusion_model", return_value=False):
        images = stub_model_facade.generate_image(prompt="test prompt")

    assert len(images) == 2
    assert images == ["image1", "image2"]
    assert stub_model_facade.usage_stats.image_usage.total_images == 2
    assert stub_model_facade.usage_stats.image_usage.has_usage is True


def test_generate_image_skip_usage_tracking(
    stub_model_facade: ModelFacade,
    stub_model_client: MagicMock,
) -> None:
    """Test that generate_image respects skip_usage_tracking flag."""
    stub_model_client.generate_image.return_value = ImageGenerationResponse(
        images=[
            ImagePayload(b64_data="image1_base64"),
            ImagePayload(b64_data="image2_base64"),
        ]
    )

    assert stub_model_facade.usage_stats.image_usage.total_images == 0

    with patch("data_designer.engine.models.facade.is_image_diffusion_model", return_value=True):
        images = stub_model_facade.generate_image(prompt="test prompt", skip_usage_tracking=True)

    assert len(images) == 2
    assert stub_model_facade.usage_stats.image_usage.total_images == 0
    assert stub_model_facade.usage_stats.image_usage.has_usage is False


def test_generate_image_no_image_data(
    stub_model_facade: ModelFacade,
    stub_model_client: MagicMock,
) -> None:
    """Test that generate_image raises ImageGenerationError when no image data in response."""
    stub_model_client.generate_image.return_value = ImageGenerationResponse(images=[])

    with patch("data_designer.engine.models.facade.is_image_diffusion_model", return_value=False):
        with pytest.raises(ImageGenerationError, match="No image data found"):
            stub_model_facade.generate_image(prompt="test prompt")

    assert stub_model_facade.usage_stats.request_usage.failed_requests == 1
    assert stub_model_facade.usage_stats.request_usage.successful_requests == 0


def test_generate_image_accumulates_usage(
    stub_model_facade: ModelFacade,
    stub_model_client: MagicMock,
) -> None:
    """Test that generate_image accumulates image usage across multiple calls."""
    response1 = ImageGenerationResponse(images=[ImagePayload(b64_data="image1"), ImagePayload(b64_data="image2")])
    response2 = ImageGenerationResponse(
        images=[ImagePayload(b64_data="image3"), ImagePayload(b64_data="image4"), ImagePayload(b64_data="image5")]
    )
    stub_model_client.generate_image.side_effect = [response1, response2]

    assert stub_model_facade.usage_stats.image_usage.total_images == 0

    with patch("data_designer.engine.models.facade.is_image_diffusion_model", return_value=True):
        images1 = stub_model_facade.generate_image(prompt="test1")
        assert len(images1) == 2
        assert stub_model_facade.usage_stats.image_usage.total_images == 2

        images2 = stub_model_facade.generate_image(prompt="test2")
        assert len(images2) == 3
        assert stub_model_facade.usage_stats.image_usage.total_images == 5


# =============================================================================
# Async behavior tests
# =============================================================================


@pytest.mark.parametrize(
    "skip_usage_tracking",
    [
        False,
        True,
    ],
)
@pytest.mark.asyncio
async def test_acompletion_success(
    stub_completion_messages: list[ChatMessage],
    stub_model_configs: Any,
    stub_model_facade: ModelFacade,
    stub_model_client: MagicMock,
    skip_usage_tracking: bool,
) -> None:
    expected_response = _make_response("Test response")
    stub_model_client.acompletion = AsyncMock(return_value=expected_response)
    result = await stub_model_facade.acompletion(stub_completion_messages, skip_usage_tracking=skip_usage_tracking)
    assert result == expected_response
    assert stub_model_client.acompletion.call_count == 1


@pytest.mark.asyncio
async def test_acompletion_with_exception(
    stub_completion_messages: list[ChatMessage],
    stub_model_facade: ModelFacade,
    stub_model_client: MagicMock,
) -> None:
    stub_model_client.acompletion = AsyncMock(side_effect=Exception("Router error"))

    with pytest.raises(Exception, match="Router error"):
        await stub_model_facade.acompletion(stub_completion_messages)


@pytest.mark.asyncio
async def test_agenerate_text_embeddings_success(
    stub_model_facade: ModelFacade,
    stub_model_client: MagicMock,
) -> None:
    expected_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    stub_model_client.aembeddings = AsyncMock(return_value=EmbeddingResponse(vectors=expected_vectors))
    input_texts = ["test1", "test2"]
    result = await stub_model_facade.agenerate_text_embeddings(input_texts)
    assert result == expected_vectors


@pytest.mark.parametrize(
    "max_correction_steps,max_conversation_restarts,total_calls",
    [
        (0, 0, 1),
        (1, 1, 4),
        (1, 2, 6),
        (5, 0, 6),
        (0, 5, 6),
        (3, 3, 16),
    ],
)
@patch.object(ModelFacade, "acompletion", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_agenerate_correction_retries(
    mock_acompletion: AsyncMock,
    stub_model_facade: ModelFacade,
    max_correction_steps: int,
    max_conversation_restarts: int,
    total_calls: int,
) -> None:
    bad_response = _make_response("bad response")
    mock_acompletion.return_value = bad_response

    def _failing_parser(response: str) -> str:
        raise ParserException("parser exception")

    with pytest.raises(ModelGenerationValidationFailureError):
        await stub_model_facade.agenerate(
            prompt="foo",
            system_prompt="bar",
            parser=_failing_parser,
            max_correction_steps=max_correction_steps,
            max_conversation_restarts=max_conversation_restarts,
        )
    assert mock_acompletion.call_count == total_calls

    with pytest.raises(ModelGenerationValidationFailureError):
        await stub_model_facade.agenerate(
            prompt="foo",
            parser=_failing_parser,
            system_prompt="bar",
            max_correction_steps=max_correction_steps,
            max_conversation_restarts=max_conversation_restarts,
        )
    assert mock_acompletion.call_count == 2 * total_calls


@patch.object(ModelFacade, "acompletion", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_agenerate_success(
    mock_acompletion: AsyncMock,
    stub_model_facade: ModelFacade,
) -> None:
    good_response = _make_response("parsed output")
    mock_acompletion.return_value = good_response

    result, trace = await stub_model_facade.agenerate(prompt="test", parser=lambda x: x)
    assert result == "parsed output"
    assert mock_acompletion.call_count == 1
    assert any(msg.role == "user" for msg in trace)
    assert any(msg.role == "assistant" and msg.content == "parsed output" for msg in trace)


# =============================================================================
# Async image generation tests
# =============================================================================


@pytest.mark.asyncio
async def test_agenerate_image_diffusion_success(
    stub_model_facade: ModelFacade,
    stub_model_client: MagicMock,
) -> None:
    """Test async image generation via diffusion API."""
    stub_model_client.agenerate_image = AsyncMock(
        return_value=ImageGenerationResponse(
            images=[ImagePayload(b64_data="image1_base64"), ImagePayload(b64_data="image2_base64")]
        )
    )

    with patch("data_designer.engine.models.facade.is_image_diffusion_model", return_value=True):
        images = await stub_model_facade.agenerate_image(prompt="test prompt")

    assert len(images) == 2
    assert images == ["image1_base64", "image2_base64"]
    assert stub_model_facade.usage_stats.image_usage.total_images == 2


@pytest.mark.asyncio
async def test_agenerate_image_chat_completion_success(
    stub_model_facade: ModelFacade,
    stub_model_client: MagicMock,
) -> None:
    """Test async image generation via chat completion API."""
    stub_model_client.agenerate_image = AsyncMock(
        return_value=ImageGenerationResponse(images=[ImagePayload(b64_data="image1")])
    )

    with patch("data_designer.engine.models.facade.is_image_diffusion_model", return_value=False):
        images = await stub_model_facade.agenerate_image(prompt="test prompt")

    assert len(images) == 1
    assert images == ["image1"]
    assert stub_model_facade.usage_stats.image_usage.total_images == 1


@pytest.mark.asyncio
async def test_agenerate_image_no_data(
    stub_model_facade: ModelFacade,
    stub_model_client: MagicMock,
) -> None:
    """Test async image generation raises error when no data."""
    stub_model_client.agenerate_image = AsyncMock(return_value=ImageGenerationResponse(images=[]))

    with patch("data_designer.engine.models.facade.is_image_diffusion_model", return_value=True):
        with pytest.raises(ImageGenerationError, match="No image data found"):
            await stub_model_facade.agenerate_image(prompt="test prompt")
