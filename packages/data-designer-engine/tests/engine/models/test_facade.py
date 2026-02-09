# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import patch

import pytest
from litellm.types.utils import Choices, EmbeddingResponse, Message, ModelResponse

from data_designer.engine.mcp.errors import MCPConfigurationError, MCPToolError
from data_designer.engine.models.errors import ModelGenerationValidationFailureError
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.models.parsers.errors import ParserException
from data_designer.engine.models.utils import ChatMessage
from data_designer.engine.testing import StubMCPFacade, StubMCPRegistry, StubMessage, StubResponse


def mock_oai_response_object(response_text: str) -> StubResponse:
    return StubResponse(StubMessage(content=response_text))


@pytest.fixture
def stub_model_facade(stub_model_configs, stub_secrets_resolver, stub_model_provider_registry):
    return ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
    )


@pytest.fixture
def stub_completion_messages() -> list[ChatMessage]:
    return [ChatMessage.as_user("test")]


@pytest.fixture
def stub_expected_completion_response():
    return ModelResponse(choices=Choices(message=Message(content="Test response")))


@pytest.fixture
def stub_expected_embedding_response():
    return EmbeddingResponse(data=[{"embedding": [0.1, 0.2, 0.3]}] * 2)


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
@patch("data_designer.engine.models.facade.ModelFacade.completion", autospec=True)
def test_generate(
    mock_completion,
    stub_model_facade,
    max_correction_steps,
    max_conversation_restarts,
    total_calls,
):
    bad_response = mock_oai_response_object("bad response")
    mock_completion.side_effect = lambda *args, **kwargs: bad_response

    def _failing_parser(response: str):
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
@patch("data_designer.engine.models.facade.ModelFacade.completion", autospec=True)
def test_generate_with_system_prompt(
    mock_completion: Any,
    stub_model_facade: ModelFacade,
    system_prompt: str,
    expected_messages: list[ChatMessage],
) -> None:
    # Capture messages at call time since they get mutated after the call
    captured_messages = []

    def capture_and_return(*args: Any, **kwargs: Any) -> ModelResponse:
        captured_messages.append(list(args[1]))  # Copy the messages list
        return ModelResponse(choices=Choices(message=Message(content="Hello!")))

    mock_completion.side_effect = capture_and_return

    stub_model_facade.generate(prompt="does not matter", system_prompt=system_prompt, parser=lambda x: x)
    assert mock_completion.call_count == 1
    assert captured_messages[0] == expected_messages


def test_model_alias_property(stub_model_facade, stub_model_configs):
    assert stub_model_facade.model_alias == stub_model_configs[0].alias


def test_usage_stats_property(stub_model_facade):
    assert stub_model_facade.usage_stats is not None
    assert hasattr(stub_model_facade.usage_stats, "model_dump")


def test_consolidate_kwargs(stub_model_configs, stub_model_facade):
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

    # Provider extra_headers
    stub_model_facade.model_provider.extra_body = None
    stub_model_facade.model_provider.extra_headers = {"hello": "world", "hola": "mundo"}
    result = stub_model_facade.consolidate_kwargs()
    assert result == {
        **stub_model_configs[0].inference_parameters.generate_kwargs,
        "extra_headers": {"hello": "world", "hola": "mundo"},
    }


@pytest.mark.parametrize(
    "skip_usage_tracking",
    [
        False,
        True,
    ],
)
@patch("data_designer.engine.models.facade.CustomRouter.completion", autospec=True)
def test_completion_success(
    mock_router_completion: Any,
    stub_completion_messages: list[ChatMessage],
    stub_model_configs: Any,
    stub_model_facade: ModelFacade,
    stub_expected_completion_response: ModelResponse,
    skip_usage_tracking: bool,
) -> None:
    mock_router_completion.side_effect = lambda self, model, messages, **kwargs: stub_expected_completion_response
    result = stub_model_facade.completion(stub_completion_messages, skip_usage_tracking=skip_usage_tracking)
    expected_messages = [message.to_dict() for message in stub_completion_messages]
    assert result == stub_expected_completion_response
    assert mock_router_completion.call_count == 1
    assert mock_router_completion.call_args[1] == {
        "model": "stub-model-text",
        "messages": expected_messages,
        **stub_model_configs[0].inference_parameters.generate_kwargs,
    }


@patch("data_designer.engine.models.facade.CustomRouter.completion", autospec=True)
def test_completion_with_exception(
    mock_router_completion: Any,
    stub_completion_messages: list[ChatMessage],
    stub_model_facade: ModelFacade,
) -> None:
    mock_router_completion.side_effect = Exception("Router error")

    with pytest.raises(Exception, match="Router error"):
        stub_model_facade.completion(stub_completion_messages)


@patch("data_designer.engine.models.facade.CustomRouter.completion", autospec=True)
def test_completion_with_kwargs(
    mock_router_completion: Any,
    stub_completion_messages: list[ChatMessage],
    stub_model_configs: Any,
    stub_model_facade: ModelFacade,
    stub_expected_completion_response: ModelResponse,
) -> None:
    captured_kwargs = {}

    def mock_completion(self: Any, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> ModelResponse:
        captured_kwargs.update(kwargs)
        return stub_expected_completion_response

    mock_router_completion.side_effect = mock_completion

    kwargs = {"temperature": 0.7, "max_tokens": 100}
    result = stub_model_facade.completion(stub_completion_messages, **kwargs)

    assert result == stub_expected_completion_response
    # completion kwargs overrides model config generate kwargs
    assert captured_kwargs == {**stub_model_configs[0].inference_parameters.generate_kwargs, **kwargs}


@patch("data_designer.engine.models.facade.CustomRouter.embedding", autospec=True)
def test_generate_text_embeddings_success(mock_router_embedding, stub_model_facade, stub_expected_embedding_response):
    mock_router_embedding.side_effect = lambda self, model, input, **kwargs: stub_expected_embedding_response
    input_texts = ["test1", "test2"]
    result = stub_model_facade.generate_text_embeddings(input_texts)
    assert result == [data["embedding"] for data in stub_expected_embedding_response.data]


@patch("data_designer.engine.models.facade.CustomRouter.embedding", autospec=True)
def test_generate_text_embeddings_with_exception(mock_router_embedding, stub_model_facade):
    mock_router_embedding.side_effect = Exception("Router error")

    with pytest.raises(Exception, match="Router error"):
        stub_model_facade.generate_text_embeddings(["test1", "test2"])


@patch("data_designer.engine.models.facade.CustomRouter.embedding", autospec=True)
def test_generate_text_embeddings_with_kwargs(
    mock_router_embedding, stub_model_configs, stub_model_facade, stub_expected_embedding_response
):
    captured_kwargs = {}

    def mock_embedding(self, model, input, **kwargs):
        captured_kwargs.update(kwargs)
        return stub_expected_embedding_response

    mock_router_embedding.side_effect = mock_embedding
    kwargs = {"temperature": 0.7, "max_tokens": 100, "input_type": "query"}
    _ = stub_model_facade.generate_text_embeddings(["test1", "test2"], **kwargs)
    assert captured_kwargs == {**stub_model_configs[0].inference_parameters.generate_kwargs, **kwargs}


def test_generate_with_mcp_tools(
    stub_model_configs: Any,
    stub_secrets_resolver: Any,
    stub_model_provider_registry: Any,
) -> None:
    tool_call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"query": "foo"}'},
    }
    responses = [
        StubResponse(StubMessage(content=None, tool_calls=[tool_call])),
        StubResponse(StubMessage(content="final result")),
    ]
    captured_calls: list[tuple[list[ChatMessage], dict[str, Any]]] = []
    registry_calls: list[tuple[str, str, dict[str, str], None]] = []

    def process_with_tracking(completion_response: Any) -> list[ChatMessage]:
        message = completion_response.choices[0].message
        if not message.tool_calls:
            return [ChatMessage.as_assistant(content=message.content or "")]
        registry_calls.append(("tools", "lookup", {"query": "foo"}, None))
        return [
            ChatMessage.as_assistant(content="", tool_calls=[tool_call]),
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

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> StubResponse:
        captured_calls.append((messages, kwargs))
        return responses.pop(0)

    model = ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
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
    stub_model_configs: Any, stub_secrets_resolver: Any, stub_model_provider_registry: Any
) -> None:
    model = ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
        mcp_registry=None,
    )

    with pytest.raises(MCPConfigurationError):
        model.generate(prompt="question", parser=lambda x: x, tool_alias="tools")


# =============================================================================
# Tool calling integration tests
# =============================================================================


def test_generate_with_tool_alias_multiple_turns(
    stub_model_configs: Any,
    stub_secrets_resolver: Any,
    stub_model_provider_registry: Any,
) -> None:
    """Multiple tool call turns before final response."""
    tool_call_1 = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": '{"query": "foo"}'}}
    tool_call_2 = {"id": "call-2", "type": "function", "function": {"name": "search", "arguments": '{"term": "bar"}'}}

    responses = [
        StubResponse(StubMessage(content="First lookup", tool_calls=[tool_call_1])),
        StubResponse(StubMessage(content="Second search", tool_calls=[tool_call_2])),
        StubResponse(StubMessage(content="final result after two tool turns")),
    ]
    call_count = 0

    facade = StubMCPFacade(max_tool_call_turns=5)
    registry = StubMCPRegistry(facade)

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> StubResponse:
        nonlocal call_count
        call_count += 1
        return responses.pop(0)

    model = ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
        mcp_registry=registry,
    )

    with patch.object(ModelFacade, "completion", new=_completion):
        result, trace = model.generate(prompt="question", parser=lambda x: x, tool_alias="tools")

    assert result == "final result after two tool turns"
    assert call_count == 3  # 2 tool turns + 1 final


def test_generate_with_tools_tracks_usage_stats(
    stub_model_configs: Any,
    stub_secrets_resolver: Any,
    stub_model_provider_registry: Any,
) -> None:
    """Tool usage stats are properly tracked with generations_with_tools incremented."""
    tool_call_1 = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": '{"query": "foo"}'}}
    tool_call_2 = {"id": "call-2", "type": "function", "function": {"name": "search", "arguments": '{"term": "bar"}'}}

    responses = [
        StubResponse(StubMessage(content="First lookup", tool_calls=[tool_call_1])),
        StubResponse(StubMessage(content="Second search", tool_calls=[tool_call_2])),
        StubResponse(StubMessage(content="final result")),
    ]

    facade = StubMCPFacade(max_tool_call_turns=5)
    registry = StubMCPRegistry(facade)

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> StubResponse:
        return responses.pop(0)

    model = ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
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
    assert model.usage_stats.tool_usage.total_tool_calls == 2  # 2 tool calls total
    assert model.usage_stats.tool_usage.total_tool_call_turns == 2  # 2 turns with tool calls
    assert model.usage_stats.tool_usage.total_generations == 1  # 1 generation
    assert model.usage_stats.tool_usage.generations_with_tools == 1  # 1 generation with tools


def test_generate_with_tools_tracks_multiple_generations(
    stub_model_configs: Any,
    stub_secrets_resolver: Any,
    stub_model_provider_registry: Any,
) -> None:
    """Tool usage is correctly tracked across multiple generations."""
    facade = StubMCPFacade(max_tool_call_turns=10)
    registry = StubMCPRegistry(facade)

    model = ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
        mcp_registry=registry,
    )

    # Generation 1: 2 tool calls across 1 turn
    tool_call_a = {"id": "call-a", "type": "function", "function": {"name": "lookup", "arguments": '{"q": "1"}'}}
    tool_call_b = {"id": "call-b", "type": "function", "function": {"name": "lookup", "arguments": '{"q": "2"}'}}
    responses_gen1 = [
        StubResponse(StubMessage(content="", tool_calls=[tool_call_a, tool_call_b])),
        StubResponse(StubMessage(content="result 1")),
    ]

    def _completion_gen1(self: Any, messages: list[ChatMessage], **kwargs: Any) -> StubResponse:
        return responses_gen1.pop(0)

    with patch.object(ModelFacade, "completion", new=_completion_gen1):
        model.generate(prompt="q1", parser=lambda x: x, tool_alias="tools")

    # Generation 2: 4 tool calls across 2 turns
    tool_call_c = {"id": "call-c", "type": "function", "function": {"name": "search", "arguments": '{"q": "3"}'}}
    tool_call_d = {"id": "call-d", "type": "function", "function": {"name": "search", "arguments": '{"q": "4"}'}}
    responses_gen2 = [
        StubResponse(StubMessage(content="", tool_calls=[tool_call_a, tool_call_b])),
        StubResponse(StubMessage(content="", tool_calls=[tool_call_c, tool_call_d])),
        StubResponse(StubMessage(content="result 2")),
    ]

    def _completion_gen2(self: Any, messages: list[ChatMessage], **kwargs: Any) -> StubResponse:
        return responses_gen2.pop(0)

    with patch.object(ModelFacade, "completion", new=_completion_gen2):
        model.generate(prompt="q2", parser=lambda x: x, tool_alias="tools")

    # Generation 3: No tool calls
    responses_gen3 = [
        StubResponse(StubMessage(content="result 3")),
    ]

    def _completion_gen3(self: Any, messages: list[ChatMessage], **kwargs: Any) -> StubResponse:
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
    stub_secrets_resolver: Any,
    stub_model_provider_registry: Any,
) -> None:
    """When max_tool_call_turns exceeded, refusal is used."""
    tool_call = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}

    # Keep returning tool calls to exceed the limit
    responses = [
        StubResponse(StubMessage(content="", tool_calls=[tool_call])),  # Turn 1
        StubResponse(StubMessage(content="", tool_calls=[tool_call])),  # Turn 2 (max)
        StubResponse(StubMessage(content="", tool_calls=[tool_call])),  # Turn 3 (exceeds, should refuse)
        StubResponse(StubMessage(content="final answer after refusal")),
    ]
    process_calls = 0
    refuse_calls = 0

    def custom_process_fn(completion_response: Any) -> list[ChatMessage]:
        nonlocal process_calls
        process_calls += 1
        message = completion_response.choices[0].message
        return [
            ChatMessage.as_assistant(content="", tool_calls=message.tool_calls or []),
            ChatMessage.as_tool(content="tool-result", tool_call_id="call-1"),
        ]

    def custom_refuse_fn(completion_response: Any) -> list[ChatMessage]:
        nonlocal refuse_calls
        refuse_calls += 1
        message = completion_response.choices[0].message
        return [
            ChatMessage.as_assistant(content="", tool_calls=message.tool_calls or []),
            ChatMessage.as_tool(content="REFUSED: Budget exceeded", tool_call_id="call-1"),
        ]

    facade = StubMCPFacade(max_tool_call_turns=2, process_fn=custom_process_fn, refuse_fn=custom_refuse_fn)
    registry = StubMCPRegistry(facade)

    response_idx = 0

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> StubResponse:
        nonlocal response_idx
        resp = responses[response_idx]
        response_idx += 1
        return resp

    model = ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
        mcp_registry=registry,
    )

    with patch.object(ModelFacade, "completion", new=_completion):
        result, _ = model.generate(prompt="question", parser=lambda x: x, tool_alias="tools")

    assert result == "final answer after refusal"
    assert process_calls == 2  # Turns 1 and 2
    assert refuse_calls == 1  # Turn 3 was refused


def test_generate_tool_turn_limit_model_responds_after_refusal(
    stub_model_configs: Any,
    stub_secrets_resolver: Any,
    stub_model_provider_registry: Any,
) -> None:
    """Model provides final answer after refusal message."""
    tool_call = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}

    responses = [
        StubResponse(StubMessage(content="", tool_calls=[tool_call])),  # Exceeds on first turn
        StubResponse(StubMessage(content="I understand, here is the answer without tools")),
    ]

    def custom_refuse_fn(completion_response: Any) -> list[ChatMessage]:
        return [
            ChatMessage.as_assistant(content="", tool_calls=[tool_call]),
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

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> StubResponse:
        nonlocal response_idx
        resp = responses[response_idx]
        response_idx += 1
        return resp

    model = ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
        mcp_registry=registry,
    )

    with patch.object(ModelFacade, "completion", new=_completion):
        result, trace = model.generate(prompt="question", parser=lambda x: x, tool_alias="tools")

    assert result == "I understand, here is the answer without tools"
    # Trace should include refusal message
    assert any(msg.content and "refused" in msg.content.lower() for msg in trace if msg.role == "tool")


def test_generate_tool_alias_not_in_registry(
    stub_model_configs: Any,
    stub_secrets_resolver: Any,
    stub_model_provider_registry: Any,
) -> None:
    """Raises error when tool_alias not found in MCPRegistry."""

    class StubMCPRegistry:
        def get_mcp(self, *, tool_alias: str) -> Any:
            raise ValueError(f"No tool config with alias {tool_alias!r} found!")

    model = ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
        mcp_registry=StubMCPRegistry(),
    )

    with pytest.raises(MCPConfigurationError, match="not registered"):
        model.generate(prompt="question", parser=lambda x: x, tool_alias="nonexistent")


def test_generate_no_tool_alias_ignores_mcp(
    stub_model_configs: Any,
    stub_secrets_resolver: Any,
    stub_model_provider_registry: Any,
) -> None:
    """When tool_alias is None, no MCP operations occur."""
    get_mcp_called = False

    class StubMCPRegistry:
        def get_mcp(self, *, tool_alias: str) -> Any:
            nonlocal get_mcp_called
            get_mcp_called = True
            raise RuntimeError("Should not be called")

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> StubResponse:
        assert "tools" not in kwargs  # No tools should be passed
        return StubResponse(StubMessage(content="response without tools"))

    model = ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
        mcp_registry=StubMCPRegistry(),
    )

    with patch.object(ModelFacade, "completion", new=_completion):
        result, _ = model.generate(prompt="question", parser=lambda x: x, tool_alias=None)

    assert result == "response without tools"
    assert get_mcp_called is False


def test_generate_tool_calls_with_parser_corrections(
    stub_model_configs: Any,
    stub_secrets_resolver: Any,
    stub_model_provider_registry: Any,
) -> None:
    """Tool calling works correctly with parser correction steps."""
    tool_call = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    parse_count = 0

    responses = [
        StubResponse(StubMessage(content="", tool_calls=[tool_call])),  # Tool call
        StubResponse(StubMessage(content="bad format")),  # Parser will fail
        StubResponse(StubMessage(content="correct format")),  # Parser will succeed
    ]

    facade = StubMCPFacade()
    registry = StubMCPRegistry(facade)

    response_idx = 0

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> StubResponse:
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
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
        mcp_registry=registry,
    )

    with patch.object(ModelFacade, "completion", new=_completion):
        result, _ = model.generate(prompt="question", parser=_parser, tool_alias="tools", max_correction_steps=1)

    assert result == "correct format"
    assert parse_count == 2  # Failed once, then succeeded


def test_generate_tool_calls_with_conversation_restarts(
    stub_model_configs: Any,
    stub_secrets_resolver: Any,
    stub_model_provider_registry: Any,
) -> None:
    """Tool calling works correctly with conversation restarts."""
    tool_call = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    messages_at_call: list[int] = []

    # First conversation: tool call + bad response
    # After restart: tool call + good response
    responses = [
        StubResponse(StubMessage(content="", tool_calls=[tool_call])),
        StubResponse(StubMessage(content="still bad")),  # Fails parser, triggers restart
        StubResponse(StubMessage(content="", tool_calls=[tool_call])),  # After restart
        StubResponse(StubMessage(content="good result")),
    ]

    facade = StubMCPFacade()
    registry = StubMCPRegistry(facade)

    response_idx = 0

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> StubResponse:
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
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
        mcp_registry=registry,
    )

    with patch.object(ModelFacade, "completion", new=_completion):
        result, _ = model.generate(
            prompt="question", parser=_parser, tool_alias="tools", max_correction_steps=0, max_conversation_restarts=1
        )

    assert result == "good result"
    # After restart, message count should preserve tool call history (restart from checkpoint)
    assert messages_at_call[2] == messages_at_call[1]  # Both should be post-tool-call message count


# =============================================================================
# Message trace tests
# =============================================================================


def test_generate_trace_includes_tool_calls(
    stub_model_configs: Any,
    stub_secrets_resolver: Any,
    stub_model_provider_registry: Any,
) -> None:
    """Returned trace includes tool call messages."""
    tool_call = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": '{"q": "test"}'}}

    responses = [
        StubResponse(StubMessage(content="Let me look that up", tool_calls=[tool_call])),
        StubResponse(StubMessage(content="Here is the answer")),
    ]

    facade = StubMCPFacade()
    registry = StubMCPRegistry(facade)

    response_idx = 0

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> StubResponse:
        nonlocal response_idx
        resp = responses[response_idx]
        response_idx += 1
        return resp

    model = ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
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
    stub_secrets_resolver: Any,
    stub_model_provider_registry: Any,
) -> None:
    """Returned trace includes tool response messages."""
    tool_call = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}

    responses = [
        StubResponse(StubMessage(content="", tool_calls=[tool_call])),
        StubResponse(StubMessage(content="final")),
    ]

    def custom_process_fn(completion_response: Any) -> list[ChatMessage]:
        return [
            ChatMessage.as_assistant(content="", tool_calls=[tool_call]),
            ChatMessage.as_tool(content="THE TOOL RESPONSE CONTENT", tool_call_id="call-1"),
        ]

    facade = StubMCPFacade(process_fn=custom_process_fn)
    registry = StubMCPRegistry(facade)

    response_idx = 0

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> StubResponse:
        nonlocal response_idx
        resp = responses[response_idx]
        response_idx += 1
        return resp

    model = ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
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
    stub_secrets_resolver: Any,
    stub_model_provider_registry: Any,
) -> None:
    """Returned trace includes refusal messages when budget exhausted."""
    tool_call = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}

    responses = [
        StubResponse(StubMessage(content="", tool_calls=[tool_call])),  # Will be refused (max_turns=0)
        StubResponse(StubMessage(content="answer without tools")),
    ]

    def custom_refuse_fn(completion_response: Any) -> list[ChatMessage]:
        return [
            ChatMessage.as_assistant(content="", tool_calls=[tool_call]),
            ChatMessage.as_tool(content="BUDGET_EXCEEDED_REFUSAL", tool_call_id="call-1"),
        ]

    facade = StubMCPFacade(
        max_tool_call_turns=0,
        process_fn=lambda _: [],
        refuse_fn=custom_refuse_fn,
    )
    registry = StubMCPRegistry(facade)

    response_idx = 0

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> StubResponse:
        nonlocal response_idx
        resp = responses[response_idx]
        response_idx += 1
        return resp

    model = ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
        mcp_registry=registry,
    )

    with patch.object(ModelFacade, "completion", new=_completion):
        _, trace = model.generate(prompt="question", parser=lambda x: x, tool_alias="tools")

    # Check for refusal message in trace
    tool_messages = [msg for msg in trace if msg.role == "tool"]
    assert any("BUDGET_EXCEEDED_REFUSAL" in msg.content for msg in tool_messages)


def test_generate_trace_preserves_reasoning_content(
    stub_model_configs: Any,
    stub_secrets_resolver: Any,
    stub_model_provider_registry: Any,
) -> None:
    """Trace messages preserve reasoning_content field."""
    response = StubResponse(
        StubMessage(
            content="The answer is 42",
            reasoning_content="Let me think about this carefully...",
        )
    )

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> StubResponse:
        return response

    model = ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
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
    stub_secrets_resolver: Any,
    stub_model_provider_registry: Any,
) -> None:
    """Handles MCP tool execution errors appropriately."""
    tool_call = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}

    responses = [StubResponse(StubMessage(content="", tool_calls=[tool_call]))]

    def error_process_fn(completion_response: Any) -> list[ChatMessage]:
        raise MCPToolError("Tool execution failed: Connection refused")

    facade = StubMCPFacade(process_fn=error_process_fn)
    registry = StubMCPRegistry(facade)

    response_idx = 0

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> StubResponse:
        nonlocal response_idx
        resp = responses[response_idx]
        response_idx += 1
        return resp

    model = ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
        mcp_registry=registry,
    )

    with patch.object(ModelFacade, "completion", new=_completion):
        with pytest.raises(MCPToolError, match="Connection refused"):
            model.generate(prompt="question", parser=lambda x: x, tool_alias="tools")


def test_generate_tool_invalid_arguments(
    stub_model_configs: Any,
    stub_secrets_resolver: Any,
    stub_model_provider_registry: Any,
) -> None:
    """Handles invalid tool arguments from LLM."""
    # Tool call with invalid JSON arguments
    tool_call = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "not valid json"}}

    responses = [StubResponse(StubMessage(content="", tool_calls=[tool_call]))]

    def error_process_fn(completion_response: Any) -> list[ChatMessage]:
        raise MCPToolError("Invalid tool arguments for 'lookup': not valid json")

    facade = StubMCPFacade(process_fn=error_process_fn)
    registry = StubMCPRegistry(facade)

    response_idx = 0

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> StubResponse:
        nonlocal response_idx
        resp = responses[response_idx]
        response_idx += 1
        return resp

    model = ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
        mcp_registry=registry,
    )

    with patch.object(ModelFacade, "completion", new=_completion):
        with pytest.raises(MCPToolError, match="Invalid tool arguments"):
            model.generate(prompt="question", parser=lambda x: x, tool_alias="tools")


# =============================================================================
# Image generation tests
# =============================================================================


@patch("data_designer.engine.models.facade.CustomRouter.image_generation", autospec=True)
def test_generate_image_diffusion_tracks_image_usage(
    mock_image_generation: Any,
    stub_model_facade: ModelFacade,
) -> None:
    """Test that generate_image tracks image usage for diffusion models."""
    from litellm.types.utils import ImageObject, ImageResponse

    # Mock response with 3 images
    mock_response = ImageResponse(
        data=[
            ImageObject(b64_json="image1_base64"),
            ImageObject(b64_json="image2_base64"),
            ImageObject(b64_json="image3_base64"),
        ]
    )
    mock_image_generation.return_value = mock_response

    # Verify initial state
    assert stub_model_facade.usage_stats.image_usage.total_images == 0

    # Generate images
    with patch("data_designer.engine.models.facade.is_image_diffusion_model", return_value=True):
        images = stub_model_facade.generate_image(prompt="test prompt", n=3)

    # Verify results
    assert len(images) == 3
    assert images == ["image1_base64", "image2_base64", "image3_base64"]

    # Verify image usage was tracked
    assert stub_model_facade.usage_stats.image_usage.total_images == 3
    assert stub_model_facade.usage_stats.image_usage.has_usage is True


@patch("data_designer.engine.models.facade.ModelFacade.completion", autospec=True)
def test_generate_image_chat_completion_tracks_image_usage(
    mock_completion: Any,
    stub_model_facade: ModelFacade,
) -> None:
    """Test that generate_image tracks image usage for chat completion models."""
    from litellm.types.utils import Choices, ImageURLListItem, Message, ModelResponse

    # Mock response with images attribute (Message requires type and index per ImageURLListItem)
    mock_message = Message(
        role="assistant",
        content="",
        images=[
            ImageURLListItem(type="image_url", image_url={"url": "data:image/png;base64,image1"}, index=0),
            ImageURLListItem(type="image_url", image_url={"url": "data:image/png;base64,image2"}, index=1),
        ],
    )
    mock_response = ModelResponse(choices=[Choices(message=mock_message)])
    mock_completion.return_value = mock_response

    # Verify initial state
    assert stub_model_facade.usage_stats.image_usage.total_images == 0

    # Generate images
    with patch("data_designer.engine.models.facade.is_image_diffusion_model", return_value=False):
        images = stub_model_facade.generate_image(prompt="test prompt")

    # Verify results
    assert len(images) == 2
    assert images == ["image1", "image2"]

    # Verify image usage was tracked
    assert stub_model_facade.usage_stats.image_usage.total_images == 2
    assert stub_model_facade.usage_stats.image_usage.has_usage is True


@patch("data_designer.engine.models.facade.CustomRouter.image_generation", autospec=True)
def test_generate_image_skip_usage_tracking(
    mock_image_generation: Any,
    stub_model_facade: ModelFacade,
) -> None:
    """Test that generate_image respects skip_usage_tracking flag."""
    from litellm.types.utils import ImageObject, ImageResponse

    mock_response = ImageResponse(
        data=[
            ImageObject(b64_json="image1_base64"),
            ImageObject(b64_json="image2_base64"),
        ]
    )
    mock_image_generation.return_value = mock_response

    # Verify initial state
    assert stub_model_facade.usage_stats.image_usage.total_images == 0

    # Generate images with skip_usage_tracking=True
    with patch("data_designer.engine.models.facade.is_image_diffusion_model", return_value=True):
        images = stub_model_facade.generate_image(prompt="test prompt", skip_usage_tracking=True)

    # Verify results
    assert len(images) == 2

    # Verify image usage was NOT tracked
    assert stub_model_facade.usage_stats.image_usage.total_images == 0
    assert stub_model_facade.usage_stats.image_usage.has_usage is False


@patch("data_designer.engine.models.facade.CustomRouter.image_generation", autospec=True)
def test_generate_image_accumulates_usage(
    mock_image_generation: Any,
    stub_model_facade: ModelFacade,
) -> None:
    """Test that generate_image accumulates image usage across multiple calls."""
    from litellm.types.utils import ImageObject, ImageResponse

    # First call - 2 images
    mock_response1 = ImageResponse(
        data=[
            ImageObject(b64_json="image1"),
            ImageObject(b64_json="image2"),
        ]
    )
    # Second call - 3 images
    mock_response2 = ImageResponse(
        data=[
            ImageObject(b64_json="image3"),
            ImageObject(b64_json="image4"),
            ImageObject(b64_json="image5"),
        ]
    )
    mock_image_generation.side_effect = [mock_response1, mock_response2]

    # Verify initial state
    assert stub_model_facade.usage_stats.image_usage.total_images == 0

    # First generation
    with patch("data_designer.engine.models.facade.is_image_diffusion_model", return_value=True):
        images1 = stub_model_facade.generate_image(prompt="test1")
        assert len(images1) == 2
        assert stub_model_facade.usage_stats.image_usage.total_images == 2

        # Second generation
        images2 = stub_model_facade.generate_image(prompt="test2")
        assert len(images2) == 3
        # Usage should accumulate
        assert stub_model_facade.usage_stats.image_usage.total_images == 5
