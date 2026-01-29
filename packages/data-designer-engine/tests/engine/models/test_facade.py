# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import patch

import pytest
from litellm.types.utils import Choices, EmbeddingResponse, Message, ModelResponse

from data_designer.engine.mcp.errors import MCPConfigurationError
from data_designer.engine.models.errors import ModelGenerationValidationFailureError
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.models.parsers.errors import ParserException
from data_designer.engine.models.utils import ChatMessage


class FakeMessage:
    """Unified fake message class for mocking LLM completion responses."""

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
    def __init__(self, message: FakeMessage) -> None:
        self.message = message


class FakeResponse:
    def __init__(self, message: FakeMessage) -> None:
        self.choices = [FakeChoice(message)]


def mock_oai_response_object(response_text: str) -> FakeResponse:
    return FakeResponse(FakeMessage(content=response_text))


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
        FakeResponse(FakeMessage(content=None, tool_calls=[tool_call])),
        FakeResponse(FakeMessage(content="final result")),
    ]
    captured_calls: list[tuple[list[ChatMessage], dict[str, Any]]] = []

    class FakeMCPRegistry:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, dict]] = []

        def get_mcp(self, *, tool_alias: str) -> Any:
            return FakeMCPFacade(self)

    class FakeMCPFacade:
        def __init__(self, registry: FakeMCPRegistry) -> None:
            self._registry = registry
            self.tool_alias = "tools"
            self.providers = ["tools"]
            self.max_tool_call_turns = 3

        def get_tool_schemas(self) -> list[dict]:
            return [
                {
                    "type": "function",
                    "function": {"name": "lookup", "description": "Lookup", "parameters": {"type": "object"}},
                }
            ]

        def tool_call_count(self, completion_response: Any) -> int:
            message = completion_response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls is None:
                return 0
            return len(tool_calls)

        def has_tool_calls(self, completion_response: Any) -> bool:
            return self.tool_call_count(completion_response) > 0

        def refuse_completion_response(self, completion_response: Any) -> list[ChatMessage]:
            message = completion_response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []
            refusal_message = "Tool call refused: maximum tool-calling turns reached."
            return [
                ChatMessage.as_assistant(content="", tool_calls=tool_calls),
                *[ChatMessage.as_tool(content=refusal_message, tool_call_id=tc["id"]) for tc in tool_calls],
            ]

        def process_completion_response(self, completion_response: Any) -> list[ChatMessage]:
            message = completion_response.choices[0].message
            if not message.tool_calls:
                return [ChatMessage.as_assistant(content=message.content or "")]

            self._registry.calls.append(("tools", "lookup", {"query": "foo"}, None))
            return [
                ChatMessage.as_assistant(
                    content="",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"query": "foo"}'},
                        }
                    ],
                ),
                ChatMessage.as_tool(content="tool-output", tool_call_id="call-1"),
            ]

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> FakeResponse:
        captured_calls.append((messages, kwargs))
        return responses.pop(0)

    fake_registry = FakeMCPRegistry()
    model = ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
        mcp_registry=fake_registry,
    )

    with patch.object(ModelFacade, "completion", new=_completion):
        result, _ = model.generate(prompt="question", parser=lambda x: x, tool_alias="tools")

    assert result == "final result"
    assert len(captured_calls) == 2
    assert "tools" in captured_calls[0][1]
    assert captured_calls[0][1]["tools"][0]["function"]["name"] == "lookup"
    assert any(message.role == "tool" for message in captured_calls[1][0])
    assert fake_registry.calls == [("tools", "lookup", {"query": "foo"}, None)]


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
        FakeResponse(FakeMessage(content="First lookup", tool_calls=[tool_call_1])),
        FakeResponse(FakeMessage(content="Second search", tool_calls=[tool_call_2])),
        FakeResponse(FakeMessage(content="final result after two tool turns")),
    ]
    call_count = 0

    class FakeMCPRegistry:
        def get_mcp(self, *, tool_alias: str) -> Any:
            return FakeMCPFacade()

    class FakeMCPFacade:
        tool_alias = "tools"
        providers = ["tools"]
        max_tool_call_turns = 5

        def get_tool_schemas(self) -> list[dict[str, Any]]:
            return [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]

        def has_tool_calls(self, completion_response: Any) -> bool:
            return completion_response.choices[0].message.tool_calls is not None

        def process_completion_response(self, completion_response: Any) -> list[ChatMessage]:
            message = completion_response.choices[0].message
            tool_calls = message.tool_calls or []
            return [
                ChatMessage.as_assistant(content=message.content or "", tool_calls=tool_calls),
                *[ChatMessage.as_tool(content="tool-result", tool_call_id=tc["id"]) for tc in tool_calls],
            ]

        def refuse_completion_response(self, completion_response: Any) -> list[ChatMessage]:
            return []

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> FakeResponse:
        nonlocal call_count
        call_count += 1
        return responses.pop(0)

    model = ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
        mcp_registry=FakeMCPRegistry(),
    )

    with patch.object(ModelFacade, "completion", new=_completion):
        result, trace = model.generate(prompt="question", parser=lambda x: x, tool_alias="tools")

    assert result == "final result after two tool turns"
    assert call_count == 3  # 2 tool turns + 1 final


def test_generate_tool_turn_limit_triggers_refusal(
    stub_model_configs: Any,
    stub_secrets_resolver: Any,
    stub_model_provider_registry: Any,
) -> None:
    """When max_tool_call_turns exceeded, refusal is used."""
    tool_call = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}

    # Keep returning tool calls to exceed the limit
    responses = [
        FakeResponse(FakeMessage(content="", tool_calls=[tool_call])),  # Turn 1
        FakeResponse(FakeMessage(content="", tool_calls=[tool_call])),  # Turn 2 (max)
        FakeResponse(FakeMessage(content="", tool_calls=[tool_call])),  # Turn 3 (exceeds, should refuse)
        FakeResponse(FakeMessage(content="final answer after refusal")),
    ]
    process_calls = 0
    refuse_calls = 0

    class FakeMCPRegistry:
        def get_mcp(self, *, tool_alias: str) -> Any:
            return FakeMCPFacade()

    class FakeMCPFacade:
        tool_alias = "tools"
        providers = ["tools"]
        max_tool_call_turns = 2  # Set limit to 2

        def get_tool_schemas(self) -> list[dict[str, Any]]:
            return [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]

        def has_tool_calls(self, completion_response: Any) -> bool:
            return completion_response.choices[0].message.tool_calls is not None

        def process_completion_response(self, completion_response: Any) -> list[ChatMessage]:
            nonlocal process_calls
            process_calls += 1
            message = completion_response.choices[0].message
            return [
                ChatMessage.as_assistant(content="", tool_calls=message.tool_calls or []),
                ChatMessage.as_tool(content="tool-result", tool_call_id="call-1"),
            ]

        def refuse_completion_response(self, completion_response: Any) -> list[ChatMessage]:
            nonlocal refuse_calls
            refuse_calls += 1
            message = completion_response.choices[0].message
            return [
                ChatMessage.as_assistant(content="", tool_calls=message.tool_calls or []),
                ChatMessage.as_tool(content="REFUSED: Budget exceeded", tool_call_id="call-1"),
            ]

    response_idx = 0

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> FakeResponse:
        nonlocal response_idx
        resp = responses[response_idx]
        response_idx += 1
        return resp

    model = ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
        mcp_registry=FakeMCPRegistry(),
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
        FakeResponse(FakeMessage(content="", tool_calls=[tool_call])),  # Exceeds on first turn
        FakeResponse(FakeMessage(content="I understand, here is the answer without tools")),
    ]

    class FakeMCPRegistry:
        def get_mcp(self, *, tool_alias: str) -> Any:
            return FakeMCPFacade()

    class FakeMCPFacade:
        tool_alias = "tools"
        providers = ["tools"]
        max_tool_call_turns = 0  # No tool turns allowed

        def get_tool_schemas(self) -> list[dict[str, Any]]:
            return [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]

        def has_tool_calls(self, completion_response: Any) -> bool:
            return completion_response.choices[0].message.tool_calls is not None

        def process_completion_response(self, completion_response: Any) -> list[ChatMessage]:
            return []  # Should not be called

        def refuse_completion_response(self, completion_response: Any) -> list[ChatMessage]:
            return [
                ChatMessage.as_assistant(content="", tool_calls=[tool_call]),
                ChatMessage.as_tool(
                    content="Tool call refused: You have reached the maximum number of tool-calling turns.",
                    tool_call_id="call-1",
                ),
            ]

    response_idx = 0

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> FakeResponse:
        nonlocal response_idx
        resp = responses[response_idx]
        response_idx += 1
        return resp

    model = ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
        mcp_registry=FakeMCPRegistry(),
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

    class FakeMCPRegistry:
        def get_mcp(self, *, tool_alias: str) -> Any:
            raise ValueError(f"No tool config with alias {tool_alias!r} found!")

    model = ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
        mcp_registry=FakeMCPRegistry(),
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

    class FakeMCPRegistry:
        def get_mcp(self, *, tool_alias: str) -> Any:
            nonlocal get_mcp_called
            get_mcp_called = True
            raise RuntimeError("Should not be called")

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> FakeResponse:
        assert "tools" not in kwargs  # No tools should be passed
        return FakeResponse(FakeMessage(content="response without tools"))

    model = ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
        mcp_registry=FakeMCPRegistry(),
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
        FakeResponse(FakeMessage(content="", tool_calls=[tool_call])),  # Tool call
        FakeResponse(FakeMessage(content="bad format")),  # Parser will fail
        FakeResponse(FakeMessage(content="correct format")),  # Parser will succeed
    ]

    class FakeMCPRegistry:
        def get_mcp(self, *, tool_alias: str) -> Any:
            return FakeMCPFacade()

    class FakeMCPFacade:
        tool_alias = "tools"
        providers = ["tools"]
        max_tool_call_turns = 3

        def get_tool_schemas(self) -> list[dict[str, Any]]:
            return [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]

        def has_tool_calls(self, completion_response: Any) -> bool:
            return completion_response.choices[0].message.tool_calls is not None

        def process_completion_response(self, completion_response: Any) -> list[ChatMessage]:
            message = completion_response.choices[0].message
            return [
                ChatMessage.as_assistant(content="", tool_calls=message.tool_calls or []),
                ChatMessage.as_tool(content="tool-result", tool_call_id="call-1"),
            ]

        def refuse_completion_response(self, completion_response: Any) -> list[ChatMessage]:
            return []

    response_idx = 0

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> FakeResponse:
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
        mcp_registry=FakeMCPRegistry(),
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
        FakeResponse(FakeMessage(content="", tool_calls=[tool_call])),
        FakeResponse(FakeMessage(content="still bad")),  # Fails parser, triggers restart
        FakeResponse(FakeMessage(content="", tool_calls=[tool_call])),  # After restart
        FakeResponse(FakeMessage(content="good result")),
    ]

    class FakeMCPRegistry:
        def get_mcp(self, *, tool_alias: str) -> Any:
            return FakeMCPFacade()

    class FakeMCPFacade:
        tool_alias = "tools"
        providers = ["tools"]
        max_tool_call_turns = 3

        def get_tool_schemas(self) -> list[dict[str, Any]]:
            return [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]

        def has_tool_calls(self, completion_response: Any) -> bool:
            return completion_response.choices[0].message.tool_calls is not None

        def process_completion_response(self, completion_response: Any) -> list[ChatMessage]:
            message = completion_response.choices[0].message
            return [
                ChatMessage.as_assistant(content="", tool_calls=message.tool_calls or []),
                ChatMessage.as_tool(content="tool-result", tool_call_id="call-1"),
            ]

        def refuse_completion_response(self, completion_response: Any) -> list[ChatMessage]:
            return []

    response_idx = 0

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> FakeResponse:
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
        mcp_registry=FakeMCPRegistry(),
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
        FakeResponse(FakeMessage(content="Let me look that up", tool_calls=[tool_call])),
        FakeResponse(FakeMessage(content="Here is the answer")),
    ]

    class FakeMCPRegistry:
        def get_mcp(self, *, tool_alias: str) -> Any:
            return FakeMCPFacade()

    class FakeMCPFacade:
        tool_alias = "tools"
        providers = ["tools"]
        max_tool_call_turns = 3

        def get_tool_schemas(self) -> list[dict[str, Any]]:
            return [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]

        def has_tool_calls(self, completion_response: Any) -> bool:
            return completion_response.choices[0].message.tool_calls is not None

        def process_completion_response(self, completion_response: Any) -> list[ChatMessage]:
            message = completion_response.choices[0].message
            return [
                ChatMessage.as_assistant(content=message.content or "", tool_calls=message.tool_calls or []),
                ChatMessage.as_tool(content="tool-output", tool_call_id="call-1"),
            ]

        def refuse_completion_response(self, completion_response: Any) -> list[ChatMessage]:
            return []

    response_idx = 0

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> FakeResponse:
        nonlocal response_idx
        resp = responses[response_idx]
        response_idx += 1
        return resp

    model = ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
        mcp_registry=FakeMCPRegistry(),
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
        FakeResponse(FakeMessage(content="", tool_calls=[tool_call])),
        FakeResponse(FakeMessage(content="final")),
    ]

    class FakeMCPRegistry:
        def get_mcp(self, *, tool_alias: str) -> Any:
            return FakeMCPFacade()

    class FakeMCPFacade:
        tool_alias = "tools"
        providers = ["tools"]
        max_tool_call_turns = 3

        def get_tool_schemas(self) -> list[dict[str, Any]]:
            return [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]

        def has_tool_calls(self, completion_response: Any) -> bool:
            return completion_response.choices[0].message.tool_calls is not None

        def process_completion_response(self, completion_response: Any) -> list[ChatMessage]:
            return [
                ChatMessage.as_assistant(content="", tool_calls=[tool_call]),
                ChatMessage.as_tool(content="THE TOOL RESPONSE CONTENT", tool_call_id="call-1"),
            ]

        def refuse_completion_response(self, completion_response: Any) -> list[ChatMessage]:
            return []

    response_idx = 0

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> FakeResponse:
        nonlocal response_idx
        resp = responses[response_idx]
        response_idx += 1
        return resp

    model = ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
        mcp_registry=FakeMCPRegistry(),
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
        FakeResponse(FakeMessage(content="", tool_calls=[tool_call])),  # Will be refused (max_turns=0)
        FakeResponse(FakeMessage(content="answer without tools")),
    ]

    class FakeMCPRegistry:
        def get_mcp(self, *, tool_alias: str) -> Any:
            return FakeMCPFacade()

    class FakeMCPFacade:
        tool_alias = "tools"
        providers = ["tools"]
        max_tool_call_turns = 0  # All tool calls refused

        def get_tool_schemas(self) -> list[dict[str, Any]]:
            return [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]

        def has_tool_calls(self, completion_response: Any) -> bool:
            return completion_response.choices[0].message.tool_calls is not None

        def process_completion_response(self, completion_response: Any) -> list[ChatMessage]:
            return []

        def refuse_completion_response(self, completion_response: Any) -> list[ChatMessage]:
            return [
                ChatMessage.as_assistant(content="", tool_calls=[tool_call]),
                ChatMessage.as_tool(content="BUDGET_EXCEEDED_REFUSAL", tool_call_id="call-1"),
            ]

    response_idx = 0

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> FakeResponse:
        nonlocal response_idx
        resp = responses[response_idx]
        response_idx += 1
        return resp

    model = ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
        mcp_registry=FakeMCPRegistry(),
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
    response = FakeResponse(
        FakeMessage(
            content="The answer is 42",
            reasoning_content="Let me think about this carefully...",
        )
    )

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> FakeResponse:
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
    from data_designer.engine.mcp.errors import MCPToolError

    tool_call = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}

    responses = [FakeResponse(FakeMessage(content="", tool_calls=[tool_call]))]

    class FakeMCPRegistry:
        def get_mcp(self, *, tool_alias: str) -> Any:
            return FakeMCPFacade()

    class FakeMCPFacade:
        tool_alias = "tools"
        providers = ["tools"]
        max_tool_call_turns = 3

        def get_tool_schemas(self) -> list[dict[str, Any]]:
            return [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]

        def has_tool_calls(self, completion_response: Any) -> bool:
            return completion_response.choices[0].message.tool_calls is not None

        def process_completion_response(self, completion_response: Any) -> list[ChatMessage]:
            raise MCPToolError("Tool execution failed: Connection refused")

        def refuse_completion_response(self, completion_response: Any) -> list[ChatMessage]:
            return []

    response_idx = 0

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> FakeResponse:
        nonlocal response_idx
        resp = responses[response_idx]
        response_idx += 1
        return resp

    model = ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
        mcp_registry=FakeMCPRegistry(),
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
    from data_designer.engine.mcp.errors import MCPToolError

    # Tool call with invalid JSON arguments
    tool_call = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "not valid json"}}

    responses = [FakeResponse(FakeMessage(content="", tool_calls=[tool_call]))]

    class FakeMCPRegistry:
        def get_mcp(self, *, tool_alias: str) -> Any:
            return FakeMCPFacade()

    class FakeMCPFacade:
        tool_alias = "tools"
        providers = ["tools"]
        max_tool_call_turns = 3

        def get_tool_schemas(self) -> list[dict[str, Any]]:
            return [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]

        def has_tool_calls(self, completion_response: Any) -> bool:
            return completion_response.choices[0].message.tool_calls is not None

        def process_completion_response(self, completion_response: Any) -> list[ChatMessage]:
            raise MCPToolError("Invalid tool arguments for 'lookup': not valid json")

        def refuse_completion_response(self, completion_response: Any) -> list[ChatMessage]:
            return []

    response_idx = 0

    def _completion(self: Any, messages: list[ChatMessage], **kwargs: Any) -> FakeResponse:
        nonlocal response_idx
        resp = responses[response_idx]
        response_idx += 1
        return resp

    model = ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
        mcp_registry=FakeMCPRegistry(),
    )

    with patch.object(ModelFacade, "completion", new=_completion):
        with pytest.raises(MCPToolError, match="Invalid tool arguments"):
            model.generate(prompt="question", parser=lambda x: x, tool_alias="tools")
