# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from unittest.mock import patch

import pytest
from litellm.types.utils import Choices, EmbeddingResponse, Message, ModelResponse

from data_designer.engine.models.errors import ModelGenerationValidationFailureError
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.models.parsers.errors import ParserException

MockMessage = namedtuple("MockMessage", ["content"])
MockChoice = namedtuple("MockChoice", ["message"])
MockCompletion = namedtuple("MockCompletion", ["choices"])


def mock_oai_response_object(response_text: str) -> MockCompletion:
    return MockCompletion(choices=[MockChoice(message=MockMessage(content=response_text))])


@pytest.fixture
def stub_model_facade(stub_model_configs, stub_secrets_resolver, stub_model_provider_registry):
    return ModelFacade(
        model_config=stub_model_configs[0],
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
    )


@pytest.fixture
def stub_completion_messages():
    return [{"role": "user", "content": "test"}]


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
        ("", [{"role": "user", "content": "does not matter"}]),
        ("hello!", [{"content": "hello!", "role": "system"}, {"role": "user", "content": "does not matter"}]),
    ],
)
@patch("data_designer.engine.models.facade.ModelFacade.completion", autospec=True)
def test_generate_with_system_prompt(mock_completion, stub_model_facade, system_prompt, expected_messages):
    mock_completion.return_value = ModelResponse(choices=Choices(message=Message(content="Hello!")))

    stub_model_facade.generate(prompt="does not matter", system_prompt=system_prompt, parser=lambda x: x)
    assert mock_completion.call_count == 1
    assert mock_completion.call_args[0][1] == expected_messages


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


@pytest.mark.parametrize(
    "skip_usage_tracking",
    [
        False,
        True,
    ],
)
@patch("data_designer.engine.models.facade.CustomRouter.completion", autospec=True)
def test_completion_success(
    mock_router_completion,
    stub_completion_messages,
    stub_model_configs,
    stub_model_facade,
    stub_expected_completion_response,
    skip_usage_tracking,
):
    mock_router_completion.side_effect = lambda self, model, messages, **kwargs: stub_expected_completion_response
    result = stub_model_facade.completion(stub_completion_messages, skip_usage_tracking=skip_usage_tracking)
    assert result == stub_expected_completion_response
    assert mock_router_completion.call_count == 1
    assert mock_router_completion.call_args[1] == {
        "model": "stub-model-text",
        "messages": stub_completion_messages,
        **stub_model_configs[0].inference_parameters.generate_kwargs,
    }


@patch("data_designer.engine.models.facade.CustomRouter.completion", autospec=True)
def test_completion_with_exception(mock_router_completion, stub_completion_messages, stub_model_facade):
    mock_router_completion.side_effect = Exception("Router error")

    with pytest.raises(Exception, match="Router error"):
        stub_model_facade.completion(stub_completion_messages)


@patch("data_designer.engine.models.facade.CustomRouter.completion", autospec=True)
def test_completion_with_kwargs(
    mock_router_completion,
    stub_completion_messages,
    stub_model_configs,
    stub_model_facade,
    stub_expected_completion_response,
):
    captured_kwargs = {}

    def mock_completion(self, model, messages, **kwargs):
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
