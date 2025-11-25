# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from unittest.mock import patch

from litellm.types.utils import Choices, Message, ModelResponse
import pytest

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
def stub_expected_response():
    return ModelResponse(choices=Choices(message=Message(content="Test response")))


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


@pytest.mark.parametrize(
    "skip_usage_tracking",
    [
        False,
        True,
    ],
)
def test_completion_success(stub_model_facade, stub_expected_response, skip_usage_tracking):
    stub_model_facade._router.completion = lambda model_name, messages, **kwargs: stub_expected_response

    messages = [{"role": "user", "content": "test"}]
    result = stub_model_facade.completion(messages, skip_usage_tracking=skip_usage_tracking)

    assert result == stub_expected_response


def test_completion_with_exception(stub_model_facade):
    def raise_exception(*args, **kwargs):
        raise Exception("Router error")

    stub_model_facade._router.completion = raise_exception

    messages = [{"role": "user", "content": "test"}]

    with pytest.raises(Exception, match="Router error"):
        stub_model_facade.completion(messages)


def test_completion_kwargs_overrides_model_config_generate_kwargs(
    stub_model_configs, stub_model_facade, stub_expected_response
):
    captured_kwargs = {}

    def mock_completion(model_name, messages, **kwargs):
        captured_kwargs.update(kwargs)
        return stub_expected_response

    stub_model_facade._router.completion = mock_completion

    messages = [{"role": "user", "content": "test"}]
    kwargs = {"temperature": 0.7, "max_tokens": 100}
    result = stub_model_facade.completion(messages, **kwargs)

    assert result == stub_expected_response
    # completion kwargs overrides model config generate kwargs
    assert captured_kwargs == {**stub_model_configs[0].inference_parameters.generate_kwargs, **kwargs}


@patch("data_designer.engine.models.facade.CustomRouter.completion", autospec=True)
def test_provider_extra_body_overrides_completion_kwargs(mock_router_completion, stub_model_configs, stub_model_facade):
    messages = [{"role": "user", "content": "test"}]
    stub_provider_extra_body = {"foo": "bar"}

    # model config has generate kwargs, completion call has no kwargs, and provider has no extra body
    _ = stub_model_facade.completion(messages)
    assert len(mock_router_completion.call_args) == 2
    assert mock_router_completion.call_args[0][1] == "stub-model-text"
    assert mock_router_completion.call_args[0][2] == messages
    assert mock_router_completion.call_args[1] == stub_model_configs[0].inference_parameters.generate_kwargs

    # model config has generate kwargs, completion call has kwargs, and provider has no extra body
    # completion kwargs overrides model config generate kwargs
    _ = stub_model_facade.completion(messages, temperature=0.1)
    assert len(mock_router_completion.call_args) == 2
    assert mock_router_completion.call_args[0][1] == "stub-model-text"
    assert mock_router_completion.call_args[0][2] == messages
    assert mock_router_completion.call_args[1] == {
        **stub_model_configs[0].inference_parameters.generate_kwargs,
        "temperature": 0.1,
    }

    # model config has generate kwargs, completion call has kwargs, and provider has extra body
    # provider extra body overrides completion kwargs
    stub_model_facade.model_provider.extra_body = stub_provider_extra_body
    _ = stub_model_facade.completion(messages, temperature=0.15, extra_body={"foo": "bat"})
    assert len(mock_router_completion.call_args) == 2
    assert mock_router_completion.call_args[0][1] == "stub-model-text"
    assert mock_router_completion.call_args[0][2] == messages
    assert mock_router_completion.call_args[1] == {
        **stub_model_configs[0].inference_parameters.generate_kwargs,
        "temperature": 0.15,
        "extra_body": stub_provider_extra_body,
    }
