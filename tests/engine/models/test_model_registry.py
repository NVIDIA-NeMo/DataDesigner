# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from litellm import AuthenticationError
import pytest

from data_designer.config.models import InferenceParameters, ModelConfig
from data_designer.engine.models.errors import ModelAuthenticationError
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.models.registry import ModelRegistry, create_model_registry
from data_designer.engine.models.usage import RequestUsageStats, TokenUsageStats


@pytest.fixture
def stub_empty_model_registry():
    return ModelRegistry(model_configs={}, secret_resolver=None, model_provider_registry=None)


@pytest.fixture
def stub_new_model_config():
    return ModelConfig(
        alias="stub-vision",
        model="stub-model-vision",
        provider="stub-model-provider",
        inference_parameters=InferenceParameters(
            temperature=0.80, top_p=0.95, max_tokens=100, max_parallel_requests=10, timeout=100
        ),
    )


@pytest.fixture
def stub_no_usage_config():
    return ModelConfig(
        alias="no-usage",
        model="no-usage-model",
        provider="stub-model-provider",
        inference_parameters=InferenceParameters(),
    )


@patch("data_designer.engine.models.registry.apply_litellm_patches", autospec=True)
def test_create_model_registry(
    mock_apply_litellm_patches, stub_model_configs, stub_secrets_resolver, stub_model_provider_registry
):
    model_registry = create_model_registry(
        model_configs=stub_model_configs,
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
    )
    assert isinstance(model_registry, ModelRegistry)
    mock_apply_litellm_patches.assert_called_once()


def test_public_props(stub_model_configs, stub_model_registry):
    assert stub_model_registry.model_configs == {
        model_config.alias: model_config for model_config in stub_model_configs
    }
    # With lazy initialization, models dict is empty until requested
    assert len(stub_model_registry.models) == 0

    # Request models to trigger lazy initialization
    stub_model_registry.get_model(model_alias="stub-text")
    stub_model_registry.get_model(model_alias="stub-reasoning")

    assert len(stub_model_registry.models) == 2
    assert all(isinstance(model, ModelFacade) for model in stub_model_registry.models.values())


def test_register_model_configs(stub_model_registry, stub_new_model_config):
    stub_model_registry.register_model_configs([stub_new_model_config])

    # Verify configs are registered
    assert len(stub_model_registry.model_configs) == 3

    # Trigger lazy initialization by requesting models
    assert stub_model_registry.get_model(model_alias="stub-text").model_name == "stub-model-text"
    assert stub_model_registry.get_model(model_alias="stub-reasoning").model_name == "stub-model-reasoning"
    assert stub_model_registry.get_model(model_alias="stub-vision").model_name == "stub-model-vision"

    assert len(stub_model_registry.models) == 3
    assert all(isinstance(model, ModelFacade) for model in stub_model_registry.models.values())


@pytest.mark.parametrize(
    "method_name,alias,expected_model_name,expected_error",
    [
        ("get_model", "stub-text", "stub-model-text", None),
        ("get_model", "invalid-alias", None, "No model config with alias 'invalid-alias' found!"),
        ("get_model_config", "stub-text", "stub-model-text", None),
        ("get_model_config", "invalid-alias", None, "No model config with alias 'invalid-alias' found!"),
    ],
)
def test_get_model_and_config(stub_model_registry, method_name, alias, expected_model_name, expected_error):
    method = getattr(stub_model_registry, method_name)

    if expected_error:
        with pytest.raises(ValueError, match=expected_error):
            method(model_alias=alias)
    else:
        result = method(model_alias=alias)
        if method_name == "get_model":
            assert result.model_name == expected_model_name
        else:  # get_model_config
            assert result.model == expected_model_name


@pytest.mark.parametrize(
    "test_case,setup_usage,expected_keys",
    [
        ("no_usage", False, []),
        ("with_usage", True, ["stub-model-text", "stub-model-reasoning"]),
        ("mixed_usage", True, ["stub-model-text"]),
    ],
)
def test_get_model_usage_stats(
    stub_model_registry, stub_empty_model_registry, stub_no_usage_config, test_case, setup_usage, expected_keys
):
    if test_case == "no_usage":
        usage_stats = stub_empty_model_registry.get_model_usage_stats(total_time_elapsed=10)
        assert usage_stats == {}
    elif test_case == "with_usage":
        # Trigger lazy initialization
        text_model = stub_model_registry.get_model(model_alias="stub-text")
        reasoning_model = stub_model_registry.get_model(model_alias="stub-reasoning")

        text_model.usage_stats.extend(
            token_usage=TokenUsageStats(prompt_tokens=10, completion_tokens=100),
            request_usage=RequestUsageStats(successful_requests=10, failed_requests=0),
        )
        reasoning_model.usage_stats.extend(
            token_usage=TokenUsageStats(prompt_tokens=5, completion_tokens=200),
            request_usage=RequestUsageStats(successful_requests=100, failed_requests=10),
        )
        usage_stats = stub_model_registry.get_model_usage_stats(total_time_elapsed=10)

        assert set(usage_stats.keys()) == set(expected_keys)
        if "stub-model-text" in usage_stats:
            assert usage_stats["stub-model-text"]["token_usage"]["prompt_tokens"] == 10
            assert usage_stats["stub-model-text"]["token_usage"]["completion_tokens"] == 100
            assert usage_stats["stub-model-text"]["token_usage"]["total_tokens"] == 110
            assert usage_stats["stub-model-text"]["request_usage"]["successful_requests"] == 10
            assert usage_stats["stub-model-text"]["request_usage"]["failed_requests"] == 0
            assert usage_stats["stub-model-text"]["request_usage"]["total_requests"] == 10
            assert usage_stats["stub-model-text"]["tokens_per_second"] == 11
            assert usage_stats["stub-model-text"]["requests_per_minute"] == 60
    else:  # mixed_usage
        stub_model_registry.register_model_configs([stub_no_usage_config])
        # Trigger lazy initialization
        text_model = stub_model_registry.get_model(model_alias="stub-text")
        text_model.usage_stats.extend(
            token_usage=TokenUsageStats(prompt_tokens=10, completion_tokens=100),
            request_usage=RequestUsageStats(successful_requests=10, failed_requests=0),
        )
        usage_stats = stub_model_registry.get_model_usage_stats(total_time_elapsed=10)
        assert set(usage_stats.keys()) == set(expected_keys)


@pytest.mark.parametrize(
    "test_case,mock_side_effect,expected_exception,expected_call_count",
    [
        ("success", None, None, 2),
        (
            "authentication_error",
            AuthenticationError("Invalid API key", llm_provider="openai", model="stub-model-text"),
            ModelAuthenticationError,
            1,
        ),
    ],
)
@patch("data_designer.engine.models.facade.ModelFacade.completion", autospec=True)
def test_run_health_check(
    mock_completion, stub_model_registry, test_case, mock_side_effect, expected_exception, expected_call_count
):
    if mock_side_effect:
        mock_completion.side_effect = mock_side_effect

    # Pass model aliases for health check
    model_aliases = {"stub-text", "stub-reasoning"}

    if expected_exception:
        with pytest.raises(expected_exception):
            stub_model_registry.run_health_check(model_aliases)
    else:
        stub_model_registry.run_health_check(model_aliases)

    assert mock_completion.call_count == expected_call_count


@pytest.mark.parametrize(
    "alias,expected_result,expected_error",
    [
        ("stub-text", True, None),
        ("invalid-alias", None, "No model config with alias 'invalid-alias' found!"),
    ],
)
def test_get_model_provider(stub_model_registry, alias, expected_result, expected_error):
    if expected_error:
        with pytest.raises(ValueError, match=expected_error):
            stub_model_registry.get_model_provider(model_alias=alias)
    else:
        provider = stub_model_registry.get_model_provider(model_alias=alias)
        assert provider is not None
