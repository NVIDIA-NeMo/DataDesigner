# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from data_designer.config.models import ChatCompletionInferenceParams, ModelConfig
from data_designer.engine.models.errors import ModelAuthenticationError
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.models.factory import create_model_registry
from data_designer.engine.models.registry import ModelRegistry
from data_designer.engine.models.usage import ModelUsageStats, RequestUsageStats, TokenUsageStats


@pytest.fixture
def stub_empty_model_registry():
    return ModelRegistry(model_configs={}, secret_resolver=None, model_provider_registry=None)


@pytest.fixture
def stub_new_model_config():
    return ModelConfig(
        alias="stub-vision",
        model="stub-model-vision",
        provider="stub-model-provider",
        inference_parameters=ChatCompletionInferenceParams(
            temperature=0.80, top_p=0.95, max_tokens=100, max_parallel_requests=10, timeout=100
        ),
    )


@pytest.fixture
def stub_no_usage_config():
    return ModelConfig(
        alias="no-usage",
        model="no-usage-model",
        provider="stub-model-provider",
        inference_parameters=ChatCompletionInferenceParams(),
    )


@patch("data_designer.engine.models.litellm_overrides.apply_litellm_patches", autospec=True)
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
    assert len(stub_model_registry.model_configs) == 4

    # Trigger lazy initialization by requesting models
    assert stub_model_registry.get_model(model_alias="stub-text").model_name == "stub-model-text"
    assert stub_model_registry.get_model(model_alias="stub-reasoning").model_name == "stub-model-reasoning"
    assert stub_model_registry.get_model(model_alias="stub-vision").model_name == "stub-model-vision"
    assert stub_model_registry.get_model(model_alias="stub-embedding").model_name == "stub-model-embedding"

    assert len(stub_model_registry.models) == 4
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
            token_usage=TokenUsageStats(input_tokens=10, output_tokens=100),
            request_usage=RequestUsageStats(successful_requests=10, failed_requests=0),
        )
        reasoning_model.usage_stats.extend(
            token_usage=TokenUsageStats(input_tokens=5, output_tokens=200),
            request_usage=RequestUsageStats(successful_requests=100, failed_requests=10),
        )
        usage_stats = stub_model_registry.get_model_usage_stats(total_time_elapsed=10)

        assert set(usage_stats.keys()) == set(expected_keys)
        if "stub-model-text" in usage_stats:
            assert usage_stats["stub-model-text"]["token_usage"]["input_tokens"] == 10
            assert usage_stats["stub-model-text"]["token_usage"]["output_tokens"] == 100
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
            token_usage=TokenUsageStats(input_tokens=10, output_tokens=100),
            request_usage=RequestUsageStats(successful_requests=10, failed_requests=0),
        )
        usage_stats = stub_model_registry.get_model_usage_stats(total_time_elapsed=10)
        assert set(usage_stats.keys()) == set(expected_keys)


@pytest.mark.parametrize(
    "test_case,expected_keys",
    [
        ("no_models", []),
        ("with_usage", ["stub-model-text", "stub-model-reasoning"]),
        ("no_usage", []),
    ],
)
def test_get_model_usage_snapshot(
    stub_model_registry: ModelRegistry,
    stub_empty_model_registry: ModelRegistry,
    test_case: str,
    expected_keys: list[str],
) -> None:
    if test_case == "no_models":
        snapshot = stub_empty_model_registry.get_model_usage_snapshot()
        assert snapshot == {}
    elif test_case == "with_usage":
        text_model = stub_model_registry.get_model(model_alias="stub-text")
        reasoning_model = stub_model_registry.get_model(model_alias="stub-reasoning")

        text_model.usage_stats.extend(
            token_usage=TokenUsageStats(input_tokens=10, output_tokens=100),
            request_usage=RequestUsageStats(successful_requests=5, failed_requests=1),
        )
        reasoning_model.usage_stats.extend(
            token_usage=TokenUsageStats(input_tokens=20, output_tokens=200),
            request_usage=RequestUsageStats(successful_requests=10, failed_requests=2),
        )

        snapshot = stub_model_registry.get_model_usage_snapshot()

        assert set(snapshot.keys()) == set(expected_keys)
        assert all(isinstance(stats, ModelUsageStats) for stats in snapshot.values())

        assert snapshot["stub-model-text"].token_usage.input_tokens == 10
        assert snapshot["stub-model-text"].token_usage.output_tokens == 100
        assert snapshot["stub-model-reasoning"].token_usage.input_tokens == 20
        assert snapshot["stub-model-reasoning"].token_usage.output_tokens == 200

        snapshot["stub-model-text"].token_usage.input_tokens = 999
        assert text_model.usage_stats.token_usage.input_tokens == 10
    else:
        stub_model_registry.get_model(model_alias="stub-text")
        stub_model_registry.get_model(model_alias="stub-reasoning")

        snapshot = stub_model_registry.get_model_usage_snapshot()
        assert snapshot == {}


@pytest.mark.parametrize(
    "test_case,expected_keys",
    [
        ("no_prior_usage", ["stub-model-text"]),
        ("with_prior_usage", ["stub-model-text"]),
        ("no_change", []),
    ],
)
def test_get_usage_deltas(
    stub_model_registry: ModelRegistry,
    test_case: str,
    expected_keys: list[str],
) -> None:
    text_model = stub_model_registry.get_model(model_alias="stub-text")

    if test_case == "no_prior_usage":
        # Empty snapshot, then add usage
        pre_snapshot: dict[str, ModelUsageStats] = {}
        text_model.usage_stats.extend(
            token_usage=TokenUsageStats(input_tokens=50, output_tokens=100),
            request_usage=RequestUsageStats(successful_requests=5, failed_requests=1),
        )

        deltas = stub_model_registry.get_usage_deltas(pre_snapshot)

        assert set(deltas.keys()) == set(expected_keys)
        assert deltas["stub-model-text"].token_usage.input_tokens == 50
        assert deltas["stub-model-text"].token_usage.output_tokens == 100
        assert deltas["stub-model-text"].request_usage.successful_requests == 5
        assert deltas["stub-model-text"].request_usage.failed_requests == 1

    elif test_case == "with_prior_usage":
        # Add initial usage, take snapshot, add more usage
        text_model.usage_stats.extend(
            token_usage=TokenUsageStats(input_tokens=100, output_tokens=200),
            request_usage=RequestUsageStats(successful_requests=10, failed_requests=2),
        )
        pre_snapshot = stub_model_registry.get_model_usage_snapshot()

        text_model.usage_stats.extend(
            token_usage=TokenUsageStats(input_tokens=50, output_tokens=75),
            request_usage=RequestUsageStats(successful_requests=3, failed_requests=1),
        )

        deltas = stub_model_registry.get_usage_deltas(pre_snapshot)

        assert set(deltas.keys()) == set(expected_keys)
        assert deltas["stub-model-text"].token_usage.input_tokens == 50
        assert deltas["stub-model-text"].token_usage.output_tokens == 75
        assert deltas["stub-model-text"].request_usage.successful_requests == 3
        assert deltas["stub-model-text"].request_usage.failed_requests == 1

    else:  # no_change
        text_model.usage_stats.extend(
            token_usage=TokenUsageStats(input_tokens=100, output_tokens=200),
            request_usage=RequestUsageStats(successful_requests=10, failed_requests=2),
        )
        pre_snapshot = stub_model_registry.get_model_usage_snapshot()

        # No additional usage after snapshot
        deltas = stub_model_registry.get_usage_deltas(pre_snapshot)
        assert deltas == {}


@patch("data_designer.engine.models.facade.ModelFacade.generate_text_embeddings", autospec=True)
@patch("data_designer.engine.models.facade.ModelFacade.completion", autospec=True)
def test_run_health_check_success(mock_completion, mock_generate_text_embeddings, stub_model_registry):
    model_aliases = {"stub-text", "stub-reasoning", "stub-embedding"}
    stub_model_registry.run_health_check(model_aliases)
    assert mock_completion.call_count == 2
    assert mock_generate_text_embeddings.call_count == 1


@patch("data_designer.engine.models.facade.ModelFacade.generate_text_embeddings", autospec=True)
@patch("data_designer.engine.models.facade.ModelFacade.completion", autospec=True)
def test_run_health_check_completion_authentication_error(
    mock_completion, mock_generate_text_embeddings, stub_model_registry
):
    auth_error = ModelAuthenticationError("Invalid API key for completion model")
    mock_completion.side_effect = auth_error
    model_aliases = ["stub-text", "stub-reasoning", "stub-embedding"]

    with pytest.raises(ModelAuthenticationError):
        stub_model_registry.run_health_check(model_aliases)

    mock_completion.assert_called_once()
    mock_generate_text_embeddings.assert_not_called()


@patch("data_designer.engine.models.facade.ModelFacade.generate_text_embeddings", autospec=True)
@patch("data_designer.engine.models.facade.ModelFacade.completion", autospec=True)
def test_run_health_check_embedding_authentication_error(
    mock_completion, mock_generate_text_embeddings, stub_model_registry
):
    auth_error = ModelAuthenticationError("Invalid API key for embedding model")
    mock_generate_text_embeddings.side_effect = auth_error
    model_aliases = ["stub-text", "stub-reasoning", "stub-embedding"]

    with pytest.raises(ModelAuthenticationError):
        stub_model_registry.run_health_check(model_aliases)

    mock_completion.call_count == 2
    mock_generate_text_embeddings.assert_called_once()


@patch("data_designer.engine.models.facade.ModelFacade.completion", autospec=True)
def test_run_health_check_skip_health_check_flag(
    mock_completion,
    stub_secrets_resolver,
    stub_model_provider_registry,
):
    # Create model configs: one with skip_health_check=True, others with default (False)
    model_configs = [
        ModelConfig(
            alias="skip-model",
            model="skip-model-id",
            provider="stub-model-provider",
            inference_parameters=ChatCompletionInferenceParams(),
            skip_health_check=True,
        ),
        ModelConfig(
            alias="check-model",
            model="check-model-id",
            provider="stub-model-provider",
            inference_parameters=ChatCompletionInferenceParams(),
            skip_health_check=False,
        ),
        ModelConfig(
            alias="default-model",
            model="default-model-id",
            provider="stub-model-provider",
            inference_parameters=ChatCompletionInferenceParams(),
        ),
    ]

    # Create a fresh model registry with the test configs
    model_registry = create_model_registry(
        model_configs=model_configs,
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
    )

    model_aliases = ["skip-model", "check-model", "default-model"]
    model_registry.run_health_check(model_aliases)

    # Only check-model and default-model should be checked (skip-model is skipped)
    assert mock_completion.call_count == 2  # check-model and default-model

    # Verify the correct models were called
    called_model_aliases = {call[0][0].model_alias for call in mock_completion.call_args_list}
    assert called_model_aliases == {"check-model", "default-model"}


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


def test_log_model_usage_no_models(stub_empty_model_registry: ModelRegistry) -> None:
    """Test log_model_usage with no models registered."""
    with patch("data_designer.engine.models.registry.logger") as mock_logger:
        stub_empty_model_registry.log_model_usage(total_time_elapsed=10.0)

        assert mock_logger.info.call_count == 2
        calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert calls[0] == "📊 Model usage summary:"
        assert calls[1] == "  |-- no model usage recorded"


def test_log_model_usage_single_model(stub_model_registry: ModelRegistry) -> None:
    """Test log_model_usage with a single model that has usage."""
    text_model = stub_model_registry.get_model(model_alias="stub-text")
    text_model.usage_stats.extend(
        token_usage=TokenUsageStats(input_tokens=1000, output_tokens=500),
        request_usage=RequestUsageStats(successful_requests=10, failed_requests=2),
    )

    with patch("data_designer.engine.models.registry.logger") as mock_logger:
        stub_model_registry.log_model_usage(total_time_elapsed=10.0)

        calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert calls[0] == "📊 Model usage summary:"
        assert calls[1] == "  |-------- stub-model-text"
        assert calls[2] == "  |-- tokens: input=1000, output=500, total=1500, tps=150"
        assert calls[3] == "  |-- requests: success=10, failed=2, total=12, rpm=72"


def test_log_model_usage_multiple_models(stub_model_registry: ModelRegistry) -> None:
    """Test log_model_usage with multiple models - verifies models are sorted by name."""
    text_model = stub_model_registry.get_model(model_alias="stub-text")
    reasoning_model = stub_model_registry.get_model(model_alias="stub-reasoning")

    text_model.usage_stats.extend(
        token_usage=TokenUsageStats(input_tokens=1000, output_tokens=500),
        request_usage=RequestUsageStats(successful_requests=10, failed_requests=0),
    )
    reasoning_model.usage_stats.extend(
        token_usage=TokenUsageStats(input_tokens=2000, output_tokens=1000),
        request_usage=RequestUsageStats(successful_requests=20, failed_requests=5),
    )

    with patch("data_designer.engine.models.registry.logger") as mock_logger:
        stub_model_registry.log_model_usage(total_time_elapsed=10.0)

        calls = [call[0][0] for call in mock_logger.info.call_args_list]

        # Header
        assert calls[0] == "📊 Model usage summary:"

        # Models should be sorted alphabetically: stub-model-reasoning before stub-model-text
        assert calls[1] == "  |-------- stub-model-reasoning"
        assert calls[2] == "  |-- tokens: input=2000, output=1000, total=3000, tps=300"
        assert calls[3] == "  |-- requests: success=20, failed=5, total=25, rpm=150"

        assert calls[4] == "  |-------- stub-model-text"
        assert calls[5] == "  |-- tokens: input=1000, output=500, total=1500, tps=150"
        assert calls[6] == "  |-- requests: success=10, failed=0, total=10, rpm=60"


def test_log_model_usage_with_tool_usage(stub_model_registry: ModelRegistry) -> None:
    """Test log_model_usage includes tool usage stats when present."""
    text_model = stub_model_registry.get_model(model_alias="stub-text")
    text_model.usage_stats.extend(
        token_usage=TokenUsageStats(input_tokens=1000, output_tokens=500),
        request_usage=RequestUsageStats(successful_requests=10, failed_requests=0),
    )
    # Add tool usage
    text_model.usage_stats.tool_usage.extend(tool_calls=4, tool_call_turns=2)
    text_model.usage_stats.tool_usage.extend(tool_calls=6, tool_call_turns=3)

    with patch("data_designer.engine.models.registry.logger") as mock_logger:
        stub_model_registry.log_model_usage(total_time_elapsed=10.0)

        calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert calls[0] == "📊 Model usage summary:"
        assert calls[1] == "  |-------- stub-model-text"
        assert calls[2] == "  |-- tokens: input=1000, output=500, total=1500, tps=150"
        assert calls[3] == "  |-- requests: success=10, failed=0, total=10, rpm=60"
        assert calls[4] == (
            "  |-- tools: calls=10, turns=5, generations=2, calls/gen=5.0 +/- 1.0, turns/gen=2.5 +/- 0.5"
        )


def test_log_model_usage_models_without_usage_excluded(stub_model_registry: ModelRegistry) -> None:
    """Test that models without usage are not included in the log."""
    # Initialize both models but only add usage to one
    stub_model_registry.get_model(model_alias="stub-text")
    reasoning_model = stub_model_registry.get_model(model_alias="stub-reasoning")

    reasoning_model.usage_stats.extend(
        token_usage=TokenUsageStats(input_tokens=500, output_tokens=250),
        request_usage=RequestUsageStats(successful_requests=5, failed_requests=1),
    )

    with patch("data_designer.engine.models.registry.logger") as mock_logger:
        stub_model_registry.log_model_usage(total_time_elapsed=10.0)

        calls = [call[0][0] for call in mock_logger.info.call_args_list]

        # Only reasoning model should appear
        assert len(calls) == 4
        assert calls[0] == "📊 Model usage summary:"
        assert calls[1] == "  |-------- stub-model-reasoning"
        assert "stub-model-text" not in str(calls)
