# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from data_designer.cli.forms.field import ValidationError
from data_designer.cli.forms.model_builder import ModelFormBuilder
from data_designer.config.models import GenerationType, ModelConfig


# Alias validation tests - test through public form interface
def test_alias_field_rejects_empty_string() -> None:
    """Test alias field rejects empty strings."""
    builder = ModelFormBuilder()
    form = builder.create_form()
    alias_field = form.get_field("alias")

    with pytest.raises(ValidationError, match="Model alias is required"):
        alias_field.value = ""


def test_alias_field_rejects_duplicate_alias() -> None:
    """Test alias field rejects aliases that already exist."""
    builder = ModelFormBuilder(existing_aliases={"gpt4", "claude"})
    form = builder.create_form()
    alias_field = form.get_field("alias")

    with pytest.raises(ValidationError, match="already exists"):
        alias_field.value = "gpt4"


def test_alias_field_accepts_unique_alias() -> None:
    """Test alias field accepts new unique aliases."""
    builder = ModelFormBuilder(existing_aliases={"gpt4", "claude"})
    form = builder.create_form()
    alias_field = form.get_field("alias")

    alias_field.value = "new-model"

    assert alias_field.value == "new-model"


def test_alias_field_accepts_any_alias_when_no_existing() -> None:
    """Test alias field accepts any alias when no existing aliases."""
    builder = ModelFormBuilder()
    form = builder.create_form()
    alias_field = form.get_field("alias")

    alias_field.value = "my-model"

    assert alias_field.value == "my-model"


# Model validation tests
def test_model_field_rejects_empty_string() -> None:
    """Test model ID field rejects empty strings."""
    builder = ModelFormBuilder()
    form = builder.create_form()
    model_field = form.get_field("model")

    with pytest.raises(ValidationError, match="Model is required"):
        model_field.value = ""


def test_model_field_accepts_any_non_empty_string() -> None:
    """Test model field accepts any non-empty string."""
    builder = ModelFormBuilder()
    form = builder.create_form()
    model_field = form.get_field("model")

    model_field.value = "gpt-4-turbo"

    assert model_field.value == "gpt-4-turbo"


# Form structure tests - conditional provider field logic
def test_form_includes_provider_field_with_multiple_providers() -> None:
    """Test form includes provider selection when multiple providers available."""
    builder = ModelFormBuilder(available_providers=["openai", "anthropic", "nvidia"])

    form = builder.create_form()

    assert form.get_field("provider") is not None


def test_form_omits_provider_field_with_single_provider() -> None:
    """Test form omits provider field when only one provider available."""
    builder = ModelFormBuilder(available_providers=["openai"])

    form = builder.create_form()

    assert form.get_field("provider") is None


def test_form_omits_provider_field_with_no_providers() -> None:
    """Test form omits provider field when no providers available."""
    builder = ModelFormBuilder(available_providers=[])

    form = builder.create_form()

    assert form.get_field("provider") is None


def test_form_has_all_required_fields() -> None:
    """Test basic form includes essential configuration fields (inference params are in separate form)."""
    builder = ModelFormBuilder()

    form = builder.create_form()

    # All required fields must be present in basic form
    assert form.get_field("alias") is not None
    assert form.get_field("model") is not None
    assert form.get_field("generation_type") is not None
    # inference_parameters are now collected in a separate form via _create_inference_params_form


# Initial data handling tests
def test_form_uses_initial_data_for_field_defaults() -> None:
    """Test form fields populate defaults from initial_data."""
    initial_data = {
        "alias": "my-model",
        "model": "gpt-4",
        "inference_parameters": {
            "generation_type": GenerationType.CHAT_COMPLETION,
            "temperature": 0.5,
            "top_p": 0.8,
            "max_tokens": 1024,
        },
    }
    builder = ModelFormBuilder()

    form = builder.create_form(initial_data)

    assert form.get_field("alias").default == "my-model"
    assert form.get_field("model").default == "gpt-4"
    assert form.get_field("generation_type").default == GenerationType.CHAT_COMPLETION


def test_form_extracts_generation_type_from_inference_parameters() -> None:
    """Test form correctly extracts generation_type from nested inference_parameters for embedding models."""
    initial_data = {
        "alias": "embedding-model",
        "model": "text-embedding-3",
        "inference_parameters": {
            "generation_type": GenerationType.EMBEDDING,
            "encoding_format": "base64",
            "dimensions": 512,
        },
        "provider": "openai",
    }
    builder = ModelFormBuilder()

    form = builder.create_form(initial_data)

    assert form.get_field("alias").default == "embedding-model"
    assert form.get_field("model").default == "text-embedding-3"
    assert form.get_field("generation_type").default == GenerationType.EMBEDDING


def test_form_uses_standard_defaults_without_initial_data() -> None:
    """Test form fields use sensible defaults when no initial_data provided."""
    builder = ModelFormBuilder()

    form = builder.create_form()

    assert form.get_field("alias").default is None
    assert form.get_field("model").default is None
    assert form.get_field("generation_type").default == GenerationType.CHAT_COMPLETION


def test_form_handles_partial_initial_data() -> None:
    """Test form gracefully handles initial_data with missing nested values."""
    initial_data = {
        "alias": "my-model",
        "model": "gpt-4",
    }
    builder = ModelFormBuilder()

    form = builder.create_form(initial_data)

    # Should use provided values
    assert form.get_field("alias").default == "my-model"
    assert form.get_field("model").default == "gpt-4"
    # Should fall back to standard defaults for missing generation_type
    assert form.get_field("generation_type").default == GenerationType.CHAT_COMPLETION


def test_form_provider_defaults_to_first_when_multiple_available() -> None:
    """Test provider field defaults to first provider when multiple available."""
    builder = ModelFormBuilder(available_providers=["openai", "anthropic", "nvidia"])

    form = builder.create_form()

    provider_field = form.get_field("provider")
    assert provider_field.default == "openai"


def test_form_provider_preserves_initial_data_preference() -> None:
    """Test provider field uses initial_data value when provided."""
    initial_data = {"provider": "anthropic"}
    builder = ModelFormBuilder(available_providers=["openai", "anthropic", "nvidia"])

    form = builder.create_form(initial_data)

    provider_field = form.get_field("provider")
    assert provider_field.default == "anthropic"


# build_config tests - provider determination logic
def test_build_config_uses_provider_from_form_data() -> None:
    """Test build_config uses provider when explicitly provided in form data."""
    builder = ModelFormBuilder(available_providers=["openai", "anthropic"])
    form_data = {
        "alias": "my-model",
        "model": "gpt-4",
        "provider": "anthropic",
        "inference_parameters": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048,
        },
    }

    config = builder.build_config(form_data)

    assert config.provider == "anthropic"


def test_build_config_infers_single_available_provider() -> None:
    """Test build_config uses single available provider when not in form data."""
    builder = ModelFormBuilder(available_providers=["openai"])
    form_data = {
        "alias": "my-model",
        "model": "gpt-4",
        "inference_parameters": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048,
        },
    }

    config = builder.build_config(form_data)

    assert config.provider == "openai"


def test_build_config_sets_provider_none_when_unavailable() -> None:
    """Test build_config sets provider to None when no providers available."""
    builder = ModelFormBuilder(available_providers=[])
    form_data = {
        "alias": "my-model",
        "model": "gpt-4",
        "inference_parameters": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048,
        },
    }

    config = builder.build_config(form_data)

    assert config.provider is None


def test_build_config_creates_valid_model_config() -> None:
    """Test build_config produces properly structured ModelConfig object."""
    builder = ModelFormBuilder(available_providers=["openai"])
    form_data = {
        "alias": "test-model",
        "model": "gpt-4-turbo",
        "inference_parameters": {
            "temperature": 0.5,
            "top_p": 0.8,
            "max_tokens": 1024,
        },
    }

    config = builder.build_config(form_data)

    assert isinstance(config, ModelConfig)
    assert config.alias == "test-model"
    assert config.model == "gpt-4-turbo"
    assert config.provider == "openai"
    assert config.inference_parameters.temperature == 0.5
    assert config.inference_parameters.top_p == 0.8
    assert config.inference_parameters.max_tokens == 1024
    assert config.inference_parameters.max_parallel_requests == 4


def test_build_config_converts_max_tokens_to_int() -> None:
    """Test build_config handles numeric values in inference parameters."""
    builder = ModelFormBuilder()
    form_data = {
        "alias": "my-model",
        "model": "gpt-4",
        "inference_parameters": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048,
        },
    }

    config = builder.build_config(form_data)

    assert config.inference_parameters.max_tokens == 2048


def test_build_config_prefers_explicit_provider_over_inference() -> None:
    """Test build_config uses form data provider even when single provider available."""
    # Edge case: form explicitly provides provider different from available
    builder = ModelFormBuilder(available_providers=["openai"])
    form_data = {
        "alias": "my-model",
        "model": "gpt-4",
        "provider": "custom",  # Explicitly overridden
        "inference_parameters": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048,
        },
    }

    config = builder.build_config(form_data)

    assert config.provider == "custom"


# Integration test - full workflow
def test_full_workflow_creates_valid_config() -> None:
    """Test complete workflow from form creation to config building."""
    builder = ModelFormBuilder(
        existing_aliases={"existing-model"},
        available_providers=["openai", "anthropic"],
    )

    # Create form with initial data
    initial_data = {
        "alias": "new-model",
        "model": "claude-3-opus",
        "provider": "anthropic",
        "inference_parameters": {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 4096,
        },
    }
    form = builder.create_form(initial_data)

    # Simulate user accepting defaults (get_values would return these)
    form.set_values(initial_data)

    # Form data now includes inference_parameters as a dict
    form_data = {
        "alias": "new-model",
        "model": "claude-3-opus",
        "provider": "anthropic",
        "inference_parameters": {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 4096,
        },
    }

    # Build config
    config = builder.build_config(form_data)

    # Verify complete config
    assert config.alias == "new-model"
    assert config.model == "claude-3-opus"
    assert config.provider == "anthropic"
    assert config.inference_parameters.temperature == 0.6
    assert config.inference_parameters.top_p == 0.95
    assert config.inference_parameters.max_tokens == 4096


# Tests for new two-step form process
def test_create_inference_params_form_for_chat_completion() -> None:
    """Test creating inference parameters form for chat completion models."""
    builder = ModelFormBuilder()

    params_form = builder.create_inference_params_form(GenerationType.CHAT_COMPLETION)

    # Should have chat completion specific fields
    assert params_form.get_field("temperature") is not None
    assert params_form.get_field("top_p") is not None
    assert params_form.get_field("max_tokens") is not None
    # Should not have embedding fields
    assert params_form.get_field("encoding_format") is None
    assert params_form.get_field("dimensions") is None


def test_create_inference_params_form_for_embedding() -> None:
    """Test creating inference parameters form for embedding models."""
    builder = ModelFormBuilder()

    params_form = builder.create_inference_params_form(GenerationType.EMBEDDING)

    # Should have embedding specific fields
    assert params_form.get_field("encoding_format") is not None
    assert params_form.get_field("dimensions") is not None
    # Should not have chat completion fields
    assert params_form.get_field("temperature") is None
    assert params_form.get_field("top_p") is None
    assert params_form.get_field("max_tokens") is None


def test_create_inference_params_form_uses_initial_params() -> None:
    """Test inference parameters form uses initial values from existing config."""
    builder = ModelFormBuilder()
    initial_params = {"temperature": 0.8, "top_p": 0.95, "max_tokens": 2048}

    params_form = builder.create_inference_params_form(GenerationType.CHAT_COMPLETION, initial_params)

    assert params_form.get_field("temperature").default == 0.8
    assert params_form.get_field("top_p").default == 0.95
    assert params_form.get_field("max_tokens").default == 2048


def test_build_inference_params_chat_completion_with_all_values() -> None:
    """Test building inference params dict from chat completion form data."""
    builder = ModelFormBuilder()
    params_data = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 1024.0}

    result = builder.build_inference_params(GenerationType.CHAT_COMPLETION, params_data)

    assert result == {"temperature": 0.7, "top_p": 0.9, "max_tokens": 1024}


def test_build_inference_params_chat_completion_with_partial_values() -> None:
    """Test building inference params dict with only some values provided."""
    builder = ModelFormBuilder()
    params_data = {"temperature": 0.7, "top_p": None, "max_tokens": None}

    result = builder.build_inference_params(GenerationType.CHAT_COMPLETION, params_data)

    # Only provided values should be included
    assert result == {"temperature": 0.7}


def test_build_inference_params_embedding_with_all_values() -> None:
    """Test building inference params dict from embedding form data."""
    builder = ModelFormBuilder()
    params_data = {"encoding_format": "float", "dimensions": 1024.0}

    result = builder.build_inference_params(GenerationType.EMBEDDING, params_data)

    assert result == {"encoding_format": "float", "dimensions": 1024}


def test_build_inference_params_embedding_with_partial_values() -> None:
    """Test building embedding inference params with only some values provided."""
    builder = ModelFormBuilder()
    params_data = {"encoding_format": "float", "dimensions": None}

    result = builder.build_inference_params(GenerationType.EMBEDDING, params_data)

    # encoding_format always included, dimensions omitted if not provided
    assert result == {"encoding_format": "float"}


def test_build_inference_params_embedding_all_cleared() -> None:
    """Test building embedding inference params when both are cleared."""
    builder = ModelFormBuilder()
    params_data = {"encoding_format": None, "dimensions": None}

    result = builder.build_inference_params(GenerationType.EMBEDDING, params_data)

    # Empty dict; Pydantic will use defaults (encoding_format="float", dimensions=None)
    assert result == {}


def test_validate_encoding_format_accepts_valid_values() -> None:
    """Test encoding format validation accepts 'float' and 'base64'."""
    builder = ModelFormBuilder()

    is_valid, error = builder.validate_encoding_format("float")
    assert is_valid is True
    assert error is None

    is_valid, error = builder.validate_encoding_format("base64")
    assert is_valid is True
    assert error is None


def test_validate_encoding_format_rejects_invalid_values() -> None:
    """Test encoding format validation rejects invalid values."""
    builder = ModelFormBuilder()

    is_valid, error = builder.validate_encoding_format("invalid")
    assert is_valid is False
    assert "float" in error and "base64" in error


def test_validate_encoding_format_accepts_empty_string() -> None:
    """Test encoding format validation accepts empty string (optional field)."""
    builder = ModelFormBuilder()

    is_valid, error = builder.validate_encoding_format("")
    assert is_valid is True
    assert error is None


def test_validate_encoding_format_accepts_clear_keywords() -> None:
    """Test encoding format validation accepts clearing keywords."""
    builder = ModelFormBuilder()

    for keyword in ("clear", "none", "default", "CLEAR", "None"):
        is_valid, error = builder.validate_encoding_format(keyword)
        assert is_valid is True, f"Failed for keyword: {keyword}"
        assert error is None
