# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from data_designer.cli.forms.field import ValidationError
from data_designer.cli.forms.provider_builder import ProviderFormBuilder
from data_designer.engine.model_provider import ModelProvider


# Name validation tests - test through public field interface
def test_name_field_rejects_empty_string() -> None:
    """Test name field rejects empty strings."""
    builder = ProviderFormBuilder()
    # Use initial_data to get manual form
    form = builder.create_form(initial_data={"name": "test"})
    name_field = form.get_field("name")

    with pytest.raises(ValidationError, match="Provider name is required"):
        name_field.value = ""


def test_name_field_rejects_duplicate_name() -> None:
    """Test name field rejects names that already exist."""
    builder = ProviderFormBuilder(existing_names={"openai", "anthropic"})
    form = builder.create_form(initial_data={"name": "test"})
    name_field = form.get_field("name")

    with pytest.raises(ValidationError, match="already exists"):
        name_field.value = "openai"


def test_name_field_accepts_unique_name() -> None:
    """Test name field accepts new unique names."""
    builder = ProviderFormBuilder(existing_names={"openai", "anthropic"})
    form = builder.create_form(initial_data={"name": "test"})
    name_field = form.get_field("name")

    name_field.value = "custom-provider"

    assert name_field.value == "custom-provider"


def test_name_field_accepts_any_name_when_no_existing() -> None:
    """Test name field accepts any name when no existing names."""
    builder = ProviderFormBuilder()
    form = builder.create_form(initial_data={"name": "test"})
    name_field = form.get_field("name")

    name_field.value = "my-provider"

    assert name_field.value == "my-provider"


# Endpoint validation tests
def test_endpoint_field_rejects_empty_string() -> None:
    """Test endpoint field rejects empty strings."""
    builder = ProviderFormBuilder()
    form = builder.create_form(initial_data={"endpoint": "test"})
    endpoint_field = form.get_field("endpoint")

    with pytest.raises(ValidationError, match="Endpoint URL is required"):
        endpoint_field.value = ""


def test_endpoint_field_rejects_invalid_url() -> None:
    """Test endpoint field rejects URLs without http/https protocol."""
    builder = ProviderFormBuilder()
    form = builder.create_form(initial_data={"endpoint": "test"})
    endpoint_field = form.get_field("endpoint")

    with pytest.raises(ValidationError, match="Invalid URL format"):
        endpoint_field.value = "not-a-url"


def test_endpoint_field_accepts_https_url() -> None:
    """Test endpoint field accepts valid HTTPS URLs."""
    builder = ProviderFormBuilder()
    form = builder.create_form(initial_data={"endpoint": "test"})
    endpoint_field = form.get_field("endpoint")

    endpoint_field.value = "https://api.example.com/v1"

    assert endpoint_field.value == "https://api.example.com/v1"


def test_endpoint_field_accepts_http_url() -> None:
    """Test endpoint field accepts valid HTTP URLs."""
    builder = ProviderFormBuilder()
    form = builder.create_form(initial_data={"endpoint": "test"})
    endpoint_field = form.get_field("endpoint")

    endpoint_field.value = "http://localhost:8000/v1"

    assert endpoint_field.value == "http://localhost:8000/v1"


# Form creation tests
def test_create_form_returns_manual_form_for_new_provider() -> None:
    """Test create_form returns manual configuration form for new provider."""
    builder = ProviderFormBuilder()

    form = builder.create_form(initial_data=None)

    # Should return manual form directly
    assert form.get_field("name") is not None
    assert form.get_field("endpoint") is not None
    assert form.get_field("provider_type") is not None
    assert form.get_field("api_key") is not None


def test_create_form_returns_manual_form_for_updates() -> None:
    """Test create_form returns manual configuration form when updating."""
    builder = ProviderFormBuilder()
    initial_data = {
        "name": "custom-provider",
        "endpoint": "https://api.example.com",
        "provider_type": "openai",
    }

    form = builder.create_form(initial_data)

    # Should return manual form with fields populated
    assert form.get_field("name") is not None
    assert form.get_field("endpoint") is not None
    assert form.get_field("provider_type") is not None
    assert form.get_field("api_key") is not None


# build_config tests
def test_build_config_creates_valid_model_provider() -> None:
    """Test build_config creates properly structured ModelProvider."""
    builder = ProviderFormBuilder()
    form_data = {
        "name": "custom-provider",
        "endpoint": "https://api.example.com/v1",
        "provider_type": "openai",
        "api_key": "SECRET_KEY",
    }

    provider = builder.build_config(form_data)

    assert isinstance(provider, ModelProvider)
    assert provider.name == "custom-provider"
    assert provider.endpoint == "https://api.example.com/v1"
    assert provider.provider_type == "openai"
    assert provider.api_key == "SECRET_KEY"


def test_build_config_handles_missing_api_key() -> None:
    """Test build_config handles optional API key."""
    builder = ProviderFormBuilder()
    form_data = {
        "name": "custom-provider",
        "endpoint": "https://api.example.com/v1",
        "provider_type": "openai",
    }

    provider = builder.build_config(form_data)

    assert provider.api_key is None


def test_build_config_with_environment_variable_api_key() -> None:
    """Test build_config accepts environment variable names as API keys."""
    builder = ProviderFormBuilder()
    form_data = {
        "name": "my-provider",
        "endpoint": "https://api.example.com/v1",
        "provider_type": "openai",
        "api_key": "CUSTOM_API_KEY_ENV",
    }

    provider = builder.build_config(form_data)

    assert provider.api_key == "CUSTOM_API_KEY_ENV"


# Integration test - full workflow
def test_full_manual_workflow_creates_valid_provider() -> None:
    """Test complete manual workflow from form creation to provider building."""
    builder = ProviderFormBuilder(existing_names={"existing-provider"})

    # Simulate user completing manual configuration
    form_data = {
        "name": "new-provider",
        "endpoint": "https://api.newprovider.com/v1",
        "provider_type": "openai",
        "api_key": "NEW_PROVIDER_KEY",
    }

    # Build provider
    provider = builder.build_config(form_data)

    # Verify complete provider
    assert provider.name == "new-provider"
    assert provider.endpoint == "https://api.newprovider.com/v1"
    assert provider.provider_type == "openai"
    assert provider.api_key == "NEW_PROVIDER_KEY"


# Edge cases
def test_endpoint_validation_rejects_url_without_protocol() -> None:
    """Test endpoint validation catches common mistake of missing protocol."""
    builder = ProviderFormBuilder()
    form = builder.create_form(initial_data={"endpoint": "test"})
    endpoint_field = form.get_field("endpoint")

    with pytest.raises(ValidationError):
        endpoint_field.value = "api.example.com/v1"


def test_endpoint_validation_accepts_localhost() -> None:
    """Test endpoint validation accepts localhost URLs."""
    builder = ProviderFormBuilder()
    form = builder.create_form(initial_data={"endpoint": "test"})
    endpoint_field = form.get_field("endpoint")

    endpoint_field.value = "http://localhost:8080/v1"

    assert endpoint_field.value == "http://localhost:8080/v1"


def test_name_validation_case_sensitive() -> None:
    """Test name validation is case-sensitive for duplicates."""
    builder = ProviderFormBuilder(existing_names={"OpenAI"})
    form = builder.create_form(initial_data={"name": "test"})
    name_field = form.get_field("name")

    # Different case should be allowed
    name_field.value = "openai"

    assert name_field.value == "openai"
