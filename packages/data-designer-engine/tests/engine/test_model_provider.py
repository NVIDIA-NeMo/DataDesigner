# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from data_designer.config.mcp import LocalStdioMCPProvider
from data_designer.engine.errors import NoModelProvidersError
from data_designer.engine.model_provider import (
    MCPProviderRegistry,
    ModelProvider,
    ModelProviderRegistry,
    UnknownProviderError,
    resolve_model_provider_registry,
)


@pytest.fixture
def stub_foo_provider():
    return ModelProvider(name="foo", endpoint="https://foo.com", provider_type="foo")


@pytest.fixture
def stub_bar_provider():
    return ModelProvider(name="bar", endpoint="https://bar.com", provider_type="bar")


def test_must_have_at_least_one_provider():
    with pytest.raises(ValueError):
        ModelProviderRegistry(providers=[], default="a")

    with pytest.raises(ValueError):
        ModelProviderRegistry(providers=[])


def test_defined_default_must_exist(stub_foo_provider: ModelProvider):
    with pytest.raises(ValueError):
        ModelProviderRegistry(providers=[stub_foo_provider], default="bar")


def test_multiple_providers_requires_explicit_default(
    stub_foo_provider: ModelProvider, stub_bar_provider: ModelProvider
):
    with pytest.raises(ValueError):
        ModelProviderRegistry(providers=[stub_foo_provider, stub_bar_provider])


def test_implicit_default(stub_foo_provider: ModelProvider):
    registry = ModelProviderRegistry(providers=[stub_foo_provider])

    assert registry.get_provider(None) == stub_foo_provider


def test_no_duplicate_provider_names(stub_foo_provider: ModelProvider):
    with pytest.raises(ValueError):
        ModelProviderRegistry(providers=[stub_foo_provider, stub_foo_provider], default="foo")


def test_get_provider(stub_foo_provider: ModelProvider, stub_bar_provider: ModelProvider):
    registry = ModelProviderRegistry(
        providers=[stub_foo_provider, stub_bar_provider],
        default="foo",
    )

    assert registry.get_provider(None) == stub_foo_provider
    assert registry.get_provider("foo") == stub_foo_provider
    assert registry.get_provider("bar") == stub_bar_provider

    with pytest.raises(UnknownProviderError):
        registry.get_provider("quux")


def test_resolve_model_provider_registry(stub_foo_provider: ModelProvider) -> None:
    """Test resolve_model_provider_registry creates a registry from providers."""
    registry = resolve_model_provider_registry([stub_foo_provider])

    assert len(registry.providers) == 1
    assert registry.get_default_provider_name() == "foo"


def test_resolve_model_provider_registry_with_explicit_default(
    stub_foo_provider: ModelProvider, stub_bar_provider: ModelProvider
) -> None:
    """Test resolve_model_provider_registry with explicit default."""
    registry = resolve_model_provider_registry([stub_foo_provider, stub_bar_provider], default_provider_name="bar")

    assert registry.get_default_provider_name() == "bar"


def test_resolve_model_provider_registry_empty_error() -> None:
    """Test resolve_model_provider_registry raises error for empty providers."""
    with pytest.raises(NoModelProvidersError, match="At least one model provider"):
        resolve_model_provider_registry([])


def test_mcp_provider_registry_empty() -> None:
    """Test MCPProviderRegistry can be created empty."""
    registry = MCPProviderRegistry()

    assert len(registry.providers) == 0


def test_mcp_provider_registry_with_providers() -> None:
    """Test MCPProviderRegistry with providers."""
    provider = LocalStdioMCPProvider(name="test-provider", command="test-cmd")
    registry = MCPProviderRegistry(providers=[provider])

    assert len(registry.providers) == 1
    assert registry.get_provider("test-provider") == provider


def test_mcp_provider_registry_duplicate_names() -> None:
    """Test MCPProviderRegistry raises error for duplicate provider names."""
    provider1 = LocalStdioMCPProvider(name="test-provider", command="test-cmd")
    provider2 = LocalStdioMCPProvider(name="test-provider", command="test-cmd-2")

    with pytest.raises(ValueError, match="duplicate"):
        MCPProviderRegistry(providers=[provider1, provider2])


def test_mcp_provider_registry_unknown_provider() -> None:
    """Test MCPProviderRegistry raises error for unknown provider."""
    registry = MCPProviderRegistry()

    with pytest.raises(UnknownProviderError, match="registered"):
        registry.get_provider("unknown-provider")


def test_mcp_provider_registry_is_empty() -> None:
    """Test MCPProviderRegistry is_empty method."""
    empty_registry = MCPProviderRegistry()
    assert empty_registry.is_empty() is True

    provider = LocalStdioMCPProvider(name="test-provider", command="test-cmd")
    registry_with_providers = MCPProviderRegistry(providers=[provider])
    assert registry_with_providers.is_empty() is False
