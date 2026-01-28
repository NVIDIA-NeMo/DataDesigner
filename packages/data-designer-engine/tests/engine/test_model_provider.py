# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from data_designer.engine.model_provider import ModelProvider, ModelProviderRegistry, UnknownProviderError


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
