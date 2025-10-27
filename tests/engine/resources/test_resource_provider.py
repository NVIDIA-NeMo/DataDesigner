# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
from unittest.mock import Mock, patch

import pytest

from data_designer.engine.resources.resource_provider import (
    ResourceProvider,
    create_resource_provider,
)


def test_resource_provider_artifact_storage_required():
    with pytest.raises(ValueError, match="Field required"):
        ResourceProvider()


@pytest.mark.parametrize(
    "test_case,expected_error",
    [
        ("model_registry_creation_error", "Model registry creation failed"),
    ],
)
def test_create_resource_provider_error_cases(test_case, expected_error):
    mock_artifact_storage = Mock()
    mock_model_configs = [Mock(), Mock()]
    mock_secret_resolver = Mock()
    mock_model_provider_registry = Mock()

    with patch("data_designer.engine.resources.resource_provider.create_model_registry") as mock_create_model_registry:
        mock_create_model_registry.side_effect = Exception(expected_error)

        with pytest.raises(Exception, match=expected_error):
            create_resource_provider(
                artifact_storage=mock_artifact_storage,
                model_configs=mock_model_configs,
                secret_resolver=mock_secret_resolver,
                model_provider_registry=mock_model_provider_registry,
            )


def test_create_resource_provider_function_exists():
    assert callable(create_resource_provider)

    sig = inspect.signature(create_resource_provider)
    params = list(sig.parameters.keys())

    expected_params = [
        "artifact_storage",
        "model_configs",
        "secret_resolver",
        "model_provider_registry",
        "datastore",
        "blob_storage",
    ]

    for param in expected_params:
        assert param in params
