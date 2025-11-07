# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from data_designer.engine.errors import SecretResolutionError
from data_designer.engine.resources.seed_dataset_source import (
    HfHubSeedDatasetSource,
    SeedDatasetRepository,
    SeedDatasetSourceRegistry,
)
from data_designer.engine.secret_resolver import EnvironmentResolver

HF_ENDPOINT = "https://huggingface.co"
NDS_ENDPOINT = "http://datastore:3000/v1/hf"


def test_hf_hub_source_resolution(monkeypatch: pytest.MonkeyPatch):
    secret_resolver = EnvironmentResolver()

    token_ref = "MY_HF_TOKEN"
    token_raw_value = "token-raw-value"
    hf_source = HfHubSeedDatasetSource(endpoint=HF_ENDPOINT, token=token_ref)

    with pytest.raises(SecretResolutionError):
        hf_source.resolve(secret_resolver)

    monkeypatch.setenv(token_ref, token_raw_value)
    resolved_hf_source = hf_source.resolve(secret_resolver)

    assert resolved_hf_source.token == token_raw_value


def test_registry_from_simple_config():
    config = {
        "sources": [
            {"source_type": "hf_hub", "endpoint": HF_ENDPOINT},
        ],
    }

    registry = SeedDatasetSourceRegistry.model_validate_json(json.dumps(config))
    assert len(registry.sources) == 1
    assert isinstance(registry.sources[0], HfHubSeedDatasetSource)
    assert registry.sources[0].name == "hf_hub"
    assert registry.sources[0].token is None


def test_registry_from_more_complex_config():
    config = {
        "default": "hf",
        "sources": [
            {"source_type": "hf_hub", "name": "hf", "endpoint": HF_ENDPOINT, "token": "HF_TOKEN"},
            {"source_type": "hf_hub", "name": "nds", "endpoint": NDS_ENDPOINT},
        ],
    }

    registry = SeedDatasetSourceRegistry.model_validate_json(json.dumps(config))
    assert len(registry.sources) == 2
    assert isinstance(registry.sources[0], HfHubSeedDatasetSource)
    assert isinstance(registry.sources[1], HfHubSeedDatasetSource)


def test_registry_validation_errors():
    with pytest.raises(ValueError) as excinfo:
        SeedDatasetSourceRegistry(
            sources=[
                HfHubSeedDatasetSource(name="hf", endpoint=HF_ENDPOINT),
                HfHubSeedDatasetSource(name="nds", endpoint=NDS_ENDPOINT),
            ]
        )
    assert "default source" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        SeedDatasetSourceRegistry(
            sources=[
                HfHubSeedDatasetSource(name="name", endpoint=HF_ENDPOINT),
                HfHubSeedDatasetSource(name="name", endpoint=NDS_ENDPOINT),
            ]
        )
    assert "duplicates" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        SeedDatasetSourceRegistry(
            default="not-defined",
            sources=[HfHubSeedDatasetSource(endpoint=HF_ENDPOINT)],
        )
    assert "not found" in str(excinfo.value)


def test_get_uri_through_repository():
    source_name = "source-name"
    registry = SeedDatasetSourceRegistry(
        sources=[HfHubSeedDatasetSource(name=source_name, endpoint=HF_ENDPOINT)],
    )
    secret_resolver = EnvironmentResolver()
    repository = SeedDatasetRepository(registry=registry, secret_resolver=secret_resolver)

    file_id = "namespace/repo/file.parquet"

    # The registry only has one source defined, so it has an implicit default
    uri1 = repository.get_dataset_uri(file_id, source_name)
    uri2 = repository.get_dataset_uri(file_id, None)

    assert uri1 == uri2 == "hf://datasets/namespace/repo/file.parquet"
