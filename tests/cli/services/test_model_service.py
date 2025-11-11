# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.cli.services.model_service import ModelService
from data_designer.config.models import ModelConfig


def test_list_all(stub_model_service: ModelService, stub_model_configs: list[ModelConfig]):
    assert stub_model_service.list_all() == stub_model_configs


def test_get_by_alias(
    stub_model_service: ModelService, stub_model_configs: list[ModelConfig], stub_new_model_config: ModelConfig
):
    assert stub_model_service.get_by_alias("test-alias-1") == stub_model_configs[0]
    assert stub_model_service.get_by_alias("test-alias-2") == stub_model_configs[1]
    assert stub_model_service.get_by_alias("test-alias-3") is None


def test_add(
    stub_model_service: ModelService, stub_model_configs: list[ModelConfig], stub_new_model_config: ModelConfig
):
    stub_model_service.add(stub_new_model_config)
    assert stub_model_service.list_all() == stub_model_configs + [stub_new_model_config]


def test_update(
    stub_model_service: ModelService, stub_model_configs: list[ModelConfig], stub_new_model_config: ModelConfig
):
    stub_model_service.update("test-alias-1", stub_new_model_config)
    assert stub_model_service.get_by_alias("test-alias-1") is None
    assert stub_model_service.get_by_alias("test-alias-3") == stub_new_model_config


def test_delete(stub_model_service: ModelService, stub_model_configs: list[ModelConfig]):
    stub_model_service.delete("test-alias-1")
    assert stub_model_service.list_all() == stub_model_configs[1:]


def test_find_by_provider(stub_model_service: ModelService, stub_model_configs: list[ModelConfig]):
    # Both test models have provider="test-provider-1"
    models = stub_model_service.find_by_provider("test-provider-1")
    assert len(models) == 2
    assert models == stub_model_configs

    # Non-existent provider should return empty list
    models = stub_model_service.find_by_provider("non-existent-provider")
    assert models == []


def test_delete_by_aliases(stub_model_service: ModelService, stub_model_configs: list[ModelConfig]):
    # Delete both models
    stub_model_service.delete_by_aliases(["test-alias-1", "test-alias-2"])
    assert stub_model_service.list_all() == []


def test_delete_by_aliases_partial(stub_model_service: ModelService, stub_model_configs: list[ModelConfig]):
    # Delete only one model
    stub_model_service.delete_by_aliases(["test-alias-1"])
    assert stub_model_service.list_all() == stub_model_configs[1:]


def test_delete_by_aliases_empty_list(stub_model_service: ModelService, stub_model_configs: list[ModelConfig]):
    # Deleting empty list should do nothing
    stub_model_service.delete_by_aliases([])
    assert stub_model_service.list_all() == stub_model_configs
