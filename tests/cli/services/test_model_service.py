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
