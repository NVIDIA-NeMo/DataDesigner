# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from data_designer.cli.repositories.model_repository import ModelConfigRegistry, ModelRepository
from data_designer.config.models import ModelConfig
from data_designer.config.utils.constants import MODEL_CONFIGS_FILE_NAME
from data_designer.config.utils.io_helpers import save_config_file


def test_config_file(tmp_path: Path):
    repository = ModelRepository(tmp_path)
    assert repository.config_file == tmp_path / MODEL_CONFIGS_FILE_NAME


def test_load_does_not_exist():
    repository = ModelRepository(Path("non_existent_path"))
    assert repository.load() is None


def test_load_exists(tmp_path: Path, stub_model_configs: list[ModelConfig]):
    model_configs_file_path = tmp_path / MODEL_CONFIGS_FILE_NAME
    save_config_file(model_configs_file_path, {"model_configs": [mc.model_dump() for mc in stub_model_configs]})
    repository = ModelRepository(tmp_path)
    assert repository.load() is not None
    assert repository.load().model_configs == stub_model_configs


def test_save(tmp_path: Path, stub_model_configs: list[ModelConfig]):
    repository = ModelRepository(tmp_path)
    repository.save(ModelConfigRegistry(model_configs=stub_model_configs))
    assert repository.load() is not None
    assert repository.load().model_configs == stub_model_configs
