# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from data_designer.cli.repositories.provider_repository import ModelProviderRegistry, ProviderRepository
from data_designer.config.models import ModelProvider
from data_designer.config.utils.constants import MODEL_PROVIDERS_FILE_NAME
from data_designer.config.utils.io_helpers import save_config_file


def test_config_file(tmp_path: Path):
    repository = ProviderRepository(tmp_path)
    assert repository.config_file == tmp_path / MODEL_PROVIDERS_FILE_NAME


def test_load_does_not_exist():
    repository = ProviderRepository(Path("non_existent_path"))
    assert repository.load() is None


def test_load_exists(tmp_path: Path, stub_model_providers: list[ModelProvider]):
    providers_file_path = tmp_path / MODEL_PROVIDERS_FILE_NAME
    save_config_file(
        providers_file_path,
        ModelProviderRegistry(providers=stub_model_providers).model_dump(exclude_none=True),
    )
    repository = ProviderRepository(tmp_path)
    assert repository.load() is not None
    assert repository.load().providers == stub_model_providers


def test_save(tmp_path: Path, stub_model_providers: list[ModelProvider]):
    repository = ProviderRepository(tmp_path)
    repository.save(ModelProviderRegistry(providers=stub_model_providers))
    assert repository.load() is not None
    assert repository.load().providers == stub_model_providers


def test_load_silently_ignores_legacy_default_key(tmp_path: Path, stub_model_providers: list[ModelProvider]) -> None:
    """Regression for #590: pydantic v2's ``extra="ignore"`` default silently
    drops the legacy ``default:`` key from older on-disk YAMLs, so existing
    user configs continue to load cleanly after the field was removed.
    """
    providers_file_path = tmp_path / MODEL_PROVIDERS_FILE_NAME
    save_config_file(
        providers_file_path,
        {
            "providers": [p.model_dump() for p in stub_model_providers],
            "default": stub_model_providers[0].name,
        },
    )
    repository = ProviderRepository(tmp_path)
    registry = repository.load()
    assert registry is not None
    assert registry.providers == stub_model_providers
    assert not hasattr(registry, "default")
