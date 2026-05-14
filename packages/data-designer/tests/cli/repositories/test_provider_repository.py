# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from pathlib import Path

import pytest

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
    # Roundtrip test for the load/save cycle. We deliberately leave ``default``
    # unset so this test does not exercise the deprecated YAML ``default:`` path
    # — that path is covered by ``test_load_with_yaml_default_emits_deprecation_warning``
    # below. See issue #589.
    providers_file_path = tmp_path / MODEL_PROVIDERS_FILE_NAME
    save_config_file(
        providers_file_path,
        ModelProviderRegistry(providers=stub_model_providers).model_dump(exclude_none=True),
    )
    repository = ProviderRepository(tmp_path)
    assert repository.load() is not None
    assert repository.load().providers == stub_model_providers


def test_save(tmp_path: Path, stub_model_providers: list[ModelProvider]):
    # As above, leave ``default`` unset so the roundtrip stays clear of the
    # YAML-default deprecation. See issue #589.
    repository = ProviderRepository(tmp_path)
    repository.save(ModelProviderRegistry(providers=stub_model_providers))
    assert repository.load() is not None
    assert repository.load().providers == stub_model_providers


def test_load_with_yaml_default_emits_deprecation_warning(
    tmp_path: Path, stub_model_providers: list[ModelProvider]
) -> None:
    """Regression for #589: when the on-disk providers YAML carries a non-None
    ``default:`` key, ``ProviderRepository.load`` must emit a
    ``DeprecationWarning`` so users see the migration nudge regardless of which
    entry point reads the file.
    """
    providers_file_path = tmp_path / MODEL_PROVIDERS_FILE_NAME
    save_config_file(
        providers_file_path,
        ModelProviderRegistry(providers=stub_model_providers, default=stub_model_providers[0].name).model_dump(),
    )
    repository = ProviderRepository(tmp_path)

    with pytest.warns(DeprecationWarning, match="'default:' key.*is deprecated"):
        registry = repository.load()
    assert registry is not None
    assert registry.default == stub_model_providers[0].name


def test_load_with_yaml_default_attributes_warning_to_caller(
    tmp_path: Path, stub_model_providers: list[ModelProvider]
) -> None:
    """Regression for PR #594 review: the YAML-default ``DeprecationWarning``
    must attribute to the *caller's* frame (this test file), not to a
    ``data_designer.cli.*`` library frame. Library-attributed
    ``DeprecationWarning`` entries fall under Python's default
    ``ignore::DeprecationWarning`` filter and are silenced, so attribution at
    a library frame == invisible warning. ``warn_at_caller`` keeps this
    visible; a regression to ``warnings.warn(stacklevel=2)`` would land on
    ``provider_repository.py`` and silently break the user nudge.
    """
    providers_file_path = tmp_path / MODEL_PROVIDERS_FILE_NAME
    save_config_file(
        providers_file_path,
        ModelProviderRegistry(providers=stub_model_providers, default=stub_model_providers[0].name).model_dump(),
    )
    repository = ProviderRepository(tmp_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        repository.load()

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1
    assert deprecations[0].filename == __file__


def test_load_without_yaml_default_does_not_warn(tmp_path: Path, stub_model_providers: list[ModelProvider]) -> None:
    """Pin the post-deprecation happy path: a YAML without a ``default:`` key
    must load cleanly with no ``DeprecationWarning``.
    """
    providers_file_path = tmp_path / MODEL_PROVIDERS_FILE_NAME
    save_config_file(
        providers_file_path,
        ModelProviderRegistry(providers=stub_model_providers).model_dump(exclude_none=True),
    )
    repository = ProviderRepository(tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        registry = repository.load()
    assert registry is not None
    assert registry.default is None
