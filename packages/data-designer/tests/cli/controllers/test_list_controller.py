# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data_designer.cli.controllers.list_controller import ListController

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def controller(tmp_path: Path) -> ListController:
    """Controller with no datasets installed and no model configs."""
    return ListController(tmp_path)


@pytest.fixture
def controller_with_datasets(tmp_path: Path) -> ListController:
    """Controller with en_US and ja_JP persona datasets installed."""
    managed = tmp_path / "managed-assets" / "datasets"
    managed.mkdir(parents=True)
    (managed / "en_US.parquet").touch()
    (managed / "ja_JP.parquet").touch()
    return ListController(tmp_path)


@pytest.fixture
def controller_all_installed(tmp_path: Path) -> ListController:
    """Controller with ALL managed persona datasets installed."""
    ctrl = ListController(tmp_path)
    managed = tmp_path / "managed-assets" / "datasets"
    managed.mkdir(parents=True)
    for locale in ctrl._persona_repository.list_all():
        (managed / f"{locale.code}.parquet").touch()
    return ctrl


def _make_model_config(alias: str, model: str, provider: str | None = None) -> MagicMock:
    mc = MagicMock()
    mc.alias = alias
    mc.model = model
    mc.provider = provider
    return mc


def _make_provider(name: str, api_key: str | None = "sk-valid-key") -> MagicMock:
    p = MagicMock()
    p.name = name
    p.api_key = api_key
    return p


def _make_provider_registry(
    providers: list[MagicMock],
    default: str | None = None,
) -> MagicMock:
    registry = MagicMock()
    registry.providers = providers
    registry.default = default
    return registry


# ---------------------------------------------------------------------------
# list_model_aliases — text
# ---------------------------------------------------------------------------


def test_model_aliases_text_empty(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ctrl = ListController(tmp_path)
    provider_reg = _make_provider_registry([_make_provider("nvidia")])
    model_reg = MagicMock()
    model_reg.model_configs = []
    with (
        patch.object(ctrl._provider_repository, "load", return_value=provider_reg),
        patch.object(ctrl._model_repository, "load", return_value=model_reg),
    ):
        ctrl.list_model_aliases()
    out = capsys.readouterr().out
    assert "No model aliases configured." in out
    assert "data-designer config models" in out


def test_model_aliases_text_with_models(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ctrl = ListController(tmp_path)
    provider_reg = _make_provider_registry(
        [_make_provider("nvidia"), _make_provider("openai")],
        default="nvidia",
    )
    model_reg = MagicMock()
    model_reg.model_configs = [
        _make_model_config("my-model", "meta/llama-3.1-8b-instruct", "nvidia"),
        _make_model_config("judge", "openai/gpt-4o", None),
    ]
    with (
        patch.object(ctrl._provider_repository, "load", return_value=provider_reg),
        patch.object(ctrl._model_repository, "load", return_value=model_reg),
    ):
        ctrl.list_model_aliases()
    out = capsys.readouterr().out
    assert "my-model" in out
    assert "meta/llama-3.1-8b-instruct" in out
    assert "nvidia" in out
    assert "judge" in out
    assert "default" in out


def test_model_aliases_text_empty_model_configs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ctrl = ListController(tmp_path)
    provider_reg = _make_provider_registry([_make_provider("nvidia")])
    model_reg = MagicMock()
    model_reg.model_configs = []
    with (
        patch.object(ctrl._provider_repository, "load", return_value=provider_reg),
        patch.object(ctrl._model_repository, "load", return_value=model_reg),
    ):
        ctrl.list_model_aliases()
    out = capsys.readouterr().out
    assert "No model aliases configured." in out


# ---------------------------------------------------------------------------
# list_model_aliases — provider validation (text)
# ---------------------------------------------------------------------------


def test_model_aliases_text_no_provider_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ctrl = ListController(tmp_path)
    with patch.object(ctrl._provider_repository, "load", return_value=None):
        ctrl.list_model_aliases()
    out = capsys.readouterr().out
    assert "No model providers configured" in out
    assert "data-designer config models" in out


def test_model_aliases_text_empty_providers(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ctrl = ListController(tmp_path)
    provider_reg = _make_provider_registry([])
    with patch.object(ctrl._provider_repository, "load", return_value=provider_reg):
        ctrl.list_model_aliases()
    out = capsys.readouterr().out
    assert "No model providers configured" in out


def test_model_aliases_text_all_providers_missing_keys(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ctrl = ListController(tmp_path)
    provider_reg = _make_provider_registry(
        [
            _make_provider("nvidia", api_key=None),
            _make_provider("openai", api_key=None),
        ]
    )
    with patch.object(ctrl._provider_repository, "load", return_value=provider_reg):
        ctrl.list_model_aliases()
    out = capsys.readouterr().out
    assert "No model providers are configured with valid API keys" in out
    assert "data-designer config models" in out


def test_model_aliases_text_filters_by_provider(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ctrl = ListController(tmp_path)
    provider_reg = _make_provider_registry(
        [_make_provider("nvidia", api_key="sk-valid"), _make_provider("openai", api_key=None)],
        default="nvidia",
    )
    model_reg = MagicMock()
    model_reg.model_configs = [
        _make_model_config("nv-model", "meta/llama-3.1-8b-instruct", "nvidia"),
        _make_model_config("oai-model", "openai/gpt-4o", "openai"),
    ]
    with (
        patch.object(ctrl._provider_repository, "load", return_value=provider_reg),
        patch.object(ctrl._model_repository, "load", return_value=model_reg),
    ):
        ctrl.list_model_aliases()
    out = capsys.readouterr().out
    assert "nv-model" in out
    assert "oai-model" not in out


def test_model_aliases_text_default_provider_resolution(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Model with provider=None resolves to default provider for filtering."""
    ctrl = ListController(tmp_path)
    provider_reg = _make_provider_registry(
        [_make_provider("nvidia", api_key="sk-valid"), _make_provider("openai", api_key=None)],
        default="nvidia",
    )
    model_reg = MagicMock()
    model_reg.model_configs = [
        _make_model_config("my-model", "meta/llama-3.1-8b-instruct", None),
    ]
    with (
        patch.object(ctrl._provider_repository, "load", return_value=provider_reg),
        patch.object(ctrl._model_repository, "load", return_value=model_reg),
    ):
        ctrl.list_model_aliases()
    out = capsys.readouterr().out
    assert "my-model" in out
    assert "default" in out


def test_model_aliases_text_default_provider_resolution_excluded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Model with provider=None is excluded when default provider lacks a valid key."""
    ctrl = ListController(tmp_path)
    provider_reg = _make_provider_registry(
        [_make_provider("nvidia", api_key=None), _make_provider("openai", api_key="sk-valid")],
        default="nvidia",
    )
    model_reg = MagicMock()
    model_reg.model_configs = [
        _make_model_config("my-model", "meta/llama-3.1-8b-instruct", None),
        _make_model_config("oai-model", "openai/gpt-4o", "openai"),
    ]
    with (
        patch.object(ctrl._provider_repository, "load", return_value=provider_reg),
        patch.object(ctrl._model_repository, "load", return_value=model_reg),
    ):
        ctrl.list_model_aliases()
    out = capsys.readouterr().out
    assert "my-model" not in out
    assert "oai-model" in out


def test_model_aliases_text_all_models_filtered(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ctrl = ListController(tmp_path)
    provider_reg = _make_provider_registry(
        [_make_provider("nvidia", api_key="sk-valid"), _make_provider("openai", api_key=None)],
        default="nvidia",
    )
    model_reg = MagicMock()
    model_reg.model_configs = [
        _make_model_config("oai-model", "openai/gpt-4o", "openai"),
    ]
    with (
        patch.object(ctrl._provider_repository, "load", return_value=provider_reg),
        patch.object(ctrl._model_repository, "load", return_value=model_reg),
    ):
        ctrl.list_model_aliases()
    out = capsys.readouterr().out
    assert "All configured model aliases use providers without valid API keys" in out
    assert "data-designer config models" in out


def test_model_aliases_text_default_from_first_provider(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """When provider_registry.default is None, first provider is used as default."""
    ctrl = ListController(tmp_path)
    provider_reg = _make_provider_registry(
        [_make_provider("nvidia", api_key="sk-valid")],
        default=None,
    )
    model_reg = MagicMock()
    model_reg.model_configs = [
        _make_model_config("my-model", "meta/llama-3.1-8b-instruct", None),
    ]
    with (
        patch.object(ctrl._provider_repository, "load", return_value=provider_reg),
        patch.object(ctrl._model_repository, "load", return_value=model_reg),
    ):
        ctrl.list_model_aliases()
    out = capsys.readouterr().out
    assert "my-model" in out


def test_model_aliases_env_var_api_key_set(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Provider whose api_key names an env var that IS set should be treated as valid."""
    ctrl = ListController(tmp_path)
    provider_reg = _make_provider_registry(
        [_make_provider("nvidia", api_key="NVIDIA_API_KEY")],
        default="nvidia",
    )
    model_reg = MagicMock()
    model_reg.model_configs = [
        _make_model_config("nv-model", "meta/llama-3.1-8b-instruct", "nvidia"),
    ]
    with (
        patch.object(ctrl._provider_repository, "load", return_value=provider_reg),
        patch.object(ctrl._model_repository, "load", return_value=model_reg),
        patch.dict(os.environ, {"NVIDIA_API_KEY": "real-key"}),
    ):
        ctrl.list_model_aliases()
    out = capsys.readouterr().out
    assert "nv-model" in out


def test_model_aliases_env_var_api_key_unset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Provider whose api_key names an env var that is NOT set should be invalid."""
    ctrl = ListController(tmp_path)
    provider_reg = _make_provider_registry(
        [_make_provider("nvidia", api_key="NVIDIA_API_KEY")],
        default="nvidia",
    )
    model_reg = MagicMock()
    model_reg.model_configs = [
        _make_model_config("nv-model", "meta/llama-3.1-8b-instruct", "nvidia"),
    ]
    env = {k: v for k, v in os.environ.items() if k != "NVIDIA_API_KEY"}
    with (
        patch.object(ctrl._provider_repository, "load", return_value=provider_reg),
        patch.object(ctrl._model_repository, "load", return_value=model_reg),
        patch.dict(os.environ, env, clear=True),
    ):
        ctrl.list_model_aliases()
    out = capsys.readouterr().out
    assert "No model providers are configured with valid API keys" in out


def test_model_aliases_model_registry_returns_none(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """When _model_repository.load() returns None, show 'No model aliases configured.'"""
    ctrl = ListController(tmp_path)
    provider_reg = _make_provider_registry([_make_provider("nvidia")])
    with (
        patch.object(ctrl._provider_repository, "load", return_value=provider_reg),
        patch.object(ctrl._model_repository, "load", return_value=None),
    ):
        ctrl.list_model_aliases()
    out = capsys.readouterr().out
    assert "No model aliases configured." in out


def test_model_aliases_multiple_models_same_provider(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """All models on a single valid provider should appear in the output."""
    ctrl = ListController(tmp_path)
    provider_reg = _make_provider_registry(
        [_make_provider("nvidia", api_key="sk-valid")],
        default="nvidia",
    )
    model_reg = MagicMock()
    model_reg.model_configs = [
        _make_model_config("model-a", "meta/llama-3.1-8b-instruct", "nvidia"),
        _make_model_config("model-b", "meta/llama-3.1-70b-instruct", "nvidia"),
        _make_model_config("model-c", "nvidia/nemotron-4-340b", "nvidia"),
    ]
    with (
        patch.object(ctrl._provider_repository, "load", return_value=provider_reg),
        patch.object(ctrl._model_repository, "load", return_value=model_reg),
    ):
        ctrl.list_model_aliases()
    out = capsys.readouterr().out
    assert "model-a" in out
    assert "model-b" in out
    assert "model-c" in out


def test_model_aliases_filtered_count_hint(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Output should contain a hint about how many aliases were hidden by filtering."""
    ctrl = ListController(tmp_path)
    provider_reg = _make_provider_registry(
        [_make_provider("nvidia", api_key="sk-valid"), _make_provider("openai", api_key=None)],
        default="nvidia",
    )
    model_reg = MagicMock()
    model_reg.model_configs = [
        _make_model_config("nv-model", "meta/llama-3.1-8b-instruct", "nvidia"),
        _make_model_config("oai-model", "openai/gpt-4o", "openai"),
    ]
    with (
        patch.object(ctrl._provider_repository, "load", return_value=provider_reg),
        patch.object(ctrl._model_repository, "load", return_value=model_reg),
    ):
        ctrl.list_model_aliases()
    out = capsys.readouterr().out
    assert "nv-model" in out
    assert "oai-model" not in out
    assert "1 model alias(es) hidden" in out


# ---------------------------------------------------------------------------
# list_persona_datasets — text
# ---------------------------------------------------------------------------


def test_persona_datasets_text_none_installed(controller: ListController, capsys: pytest.CaptureFixture[str]) -> None:
    controller.list_persona_datasets()
    out = capsys.readouterr().out
    assert "Nemotron-Personas Datasets" in out
    assert "not installed" in out


def test_persona_datasets_text_some_installed(
    controller_with_datasets: ListController, capsys: pytest.CaptureFixture[str]
) -> None:
    controller_with_datasets.list_persona_datasets()
    out = capsys.readouterr().out
    assert "en_US" in out
    assert "installed" in out
    assert "ja_JP" in out


def test_persona_datasets_text_all_installed(
    controller_all_installed: ListController, capsys: pytest.CaptureFixture[str]
) -> None:
    controller_all_installed.list_persona_datasets()
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    locale_lines = [line for line in lines if "installed" in line and "---" not in line and "status" not in line]
    assert len(locale_lines) > 0
    for line in locale_lines:
        assert "not installed" not in line


# ---------------------------------------------------------------------------
# list_column_types — text
# ---------------------------------------------------------------------------


def test_column_types_text(controller: ListController, capsys: pytest.CaptureFixture[str]) -> None:
    controller.list_column_types()
    out = capsys.readouterr().out
    assert "column_type" in out
    assert "config_class" in out
    assert "llm-text" in out
    assert "sampler" in out
    assert "data-designer inspect column" in out


# ---------------------------------------------------------------------------
# list_sampler_types — text
# ---------------------------------------------------------------------------


def test_sampler_types_text(controller: ListController, capsys: pytest.CaptureFixture[str]) -> None:
    controller.list_sampler_types()
    out = capsys.readouterr().out
    assert "sampler_type" in out
    assert "params_class" in out
    assert "category" in out
    assert "data-designer inspect sampler" in out


# ---------------------------------------------------------------------------
# list_validator_types — text
# ---------------------------------------------------------------------------


def test_validator_types_text(controller: ListController, capsys: pytest.CaptureFixture[str]) -> None:
    controller.list_validator_types()
    out = capsys.readouterr().out
    assert "validator_type" in out
    assert "params_class" in out
    assert "data-designer inspect validator" in out


# ---------------------------------------------------------------------------
# list_processor_types — text
# ---------------------------------------------------------------------------


def test_processor_types_text(controller: ListController, capsys: pytest.CaptureFixture[str]) -> None:
    controller.list_processor_types()
    out = capsys.readouterr().out
    assert "processor_type" in out
    assert "config_class" in out
    assert "data-designer inspect processor" in out
