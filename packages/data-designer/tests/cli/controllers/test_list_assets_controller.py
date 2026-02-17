# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_designer.cli.controllers.list_assets_controller import ListAssetsController

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def controller(tmp_path: Path) -> ListAssetsController:
    """Create a controller with no datasets installed."""
    return ListAssetsController(tmp_path)


@pytest.fixture
def controller_with_datasets(tmp_path: Path) -> ListAssetsController:
    """Create a controller with en_US and ja_JP already installed."""
    managed = tmp_path / "managed-assets" / "datasets"
    managed.mkdir(parents=True)
    (managed / "en_US.parquet").touch()
    (managed / "ja_JP.parquet").touch()
    return ListAssetsController(tmp_path)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


def test_init(tmp_path: Path) -> None:
    """Controller sets up repository and service."""
    ctrl = ListAssetsController(tmp_path)
    assert ctrl.persona_repository is not None
    assert ctrl.service.config_dir == tmp_path


# ---------------------------------------------------------------------------
# text format
# ---------------------------------------------------------------------------


def test_text_none_installed(controller: ListAssetsController, capsys: pytest.CaptureFixture[str]) -> None:
    """Text output shows no-installed message when nothing is downloaded."""
    controller.list_assets("text")
    out = capsys.readouterr().out

    assert "Nemotron-Persona Datasets" in out
    assert "No persona datasets installed." in out
    assert "Not installed:" in out
    assert "The user can run" in out


def test_text_some_installed(
    controller_with_datasets: ListAssetsController, capsys: pytest.CaptureFixture[str]
) -> None:
    """Text output lists usable locales and not-installed ones."""
    controller_with_datasets.list_assets("text")
    out = capsys.readouterr().out

    assert "Usable locales in PersonSamplerParams:" in out
    assert "en_US" in out
    assert "ja_JP" in out
    assert "Not installed:" in out


def test_text_all_installed_omits_not_installed_section(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """When every locale is installed the not-installed section is omitted."""
    managed = tmp_path / "managed-assets" / "datasets"
    managed.mkdir(parents=True)
    ctrl = ListAssetsController(tmp_path)
    for locale in ctrl.persona_repository.list_all():
        (managed / f"{locale.code}.parquet").touch()

    ctrl.list_assets("text")
    out = capsys.readouterr().out

    assert "Usable locales in PersonSamplerParams:" in out
    assert "Not installed" not in out


# ---------------------------------------------------------------------------
# json format
# ---------------------------------------------------------------------------


def test_json_structure(controller: ListAssetsController, capsys: pytest.CaptureFixture[str]) -> None:
    """JSON output has the expected keys and types."""
    controller.list_assets("json")
    data = json.loads(capsys.readouterr().out)

    assert isinstance(data["installed"], list)
    assert isinstance(data["not_installed"], list)


def test_json_partitions_correctly(
    controller_with_datasets: ListAssetsController, capsys: pytest.CaptureFixture[str]
) -> None:
    """JSON output places downloaded locales in installed and the rest in not_installed."""
    controller_with_datasets.list_assets("json")
    data = json.loads(capsys.readouterr().out)

    assert "en_US" in data["installed"]
    assert "ja_JP" in data["installed"]
    assert "en_US" not in data["not_installed"]
    assert "ja_JP" not in data["not_installed"]
    assert len(data["installed"]) + len(data["not_installed"]) == len(
        controller_with_datasets.persona_repository.list_all()
    )


def test_json_none_installed(controller: ListAssetsController, capsys: pytest.CaptureFixture[str]) -> None:
    """JSON output when nothing is installed."""
    controller.list_assets("json")
    data = json.loads(capsys.readouterr().out)

    assert data["installed"] == []
    assert len(data["not_installed"]) > 0
